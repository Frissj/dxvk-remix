/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

// NV-DXVK: This file originates from nvpro-samples/vk_lod_clusters
// (src/resources.hpp) and was trimmed for the RTX Remix integration:
// - The sample's viewer subsystems are removed (swapchain framebuffer, color/
//   depth targets, HBAO, DLSS denoiser/upscaler, background/bbox rendering).
//   Remix owns the frame; only the HiZ images used for occlusion culling
//   remain in the reduced FrameBuffer struct.
// - Runtime shaderc compilation is replaced by a prebuilt-SPIR-V variant
//   lookup (see nvpro_shims/nvvkglsl/glsl.hpp). All shader-facing signatures
//   are unchanged.
// Everything kept below is unchanged from the sample wherever possible so
// the ported scene/streaming/renderer code compiles against the exact same
// API.

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <span>

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/physical_device.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>
#include <vk_radix_sort.h>

#if VK_HEADER_VERSION < 309
#error Update Vulkan SDK >= 1.4.309.0
#endif

#include "nvhiz_vk.hpp"
#include "shaderio.h"

namespace lodclusters {

struct FrameConfig
{
  VkExtent2D windowSize;

  bool  showInstanceBboxes = false;
  bool  showClusterBboxes  = false;
  bool  freezeCulling      = false;
  bool  freezeLoD          = false;
  float lodPixelError      = 1.0f;
  // increase error by this for instances not having primary visibility in ray tracing
  float culledErrorScale = 2.0f;
  // if less pixels than this, use sw raster
  float swRasterThreshold = 8.0f;

  // how many frames until we schedule a group for unloading
  uint32_t streamingAgeThreshold = 16;

  // how much threads to use in the persistent kernels
  uint32_t traversalPersistentThreads = 2048;

  uint32_t sharingTolerantLevels = 7;
  uint32_t sharingEnabledLevels  = 8;
  bool     sharingPushCulled     = true;

  uint32_t cachingEnabledLevels = 8;
  uint32_t cachingAgeThreshold  = 16;

  uint32_t visualize = VISUALIZE_LOD;

  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
  float                    traversalFov;
  float                    traversalViewHeight;
  glm::mat4                traversalViewMatrix;
  glm::mat4                cullViewProjMatrix;
  glm::mat4                cullViewProjMatrixLast;
};

//////////////////////////////////////////////////////////////////////////

inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

std::string formatMemorySize(size_t sizeInBytes);

// NV-DXVK: process-wide secondary device-lost dump. The animated template
// system (Path B) registers its forensic dump here; the LOD renderer's
// primary dump chains into it so one device-lost flushes both captures no
// matter which system's thread notices the loss first.
std::function<void()>& deviceLostAuxDumpFn();

// NV-DXVK: same dump, reachable from DxvkSubmissionQueue's device-lost site.
// When no lodclusters temp submit is in flight at the loss, neither the
// tempSyncSubmit fence wait nor NVVK_CHECK ever observes it - the dxvk submit
// thread is then the ONLY observer, and it must be able to flush [BlasCapture].
std::function<void()>& deviceLostQueueDumpFn();

// NV-DXVK: [HeadWatch] registry of AS memory pools (Path-A per-frame BLAS ring
// slots, low-detail CLAS/BLAS batches, ...) so AccelManager's per-frame scan
// can re-read the first bytes AT every TLAS cluster reference each frame.
// [BlasHeadScan] proved the heads are valid at assign time; the fault appears
// frames later - this watch catches the frame the content gets stomped.
// Thread-safe: registration from worker threads, lookup from the CS thread.
struct WatchedAsPool
{
  VkBuffer buffer  = VK_NULL_HANDLE;
  uint64_t address = 0;
  uint64_t size    = 0;
};
void registerWatchedAsPool(VkBuffer buffer, uint64_t address, uint64_t size);
void unregisterWatchedAsPool(VkBuffer buffer);
// snapshot copy for race-free iteration
std::vector<WatchedAsPool> getWatchedAsPools();

// NV-DXVK: label this thread's next temp op(s) so [TempSubmit] names which
// operation submitted the faulting work (device-lost forensics).
void dbgSetTempLabel(const char* label);

inline size_t logMemoryUsage(size_t size, const char* memtype, const char* what)
{
  LOGI("%s memory: %s - %s\n", memtype, formatMemorySize(size).c_str(), what);
  return size;
}

//////////////////////////////////////////////////////////////////////////

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvutils::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }

  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }
  void splitOverlap()
  {
    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};

//////////////////////////////////////////////////////////////////////////

class QueueState
{
public:
  VkDevice    m_device            = nullptr;
  VkQueue     m_queue             = nullptr;
  uint32_t    m_familyIndex       = 0;
  VkSemaphore m_timelineSemaphore = nullptr;
  uint64_t    m_timelineValue     = 1;

  std::vector<VkSemaphoreSubmitInfo> m_pendingWaits;

  void init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue);
  void deinit();

  VkResult getTimelineValue(uint64_t& timelineValue) const
  {
    return vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &timelineValue);
  }

  nvvk::SemaphoreState getCurrentState() const
  {
    return nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineValue);
  }

  VkSemaphoreSubmitInfo getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0) const;

  // increments timeline
  VkSemaphoreSubmitInfo advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0);
};

struct QueueStateManager
{
  QueueState primary;
  QueueState transfer;
};

//////////////////////////////////////////////////////////////////////////

class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

  // NV-DXVK: reduced from the sample's full swapchain framebuffer to the
  // pieces the cluster system itself consumes. Remix owns color/depth; the
  // HiZ far images are built from Remix's previous-frame primary depth.
  struct FrameBuffer
  {
    VkExtent2D renderSize{};
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};

    glm::vec2 renderScale = {1.0f, 1.0f};
    float     pixelScale  = 1;

    nvvk::Image imgHizFar[2] = {};

    // NV-DXVK P4: reversed-Z copy of Remix's previous-frame primary depth,
    // written by the remix_depth_flip kernel and consumed as the NVHizVK
    // source (replaces the sample's own depth-stencil attachment)
    nvvk::Image imgHizSource = {};
  };

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer);
  void deinit();

  glm::vec2 getFramebufferWindow2RenderScale() const;

  void beginFrame(uint32_t cycleIndex);
  void endFrame();

  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx);

  // NV-DXVK P4: HiZ occlusion feed from Remix's previous-frame primary depth.
  // hizResolutionDiffers/ensureHizResources manage the render-resolution-sized
  // HiZ source + far pyramid (ensure waits for device idle when recreating -
  // the caller must hold Remix's submission lock); cmdBuildHizFromDepth
  // records depth conversion (remix_depth_flip: game convention -> reversed-Z)
  // followed by NVHizVK's far-mip build.
  bool hizResolutionDiffers(uint32_t width, uint32_t height) const
  {
    return m_frameBuffer.renderSize.width != width || m_frameBuffer.renderSize.height != height
           || !m_frameBuffer.imgHizSource.image;
  }
  void ensureHizResources(uint32_t width, uint32_t height);
  void cmdBuildHizFromDepth(VkCommandBuffer cmd, VkImageView sourceDepthView, bool flipDepth, nvvk::ProfilerGpuTimer& profiler);

  // some vulkan implementations only support 16 bit per grid component
  // need to convert the 1D intended launch into a grid.
  void cmdLinearDispatch(VkCommandBuffer cmd, uint32_t count) const
  {
    if(!count)
      return;

    if(!m_use16bitDispatch || count <= 0xFFFF)
    {
      vkCmdDispatch(cmd, count, 1, 1);
    }
    else
    {
      glm::uvec3 grid = shaderio::fit16bitLaunchGrid(count);
      assert(grid.x <= 0xFFFF && grid.y <= 0xFFFF && grid.z <= 0xFFFF);
      vkCmdDispatch(cmd, grid.x, grid.y, grid.z);
    }
  }

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  // NV-DXVK: resolves to a build-time compiled shader variant instead of
  // invoking shaderc; signature unchanged from the sample.
  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options = nullptr);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, shaderc::SpvCompilationResult* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(shaderc::SpvCompilationResult), (shaderc::SpvCompilationResult*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  template <typename T>
  VkResult createBufferTyped(nvvk::BufferTyped<T>&     buffer,
                             size_t                    elementCount,
                             VkBufferUsageFlagBits2    bufferUsageFlags,
                             VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                             VmaAllocationCreateFlags  vmaAllocFlags = 0,
                             VkDeviceSize              minAlignment  = 0,
                             std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, elementCount * nvvk::BufferTyped<T>::value_size, bufferUsageFlags,
                                    vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createBuffer(nvvk::Buffer&             buffer,
                        VkDeviceSize              bufferSize,
                        VkBufferUsageFlagBits2    bufferUsageFlags,
                        VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                        VmaAllocationCreateFlags  vmaAllocFlags = 0,
                        VkDeviceSize              minAlignment  = 0,
                        std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, bufferSize, bufferUsageFlags, vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createLargeBuffer(nvvk::LargeBuffer& buffer, VkDeviceSize bufferSize, VkBufferUsageFlagBits2 bufferUsageFlags)
  {
    return m_allocator.createLargeBuffer(buffer, bufferSize, bufferUsageFlags, m_queue.queue);
  }

  VkDeviceSize getDeviceLocalHeapSize() const;

  bool isBufferSizeValid(VkDeviceSize size) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, offset, sz, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class BatchedUploader
  {
  public:
    BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
        : m_resources(resources)
        , m_maxBatchSize(maxBatchSize)
    {
    }

    VkCommandBuffer getCmd()
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      return m_cmd;
    }

    void checkedFlush(size_t sz)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz))
        {
          flush();
        }
      }
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        if(!m_cmd)
        {
          m_cmd = m_resources.createTempCmdBuffer();
        }
        T* mapping = nullptr;
        NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

        if(src)
        {
          memcpy(mapping, src, sz);
        }

        return mapping;
      }
      return nullptr;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
    }

    void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.m_uploader.cmdUploadAppended(m_cmd);
        m_resources.tempSyncSubmit(m_cmd);
        m_resources.m_uploader.releaseStaging();
        m_cmd = nullptr;
      }
    }

    void abort()
    {
      m_resources.m_uploader.cancelAppended();
      m_resources.m_uploader.releaseStaging();
    }

    ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////

  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

  // NV-DXVK P4c: optional submission-lock callbacks. When set, tempSyncSubmit
  // takes them ONLY around its raw vkQueueSubmit2 - the fence wait runs
  // unlocked, so a caller no longer blocks dxvk's render-thread submissions
  // for the GPU duration of its temp work. When unset, callers must
  // externally synchronize queue access (the original P2 contract, still
  // used by the render system's generation builds).
  std::function<void()> submitLockFn;
  std::function<void()> submitUnlockFn;

  // NV-DXVK: fired once when tempSyncSubmit's fence wait returns
  // VK_ERROR_DEVICE_LOST - both observed streaming device-losts were noticed
  // here. The renderer hooks its BLAS-input forensic dump into this so the
  // faulting frame's captured build inputs reach the log before NVVK_CHECK
  // exit()s the process. The mutex keeps a concurrently-failing thread from
  // exiting mid-dump.
  std::function<void()> deviceLostDumpFn;
  std::mutex            m_deviceLostDumpMutex;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  nvvk::QueueInfo  m_queueTransfer   = {};
  VkCommandPool    m_tempCommandPool = {};
  // NV-DXVK: per-pool in-flight temp-op counter (device-lost race probe). Unlike
  // a global counter this is >1 ONLY when two threads share THIS pool.
  std::atomic<int> m_tempInFlight{0};

  nvvk::ResourceAllocator m_allocator        = {};
  nvvk::SamplerPool       m_samplerPool      = {};
  VkSampler               m_samplerBiLinear  = {};
  VkSampler               m_samplerTriLinear = {};
  nvvkglsl::GlslCompiler  m_glslCompiler     = {};
  nvvk::StagingUploader   m_uploader         = {};

  FrameBuffer m_frameBuffer;
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  nvvk::PhysicalDeviceInfo         m_physicalDeviceInfo = {};
  VkPhysicalDeviceMemoryProperties m_memoryProperties   = {};
  uint32_t                         m_cycleIndex         = 0;

  bool m_use16bitDispatch          = false;
  bool m_supportsClusterRaytracing = false;

  NVHizVK                       m_hiz;
  NVHizVK::Update               m_hizUpdate[2];
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

  // NV-DXVK P4: depth-conversion pass feeding the HiZ source (Remix glue).
  // The descriptor sets form a ring over the frame cycle so a set that an
  // in-flight frame still references is never rewritten.
  static constexpr uint32_t kDepthFlipSets = 4;

  shaderc::SpvCompilationResult m_depthFlipShader;
  VkDescriptorSetLayout         m_depthFlipSetLayout = {};
  VkDescriptorPool              m_depthFlipPool = {};
  VkDescriptorSet               m_depthFlipSets[kDepthFlipSets] = {};
  VkPipelineLayout              m_depthFlipPipelineLayout = {};
  VkPipeline                    m_depthFlipPipeline = {};

  QueueStateManager m_queueStates;
  VrdxSorter        m_vrdxSorter{};

private:
};


}  // namespace lodclusters
