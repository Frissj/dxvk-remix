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
// (src/resources.cpp, with formatMemorySize transplanted from
// lodclusters_ui.cpp) and was trimmed for the RTX Remix integration.
// Removed: swapchain framebuffer management, HBAO, DLSS, viewer rendering
// helpers, shaderc search-path setup. compileShader resolves prebuilt
// build-time variants (see nvpro_shims/nvvkglsl/glsl.hpp). All kept function
// bodies are unchanged from the sample.

#include <algorithm>

#include <volk.h>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <fmt/format.h>

#include "resources.hpp"

namespace lodclusters {

std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

void Resources::beginFrame(uint32_t cycleIndex)
{
  m_cycleIndex = cycleIndex;
}

void Resources::endFrame() {}

void Resources::init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer)
{
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_queue          = queue;
  m_queueTransfer  = queueTransfer;

  m_physicalDeviceInfo.init(physicalDevice);
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &m_memoryProperties);

  m_use16bitDispatch = m_physicalDeviceInfo.properties10.limits.maxComputeWorkGroupCount[0] < (1 << 30);

  {
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physicalDevice,
        .device         = device,
        .instance       = instance,
    };

    NVVK_CHECK(m_allocator.init(allocatorInfo));
  }

  m_uploader.init(&m_allocator);

  m_samplerPool.init(device);
  m_samplerPool.acquireSampler(m_samplerBiLinear);

  VkSamplerCreateInfo samplerCreateInfo = DEFAULT_VkSamplerCreateInfo;
  m_samplerPool.acquireSampler(m_samplerTriLinear, samplerCreateInfo);

  // temp command pool
  {
    VkCommandPoolCreateInfo createInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_queue.familyIndex,
    };

    NVVK_CHECK(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_tempCommandPool));
  }

  // common resources
  {
    m_allocator.createBuffer(m_commonBuffers.frameConstants, sizeof(shaderio::FrameConstants),
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    m_allocator.createBuffer(m_commonBuffers.readBack, sizeof(shaderio::Readback),
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_allocator.createBuffer(m_commonBuffers.readBackHost, sizeof(shaderio::Readback) * 4,
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                             VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  }

  {
    NVHizVK::Config config;
    config.msaaSamples             = 0;
    config.reversedZ               = true;
    config.supportsMinmaxFilter    = true;
    config.supportsSubGroupShuffle = true;
    m_hiz.init(m_device, config, 2);

    shaderc::SpvCompilationResult shaderResults[NVHizVK::SHADER_COUNT];
    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {
      shaderc::CompileOptions options = makeCompilerOptions();
      m_hiz.appendShaderDefines(i, options);
      compileShader(shaderResults[i], VK_SHADER_STAGE_COMPUTE_BIT, "nvhiz-update.comp.glsl", &options);
    }
    m_hiz.initPipelines(shaderResults);
  }
  {
    VrdxSorterCreateInfo sorterCreateInfo;
    sorterCreateInfo.device         = m_device;
    sorterCreateInfo.physicalDevice = m_physicalDevice;
    sorterCreateInfo.pipelineCache  = nullptr;

    vrdxCreateSorter(&sorterCreateInfo, &m_vrdxSorter);
  }
  {
    // NV-DXVK P4: depth-conversion pass (remix_depth_flip) feeding the HiZ
    // source image - Remix's primary depth carries the game's depth
    // convention, the cluster kernels assume the sample's reversed-Z.
    compileShader(m_depthFlipShader, VK_SHADER_STAGE_COMPUTE_BIT, "remix_depth_flip.comp.glsl", nullptr);

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings    = bindings;
    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_depthFlipSetLayout));

    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kDepthFlipSets},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kDepthFlipSets},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets       = kDepthFlipSets;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_depthFlipPool));

    VkDescriptorSetLayout setLayouts[kDepthFlipSets];
    for(uint32_t i = 0; i < kDepthFlipSets; i++)
    {
      setLayouts[i] = m_depthFlipSetLayout;
    }
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = m_depthFlipPool;
    allocInfo.descriptorSetCount = kDepthFlipSets;
    allocInfo.pSetLayouts        = setLayouts;
    NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, m_depthFlipSets));

    VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2 + sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_depthFlipSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_depthFlipPipelineLayout));

    VkShaderModuleCreateInfo shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_depthFlipShader);
    VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName = "main";
    compInfo.stage.pNext = &shaderInfo;
    compInfo.layout      = m_depthFlipPipelineLayout;
    NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_depthFlipPipeline));
  }
  {
    // NV-DXVK P4: bootstrap HiZ resources so the renderer's BINDINGS_HIZ_TEX
    // descriptor is valid before the first frame provides the real render
    // resolution (the bootstrap far image is cleared to 0 = reversed-Z far
    // plane and never falsely occludes; the first recordFrame resizes).
    ensureHizResources(64, 64);
  }
  {
    m_queueStates.primary.init(m_device, m_queue.queue, m_queue.familyIndex, 0);
    NVVK_DBG_NAME(m_queueStates.primary.m_timelineSemaphore);

    m_queueStates.transfer.init(m_device, m_queueTransfer.queue, m_queueTransfer.familyIndex, 0);
    NVVK_DBG_NAME(m_queueStates.transfer.m_timelineSemaphore);
  }
}

void Resources::deinit()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  m_allocator.destroyBuffer(m_commonBuffers.frameConstants);
  m_allocator.destroyBuffer(m_commonBuffers.readBack);
  m_allocator.destroyBuffer(m_commonBuffers.readBackHost);

  vkDestroyCommandPool(m_device, m_tempCommandPool, nullptr);

  // NV-DXVK P4: HiZ feed resources
  m_hiz.deinitUpdateViews(m_hizUpdate[0]);
  for(uint32_t i = 0; i < 2; i++)
  {
    if(m_frameBuffer.imgHizFar[i].image)
    {
      m_allocator.destroyImage(m_frameBuffer.imgHizFar[i]);
    }
  }
  if(m_frameBuffer.imgHizSource.image)
  {
    m_allocator.destroyImage(m_frameBuffer.imgHizSource);
  }

  vkDestroyPipeline(m_device, m_depthFlipPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_depthFlipPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_depthFlipPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_depthFlipSetLayout, nullptr);

  m_hiz.deinit();
  vrdxDestroySorter(m_vrdxSorter);
  m_queueStates.primary.deinit();
  m_queueStates.transfer.deinit();

  m_samplerPool.releaseSampler(m_samplerBiLinear);
  m_samplerPool.releaseSampler(m_samplerTriLinear);
  m_samplerPool.deinit();
  m_uploader.deinit();
  m_allocator.deinit();
}

glm::vec2 Resources::getFramebufferWindow2RenderScale() const
{
  return m_frameBuffer.renderScale;
}

void Resources::getReadbackData(shaderio::Readback& readback)
{
  const shaderio::Readback* pReadback = m_commonBuffers.readBackHost.data();
  readback                            = pReadback[m_cycleIndex];
}

void Resources::cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "HiZ");

  // NV-DXVK: the depth source is Remix's previous-frame primary depth; its
  // layout transition is handled by the caller (ClusterLodManager) before
  // this point, replacing the sample's own depth-stencil transition here.

  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate[idx], idx);
}

// NV-DXVK P4: (re)creates the HiZ source image and far pyramid for the given
// render resolution. Mirrors the sample's initFramebuffer HiZ section
// (resources.cpp ~620-650), with the sample's depth-stencil attachment
// replaced by our own R32F source image that remix_depth_flip writes.
void Resources::ensureHizResources(uint32_t width, uint32_t height)
{
  width  = std::max(width, 8u);
  height = std::max(height, 8u);

  if(!hizResolutionDiffers(width, height))
  {
    return;
  }

  // in-flight frames may still reference the previous images through the
  // renderer's HIZ descriptor and NVHizVK's sets (same reasoning as the
  // sample's updatedFrameBuffer, which also waits for idle on resize)
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  m_hiz.deinitUpdateViews(m_hizUpdate[0]);
  if(m_frameBuffer.imgHizFar[0].image)
  {
    m_allocator.destroyImage(m_frameBuffer.imgHizFar[0]);
  }
  if(m_frameBuffer.imgHizSource.image)
  {
    m_allocator.destroyImage(m_frameBuffer.imgHizSource);
  }

  m_frameBuffer.renderSize  = {width, height};
  m_frameBuffer.targetSize  = m_frameBuffer.renderSize;
  m_frameBuffer.windowSize  = m_frameBuffer.renderSize;
  m_frameBuffer.renderScale = {1.0f, 1.0f};
  m_frameBuffer.pixelScale  = 1.0f;

  m_hizUpdate[0] = NVHizVK::Update();
  m_hiz.setupUpdateInfos(m_hizUpdate[0], width, height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

  // reversed-Z copy of Remix's primary depth (written by remix_depth_flip)
  {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType   = VK_IMAGE_TYPE_2D;
    imageInfo.format      = VK_FORMAT_R32_SFLOAT;
    imageInfo.extent      = {width, height, 1};
    imageInfo.mipLevels   = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format           = imageInfo.format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgHizSource, imageInfo, viewInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgHizSource.image);
  }

  // far pyramid sized by NVHizVK (sample-identical creation)
  {
    VkImageCreateInfo hizImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    hizImageInfo.format            = m_hizUpdate[0].farInfo.format;
    hizImageInfo.extent.width      = m_hizUpdate[0].farInfo.width;
    hizImageInfo.extent.height     = m_hizUpdate[0].farInfo.height;
    hizImageInfo.mipLevels         = m_hizUpdate[0].farInfo.mipLevels;
    hizImageInfo.extent.depth      = 1;
    hizImageInfo.arrayLayers       = 1;
    hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    hizImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    hizImageInfo.flags = 0;
    hizImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgHizFar[0], hizImageInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgHizFar[0].image);
  }

  m_hizUpdate[0].sourceImage = m_frameBuffer.imgHizSource.image;
  m_hizUpdate[0].farImage    = m_frameBuffer.imgHizFar[0].image;
  m_hizUpdate[0].nearImage   = VK_NULL_HANDLE;

  m_hiz.initUpdateViews(m_hizUpdate[0]);
  m_hiz.updateDescriptorSet(m_hizUpdate[0], 0);

  {
    VkCommandBuffer cmd = createTempCmdBuffer();

    // far pyramid starts cleared to 0 = reversed-Z far plane: until real
    // depth arrives, every occlusion test passes and nothing is falsely
    // culled (the sample clears its HiZ images the same way)
    VkClearColorValue clear = {};
    VkImageSubresourceRange subResourceRange;
    subResourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subResourceRange.baseArrayLayer = 0;
    subResourceRange.baseMipLevel   = 0;
    subResourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
    subResourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;

    cmdImageTransition(cmd, m_frameBuffer.imgHizFar[0], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdClearColorImage(cmd, m_frameBuffer.imgHizFar[0].image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &subResourceRange);

    cmdImageTransition(cmd, m_frameBuffer.imgHizSource, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdClearColorImage(cmd, m_frameBuffer.imgHizSource.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &subResourceRange);
    // the layout NVHizVK's source descriptor expects; cmdBuildHizFromDepth
    // cycles GENERAL (flip write) -> SHADER_READ_ONLY (nvhiz read) per frame
    cmdImageTransition(cmd, m_frameBuffer.imgHizSource, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);

    tempSyncSubmit(cmd);
  }

  LOGI("lodclusters: HiZ resources %ux%u (far %ux%u, %u mips)\n", width, height, m_hizUpdate[0].farInfo.width,
       m_hizUpdate[0].farInfo.height, m_hizUpdate[0].farInfo.mipLevels);
}

// NV-DXVK P4: records the per-frame HiZ build - depth conversion (game
// convention -> reversed-Z) into imgHizSource, then NVHizVK's far-mip
// reduction. The source view is Remix's previous-frame primary depth (R32F
// color, GENERAL layout, render resolution); the caller skips this entirely
// when culling is frozen or no depth exists yet (far pyramid then keeps its
// last/cleared content, matching the sample's freezeCulling behavior).
void Resources::cmdBuildHizFromDepth(VkCommandBuffer cmd, VkImageView sourceDepthView, bool flipDepth, nvvk::ProfilerGpuTimer& profiler)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "HiZ");

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  // the depth was written by the previous frame's gbuffer pass on this same
  // queue (earlier submission) - make those writes visible to our read
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // ring slot for this frame cycle - a set an in-flight frame uses is never rewritten
  const uint32_t slot = m_cycleIndex % kDepthFlipSets;
  {
    VkDescriptorImageInfo srcInfo{};
    srcInfo.sampler     = m_samplerBiLinear;
    srcInfo.imageView   = sourceDepthView;
    srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // Remix keeps its RT resources in GENERAL

    VkDescriptorImageInfo dstInfo{};
    dstInfo.imageView   = m_frameBuffer.imgHizSource.descriptor.imageView;
    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = m_depthFlipSets[slot];
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo      = &srcInfo;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = m_depthFlipSets[slot];
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &dstInfo;

    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
  }

  cmdImageTransition(cmd, m_frameBuffer.imgHizSource, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, true);

  struct
  {
    int32_t  resolution[2];
    uint32_t flipDepth;
  } push;
  push.resolution[0] = int32_t(m_frameBuffer.renderSize.width);
  push.resolution[1] = int32_t(m_frameBuffer.renderSize.height);
  push.flipDepth     = flipDepth ? 1u : 0u;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_depthFlipPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_depthFlipPipelineLayout, 0, 1, &m_depthFlipSets[slot], 0, nullptr);
  vkCmdPushConstants(cmd, m_depthFlipPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(cmd, (m_frameBuffer.renderSize.width + 7) / 8, (m_frameBuffer.renderSize.height + 7) / 8, 1);

  // flip write -> nvhiz source read; also the layout its source descriptor uses
  cmdImageTransition(cmd, m_frameBuffer.imgHizSource, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);

  // uint32_t picks the set-index overload (a literal 0 is ambiguous against
  // the VkDescriptorSet overload - null pointer constant on x64)
  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate[0], uint32_t(0));

  // far-mip writes -> the traversal kernels' sampled reads within this same
  // command buffer (the sample built HiZ at frame end, one submission ahead
  // of its consumer, so it never needed this trailing barrier)
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
}

bool Resources::compileShader(shaderc::SpvCompilationResult& compiled,
                              VkShaderStageFlagBits          shaderStage,
                              const std::filesystem::path&   filePath,
                              shaderc::CompileOptions*       options)
{
  // NV-DXVK: shaders are compiled at build time by compile_shaders.py into
  // one SPIR-V blob per //!variant combination. Resolve the sample's original
  // (file, macro set) request to the matching prebuilt variant.
  (void)shaderStage;

  const shaderc::CompileOptions  emptyOptions;
  const shaderc::CompileOptions& lookupOptions = options ? *options : emptyOptions;

  const std::string fileName = filePath.filename().string();
  compiled                   = lookupPrebuiltShader(fileName.c_str(), lookupOptions);

  if(compiled.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    return true;
  }

  LOGE("lodclusters: no prebuilt shader variant matches '%s' with the requested defines\n", fileName.c_str());
  return false;
}

VkCommandBuffer Resources::createTempCmdBuffer()
{
  VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool                 = m_tempCommandPool;
  allocInfo.commandBufferCount          = 1;

  VkCommandBuffer cmd;
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &cmd));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  beginInfo.pInheritanceInfo         = nullptr;

  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

  return cmd;
}

void Resources::tempSyncSubmit(VkCommandBuffer cmd)
{
  vkEndCommandBuffer(cmd);

  VkCommandBufferSubmitInfo cmdInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
  };

  VkSubmitInfo2 submitInfo2 = {
      .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .flags                  = 0,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos    = &cmdInfo,
  };

  // NV-DXVK P2.5: wait on this submission's fence instead of the sample's
  // vkDeviceWaitIdle. Every guarantee callers rely on (the temp commands and
  // staging copies completed) is provided by the fence; unlike a device-wide
  // idle it does not stall against Remix's other queues (present, transfer),
  // which matters now that geometry appends run mid-session.
  VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence           fence     = nullptr;
  NVVK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &fence));

  NVVK_CHECK(vkQueueSubmit2(m_queue.queue, 1, &submitInfo2, fence));
  NVVK_CHECK(vkWaitForFences(m_device, 1, &fence, VK_TRUE, ~0ULL));

  vkDestroyFence(m_device, fence, nullptr);
  vkFreeCommandBuffers(m_device, m_tempCommandPool, 1, &cmd);
}

void Resources::cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier) const
{
  if(newLayout == rimg.descriptor.imageLayout && !needBarrier)
    return;

  nvvk::ImageMemoryBarrierParams imageBarrier;
  imageBarrier.image                       = rimg.image;
  imageBarrier.oldLayout                   = rimg.descriptor.imageLayout;
  imageBarrier.newLayout                   = newLayout;
  imageBarrier.subresourceRange.aspectMask = aspects;

  nvvk::cmdImageMemoryBarrier(cmd, imageBarrier);

  rimg.descriptor.imageLayout = newLayout;
}

VkDeviceSize Resources::getDeviceLocalHeapSize() const
{
  const VkPhysicalDeviceMemoryProperties& memProperties = m_memoryProperties;

  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {
    // find the heap that is purely tagged as device local
    if(memProperties.memoryTypes[type].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }

  // otherwise take something that is device local and host visible
  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {
    if((memProperties.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
       == (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }
  assert(0);
  return 0;
}

bool Resources::isBufferSizeValid(VkDeviceSize size) const
{
  return size <= m_physicalDeviceInfo.properties13.maxBufferSize && size <= m_physicalDeviceInfo.properties11.maxMemoryAllocationSize;
}

void QueueState::init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue)
{
  assert(m_device == nullptr);

  m_device      = device;
  m_queue       = queue;
  m_familyIndex = familyIndex;

  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = initialValue};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};

  vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_timelineSemaphore);

  m_device        = device;
  m_timelineValue = initialValue + 1;
}

void QueueState::deinit()
{
  if(!m_device)
    return;
  vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
}

VkSemaphoreSubmitInfo QueueState::getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex /*= 0*/) const
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}

VkSemaphoreSubmitInfo QueueState::advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex /*= 0*/)
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue++,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}

}  // namespace lodclusters
