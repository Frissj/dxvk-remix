/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// ClusterRenderSystem: the GPU half of the C++17-clean lodclusters_remix
// boundary (P2, preloaded path). Translates plain boundary types into
// NVIDIA's lodclusters types and owns Resources (volk/VMA), the generation's
// combined Scene + RenderScene and the cluster renderer. Compiles as part of
// the C++20 lodclusters library.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

#include <volk.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/profiler.hpp>
#include <nvvk/profiler_vk.hpp>

#include "lodclusters_remix.h"
#include "renderer.hpp"

namespace lodclusters_remix {

namespace {

// the library's own function pointers (volk); Remix's dxvk loader is separate
void loadVolk(VkInstance instance, VkDevice device)
{
  static std::once_flag s_volkOnce;
  std::call_once(s_volkOnce, [&] {
    if(volkInitialize() != VK_SUCCESS)
    {
      LOGE("ClusterRenderSystem: volkInitialize failed\n");
      return;
    }
    volkLoadInstance(instance);
  });

  // device functions may be reloaded if the device ever changes (device reset)
  volkLoadDevice(device);
}

glm::mat4 toMat4(const float m[16])
{
  glm::mat4 out;
  memcpy(&out[0][0], m, sizeof(float) * 16);
  return out;
}

uint32_t nextPowerOfTwo(uint32_t value)
{
  uint32_t result = 1;
  while(result < value)
  {
    result <<= 1;
  }
  return result;
}

// P4: detects the game's depth-buffer direction from its projection matrix.
// Remix's primary depth stores post-projection z/w in the game's own
// convention - standard-Z (near = 0, far = 1) for virtually all D3D9 titles,
// but reversed-Z projections exist. The view-space forward sign is probed via
// clip w (positive = point in front of the camera).
bool detectReversedZ(const glm::mat4& projMatrix, float nearPlane, float farPlane)
{
  auto ndcZ = [&](float distance) {
    glm::vec4 clip = projMatrix * glm::vec4(0.0f, 0.0f, -distance, 1.0f);
    if(clip.w <= 0.0f)
    {
      clip = projMatrix * glm::vec4(0.0f, 0.0f, distance, 1.0f);
    }
    return clip.w != 0.0f ? clip.z / clip.w : 0.0f;
  };

  return ndcZ(std::max(nearPlane, 0.001f)) > ndcZ(std::max(farPlane, nearPlane + 1.0f));
}

// P4: clip-space transform appending z' = w - z (NDC z' = 1 - z). Rewrites the
// game's standard depth range into the reversed-Z convention NVIDIA's culling
// kernels assume (culling.glsl intersectHiz / getCullBits) - the exact clip
// volume the sample's glm::perspectiveRH_ZO(fov, aspect, FAR, NEAR) produces.
// The HiZ source gets the matching value flip in remix_depth_flip.comp.
glm::mat4 makeReversedZFlip()
{
  glm::mat4 flip(1.0f);
  flip[2][2] = -1.0f;  // z' = -z ...
  flip[3][2] = 1.0f;   //          ... + w
  return flip;
}

}  // namespace

struct ClusterRenderSystem::Impl
{
  bool initialized = false;

  RenderDeviceInfo deviceInfo;
  RenderConfig     config;

  lodclusters::Resources res;

  nvutils::ProfilerManager   profilerManager;
  nvutils::ProfilerTimeline* profilerTimeline = nullptr;
  nvvk::ProfilerGpuTimer     profilerGpuTimer;

  // current generation
  std::unique_ptr<lodclusters::Scene>       scene;
  std::unique_ptr<lodclusters::RenderScene> rscene;
  std::unique_ptr<lodclusters::Renderer>    renderer;
  std::vector<GeometryRenderInfo>           geometryInfos;
  uint32_t                                  maxRenderInstances = 0;
  uint32_t                                  geometryCapacity   = 0;
  bool                                      hasGeneration      = false;

  // per-frame host-visible staging ring for the render instance array and the
  // CPU-known TlasInstance fields. 4 slots cover Remix's frames in flight.
  static constexpr uint32_t kStagingSlots = 4;
  nvvk::Buffer stagingBuffers[kStagingSlots];
  size_t       stagingTlasOffset = 0;

  uint32_t frameIndex = 0;
  bool     anyFrameRendered = false;

  // P4: freeze-latch state - the sample keeps these in its app frame loop
  // (lodclusters.cpp: matrices only advance while not frozen, so the frozen
  // HiZ content and the cull matrices stay consistent)
  glm::mat4 cullViewProjLatched = glm::mat4(1.0f);
  glm::mat4 cullViewProjLastLatched = glm::mat4(1.0f);
  bool      cullLatchValid = false;
  glm::mat4 traversalViewLatched = glm::mat4(1.0f);
  float     traversalFovLatched = 1.0f;
  float     traversalViewHeightLatched = 1.0f;
  bool      lodLatchValid = false;
  bool      depthConventionLogged = false;

  // ---- P4c rigid-capture promotion (plan 7.7 spec) ----
  // System scope (survives generation swaps). SoA state: matrices hold M +
  // prevM (96 B/slot, prevM read by the hit side for motion vectors), status
  // holds the 36 B/slot compact state the CPU reads back. Entries ride a
  // host-visible BDA ring; the status array is copied into a host readback
  // ring each frame promotion work ran.
  static constexpr uint32_t kPromoMatricesStride = 160;  // M + prevM + lastRigidM + eigen baseline (kernel mBase = slot*40 floats)
  static constexpr uint32_t kPromoStatusStride   = 80;   // 8 base + gateOver/gateStale/temporalDeform + meanDev/dirCoh/normAlign/solveInfo (DIAG) + capSig ([ShapeClass]) + eigDrift/eigFrame/eigLam1Hat/eigLam2Hat (Option 1)
  static constexpr uint32_t kPromoEntryStride    = sizeof(PromotionEntry);

  shaderc::SpvCompilationResult promoShader;
  VkPipelineLayout promoPipelineLayout = VK_NULL_HANDLE;
  VkPipeline       promoPipeline       = VK_NULL_HANDLE;
  nvvk::Buffer promoMatricesBuffer;
  nvvk::Buffer promoStatusBuffer;
  nvvk::Buffer promoLastSampleBuffer;  // [slot][kPromoSolveSamples][3]: last frame's solve-sample
                                       // capture, for the inter-frame deformation gate
  nvvk::Buffer promoLastCaptureVaBuffer;  // [slot] u64: the captureVa the stored last-frame samples
                                          // came from. The game double-buffers the vertex capture
                                          // (captureVa ping-pongs for a stable instance), so the
                                          // inter-frame deformation gate must only compare frames
                                          // that read the SAME capture allocation - else it diffs
                                          // two unrelated buffers and reports phantom deformation.
  static constexpr uint32_t kPromoSolveSamples = 64;  // mirrors PROMO_SOLVE_SAMPLES in the shader
  nvvk::Buffer promoDumpReadback[kStagingSlots];      // DIAG: raw-dump ring (one slot's samples)
  uint32_t promoDumpFramesRecorded = 0;
  // [SolveDump] M + per-validation (ref,cap,dev) of one traced slot: 16 header
  // floats + SOLVE_DUMP_MAXVAL(16) * SOLVE_DUMP_STRIDE(10). Device buffer the
  // kernel writes, copied to a host ring for readback.
  // 16 header + SOLVE_DUMP_MAXVAL(16)*SOLVE_DUMP_STRIDE(10) = 176 for the M/validation
  // dump; + [176,177]=captureVa lo/hi, [178]=capSigVar, [179]=sigN, [180 + i*3]=the
  // actual capSig-sampled vertex positions (i<32) -> [CapSigDump] buffer-consistency probe
  static constexpr uint32_t kPromoSolveDumpFloats = 16 + 16 * 10 + 4 + 32 * 3;
  nvvk::Buffer promoSolveDumpBuffer;
  nvvk::Buffer promoSolveDumpReadback[kStagingSlots];
  uint32_t promoSolveDumpFramesRecorded = 0;
  nvvk::Buffer promoEntryBuffers[kStagingSlots];
  nvvk::Buffer promoReadbackBuffers[kStagingSlots];
  uint32_t promoUsedSlots = 0;                        // stateSlot high-water + 1
  uint32_t promoReadbackUsedSlots[kStagingSlots] = {};
  uint32_t promoFramesRecorded = 0;
  bool     promoStateCleared = false;
  bool     promoReady = false;

  bool initPromotion();
  void deinitPromotion();
  void recordPromotion(VkCommandBuffer cmd, const FrameParams& frame, uint32_t instanceCount);

  void destroyGeneration()
  {
    if(renderer)
    {
      renderer->deinit(res);
      renderer.reset();
    }
    if(rscene)
    {
      rscene->deinit();
      rscene.reset();
    }
    if(scene)
    {
      scene->deinit();
      scene.reset();
    }
    for(nvvk::Buffer& staging : stagingBuffers)
    {
      if(staging.buffer)
      {
        res.m_allocator.destroyBuffer(staging);
      }
    }
    geometryInfos.clear();
    hasGeneration = false;
  }

  // fills geometryInfos [firstGeometry, firstGeometry + hashes.size()) from
  // the scene's views and the active path's geometry table (low-detail BLAS
  // address = safe blasReference default the assign kernel expects)
  void appendGeometryInfos(const std::vector<uint64_t>& hashes, size_t firstGeometry)
  {
    const std::vector<shaderio::Geometry>& shaderGeometries =
        rscene->useStreaming ? rscene->sceneStreaming.getShaderGeometries() : rscene->scenePreloaded.getShaderGeometries();

    geometryInfos.resize(firstGeometry + hashes.size());
    for(size_t i = 0; i < hashes.size(); i++)
    {
      const size_t                            g    = firstGeometry + i;
      const lodclusters::Scene::GeometryView& view = scene->getActiveGeometry(g);

      GeometryRenderInfo& info       = geometryInfos[g];
      info.geometryHash              = hashes[i];
      info.lodLevelsCount            = view.lodLevelsCount;
      info.lowDetailClusterStateBits = view.lowDetailClusterStateBits;
      info.lowDetailBlasAddress      = shaderGeometries[g].lowDetailBlasAddress;
      info.totalClusters             = view.totalClustersCount;
    }
  }
};

// ---- P4c rigid-capture promotion (plan 7.7 spec) -----------------------------

// mirrors promotion_solve.comp's push_constant block (scalar layout - u64s
// first, then 32-bit fields, identical order)
struct PromoPush
{
  uint64_t entriesVa;
  uint64_t matricesVa;
  uint64_t statusVa;
  uint64_t renderInstancesVa;
  uint64_t tlasInstancesVa;
  uint64_t lastSampleVa;
  uint32_t entryCount;
  uint32_t frameId;
  uint32_t riStrideBytes;
  uint32_t riWorldMatrixOffset;
  uint32_t riWorldMatrixIOffset;
  uint32_t riFlipWindingOffset;
  float    residualEpsilon;
  uint32_t gateEntryIndex;
  uint32_t promoScanEnable;          // DIAG: 1 = run the correspondence offset scan
  float    temporalEpsilon;          // max inter-frame sample-distance drift for "rigid"
  float    demoteHysteresis;         // promoted instances demote only past eps*this
  uint64_t solveDumpVa;              // [SolveDump] M + per-validation dump target (0 = off)
  uint32_t solveDumpSlot;            // [SolveDump] traced stateSlot (~0 = none)
  uint32_t solveDumpPad;             // std430 8-byte align of the trailing u64/u32 pair
  uint64_t lastCaptureVaVa;          // [slot] last frame's captureVa - tDeform buffer-phase gate
};

bool ClusterRenderSystem::Impl::initPromotion()
{
  // pipeline: push constants + buffer device addresses only, no descriptors
  if(!res.compileShader(promoShader, VK_SHADER_STAGE_COMPUTE_BIT, "promotion_solve.comp.glsl", nullptr))
  {
    LOGE("ClusterRenderSystem: promotion_solve shader missing from the variant table\n");
    return false;
  }

  VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PromoPush)};
  VkPipelineLayoutCreateInfo pipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeLayoutInfo.pushConstantRangeCount = 1;
  pipeLayoutInfo.pPushConstantRanges    = &pushRange;
  NVVK_CHECK(vkCreatePipelineLayout(deviceInfo.device, &pipeLayoutInfo, nullptr, &promoPipelineLayout));

  VkShaderModuleCreateInfo shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(promoShader);
  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.stage       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pName = "main";
  compInfo.stage.pNext = &shaderInfo;
  compInfo.layout      = promoPipelineLayout;
  NVVK_CHECK(vkCreateComputePipelines(deviceInfo.device, nullptr, 1, &compInfo, nullptr, &promoPipeline));

  NVVK_CHECK(res.createBuffer(promoMatricesBuffer, size_t(kPromoMatricesStride) * kPromotionSlotCapacity,
                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  NVVK_CHECK(res.createBuffer(promoStatusBuffer, size_t(kPromoStatusStride) * kPromotionSlotCapacity,
                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
  NVVK_CHECK(res.createBuffer(promoLastSampleBuffer,
                              size_t(kPromoSolveSamples) * 3 * sizeof(float) * kPromotionSlotCapacity,
                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
  NVVK_CHECK(res.createBuffer(promoLastCaptureVaBuffer,
                              sizeof(uint64_t) * kPromotionSlotCapacity,
                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    // DIAG raw-dump readback: one slot's 64 solve-sample capture positions
    NVVK_CHECK(res.createBuffer(promoDumpReadback[i], size_t(kPromoSolveSamples) * 3 * sizeof(float),
                                VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
  }
  // [SolveDump] device buffer the kernel writes + host ring for readback
  NVVK_CHECK(res.createBuffer(promoSolveDumpBuffer, size_t(kPromoSolveDumpFloats) * sizeof(float),
                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    NVVK_CHECK(res.createBuffer(promoSolveDumpReadback[i], size_t(kPromoSolveDumpFloats) * sizeof(float),
                                VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
  }

  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    NVVK_CHECK(res.createBuffer(promoEntryBuffers[i], size_t(kPromoEntryStride) * kPromotionSlotCapacity,
                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
    NVVK_CHECK(res.createBuffer(promoReadbackBuffers[i], size_t(kPromoStatusStride) * kPromotionSlotCapacity,
                                VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
  }

  promoReady = promoPipeline != VK_NULL_HANDLE && promoMatricesBuffer.buffer != VK_NULL_HANDLE
               && promoStatusBuffer.buffer != VK_NULL_HANDLE;
  return promoReady;
}

void ClusterRenderSystem::Impl::deinitPromotion()
{
  if(promoPipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(deviceInfo.device, promoPipeline, nullptr);
    promoPipeline = VK_NULL_HANDLE;
  }
  if(promoPipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(deviceInfo.device, promoPipelineLayout, nullptr);
    promoPipelineLayout = VK_NULL_HANDLE;
  }
  if(promoMatricesBuffer.buffer)
  {
    res.m_allocator.destroyBuffer(promoMatricesBuffer);
  }
  if(promoStatusBuffer.buffer)
  {
    res.m_allocator.destroyBuffer(promoStatusBuffer);
  }
  if(promoLastSampleBuffer.buffer)
  {
    res.m_allocator.destroyBuffer(promoLastSampleBuffer);
  }
  if(promoLastCaptureVaBuffer.buffer)
  {
    res.m_allocator.destroyBuffer(promoLastCaptureVaBuffer);
  }
  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    if(promoDumpReadback[i].buffer)
    {
      res.m_allocator.destroyBuffer(promoDumpReadback[i]);
    }
  }
  if(promoSolveDumpBuffer.buffer)
  {
    res.m_allocator.destroyBuffer(promoSolveDumpBuffer);
  }
  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    if(promoSolveDumpReadback[i].buffer)
    {
      res.m_allocator.destroyBuffer(promoSolveDumpReadback[i]);
    }
  }
  for(uint32_t i = 0; i < kStagingSlots; i++)
  {
    if(promoEntryBuffers[i].buffer)
    {
      res.m_allocator.destroyBuffer(promoEntryBuffers[i]);
    }
    if(promoReadbackBuffers[i].buffer)
    {
      res.m_allocator.destroyBuffer(promoReadbackBuffers[i]);
    }
  }
  promoReady = false;
}

void ClusterRenderSystem::Impl::recordPromotion(VkCommandBuffer cmd, const FrameParams& frame, uint32_t instanceCount)
{
  if(!promoReady)
  {
    return;
  }

  const uint32_t slot = frameIndex % kStagingSlots;

  // one-time zero init of the persistent state (recorded before any use)
  const bool needsClear = !promoStateCleared;
  if(needsClear)
  {
    vkCmdFillBuffer(cmd, promoMatricesBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, promoStatusBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, promoLastSampleBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, promoLastCaptureVaBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    promoStateCleared = true;
  }

  uint32_t entryCount = frame.promotionEntryCount;
  if(entryCount > kPromotionSlotCapacity)
  {
    entryCount = kPromotionSlotCapacity;
  }
  if(entryCount == 0 || frame.promotionEntries == nullptr)
  {
    promoReadbackUsedSlots[slot] = 0;
    return;
  }

  // stage this frame's entries (host-coherent mapped ring, read via BDA);
  // drop entries whose patch target would exceed this frame's instance range
  PromotionEntry* stagedEntries = reinterpret_cast<PromotionEntry*>(promoEntryBuffers[slot].mapping);
  bool anyGate = false;
  uint32_t usedSlots = promoUsedSlots;
  for(uint32_t i = 0; i < entryCount; i++)
  {
    PromotionEntry entry = frame.promotionEntries[i];
    if(entry.patchSlot != 0xFFFFFFFFu && entry.patchSlot >= instanceCount)
    {
      entry.patchSlot = 0xFFFFFFFFu;
    }
    if(entry.stateSlot >= kPromotionSlotCapacity)
    {
      entry.patchSlot = 0xFFFFFFFFu;
      entry.stateSlot = kPromotionSlotCapacity - 1;
    }
    anyGate |= entry.mode == 1 || entry.mode == 2;  // gates AND eigen sweeps use the per-entry pass
    usedSlots = std::max(usedSlots, entry.stateSlot + 1);
    stagedEntries[i] = entry;
  }
  promoUsedSlots = std::min(usedSlots, kPromotionSlotCapacity);

  // gate entries: reset their max-residual accumulators
  for(uint32_t i = 0; i < entryCount; i++)
  {
    if(stagedEntries[i].mode == 1)
    {
      const VkDeviceSize base = VkDeviceSize(stagedEntries[i].stateSlot) * kPromoStatusStride;
      vkCmdFillBuffer(cmd, promoStatusBuffer.buffer, base + 4,  4, 0);   // gateResidualBits
      vkCmdFillBuffer(cmd, promoStatusBuffer.buffer, base + 32, 4, 0);   // gateOverCount (DIAG)
    }
  }

  // fills + this frame's instance-array upload -> kernel reads/writes
  VkMemoryBarrier memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  auto timerSection = profilerGpuTimer.cmdFrameSection(cmd, "Promotion Solve");

  PromoPush push{};
  push.entriesVa            = promoEntryBuffers[slot].address;
  push.matricesVa           = promoMatricesBuffer.address;
  push.statusVa             = promoStatusBuffer.address;
  push.renderInstancesVa    = renderer->getRenderInstanceBuffer().address;
  push.tlasInstancesVa      = renderer->getTlasInstancesBuffer().address;
  push.entryCount           = entryCount;
  push.frameId              = frameIndex;
  push.riStrideBytes        = uint32_t(sizeof(shaderio::RenderInstance));
  push.riWorldMatrixOffset  = uint32_t(offsetof(shaderio::RenderInstance, worldMatrix));
  push.riWorldMatrixIOffset = uint32_t(offsetof(shaderio::RenderInstance, worldMatrixI));
  // dword packing flipWinding|twoSided|multiMaterial|opaqueStatus - the kernel
  // RMWs it, so the byte layout must stay dword-aligned
  static_assert(offsetof(shaderio::RenderInstance, flipWinding) % 4 == 0
                && offsetof(shaderio::RenderInstance, twoSided) == offsetof(shaderio::RenderInstance, flipWinding) + 1,
                "promotion_solve RMWs this dword");
  push.riFlipWindingOffset  = uint32_t(offsetof(shaderio::RenderInstance, flipWinding));
  push.residualEpsilon      = frame.promotionResidualEpsilon;
  push.gateEntryIndex       = 0xFFFFFFFFu;
  push.promoScanEnable      = frame.promotionCorrespondenceScan ? 1u : 0u;
  push.lastSampleVa         = promoLastSampleBuffer.address;
  push.lastCaptureVaVa      = promoLastCaptureVaBuffer.address;
  push.temporalEpsilon      = frame.promotionTemporalEpsilon;
  push.demoteHysteresis     = frame.promotionDemoteHysteresis;
  // [SolveDump] the kernel writes M + per-validation (ref,cap,dev) for the traced
  // slot into this buffer; copied to the host ring below. Reuses the resolved
  // dump slot (frame.promotionDumpStateSlot). ~0u slot => no write.
  push.solveDumpVa   = promoSolveDumpBuffer.address;
  push.solveDumpSlot = frame.promotionDumpStateSlot;
  push.solveDumpPad  = 0u;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, promoPipeline);
  vkCmdPushConstants(cmd, promoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PromoPush), &push);
  vkCmdDispatch(cmd, entryCount, 1, 1);

  // full-mesh gate sweeps read the matrices the solve above just wrote
  if(anyGate)
  {
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    for(uint32_t i = 0; i < entryCount; i++)
    {
      const PromotionEntry& entry = stagedEntries[i];
      if(entry.mode == 1 && entry.vertexCount != 0)
      {
        // full-mesh residual gate: grid over vertexCount
        push.gateEntryIndex = i;
        vkCmdPushConstants(cmd, promoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PromoPush), &push);
        vkCmdDispatch(cmd, (entry.vertexCount + 63u) / 64u, 1, 1);
      }
      else if(entry.mode == 2)
      {
        // Option 1 eigen sweep: ONE workgroup strides the full referenced set
        push.gateEntryIndex = i;
        vkCmdPushConstants(cmd, promoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PromoPush), &push);
        vkCmdDispatch(cmd, 1, 1, 1);
      }
    }
  }

  // patch writes -> downstream kernels + copy-out; status writes -> readback
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  // status snapshot for the CPU (readPromotionStates drains with the ring lag)
  {
    VkBufferCopy region{0, 0, VkDeviceSize(promoUsedSlots) * kPromoStatusStride};
    vkCmdCopyBuffer(cmd, promoStatusBuffer.buffer, promoReadbackBuffers[slot].buffer, 1, &region);
    promoReadbackUsedSlots[slot] = promoUsedSlots;
  }

  // DIAG raw dump: the requested slot's solve-sample capture positions (the solve
  // kernel stored them in the last-sample buffer this frame; barrier above covers it)
  if(frame.promotionDumpStateSlot != ~0u && frame.promotionDumpStateSlot < kPromotionSlotCapacity)
  {
    const VkDeviceSize bytes = VkDeviceSize(kPromoSolveSamples) * 3 * sizeof(float);
    if(frame.promotionDumpTargetBuffer != VK_NULL_HANDLE)
    {
      // [RestCapProbe] one-shot: the manager supplied its own staging so this
      // frame's solve view survives until its paired rest-capture copy drains
      // (the ring below is per-frame transient). Does not touch the ring or
      // its recorded-frames counter - the config [PromoDump] path is unaffected.
      VkBufferCopy region{VkDeviceSize(frame.promotionDumpStateSlot) * bytes,
                          frame.promotionDumpTargetOffset, bytes};
      vkCmdCopyBuffer(cmd, promoLastSampleBuffer.buffer, frame.promotionDumpTargetBuffer, 1, &region);
    }
    else if(promoDumpReadback[slot].buffer != VK_NULL_HANDLE)
    {
      VkBufferCopy region{VkDeviceSize(frame.promotionDumpStateSlot) * bytes, 0, bytes};
      vkCmdCopyBuffer(cmd, promoLastSampleBuffer.buffer, promoDumpReadback[slot].buffer, 1, &region);
      promoDumpFramesRecorded++;
    }

    // [SolveDump] M + per-validation (ref,cap,dev) the kernel wrote for this slot
    if(promoSolveDumpReadback[slot].buffer != VK_NULL_HANDLE)
    {
      VkBufferCopy region{0, 0, VkDeviceSize(kPromoSolveDumpFloats) * sizeof(float)};
      vkCmdCopyBuffer(cmd, promoSolveDumpBuffer.buffer, promoSolveDumpReadback[slot].buffer, 1, &region);
      promoSolveDumpFramesRecorded++;
    }
  }

  promoFramesRecorded++;
}

ClusterRenderSystem::ClusterRenderSystem()
    : m_impl(std::make_unique<Impl>())
{
}

ClusterRenderSystem::~ClusterRenderSystem()
{
  deinit();
}

bool ClusterRenderSystem::init(const RenderDeviceInfo& deviceInfo, const RenderConfig& config)
{
  Impl& impl = *m_impl;

  if(impl.initialized)
  {
    return true;
  }

  impl.deviceInfo = deviceInfo;
  impl.config     = config;

  loadVolk(deviceInfo.instance, deviceInfo.device);
  if(!vkGetDeviceProcAddr || !vkCmdBuildClusterAccelerationStructureIndirectNV)
  {
    LOGE("ClusterRenderSystem: volk did not resolve VK_NV_cluster_acceleration_structure entry points\n");
    return false;
  }

  nvvk::QueueInfo queueGraphics;
  queueGraphics.familyIndex = deviceInfo.graphicsQueueFamilyIndex;
  queueGraphics.queueIndex  = 0;
  queueGraphics.queue       = deviceInfo.graphicsQueue;

  nvvk::QueueInfo queueTransfer;
  queueTransfer.familyIndex = deviceInfo.transferQueueFamilyIndex;
  queueTransfer.queueIndex  = 0;
  queueTransfer.queue       = deviceInfo.transferQueue;

  impl.res.init(deviceInfo.device, deviceInfo.physicalDevice, deviceInfo.instance, queueGraphics, queueTransfer);

  nvutils::ProfilerTimeline::CreateInfo timelineInfo;
  timelineInfo.name     = "clusterlod";
  impl.profilerTimeline = impl.profilerManager.createTimeline(timelineInfo);
  impl.profilerGpuTimer.init(impl.profilerTimeline, deviceInfo.device, deviceInfo.physicalDevice,
                             int(deviceInfo.graphicsQueueFamilyIndex), false);

  // P4c: promotion solve pipeline + state buffers (failure leaves promotion
  // dormant - candidates then simply stay on Path B)
  if(!impl.initPromotion())
  {
    LOGW("ClusterRenderSystem: rigid-capture promotion unavailable\n");
  }

  impl.initialized = true;
  return true;
}

void ClusterRenderSystem::deinit()
{
  Impl& impl = *m_impl;

  if(!impl.initialized)
  {
    return;
  }

  vkDeviceWaitIdle(impl.deviceInfo.device);

  impl.destroyGeneration();
  impl.deinitPromotion();

  impl.profilerGpuTimer.deinit();
  if(impl.profilerTimeline)
  {
    impl.profilerManager.destroyTimeline(impl.profilerTimeline);
    impl.profilerTimeline = nullptr;
  }

  impl.res.deinit();
  impl.initialized = false;
}

bool ClusterRenderSystem::buildGeneration(const std::vector<std::string>& cacheFilesUtf8,
                                          const std::vector<uint64_t>&    geometryHashes,
                                          uint32_t                        maxRenderInstances)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || cacheFilesUtf8.empty() || cacheFilesUtf8.size() != geometryHashes.size())
  {
    return false;
  }

  // P2 bring-up: the previous generation's resources may still be referenced by
  // in-flight frames; wait for idle before replacing them. P5 refines this into
  // fully overlapped generation swaps.
  vkDeviceWaitIdle(impl.deviceInfo.device);

  impl.destroyGeneration();

  std::vector<std::filesystem::path> cachePaths;
  cachePaths.reserve(cacheFilesUtf8.size());
  for(const std::string& fileUtf8 : cacheFilesUtf8)
  {
    cachePaths.push_back(nvutils::pathFromUtf8(fileUtf8));
  }

  // assemble the combined Scene from the memory-mapped per-geometry caches
  impl.scene = std::make_unique<lodclusters::Scene>();

  lodclusters::SceneConfig sceneConfig = {};  // adopted from the cache files

  lodclusters::SceneLoaderConfig loaderConfig = {};
  loaderConfig.autoLoadCache                  = true;
  loaderConfig.autoSaveCache                  = false;
  loaderConfig.memoryMappedCache              = true;
  // 0 = "use the pool as-is": the struct default (0.5) made ProcessingInfo::
  // init/deinit reset the whole shared thread pool down and back up around
  // every generation build/append - a hidden main-thread hitch (rule 3). The
  // pool is sized once by configureProcessingThreadPool on the processing path.
  loaderConfig.processingThreadsPct           = 0.0f;

  lodclusters::Scene::Result result = impl.scene->initFromCachedGeometries(cachePaths, sceneConfig, loaderConfig);
  if(result != lodclusters::Scene::SCENE_RESULT_SUCCESS)
  {
    LOGE("ClusterRenderSystem: combined scene assembly failed (%d)\n", int(result));
    impl.destroyGeneration();
    return false;
  }

  // upload (preloaded path) or seed persistent low-detail data (streaming path)
  impl.rscene = std::make_unique<lodclusters::RenderScene>();

  lodclusters::StreamingConfig streamingConfig = {};  // preloaded mode uses only the CLAS fields
  streamingConfig.clasBuildFlags               = VkBuildAccelerationStructureFlagsKHR(impl.config.clasBuildFlags);
  streamingConfig.clasPositionTruncateBits     = impl.config.clasPositionTruncateBits;
  // P3: full streaming configuration (defaults = sample defaults)
  streamingConfig.useAsyncTransfer             = impl.config.useAsyncTransfer;
  streamingConfig.useDecoupledAsyncTransfer    = impl.config.useAsyncTransfer && impl.config.useDecoupledAsyncTransfer;
  streamingConfig.usePersistentClasAllocator   = impl.config.usePersistentClasAllocator;
  streamingConfig.maxPerFrameLoadRequests      = impl.config.maxPerFrameLoadRequests;
  streamingConfig.maxPerFrameUnloadRequests    = impl.config.maxPerFrameUnloadRequests;
  streamingConfig.maxGroups                    = impl.config.streamingMaxGroups;
  streamingConfig.maxClusters                  = impl.config.streamingMaxClusters;
  streamingConfig.maxTransferMegaBytes         = size_t(impl.config.maxTransferMegaBytes);
  streamingConfig.maxGeometryMegaBytes         = size_t(impl.config.maxGeometryMegaBytes);
  streamingConfig.maxClasMegaBytes             = size_t(impl.config.maxClasMegaBytes);
  streamingConfig.clasAllocatorSectorSizeShift  = impl.config.clasAllocatorSectorSizeShift;
  streamingConfig.clasAllocatorGranularityShift = impl.config.clasAllocatorGranularityShift;
  // P4: BLAS caching (streaming-only; the renderer additionally requires BLAS
  // sharing). allowBlasCaching arms SceneStreaming's cached-BLAS host logic +
  // sub-allocator; the per-frame FrameSettings gate what actually runs.
  streamingConfig.allowBlasCaching =
      impl.config.preferStreaming && impl.config.useBlasSharing && impl.config.useBlasCaching;
  streamingConfig.maxBlasCachingMegaBytes = size_t(impl.config.maxBlasCachingMegaBytes);

  // P2.5: geometry-slot capacity for the generation - headroom above the
  // initial count so later discoveries append in O(new) instead of forcing
  // full rebuilds. Exceeding it triggers a rebuild with the next capacity.
  impl.geometryCapacity =
      std::max(impl.config.maxGeometries, nextPowerOfTwo(uint32_t(cacheFilesUtf8.size())));

  if(!impl.rscene->init(&impl.res, impl.scene.get(), streamingConfig, impl.config.preferStreaming, impl.geometryCapacity))
  {
    LOGE("ClusterRenderSystem: RenderScene init failed\n");
    impl.destroyGeneration();
    return false;
  }

  // renderer (builds all CLAS + low-detail BLAS via updateClasRequired)
  impl.maxRenderInstances = std::max(impl.config.maxRenderInstances, maxRenderInstances);

  lodclusters::RendererConfig rendererConfig    = {};
  rendererConfig.useSorting                     = impl.config.useSorting;
  rendererConfig.useCulling                     = impl.config.useCulling;
  rendererConfig.useBlasSharing                 = impl.config.useBlasSharing;
  // requires streaming (the renderer force-disables both without it)
  rendererConfig.useBlasMerging                 = impl.config.useBlasMerging;
  rendererConfig.useBlasCaching                 = impl.config.useBlasCaching;
  rendererConfig.usePersistentTraversal         = impl.config.usePersistentTraversal;
  rendererConfig.useRenderStats                 = impl.config.useRenderStats;
  rendererConfig.useForcedInvisibleCulling      = impl.config.useForcedInvisibleCulling;
  rendererConfig.numRenderClusterBits           = impl.config.numRenderClusterBits;
  rendererConfig.numTraversalTaskBits           = impl.config.numTraversalTaskBits;
  rendererConfig.clusterBlasFlags               = VkBuildAccelerationStructureFlagsKHR(impl.config.clusterBlasFlags);
  rendererConfig.maxRenderInstances             = impl.maxRenderInstances;
  rendererConfig.maxGeometries                  = impl.geometryCapacity;

  impl.renderer = lodclusters::makeRendererRayTraceClustersLod();
  if(!impl.renderer->init(impl.res, *impl.rscene, rendererConfig))
  {
    LOGE("ClusterRenderSystem: renderer init failed\n");
    impl.destroyGeneration();
    return false;
  }

  // per-geometry render info for Remix
  impl.appendGeometryInfos(geometryHashes, 0);

  // host-visible staging ring for per-frame uploads
  const size_t renderInstancesBytes = sizeof(shaderio::RenderInstance) * size_t(impl.maxRenderInstances);
  impl.stagingTlasOffset            = nvutils::align_up(renderInstancesBytes, size_t(16));
  const size_t stagingBytes = impl.stagingTlasOffset + sizeof(VkAccelerationStructureInstanceKHR) * size_t(impl.maxRenderInstances);

  for(nvvk::Buffer& staging : impl.stagingBuffers)
  {
    NVVK_CHECK(impl.res.m_allocator.createBuffer(
        staging, stagingBytes, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
  }

  impl.hasGeneration = true;

  LOGI("ClusterRenderSystem: generation ready - %zu geometries (capacity %u), %u instance capacity, %s\n",
       impl.geometryInfos.size(), impl.geometryCapacity, impl.maxRenderInstances,
       impl.config.preferStreaming ? "streaming" : "preloaded");

  return true;
}

ClusterRenderSystem::AppendResult ClusterRenderSystem::appendToGeneration(const std::vector<std::string>& cacheFilesUtf8,
                                                                          const std::vector<uint64_t>& geometryHashes)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || !impl.hasGeneration)
  {
    return AppendResult::NeedsRebuild;
  }

  if(cacheFilesUtf8.empty() || cacheFilesUtf8.size() != geometryHashes.size())
  {
    return AppendResult::Failed;
  }

  const size_t firstGeometry = impl.scene->getActiveGeometryCount();

  // cheap capacity pre-check before touching anything
  if(firstGeometry + cacheFilesUtf8.size() > impl.geometryCapacity)
  {
    LOGI("ClusterRenderSystem: append exceeds geometry capacity (%zu + %zu > %u) - full rebuild\n", firstGeometry,
         cacheFilesUtf8.size(), impl.geometryCapacity);
    return AppendResult::NeedsRebuild;
  }

  std::vector<std::filesystem::path> cachePaths;
  cachePaths.reserve(cacheFilesUtf8.size());
  for(const std::string& fileUtf8 : cacheFilesUtf8)
  {
    cachePaths.push_back(nvutils::pathFromUtf8(fileUtf8));
  }

  // extend the combined Scene in place (transactional: a bad file leaves the
  // rendering Scene untouched, so the generation keeps running as-is)
  lodclusters::Scene::Result result = impl.scene->appendCachedGeometries(cachePaths);
  if(result != lodclusters::Scene::SCENE_RESULT_SUCCESS)
  {
    LOGE("ClusterRenderSystem: scene append failed (%d) - dropping %zu geometries\n", int(result), cachePaths.size());
    return AppendResult::Failed;
  }

  // the renderer's init-time sizing (per-BLAS cluster maximum, geometry
  // capacity of the sharing buffers) must still cover the grown scene. On a
  // rebuild the whole Scene is re-assembled from scratch, so the mutation
  // above is not a problem for the fallback path.
  if(!impl.renderer->canRenderScene(*impl.rscene))
  {
    LOGI("ClusterRenderSystem: appended geometries exceed the renderer's sizing - full rebuild\n");
    return AppendResult::NeedsRebuild;
  }

  // upload + CLAS + low-detail BLAS for the new range only (O(new); resident
  // geometry untouched, the geometry table buffer and its address are stable).
  // P3: in streaming mode this seeds the persistent lowest-detail data into
  // the reserved capacity; higher detail streams in on demand.
  if(!impl.rscene->appendGeometries(firstGeometry, cachePaths.size()))
  {
    LOGI("ClusterRenderSystem: render-scene append rejected - full rebuild\n");
    return AppendResult::NeedsRebuild;
  }

  // only now do the new geometryIDs become visible to recordFrame (its
  // per-frame numGeometries advances with geometryInfos)
  impl.appendGeometryInfos(geometryHashes, firstGeometry);

  LOGI("ClusterRenderSystem: generation grew by %zu geometries - %zu total (capacity %u)\n", cachePaths.size(),
       impl.geometryInfos.size(), impl.geometryCapacity);

  return AppendResult::Ok;
}

bool ClusterRenderSystem::hasGeneration() const
{
  return m_impl->hasGeneration;
}

bool ClusterRenderSystem::hizResolutionDiffers(uint32_t width, uint32_t height) const
{
  const Impl& impl = *m_impl;
  return impl.initialized && impl.res.hizResolutionDiffers(width, height);
}

void ClusterRenderSystem::updateHizResolution(uint32_t width, uint32_t height)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || !impl.res.hizResolutionDiffers(width, height))
  {
    return;
  }

  // waits for device idle internally; the caller holds Remix's submission lock
  impl.res.ensureHizResources(width, height);

  // the renderer's HIZ descriptor references the recreated far pyramid
  if(impl.renderer)
  {
    impl.renderer->updatedFrameBuffer(impl.res, *impl.rscene);
  }
}

const std::vector<GeometryRenderInfo>& ClusterRenderSystem::getGeometryRenderInfos() const
{
  return m_impl->geometryInfos;
}

uint32_t ClusterRenderSystem::getMaxRenderInstances() const
{
  return m_impl->maxRenderInstances;
}

void ClusterRenderSystem::recordFrame(VkCommandBuffer                           cmd,
                                      const FrameParams&                        frame,
                                      const InstanceInput*                      instances,
                                      const VkAccelerationStructureInstanceKHR* tlasInstances,
                                      uint32_t                                  count,
                                      FrameSubmitSync*                          outSubmitSync)
{
  Impl& impl = *m_impl;

  if(outSubmitSync)
  {
    *outSubmitSync = {};
  }

  if(!impl.hasGeneration || count > impl.maxRenderInstances
     || (count == 0 && frame.promotionEntryCount == 0))
  {
    return;
  }

  impl.res.beginFrame(impl.frameIndex % 4);
  impl.profilerTimeline->frameAdvance();

  // P4c: promotion-only frames (candidates probing while no promoted/static
  // instance rendered) - record just the solve + readback and advance the ring
  if(count == 0)
  {
    impl.recordPromotion(cmd, frame, 0);
    impl.frameIndex++;
    return;
  }

  // stage this frame's instance data
  nvvk::Buffer& staging = impl.stagingBuffers[impl.frameIndex % Impl::kStagingSlots];

  shaderio::RenderInstance* renderInstances = reinterpret_cast<shaderio::RenderInstance*>(staging.mapping);
  for(uint32_t i = 0; i < count; i++)
  {
    const InstanceInput& input = instances[i];

    const glm::mat4 worldMatrix = toMat4(input.worldMatrix);

    const GeometryRenderInfo& geometryInfo = impl.geometryInfos[input.geometryID];

    shaderio::RenderInstance& renderInstance = renderInstances[i];
    renderInstance                           = {};
    renderInstance.worldMatrix               = glm::mat4x3(worldMatrix);
    renderInstance.worldMatrixI              = glm::mat4x3(glm::inverse(worldMatrix));
    renderInstance.geometryID                = input.geometryID;

    // Remix materials shade; the cluster-side material system stays on the
    // single dummy entry (see renderer.cpp)
    renderInstance.materialID       = 0;
    renderInstance.multiMaterial    = 0;
    renderInstance.twoSided         = input.twoSided ? 1 : 0;
    renderInstance.alphaMaskTexture = 0xFFFF;
    renderInstance.opaqueStatus     = uint8_t(input.opaqueStatus);

    renderInstance.maxLodLevelRcp =
        geometryInfo.lodLevelsCount > 1 ? 1.0f / float(geometryInfo.lodLevelsCount - 1) : 0.0f;
    renderInstance.packedColor = 0xFFFFFFFFu;

    renderInstance.lowDetailClusterStateBits = uint8_t(geometryInfo.lowDetailClusterStateBits);

    renderInstance.flipWinding = (!renderInstance.twoSided && (glm::determinant(worldMatrix) <= 0)) ? 1 : 0;
  }

  memcpy(staging.mapping + impl.stagingTlasOffset, tlasInstances, sizeof(VkAccelerationStructureInstanceKHR) * count);

  // upload into the renderer's device buffers; the renderer's first barrier
  // (TRANSFER -> COMPUTE|TRANSFER) covers these writes
  {
    const nvvk::Buffer& renderInstanceBuffer = impl.renderer->getRenderInstanceBuffer();
    const nvvk::Buffer& tlasInstancesBuffer  = impl.renderer->getTlasInstancesBuffer();

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = sizeof(shaderio::RenderInstance) * count;
    vkCmdCopyBuffer(cmd, staging.buffer, renderInstanceBuffer.buffer, 1, &region);

    region.srcOffset = impl.stagingTlasOffset;
    region.dstOffset = 0;
    region.size      = sizeof(VkAccelerationStructureInstanceKHR) * count;
    vkCmdCopyBuffer(cmd, staging.buffer, tlasInstancesBuffer.buffer, 1, &region);
  }

  // P4c rigid-capture promotion (plan 7.7): solve M per candidate against the
  // capture output and patch promoted slots' worldMatrix/TLAS transform -
  // recorded HERE, after this frame's instance data landed in the device
  // arrays and before any kernel consumes them
  impl.recordPromotion(cmd, frame, count);

  const glm::mat4 viewMatrix     = toMat4(frame.viewMatrix);
  const glm::mat4 projMatrix     = toMat4(frame.projMatrix);
  const glm::mat4 viewProjMatrix = toMat4(frame.viewProjMatrix);

  // ---- P4: cull space + HiZ occlusion feed ----
  // NVIDIA's culling kernels assume the sample's reversed-Z clip convention.
  // The game's matrices/depth stay untouched; the CULL matrices get a
  // z' = w - z flip appended and the HiZ source gets the matching 1 - depth
  // conversion (remix_depth_flip), so every kernel stays byte-identical. A
  // game already rendering reversed-Z passes through unchanged.
  const bool depthIsReversed = detectReversedZ(projMatrix, frame.nearPlane, frame.farPlane);
  if(!impl.depthConventionLogged)
  {
    LOGI("ClusterRenderSystem: game depth convention detected as %s-Z\n", depthIsReversed ? "reversed" : "standard");
    impl.depthConventionLogged = true;
  }

  const glm::mat4 zFlip = depthIsReversed ? glm::mat4(1.0f) : makeReversedZFlip();

  // freeze latches (sample parity: lodclusters.cpp only advances these while
  // not frozen, keeping cull matrices consistent with the frozen HiZ content)
  if(!frame.freezeCulling || !impl.cullLatchValid)
  {
    impl.cullViewProjLatched     = zFlip * viewProjMatrix;
    impl.cullViewProjLastLatched = zFlip * toMat4(frame.prevViewProjMatrix);
    impl.cullLatchValid          = true;
  }
  if(!frame.freezeLoD || !impl.lodLatchValid)
  {
    impl.traversalViewLatched       = viewMatrix;
    impl.traversalFovLatched        = frame.fovRadians;
    impl.traversalViewHeightLatched = float(frame.viewportHeight);
    impl.lodLatchValid              = true;
  }

  // HiZ far-mip build from the previous frame's primary depth. Skipped while
  // culling is frozen (sample parity), without a depth source (first frames -
  // the cleared far pyramid passes every occlusion test) or on a resolution
  // mismatch (the manager resizes via updateHizResolution before recordFrame).
  if(impl.config.useCulling && !frame.freezeCulling && frame.depthView != VK_NULL_HANDLE
     && !impl.res.hizResolutionDiffers(frame.depthWidth, frame.depthHeight))
  {
    impl.res.cmdBuildHizFromDepth(cmd, frame.depthView, !depthIsReversed, impl.profilerGpuTimer);
  }

  // frame configuration from Remix's camera
  lodclusters::FrameConfig frameConfig = {};

  frameConfig.windowSize                 = {frame.viewportWidth, frame.viewportHeight};
  frameConfig.lodPixelError              = frame.lodPixelError;
  frameConfig.culledErrorScale           = frame.culledErrorScale;
  frameConfig.freezeCulling              = frame.freezeCulling;
  frameConfig.freezeLoD                  = frame.freezeLoD;
  frameConfig.traversalPersistentThreads = frame.traversalPersistentThreads;
  frameConfig.streamingAgeThreshold      = frame.streamingAgeThreshold;
  frameConfig.traversalFov               = impl.traversalFovLatched;
  frameConfig.traversalViewHeight        = impl.traversalViewHeightLatched;
  frameConfig.traversalViewMatrix        = impl.traversalViewLatched;
  frameConfig.cullViewProjMatrix         = impl.cullViewProjLatched;
  frameConfig.cullViewProjMatrixLast     = impl.cullViewProjLastLatched;

  // P4: per-frame BLAS sharing / caching tuning
  frameConfig.sharingPushCulled     = frame.sharingPushCulled;
  frameConfig.sharingTolerantLevels = frame.sharingTolerantLevels;
  frameConfig.sharingEnabledLevels  = frame.sharingEnabledLevels;
  frameConfig.cachingAgeThreshold   = frame.cachingAgeThreshold;
  frameConfig.cachingEnabledLevels  = frame.cachingEnabledLevels;

  shaderio::FrameConstants& frameConstants = frameConfig.frameConstants;
  frameConstants                           = {};
  frameConstants.projMatrix                = projMatrix;
  frameConstants.projMatrixI               = glm::inverse(projMatrix);
  frameConstants.viewProjMatrix            = viewProjMatrix;
  frameConstants.viewProjMatrixI           = glm::inverse(viewProjMatrix);
  frameConstants.viewProjMatrixRender      = viewProjMatrix;
  frameConstants.viewMatrix                = viewMatrix;
  frameConstants.viewMatrixI               = glm::inverse(viewMatrix);
  // game convention (NOT the z-flipped cull matrix): kernels don't read this,
  // it exists for the sample's motion vectors
  frameConstants.viewProjMatrixPrev        = toMat4(frame.prevViewProjMatrix);
  frameConstants.viewPos                   = glm::vec4(frame.viewPos[0], frame.viewPos[1], frame.viewPos[2], 1.0f);
  frameConstants.viewDir                   = -glm::vec4(glm::vec3(glm::transpose(viewMatrix)[2]), 0.0f);
  frameConstants.viewport                  = glm::ivec2(frame.viewportWidth, frame.viewportHeight);
  frameConstants.viewportf                 = glm::vec2(frame.viewportWidth, frame.viewportHeight);
  frameConstants.viewPixelSize             = 1.0f / glm::vec2(frame.viewportWidth, frame.viewportHeight);
  frameConstants.fov                       = frame.fovRadians;
  frameConstants.nearPlane                 = frame.nearPlane;
  frameConstants.farPlane                  = frame.farPlane;
  frameConstants.frame                     = impl.frameIndex;
  // P4: real far-pyramid lookup factors (sample: lodclusters.cpp ~1108)
  impl.res.m_hizUpdate[0].farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
  impl.res.m_hizUpdate[0].farInfo.getSize((float*)&frameConstants.hizSize);
  frameConstants.nearSizeFactors           = glm::vec4(1.0f);

  lodclusters::RendererFrameInput frameInput;
  frameInput.numRenderInstances   = count;
  // P2.5: advances only when an append fully completed (see appendToGeneration)
  frameInput.numGeometries        = uint32_t(impl.geometryInfos.size());
  frameInput.tlasInstancesAddress = impl.renderer->getTlasInstancesBuffer().address;

  impl.renderer->render(cmd, impl.res, *impl.rscene, frameConfig, frameInput, impl.profilerGpuTimer);

  // P3, streaming mode: the streaming task queues track this frame through
  // the primary QueueState timeline - mirror the sample's frame loop, which
  // signals the advanced timeline value with this command buffer's submission
  // and forwards any async-transfer waits onto it (lodclusters.cpp).
  if(impl.rscene->useStreaming)
  {
    if(outSubmitSync)
    {
      VkSemaphoreSubmitInfo signalInfo =
          impl.res.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
      outSubmitSync->signal.semaphore = signalInfo.semaphore;
      outSubmitSync->signal.value     = signalInfo.value;

      while(!impl.res.m_queueStates.primary.m_pendingWaits.empty())
      {
        const VkSemaphoreSubmitInfo& waitInfo = impl.res.m_queueStates.primary.m_pendingWaits.back();
        outSubmitSync->waits.push_back({waitInfo.semaphore, waitInfo.value});
        impl.res.m_queueStates.primary.m_pendingWaits.pop_back();
      }
    }
    else
    {
      // without the signal the task queues would deadlock waiting for frames
      LOGE("ClusterRenderSystem::recordFrame streaming requires outSubmitSync\n");
    }
  }

  impl.res.endFrame();
  impl.frameIndex++;
  impl.anyFrameRendered = true;
}

VkBuffer ClusterRenderSystem::getTlasInstancesBuffer() const
{
  Impl& impl = *m_impl;
  if(!impl.hasGeneration)
  {
    return VK_NULL_HANDLE;
  }
  return impl.renderer->getTlasInstancesBuffer().buffer;
}

uint64_t ClusterRenderSystem::getGeometriesTableAddress() const
{
  Impl& impl = *m_impl;
  if(!impl.hasGeneration)
  {
    return 0;
  }
  return impl.rscene->getShaderGeometriesBuffer().address;
}

uint64_t ClusterRenderSystem::getPromotionStateAddress() const
{
  // 160 B per slot: M rows (row-major 3x4) at +0, prevM rows at +48,
  // last-RIGID M at +96, eigen baseline at +144 (kernel-internal)
  Impl& impl = *m_impl;
  return impl.promoReady ? impl.promoMatricesBuffer.address : 0;
}

bool ClusterRenderSystem::readPromotionSampleDump(float* outPositions192)
{
  Impl& impl = *m_impl;
  if(!impl.promoReady || outPositions192 == nullptr || impl.promoDumpFramesRecorded < Impl::kStagingSlots)
  {
    return false;
  }
  const uint32_t slot = (impl.frameIndex + 1u) % Impl::kStagingSlots;
  const void* src = impl.promoDumpReadback[slot].mapping;
  if(src == nullptr)
  {
    return false;
  }
  memcpy(outPositions192, src, size_t(Impl::kPromoSolveSamples) * 3 * sizeof(float));
  return true;
}

uint32_t ClusterRenderSystem::promotionSolveDumpFloatCount()
{
  return Impl::kPromoSolveDumpFloats;
}

bool ClusterRenderSystem::readPromotionSolveDump(float* outFloats)
{
  Impl& impl = *m_impl;
  if(!impl.promoReady || outFloats == nullptr || impl.promoSolveDumpFramesRecorded < Impl::kStagingSlots)
  {
    return false;
  }
  const uint32_t slot = (impl.frameIndex + 1u) % Impl::kStagingSlots;
  const void* src = impl.promoSolveDumpReadback[slot].mapping;
  if(src == nullptr)
  {
    return false;
  }
  memcpy(outFloats, src, size_t(Impl::kPromoSolveDumpFloats) * sizeof(float));
  return true;
}

bool ClusterRenderSystem::readPromotionStates(PromotionStateView* outStates)
{
  Impl& impl = *m_impl;

  // the ring holds one snapshot per in-flight frame; only trust it once every
  // slot has been written at least once (the +1 slot below is then the oldest
  // complete snapshot = safely retired by the frames-in-flight window)
  if(!impl.promoReady || outStates == nullptr || impl.promoFramesRecorded < Impl::kStagingSlots)
  {
    return false;
  }

  const uint32_t slot = (impl.frameIndex + 1u) % Impl::kStagingSlots;
  const uint32_t used = std::min(impl.promoReadbackUsedSlots[slot], kPromotionSlotCapacity);
  const uint8_t* src  = reinterpret_cast<const uint8_t*>(impl.promoReadbackBuffers[slot].mapping);
  if(src == nullptr)
  {
    return false;
  }

  for(uint32_t i = 0; i < kPromotionSlotCapacity; i++)
  {
    outStates[i] = {};
  }
  for(uint32_t i = 0; i < used; i++)
  {
    const uint8_t* s = src + size_t(i) * Impl::kPromoStatusStride;
    PromotionStateView& v = outStates[i];
    memcpy(&v.residualRel, s + 0, sizeof(float));
    uint32_t gateBits = 0;
    memcpy(&gateBits, s + 4, sizeof(uint32_t));
    memcpy(&v.gateResidualRel, &gateBits, sizeof(float));  // ordered-uint == float bits for non-negatives
    memcpy(&v.rigidStreak, s + 8, sizeof(uint32_t));
    memcpy(&v.flags, s + 12, sizeof(uint32_t));
    memcpy(&v.lastFrame, s + 16, sizeof(uint32_t));
    uint32_t shearBits = 0;
    memcpy(&shearBits, s + 20, sizeof(uint32_t));
    memcpy(&v.affineNonRigid, &shearBits, sizeof(float));  // ordered-uint == float bits for non-negatives
    memcpy(&v.diagGuard, s + 24, sizeof(uint32_t));
    memcpy(&v.diagAux, s + 28, sizeof(uint32_t));
    memcpy(&v.gateOverCount, s + 32, sizeof(uint32_t));
    memcpy(&v.gateStaleFrames, s + 36, sizeof(uint32_t));
    uint32_t tdBits = 0;
    memcpy(&tdBits, s + 40, sizeof(uint32_t));
    memcpy(&v.temporalDeformRel, &tdBits, sizeof(float));  // ordered-uint == float bits for non-negatives
    memcpy(&v.meanDevRel, s + 44, sizeof(float));
    memcpy(&v.dirCoherence, s + 48, sizeof(float));
    memcpy(&v.normAlign, s + 52, sizeof(float));
    memcpy(&v.solveInfo, s + 56, sizeof(uint32_t));
    memcpy(&v.capSig, s + 60, sizeof(float));  // [ShapeClass] probe-independent signature
    // Option 1 eigen verdict + content-identity key (plain float bits - single writer)
    memcpy(&v.eigDrift, s + 64, sizeof(float));
    memcpy(&v.eigFrame, s + 68, sizeof(uint32_t));
    memcpy(&v.eigLam1Hat, s + 72, sizeof(float));
    memcpy(&v.eigLam2Hat, s + 76, sizeof(float));
  }
  return true;
}

bool ClusterRenderSystem::getFrameStats(FrameStats& outStats) const
{
  Impl& impl = *m_impl;

  outStats = {};

  if(!impl.hasGeneration || !impl.anyFrameRendered)
  {
    return false;
  }

  shaderio::Readback readback;
  impl.res.getReadbackData(readback);

  outStats.numRenderClusters   = readback.numRenderClusters;
  outStats.numTraversalTasks   = readback.numTraversalTasks;
  outStats.numBlasBuilds       = readback.numBlasBuilds;
  outStats.blasActualSizeBytes = readback.blasActualSizes;

  const lodclusters::Renderer::ResourceUsageInfo usage = impl.renderer->getResourceUsage(true);
  outStats.reservedClasBytes                           = usage.rtClasMemBytes;
  outStats.reservedBlasBytes                           = usage.rtBlasMemBytes;
  outStats.reservedGeometryBytes                       = usage.geometryMemBytes;
  outStats.reservedOperationsBytes                     = usage.operationsMemBytes;

  // P3: streaming statistics
  if(impl.rscene->useStreaming)
  {
    lodclusters::StreamingStats streamingStats;
    impl.rscene->sceneStreaming.getStats(streamingStats);

    outStats.streaming = true;

    outStats.residentGroups   = streamingStats.residentGroups;
    outStats.residentClusters = streamingStats.residentClusters;
    outStats.maxGroups        = streamingStats.maxGroups;
    outStats.maxClusters      = streamingStats.maxClusters;
    outStats.persistentGroups = streamingStats.persistentGroups;

    outStats.usedDataBytes       = streamingStats.usedDataBytes;
    outStats.reservedDataBytes   = streamingStats.reservedDataBytes;
    outStats.maxDataBytes        = streamingStats.maxDataBytes;
    outStats.persistentDataBytes = streamingStats.persistentDataBytes;

    outStats.usedClasBytes   = streamingStats.usedClasBytes;
    outStats.wastedClasBytes = streamingStats.wastedClasBytes;

    outStats.transferBytes        = streamingStats.transferBytes;
    outStats.transferCount        = streamingStats.transferCount;
    outStats.loadCount            = streamingStats.loadCount;
    outStats.unloadCount          = streamingStats.unloadCount;
    outStats.uncompletedLoadCount = streamingStats.uncompletedLoadCount;

    outStats.couldNotAllocateGroup = streamingStats.couldNotAllocateGroup;
    outStats.couldNotAllocateClas  = streamingStats.couldNotAllocateClas;
    outStats.couldNotTransfer      = streamingStats.couldNotTransfer;
    outStats.couldNotStore         = streamingStats.couldNotStore;
  }

  return true;
}

// shared with the Path B report (see renderer_raytrace_clusters.cpp): formats
// one profiler timeline snapshot as '\n'-separated section lines. Values come
// out of nvutils::ProfilerTimeline in microseconds; reported in milliseconds.
bool formatProfilerReportUtf8(const nvutils::ProfilerTimeline* timeline, std::string& outReport)
{
  outReport.clear();

  if(timeline == nullptr)
  {
    return false;
  }

  nvutils::ProfilerTimeline::Snapshot snapshot;
  timeline->getFrameSnapshot(snapshot);

  char line[256];
  for(size_t i = 0; i < snapshot.timerInfos.size(); i++)
  {
    const nvutils::ProfilerTimeline::TimerInfo& info = snapshot.timerInfos[i];
    if(info.numAveraged == 0)
    {
      continue;
    }

    // level indents nested sections under their parent (e.g. streaming
    // sub-phases under "Streaming"); gpu max exposes hitchy sections that a
    // clean average would hide
    snprintf(line, sizeof(line), "%*s%s: cpu %.3f ms, gpu %.3f ms (gpu max %.3f, avg of %u)",
             int(info.level) * 2, "", snapshot.timerNames[i].c_str(),
             info.cpu.average * 1e-3, info.gpu.average * 1e-3, info.gpu.absMaxValue * 1e-3, info.numAveraged);

    if(!outReport.empty())
    {
      outReport += '\n';
    }
    outReport += line;
  }

  return !outReport.empty();
}

bool ClusterRenderSystem::getProfilerReportUtf8(std::string& outReport) const
{
  const Impl& impl = *m_impl;

  if(!impl.initialized)
  {
    outReport.clear();
    return false;
  }

  return formatProfilerReportUtf8(impl.profilerTimeline, outReport);
}

}  // namespace lodclusters_remix
