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
// (src/renderer_raytrace_clusters_lod.cpp) and is the core of the RTX Remix
// cluster LOD renderer. Edits against the sample ("heavy but mechanical",
// INTEGRATION_PLAN.txt section 4):
// - The sample's ray tracing pipeline (rgen/chit/ahit/miss + SBT) and its
//   TLAS are removed: Remix's path tracer consumes the cluster BLASes through
//   AccelManager's TLAS. instance_assign_blas patches an externally provided
//   shaderio::TlasInstance array (m_tlasInstancesBuffer, staged here, copied
//   into AccelManager's instance buffer by ClusterLodManager).
// - Shader "compilation" resolves to build-time SPIR-V variants
//   (see lodclusters_shader_table.cpp); call sites unchanged.
// - Render instances are provided per frame by Remix; buffers are sized for
//   RendererConfig::maxRenderInstances.
// - The viewer's DLSS/HiZ-build/bbox/present sections are removed. The HiZ
//   occlusion feed (P4) is built from Remix's previous-frame primary depth at
//   the START of recordFrame (Resources::cmdBuildHizFromDepth) instead of the
//   sample's end-of-frame cmdBuildHiz; updatedFrameBuffer rewires the HIZ
//   descriptor when the render resolution changes.
// The compute dispatch sequences and all barriers are preserved verbatim.

#include <volk.h>
#include <nvutils/parallel_work.hpp>
#include <nvutils/alignment.hpp>
#include <nvutils/logger.hpp>
#include <fmt/format.h>

#include <mutex>

#include <nvvk/check_error.hpp>

#include "renderer.hpp"

#define USE_LARGE_BUFFER_BLAS 1

//////////////////////////////////////////////////////////////////////////

namespace lodclusters {

class RendererRayTraceClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, const RendererFrameInput& frameInput, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;

  // NV-DXVK: Remix-facing accessors (see ClusterLodManager)
  virtual const nvvk::Buffer& getTlasInstancesBuffer() const override { return m_tlasInstancesBuffer; }

  // NV-DXVK P2.5: whether the init-time sizing still fits the (appended-to)
  // scene: geometry capacity for the sharing buffers and the per-BLAS cluster
  // maximum the BLAS pool and scratch were sized with.
  virtual bool canRenderScene(const RenderScene& rscene) const override
  {
    const uint32_t requiredClustersPerBlas = std::min(rscene.scene->m_maxPerGeometryClusters, m_maxRenderClusters);

    return uint32_t(rscene.scene->getActiveGeometryCount()) <= m_maxGeometries
           && requiredClustersPerBlas <= m_blasInput.maxClusterCountPerAccelerationStructure;
  }

private:
  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);

  bool initRayTracingBlas(Resources& res, RenderScene& scene, const RendererConfig& config, VkDeviceSize& scratchSize);

  struct Shaders
  {
    shaderc::SpvCompilationResult computeTraversalPresort;
    shaderc::SpvCompilationResult computeTraversalInit;
    shaderc::SpvCompilationResult computeTraversalRun;
    shaderc::SpvCompilationResult computeTraversalGroups;
    shaderc::SpvCompilationResult computeTraversalMerge;
    shaderc::SpvCompilationResult computeBuildSetup;

    shaderc::SpvCompilationResult computeBlasInsertClusters;
    shaderc::SpvCompilationResult computeBlasSetupInsertion;
    shaderc::SpvCompilationResult computeBlasCachingSetupCopy;
    shaderc::SpvCompilationResult computeBlasCachingSetupBuild;

    shaderc::SpvCompilationResult computeInstanceAssignBlas;
    shaderc::SpvCompilationResult computeInstanceClassifyLod;
    shaderc::SpvCompilationResult computeGeometryBlasSharing;
  };

  struct Pipelines
  {
    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeTraversalMerge   = nullptr;
    VkPipeline computeBuildSetup       = nullptr;

    VkPipeline computeBlasInsertClusters    = nullptr;
    VkPipeline computeBlasSetupInsertion    = nullptr;
    VkPipeline computeBlasCachingSetupCopy  = nullptr;
    VkPipeline computeBlasCachingSetupBuild = nullptr;
    VkPipeline computeInstanceAssignBlas    = nullptr;
    VkPipeline computeInstanceClassifyLod   = nullptr;
    VkPipeline computeGeometryBlasSharing   = nullptr;
  };

  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_rtClasProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};

  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  VkPipelineLayout   m_pipelineLayout{};

  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer m_sceneBuildBuffer;
  nvvk::Buffer m_sceneDataBuffer;

  // NV-DXVK (risk R17): the implicit-destination BLAS pool is TRACE-READ.
  // Frame N's ray trace dereferences BLASes in this pool while frame N+1's
  // dispatchBuild records the next implicit build - single-buffered, frame
  // N+1 would overwrite what frame N still reads (the exact race Path B hit
  // and fixed by ringing its CLAS/BLAS pools; first triggered when Path A
  // activated under streaming, VK_ERROR_DEVICE_LOST 2026-07-04). Ring one
  // pool per frame-in-flight slot; everything else the trace reads (CLAS,
  // cached BLAS memory, low-detail BLAS) is persistent and unaffected.
  //
  // Correctness bound: kBlasRingSlots >= max frames-in-flight. dxvk's hard cap
  // is dxvk::kMaxFramesInFlight = 4 (rtx_utils.h) - default frame latency is 3,
  // but SetMaximumFrameLatency / d3d9.maxFrameLatency can request up to 4, so a
  // title can run 4 frames deep. 4 is therefore the *minimum* safe value here,
  // not an over-provision: do NOT shrink to 3 - that reintroduces the exact
  // device-lost race above whenever frames-in-flight hits 4. (Matches Path B's
  // kRingSlots = 4 in animated/renderer_raytrace_clusters.cpp.)
  static constexpr uint32_t kBlasRingSlots = 4;
#if USE_LARGE_BUFFER_BLAS
  nvvk::LargeBuffer m_sceneBlasDataBuffers[kBlasRingSlots];
#else
  nvvk::Buffer m_sceneBlasDataBuffers[kBlasRingSlots];
#endif
  nvvk::Buffer            m_sceneTraversalBuffer;
  nvvk::Buffer            m_sceneGeometryHistogramBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  VkClusterAccelerationStructureClustersBottomLevelInputNV m_blasInput{};
  VkDeviceSize                                             m_blasDataSize = 0;

  // NV-DXVK P2.5: geometry capacity the geometry-count dependent buffers were
  // sized with (config.maxGeometries resolved against the init-time count)
  uint32_t m_maxGeometries = 0;

  VkClusterAccelerationStructureMoveObjectsInputNV m_blasMoveInput{};

  // NV-DXVK: staging array of shaderio::TlasInstance the assign kernel patches;
  // Remix uploads the CPU-known fields each frame and copies the patched
  // entries into AccelManager's instance buffer regions afterwards.
  nvvk::Buffer m_tlasInstancesBuffer;

  nvvk::Buffer m_scratchBuffer;

  // NV-DXVK: device-lost forensics (Path A). Every frame the per-frame cluster
  // BLAS build's inputs (SceneBuilding counters, the BlasBuildInfo array, the
  // CLAS reference lists) are copied to a host mirror BEFORE the build runs,
  // with an execution barrier so the copies complete before the build can
  // page-fault the device. Pre/post stamps bracket the build; on
  // VK_ERROR_DEVICE_LOST the slot whose post-stamp is missing holds exactly
  // what the faulting build consumed, and debugDumpBlasInputCapture classifies
  // every cluster reference against the valid CLAS pools.
  // NV-DXVK: 8 slots (was 2) - the null CLAS ref is BAKED at the promotion-wave
  // build frame but the device-lost happens 2+ frames later (in the volume
  // ReSTIR visibility traversal, per Aftermath markers), so a 2-frame ring had
  // always evicted the bake by dump time.
  static constexpr uint32_t kDbgCaptureSlots        = 8;
  static constexpr uint32_t kDbgCaptureMaxAddresses = 1u << 21;  // 16 MiB of refs per slot
  // NV-DXVK: [BlasHeadScan] Resources access for the device-lost dump (reads the
  // Readback host ring's dbgBlasRefs/dbgBlasHeads written by instance_assign_blas)
  Resources* m_dbgRes = nullptr;
  nvvk::Buffer m_dbgCaptureHost[kDbgCaptureSlots];
  nvvk::Buffer m_dbgStampHost;
  uint32_t     m_dbgCaptureAddressCount = 0;
  VkDeviceSize m_dbgCaptureInfosOffset  = 0;
  VkDeviceSize m_dbgCaptureAddrsOffset  = 0;
  std::vector<std::pair<uint64_t, uint64_t>> m_dbgClasRanges;

  void debugRecordBlasInputCapture(VkCommandBuffer cmd, uint32_t slot);
  void debugStampBuildCompleted(VkCommandBuffer cmd, uint32_t slot);
  void debugDumpBlasInputCapture();
};

bool RendererRayTraceClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.makeCompilerOptions();

  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));
  // NV-DXVK: the sample compiles at runtime with the scene's observed cluster
  // maxima as tight bounds; Remix's build-time variant axis is the SceneConfig
  // points {64,128} (plan 6.3), a correct superset of any observed maximum.
  // Using the config here also keeps appended geometries (P2.5) from ever
  // invalidating the compiled pipelines, since the config is fixed per
  // generation while observed maxima may grow.
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", rscene.scene->m_config.clusterVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", rscene.scene->m_config.clusterTriangles));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "0");
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");
  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0");
  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_PASS_CULLING", "0");
  options.AddMacroDefinition("USE_BLAS_SHARING", config.useBlasSharing ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_MERGING", config.useBlasSharing && config.useBlasMerging ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_CACHING", config.useBlasSharing && config.useBlasCaching ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_SIDED", rscene.scene->m_hasTwoSided && !config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_TWO_SIDED", config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_INVISIBLE_CULLING", config.useForcedInvisibleCulling ? "1" : "0");
  options.AddMacroDefinition("USE_PERSISTENT_TRAVERSAL_KERNEL", config.usePersistentTraversal ? "1" : "0");
  options.AddMacroDefinition("HAS_ALPHA_TEST", rscene.scene->m_hasAlphaMask ? "1" : "0");

  if(m_config.useSorting)
  {
    res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options);
  }

  if(m_config.useBlasSharing)
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init_blas_sharing.comp.glsl", &options);
    if(m_config.useBlasMerging)
    {
      res.compileShader(m_shaders.computeTraversalMerge, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_blas_merging.comp.glsl", &options);
    }
  }
  else
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options);
  }

  res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run_groups.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasInsertClusters, VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasSetupInsertion, VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", &options);
  res.compileShader(m_shaders.computeInstanceAssignBlas, VK_SHADER_STAGE_COMPUTE_BIT, "instance_assign_blas.comp.glsl", &options);

  if(m_config.useBlasSharing)
  {
    res.compileShader(m_shaders.computeInstanceClassifyLod, VK_SHADER_STAGE_COMPUTE_BIT, "instance_classify_lod.comp.glsl", &options);
    res.compileShader(m_shaders.computeGeometryBlasSharing, VK_SHADER_STAGE_COMPUTE_BIT, "geometry_blas_sharing.comp.glsl", &options);
    if(m_config.useBlasCaching)
    {
      res.compileShader(m_shaders.computeBlasCachingSetupCopy, VK_SHADER_STAGE_COMPUTE_BIT,
                        "blas_caching_setup_copy.comp.glsl", &options);
      res.compileShader(m_shaders.computeBlasCachingSetupBuild, VK_SHADER_STAGE_COMPUTE_BIT,
                        "blas_caching_setup_build.comp.glsl", &options);
    }
  }

  return res.verifyShaders(m_shaders);
}

bool RendererRayTraceClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_resourceReservedUsage = {};
  m_config                = config;
  m_maxRenderClusters     = 1u << config.numRenderClusterBits;
  m_maxTraversalTasks     = 1u << config.numTraversalTaskBits;

  // NV-DXVK P2.5: geometry capacity for the geometry-count dependent buffers;
  // 0 keeps the sample's exact-count sizing
  m_maxGeometries = std::max(config.maxGeometries, uint32_t(rscene.scene->getActiveGeometryCount()));

  if(!rscene.useStreaming)
  {
    m_config.useBlasMerging = false;
    m_config.useBlasCaching = false;
  }

  if(!initShaders(res, rscene, m_config))
  {
    LOGE("RendererRayTraceClustersLod shaders failed\n");
    return false;
  }

  if(!rscene.updateClasRequired(true))
  {
    LOGE("RendererRayTraceClustersLod rscene.updateClasRequired failed\n");
    return false;
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.resetCachedBlas();
  }

  initBasics(res, rscene, m_config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.rtClasMemBytes     = rscene.getClasSize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");

  {
    // get ray tracing properties

    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_accProperties};
    m_accProperties.pNext = &m_rtClasProperties;
    vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &prop2);

    VkDeviceSize scratchSize = 0;
    if(rscene.useStreaming)
    {
      scratchSize = rscene.sceneStreaming.getRequiredClasScratchSize();
      LOGI("raytracer: CLAS scratchsize %d KiB\n", uint32_t((scratchSize + 1023) / 1024));
    }

    if(!initRayTracingBlas(res, rscene, m_config, scratchSize))
    {
      LOGE("Resources exceeding max buffer allocation size\n");
      deinit(res);
      return false;
    }

    // NV-DXVK: TlasInstance staging array replaces the sample's TLAS instance
    // buffer; contents are CPU-provided each frame, blasReference patched by
    // instance_assign_blas, then copied into AccelManager's instance buffer.
    res.createBuffer(m_tlasInstancesBuffer, sizeof(VkAccelerationStructureInstanceKHR) * m_config.maxRenderInstances,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    NVVK_DBG_NAME(m_tlasInstancesBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_tlasInstancesBuffer.bufferSize, "operations", "rt instances");

    // streaming also stores newly built clas in scratch
    res.createBuffer(m_scratchBuffer, scratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, m_accProperties.minAccelerationStructureScratchOffsetAlignment);
    NVVK_DBG_NAME(m_scratchBuffer.buffer);

    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_scratchBuffer.bufferSize, "operations", "rt scratch");
  }

  // scene building data

  {
    // NV-DXVK: TRANSFER_SRC so debugRecordBlasInputCapture's forensic mirror
    // copy is VALID (it was silently no-op'ing without it - VUID 00118 - which
    // is why [BlasCapture] reported "0 builds" on every device-lost dump).
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                         | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    NVVK_DBG_NAME(m_sceneBuildBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = m_config.maxRenderInstances;
    m_sceneBuildShaderio.maxRenderClusters  = m_maxRenderClusters;
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);
    m_sceneBuildShaderio.tlasInstances      = m_tlasInstancesBuffer.address;
    m_sceneBuildShaderio.numGeometries      = uint32_t(rscene.scene->getActiveGeometryCount());

    m_sceneBuildShaderio.indirectDispatchGroups.gridY        = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ        = 1;
    m_sceneBuildShaderio.indirectDispatchBlasInsertion.gridY = 1;
    m_sceneBuildShaderio.indirectDispatchBlasInsertion.gridZ = 1;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    m_sceneBuildShaderio.instanceVisibility =
        mem.append(sizeof(uint8_t) * nvutils::align_up(size_t(m_config.maxRenderInstances), TRAVERSAL_INIT_WORKGROUP), 4);
    m_sceneBuildShaderio.blasBuildInfos = mem.append(sizeof(shaderio::BlasBuildInfo) * m_maxBlasBuilds, 16);
    m_sceneBuildShaderio.instanceBuildInfos = mem.append(sizeof(shaderio::InstanceBuildInfo) * m_config.maxRenderInstances, 16);

    if(m_config.useBlasSharing)
    {
      // NV-DXVK P2.5: sized by geometry capacity so appended geometries fit
      m_sceneBuildShaderio.geometryBuildInfos =
          mem.append(sizeof(shaderio::GeometryBuildInfo) * m_maxGeometries, 16);

      if(m_config.useBlasCaching)
      {
        m_sceneBuildShaderio.cachedBlasClusterAddressesDst =
            mem.append(sizeof(uint64_t) * rscene.sceneStreaming.getMaxCachedBlasBuilds(), 8);
        m_sceneBuildShaderio.cachedBlasClusterAddressesSrc =
            mem.append(sizeof(uint64_t) * rscene.sceneStreaming.getMaxCachedBlasBuilds(), 8);
      }
    }

    if(m_config.useSorting)
    {
      // can alias some data required for sorting, with other data used at traversal/blas time.
      mem.beginOverlap();
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_config.maxRenderInstances, 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_config.maxRenderInstances, 4);
      mem.splitOverlap();
    }

    m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    m_sceneBuildShaderio.blasBuildSizes      = mem.append(sizeof(uint32_t) * m_maxBlasBuilds, 4);
    m_sceneBuildShaderio.blasBuildAddresses  = mem.append(sizeof(uint64_t) * m_maxBlasBuilds, 8);

    m_sceneBuildShaderio.blasClusterAddresses = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(m_config.useSorting)
    {
      mem.endOverlap();
    }

    // NV-DXVK: TRANSFER_SRC for debugRecordBlasInputCapture (see m_sceneBuildBuffer)
    res.createBuffer(m_sceneDataBuffer, mem.getSize(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                         | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    NVVK_DBG_NAME(m_sceneDataBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    if(m_config.useBlasSharing)
    {
      m_sceneBuildShaderio.geometryBuildInfos += m_sceneDataBuffer.address;
      if(m_config.useBlasCaching)
      {
        m_sceneBuildShaderio.cachedBlasClusterAddressesDst += m_sceneDataBuffer.address;
        m_sceneBuildShaderio.cachedBlasClusterAddressesSrc += m_sceneDataBuffer.address;
      }
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    NVVK_DBG_NAME(m_sceneTraversalBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;

    for(uint32_t ringSlot = 0; ringSlot < kBlasRingSlots; ringSlot++)
    {
#if USE_LARGE_BUFFER_BLAS
      res.createLargeBuffer(m_sceneBlasDataBuffers[ringSlot], m_blasDataSize,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
#else
      res.createBuffer(m_sceneBlasDataBuffers[ringSlot], m_blasDataSize,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
#endif
      NVVK_DBG_NAME(m_sceneBlasDataBuffers[ringSlot].buffer);
      // NV-DXVK: [HeadWatch] Path-A per-frame BLAS ring - the pool the fatal
      // TLAS refs live in; AccelManager re-reads heads at these refs per frame
      lodclusters::registerWatchedAsPool(m_sceneBlasDataBuffers[ringSlot].buffer, m_sceneBlasDataBuffers[ringSlot].address,
                                         m_sceneBlasDataBuffers[ringSlot].bufferSize);
    }

    // NV-DXVK: device-lost forensics host mirrors (see the member block)
    {
      m_dbgCaptureAddressCount = std::min(m_sceneBuildShaderio.maxRenderClusters, kDbgCaptureMaxAddresses);
      m_dbgCaptureInfosOffset  = nvutils::align_up(VkDeviceSize(sizeof(shaderio::SceneBuilding)), VkDeviceSize(16));
      m_dbgCaptureAddrsOffset =
          m_dbgCaptureInfosOffset + nvutils::align_up(VkDeviceSize(sizeof(shaderio::BlasBuildInfo)) * m_maxBlasBuilds, VkDeviceSize(16));
      const VkDeviceSize captureSize = m_dbgCaptureAddrsOffset + sizeof(uint64_t) * VkDeviceSize(m_dbgCaptureAddressCount);

      for(uint32_t slot = 0; slot < kDbgCaptureSlots; slot++)
      {
        res.createBuffer(m_dbgCaptureHost[slot], captureSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
      }
      res.createBuffer(m_dbgStampHost, sizeof(uint32_t) * 2 * kDbgCaptureSlots, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
      if(m_dbgStampHost.mapping)
      {
        memset(m_dbgStampHost.mapping, 0, sizeof(uint32_t) * 2 * kDbgCaptureSlots);
      }

      if(m_dbgCaptureAddressCount < m_sceneBuildShaderio.maxRenderClusters)
      {
        LOGI("[BlasCapture] cluster reference capture capped at %u of %u possible entries\n", m_dbgCaptureAddressCount,
             m_sceneBuildShaderio.maxRenderClusters);
      }

      res.deviceLostDumpFn = [this]() { debugDumpBlasInputCapture(); };
      // NV-DXVK: also reachable from DxvkSubmissionQueue's device-lost site
      // (the only observer when no temp submit is in flight at the loss).
      // debugDumpBlasInputCapture itself guards against double-dump.
      deviceLostQueueDumpFn() = [this]() { debugDumpBlasInputCapture(); };
      // NV-DXVK: [BlasHeadScan] readback-ring access for the dump
      m_dbgRes = &res;

      // NV-DXVK: NVVK_CHECK exit()s the process on the FIRST failing thread -
      // including the template system's Resources, which has no dump hook. This
      // process-wide pre-exit callback ensures whichever thread fails first
      // writes the dump (or blocks inside it until it is complete).
      nvvk::CheckError::getInstance().setCallbackFunction([this](VkResult result) {
        if(result == VK_ERROR_DEVICE_LOST)
        {
          debugDumpBlasInputCapture();
        }
      });
    }

    if(m_config.useBlasSharing)
    {
      // NV-DXVK P2.5: sized by geometry capacity so appended geometries fit
      res.createBuffer(m_sceneGeometryHistogramBuffer, sizeof(shaderio::GeometryBuildHistogram) * m_maxGeometries,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      NVVK_DBG_NAME(m_sceneGeometryHistogramBuffer.buffer);
      m_resourceReservedUsage.operationsMemBytes +=
          logMemoryUsage(m_sceneGeometryHistogramBuffer.bufferSize, "operations", "build geo");

      m_sceneBuildShaderio.geometryHistograms = m_sceneGeometryHistogramBuffer.address;
    }
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.updateBindings(m_sceneBuildBuffer);
  }

  // use a single common descriptor set for all operations

  {
    // NV-DXVK: compute-only; the sample's RT stages are gone
    m_stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERMATERIALS_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    if(rscene.useStreaming)
    {
      bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }

    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});

    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERMATERIALS_SSBO), m_renderMaterialBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), &res.m_hizUpdate[0].farImageInfo);
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }

    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  // initialize traversal pipeline

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    if(config.useSorting)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalGroups);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalGroups);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasInsertClusters);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasInsertClusters);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasSetupInsertion);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasSetupInsertion);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeInstanceAssignBlas);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstanceAssignBlas);

    if(m_config.useBlasSharing)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeInstanceClassifyLod);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstanceClassifyLod);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeGeometryBlasSharing);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeGeometryBlasSharing);
      if(m_config.useBlasMerging)
      {
        shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalMerge);
        vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalMerge);
      }
      if(m_config.useBlasCaching)
      {
        shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasCachingSetupCopy);
        vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasCachingSetupCopy);

        shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasCachingSetupBuild);
        vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasCachingSetupBuild);
      }
    }
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRayTraceClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, const RendererFrameInput& frameInput, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  {
    glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();
    float     pixelScale  = std::min(renderScale.x, renderScale.y);

    m_sceneBuildShaderio.errorOverDistanceThreshold =
        clusterLodErrorOverDistance(frame.lodPixelError * pixelScale, frame.traversalFov, frame.traversalViewHeight);
  }

  m_sceneBuildShaderio.traversalViewMatrix    = frame.traversalViewMatrix;
  m_sceneBuildShaderio.cullViewProjMatrix     = frame.cullViewProjMatrix;
  m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrixLast;

  m_sceneBuildShaderio.frameIndex            = m_frameIndex;
  m_sceneBuildShaderio.culledErrorScale      = std::max(1.0f, frame.culledErrorScale);
  m_sceneBuildShaderio.sharingPushCulled     = frame.sharingPushCulled;
  m_sceneBuildShaderio.sharingTolerantLevels = frame.sharingTolerantLevels;
  m_sceneBuildShaderio.sharingEnabledLevels  = frame.sharingEnabledLevels;

  // NV-DXVK: per-frame instance count and external TlasInstance array
  m_sceneBuildShaderio.numRenderInstances = frameInput.numRenderInstances;
  m_sceneBuildShaderio.maxRenderClusters  = m_maxRenderClusters;
  m_sceneBuildShaderio.tlasInstances      = frameInput.tlasInstancesAddress;

  // NV-DXVK P2.5: geometry count may grow between frames via appendGeometries;
  // the caller advances it only once an appended range is fully uploaded
  if(frameInput.numGeometries)
  {
    assert(frameInput.numGeometries <= m_maxGeometries);
    m_sceneBuildShaderio.numGeometries = std::min(frameInput.numGeometries, m_maxGeometries);
  }

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants),
                    (const uint32_t*)&frame.frameConstants);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(m_config.useBlasSharing)
  {
    vkCmdFillBuffer(cmd, m_sceneGeometryHistogramBuffer.buffer, 0, m_sceneGeometryHistogramBuffer.bufferSize, 0);
  }

  if(rscene.useStreaming)
  {
    SceneStreaming::FrameSettings settings;
    settings.ageThreshold          = frame.streamingAgeThreshold;
    settings.useBlasCaching        = m_config.useBlasSharing && m_config.useBlasCaching;
    settings.blasCacheFlags        = m_config.clusterBlasFlags;
    settings.blasCacheMaxClusters  = m_maxRenderClusters;
    settings.blasCacheMaxBuilds    = m_maxBlasBuilds;
    settings.blasCacheAgeThreshold = frame.cachingAgeThreshold;
    settings.blasCacheMinLevel     = frame.cachingEnabledLevels;

    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer, settings, profiler);

    if(m_config.useBlasSharing && m_config.useBlasCaching)
    {
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      // can't add as much dynamic clusters as we reduce the budget with the once from the cached geometry builds
      m_sceneBuildShaderio.maxRenderClusters = m_maxRenderClusters - shaderData.update.patchCachedClustersCount;
    }
  }


  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);


  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);


  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, m_scratchBuffer.address, profiler);
  }


  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

    if(m_config.useSorting)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalPresort);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_PRESORT_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vrdxCmdSortKeyValue(cmd, res.m_vrdxSorter, m_sceneBuildShaderio.numRenderInstances, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortKeys - m_sceneDataBuffer.address, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortValues - m_sceneDataBuffer.address,
                          m_sortingAuxBuffer.buffer, 0, nullptr, 0);
    }

    if(m_config.useBlasSharing)
    {
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Instance Classify");

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceClassifyLod);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_CLASSIFY_LOD_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Geometry Blas Sharing");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeGeometryBlasSharing);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numGeometries, GEOMETRY_BLAS_SHARING_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }
    }

    if(m_config.useBlasSharing || m_config.useSorting)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    // we prepare traversal by filling in instance root nodes into the traversal queue
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP));

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in traversal queue

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");

    // this does the main traversal
    // it returns a list of render clusters

    if(m_config.usePersistentTraversal)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
    }
    else
    {
      // this is typically faster
      constexpr bool batchGroupsAtEnd = true;

      for(uint32_t p = 0; p < rscene.scene->m_maxNodeTreeDepth; p++)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchNodes));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        // first time we hit hiNodes == 1, is when we built the tree over the groups of highest detail lod
        // second time is when we link that node to the root node
        bool isLast = rscene.scene->m_maxNodeTreeDepth - 1 == p;

        uint32_t buildSetupID = !isLast && batchGroupsAtEnd ? BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY :
                                                              BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
        vkCmdDispatch(cmd, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        if(buildSetupID == BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED)
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
          vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
        }
      }
    }

    bool useBlasMerging = m_config.useBlasSharing && m_config.useBlasMerging;
    if(useBlasMerging)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(useBlasMerging)
    {
      // This kernel builds all clusters for merged instances based on residency of cluster groups.
      // It also does the age update for resident groups.

      assert(rscene.useStreaming);
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalMerge);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(shaderData.resident.activeGroupsCount, TRAVERSAL_BLAS_MERGING_WORKGROUP));
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in render list

    uint32_t buildSetupID = BUILD_SETUP_BLAS_INSERTION;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  if(rscene.useStreaming)
  {
    // This operation gives us new CLAS addresses

    // The age filter is skipped as it was handled during traversal already
    bool runAgeFilter = !(m_config.useBlasSharing && m_config.useBlasMerging);

    rscene.sceneStreaming.cmdPostTraversal(cmd, m_scratchBuffer.address, runAgeFilter, profiler);

    // no barrier needed here, given the critical barrier prior using these addresses
    // is directly prior running `m_pipelines.computeBlasInsertCluster`
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build Preparation");
    // this kernel prepares the per-blas clas reference list starting position.
    // it also resets the per-blas clas counters.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetupInsertion);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, BLAS_SETUP_INSERTION_WORKGROUP));

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // let's fill in the clusters from the unsorted render list, into the per-blas clas reference lists.

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInsertClusters);
    vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchBlasInsertion));


    if(m_config.useBlasSharing && m_config.useBlasCaching)
    {
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      if(shaderData.update.patchCachedBlasCount)
      {
        // seed blas builds
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasCachingSetupBuild);
        // one work group per geometry
        res.cmdLinearDispatch(cmd, shaderData.update.patchCachedBlasCount);
      }
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT
                               | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                             | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  if(rscene.useStreaming)
  {
    // initialize download work here so it can overlap with next
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  // NV-DXVK: forensic capture of this frame's BLAS build inputs before the build
  m_dbgClasRanges.clear();
  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.appendClasRanges(m_dbgClasRanges);
  }
  debugRecordBlasInputCapture(cmd, m_frameIndex % kDbgCaptureSlots);

  // what is this? nah we never had any bugs in building and allocating the cluster data, totally not needed
#if !STREAMING_DEBUG_WITHOUT_RT
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build");

    // after we prepared the build information for the blas we can run it.

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV& inputs = cmdInfo.input;
    inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    // setup blas inputs
    inputs.maxAccelerationStructureCount = m_maxBlasBuilds;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel  = &m_blasInput;
    inputs.flags                         = m_config.clusterBlasFlags;

    // input
    // we may actually build less BLAS than instances, due pre-build low detail blas or recycling
    cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, blasBuildCounter);

    cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.blasBuildInfos;
    cmdInfo.srcInfosArray.size =
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * m_config.maxRenderInstances;
    cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    // output
    // the blas addresses are later provided to the ray instances
    cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.blasBuildAddresses;
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_config.maxRenderInstances;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    // for statistics we keep track of blas sizes
    cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.blasBuildSizes;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_config.maxRenderInstances;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    // in implicit mode we provide one big chunk from which outputs are sub-allocated
    // NV-DXVK (R17): rotate the pool so in-flight traces never see this build
    cmdInfo.dstImplicitData = m_sceneBlasDataBuffers[m_frameIndex % kBlasRingSlots].address;

    cmdInfo.scratchData = m_scratchBuffer.address;

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    // NV-DXVK: post-stamp - executes only if the build did not kill the device
    debugStampBuildCompleted(cmd, m_frameIndex % kDbgCaptureSlots);

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    if(m_config.useBlasSharing && m_config.useBlasCaching)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Blas Copy");

      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      if(shaderData.update.patchCachedBlasCount)
      {
        // prepare copy and then execute copy
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasCachingSetupCopy);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(shaderData.update.patchCachedBlasCount, BLAS_CACHING_SETUP_COPY_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
        inputs  = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

        // setup move inputs
        inputs.maxAccelerationStructureCount = shaderData.update.patchCachedBlasCount;
        inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
        inputs.opInput.pMoveObjects          = &m_blasMoveInput;
        inputs.flags                         = 0;

        // input
        cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, cachedBlasCopyCounter);

        cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.cachedBlasClusterAddressesSrc;
        cmdInfo.srcInfosArray.size = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV) * m_config.maxRenderInstances;
        cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);

        // output
        cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.cachedBlasClusterAddressesDst;
        cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_config.maxRenderInstances;
        cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

        cmdInfo.scratchData = m_scratchBuffer.address;

        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }
    }
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Preparation");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceAssignBlas);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_ASSIGN_BLAS_WORKGROUP));

    // NV-DXVK: the patched TlasInstances are consumed by a transfer copy into
    // AccelManager's instance buffer (then its TLAS build) instead of a local
    // TLAS build; the readback is copied to host memory below.
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }
#endif

  // NV-DXVK: copy this frame's readback into the cycled host buffer (the
  // sample's viewer app did this in its frame loop).
  {
    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = sizeof(shaderio::Readback) * res.m_cycleIndex;
    region.size      = sizeof(shaderio::Readback);
    vkCmdCopyBuffer(cmd, res.m_commonBuffers.readBack.buffer, res.m_commonBuffers.readBackHost.buffer, 1, &region);
  }

  {
    // reservation for geometry may change
    m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
    // reservation for blas may change (R17: the implicit pool is ringed)
    m_resourceReservedUsage.rtBlasMemBytes = m_blasDataSize * kBlasRingSlots + rscene.getBlasSize(true);

    m_resourceActualUsage                  = m_resourceReservedUsage;
    m_resourceActualUsage.geometryMemBytes = rscene.getGeometrySize(false);
    m_resourceActualUsage.rtClasMemBytes   = rscene.getClasSize(false);

    shaderio::Readback readback;
    res.getReadbackData(readback);
    m_resourceActualUsage.rtBlasMemBytes = readback.blasActualSizes + rscene.getBlasSize(false);

    // NV-DXVK: [ZeroRefScan] a render instance reached the TLAS build with a 0
    // blasReference (the null read as GPU VA=0). Name it so we can trace which
    // geometry/build-index produced it. Readback lags a couple frames.
    if(readback.zeroRefCount != 0)
    {
      LOGE("[ZeroRefScan] %u render instance(s) have blasReference 0 | first: instanceId %u geometryId %u buildIndex 0x%x"
           "  *** NULL TLAS REF ***\n",
           readback.zeroRefCount, readback.zeroRefInstanceId, readback.zeroRefGeometryId, readback.zeroRefBuildIndex);
    }
  }

  // NV-DXVK: [BlasWave] every render() is a Path-A cluster BLAS build wave into
  // pool slot m_frameIndex%kBlasRingSlots. The BlasCapture stamps proved this
  // ran ONCE by crash time while streaming churns CLAS residency every frame -
  // log each wave so the fatal frame's TLAS refs can be dated against the wave
  // that built them (a stale wave + CLAS unloads in between = UAF underneath a
  // frozen BLAS).
  LOGI("[BlasWave] wave %u done (pool slot %u)\n", m_frameIndex, m_frameIndex % kBlasRingSlots);

  m_frameIndex++;
}

// NV-DXVK: see the forensic-capture member block
void RendererRayTraceClustersLod::debugRecordBlasInputCapture(VkCommandBuffer cmd, uint32_t slot)
{
  nvvk::Buffer& dst = m_dbgCaptureHost[slot];
  if(dst.buffer == VK_NULL_HANDLE)
  {
    return;
  }

  // pre-stamp: +1 so a raw 0 means "slot never used"
  const uint32_t frameStamp = m_frameIndex + 1;
  vkCmdFillBuffer(cmd, m_dbgStampHost.buffer, sizeof(uint32_t) * (slot * 2 + 0), sizeof(uint32_t), frameStamp);

  VkBufferCopy region;
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size      = sizeof(shaderio::SceneBuilding);
  vkCmdCopyBuffer(cmd, m_sceneBuildBuffer.buffer, dst.buffer, 1, &region);

  VkBufferCopy regions[2];
  regions[0].srcOffset = m_sceneBuildShaderio.blasBuildInfos - m_sceneDataBuffer.address;
  regions[0].dstOffset = m_dbgCaptureInfosOffset;
  regions[0].size      = sizeof(shaderio::BlasBuildInfo) * VkDeviceSize(m_maxBlasBuilds);
  regions[1].srcOffset = m_sceneBuildShaderio.blasClusterAddresses - m_sceneDataBuffer.address;
  regions[1].dstOffset = m_dbgCaptureAddrsOffset;
  regions[1].size      = sizeof(uint64_t) * VkDeviceSize(m_dbgCaptureAddressCount);
  vkCmdCopyBuffer(cmd, m_sceneDataBuffer.buffer, dst.buffer, 2, regions);

  // the mirrors must be fully written before the build gets a chance to
  // page-fault the device, else the capture of the faulting frame is lost
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
}

// NV-DXVK
void RendererRayTraceClustersLod::debugStampBuildCompleted(VkCommandBuffer cmd, uint32_t slot)
{
  if(m_dbgStampHost.buffer == VK_NULL_HANDLE)
  {
    return;
  }

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
  vkCmdFillBuffer(cmd, m_dbgStampHost.buffer, sizeof(uint32_t) * (slot * 2 + 1), sizeof(uint32_t), m_frameIndex + 1);
}

// NV-DXVK
void RendererRayTraceClustersLod::debugDumpBlasInputCapture()
{
  // one dump per process run; any other failing thread blocks here until the
  // dump is written, so its NVVK_CHECK exit() cannot truncate it
  static std::mutex s_dumpMutex;
  static bool       s_dumped = false;
  std::lock_guard<std::mutex> lock(s_dumpMutex);
  if(s_dumped)
  {
    return;
  }
  s_dumped = true;

  if(m_dbgStampHost.mapping == nullptr)
  {
    return;
  }

  const uint32_t* stamps = reinterpret_cast<const uint32_t*>(m_dbgStampHost.mapping);

  LOGE("[BlasCapture] ==== device lost - dumping captured cluster BLAS build inputs ====\n");
  for(const std::pair<uint64_t, uint64_t>& range : m_dbgClasRanges)
  {
    LOGE("[BlasCapture] valid clas range 0x%llx..0x%llx\n", (unsigned long long)range.first, (unsigned long long)range.second);
  }

  // NV-DXVK: a cluster REFERENCE is a CLAS (cluster-level AS) address, aligned to
  // clusterByteAlignment - NOT clusterBottomLevelByteAlignment, which is the BLAS
  // OUTPUT alignment. The reference sample packs CLAS at clusterByteAlignment
  // granularity (scene_streaming.cpp CLAS allocator) and never rounds refs to the
  // bottom-level alignment. Checking against the stricter bottom-level value made
  // valid 128-aligned refs read as MISALIGNED (false positive). Log both so the
  // value is ground-truth in the dump, and validate against the correct one.
  LOGE("[BlasCapture] CLAS alignment: clusterByteAlignment %u, clusterBottomLevelByteAlignment %u (refs checked vs clusterByteAlignment)\n",
       m_rtClasProperties.clusterByteAlignment, m_rtClasProperties.clusterBottomLevelByteAlignment);
  const uint32_t clasAlignment = std::max(1u, m_rtClasProperties.clusterByteAlignment);

  for(uint32_t slot = 0; slot < kDbgCaptureSlots; slot++)
  {
    const uint32_t pre  = stamps[slot * 2 + 0];
    const uint32_t post = stamps[slot * 2 + 1];

    LOGE("[BlasCapture] slot %u: pre-stamp %u post-stamp %u -> %s\n", slot, pre, post,
         pre == 0 ? "never used" : (pre == post ? "build COMPLETED" : "build DID NOT COMPLETE (faulting frame)"));

    if(pre == 0 || m_dbgCaptureHost[slot].mapping == nullptr)
    {
      continue;
    }

    const uint8_t*                  base  = m_dbgCaptureHost[slot].mapping;
    const shaderio::SceneBuilding&  sb    = *reinterpret_cast<const shaderio::SceneBuilding*>(base);
    const shaderio::BlasBuildInfo*  infos = reinterpret_cast<const shaderio::BlasBuildInfo*>(base + m_dbgCaptureInfosOffset);
    const uint64_t*                 addrs = reinterpret_cast<const uint64_t*>(base + m_dbgCaptureAddrsOffset);
    const uint64_t                  addrsBase = m_sceneBuildShaderio.blasClusterAddresses;
    const uint64_t                  addrsEnd  = addrsBase + sizeof(uint64_t) * uint64_t(m_dbgCaptureAddressCount);

    LOGE("[BlasCapture] slot %u counters: blasBuildCounter %u (max %u), blasClasCounter %u, renderClusterCounter %u\n",
         slot, sb.blasBuildCounter, m_maxBlasBuilds, sb.blasClasCounter, sb.renderClusterCounter);

    const uint32_t buildCount = std::min(sb.blasBuildCounter, m_maxBlasBuilds);
    uint32_t       badBuilds = 0, badRefs = 0, loggedLines = 0;

    for(uint32_t i = 0; i < buildCount; i++)
    {
      const shaderio::BlasBuildInfo& info = infos[i];

      const bool ptrOk = info.clusterReferences >= addrsBase
                         && info.clusterReferences + uint64_t(info.clusterReferencesCount) * 8 <= addrsEnd;
      const bool metaOk = info.clusterReferencesStride == 8 && info.clusterReferencesCount > 0
                          && info.clusterReferencesCount <= m_blasInput.maxClusterCountPerAccelerationStructure;

      if(!ptrOk || !metaOk)
      {
        badBuilds++;
        if(loggedLines < 48)
        {
          LOGE("[BlasCapture]  BAD BUILD %u: count %u stride %u references 0x%llx (array 0x%llx..0x%llx)\n", i,
               info.clusterReferencesCount, info.clusterReferencesStride, (unsigned long long)info.clusterReferences,
               (unsigned long long)addrsBase, (unsigned long long)addrsEnd);
          loggedLines++;
        }
      }
      if(!ptrOk)
      {
        continue;
      }

      const uint64_t* refList = addrs + (info.clusterReferences - addrsBase) / 8;
      for(uint32_t c = 0; c < info.clusterReferencesCount; c++)
      {
        const uint64_t a       = refList[c];
        bool           inRange = false;
        for(const std::pair<uint64_t, uint64_t>& range : m_dbgClasRanges)
        {
          if(a >= range.first && a < range.second)
          {
            inRange = true;
            break;
          }
        }
        if(a == 0 || (a % clasAlignment) != 0 || !inRange)
        {
          badRefs++;
          if(loggedLines < 96)
          {
            LOGE("[BlasCapture]  BAD REF build %u ref %u/%u: 0x%llx%s%s%s\n", i, c, info.clusterReferencesCount,
                 (unsigned long long)a, a == 0 ? " NULL" : "", (a % clasAlignment) != 0 ? " MISALIGNED" : "",
                 !inRange && a != 0 ? " OUT-OF-POOL" : "");
            loggedLines++;
          }
        }
      }
    }

    LOGE("[BlasCapture] slot %u summary: %u builds checked, %u bad build infos, %u bad clas refs\n", slot, buildCount,
         badBuilds, badRefs);
  }
  LOGE("[BlasCapture] ==== dump end ====\n");

  // NV-DXVK: [BlasHeadScan] the GPU-side content probe: instance_assign_blas
  // mirrored every render instance's FINAL blasReference and the u64 AT that
  // address into the Readback ring (4 frames of history). A nonzero ref with a
  // ZERO head = the TLAS referenced AS memory nothing had built at assign time.
  if(m_dbgRes != nullptr && m_dbgRes->m_commonBuffers.readBackHost.buffer)
  {
    const shaderio::Readback* ring = m_dbgRes->m_commonBuffers.readBackHost.data();
    LOGE("[BlasHeadScan] ==== readback ring (cycle now %u) ====\n", m_dbgRes->m_cycleIndex);
    for(uint32_t r = 0; r < 4; r++)
    {
      const shaderio::Readback& rb = ring[r];
      uint32_t nonzeroRefs = 0, zeroHeads = 0;
      int      firstBad = -1;
      for(uint32_t i = 0; i < 64; i++)
      {
        if(rb.dbgBlasRefs[i] == 0)
          continue;
        nonzeroRefs++;
        if(rb.dbgBlasHeads[i] == 0)
        {
          zeroHeads++;
          if(firstBad < 0) firstBad = int(i);
        }
      }
      LOGE("[BlasHeadScan] ring %u: numBlasBuilds %u zeroRefCount %u | refs %u zeroHeads %u first %d%s\n", r,
           rb.numBlasBuilds, rb.zeroRefCount, nonzeroRefs, zeroHeads, firstBad,
           zeroHeads ? "  *** REF INTO UNBUILT AS MEMORY ***" : "");
      for(uint32_t i = 0; i < 64 && zeroHeads != 0; i++)
      {
        if(rb.dbgBlasRefs[i] != 0)
        {
          LOGE("[BlasHeadScan]   ring %u inst %u ref 0x%llx head 0x%llx%s\n", r, i,
               (unsigned long long)rb.dbgBlasRefs[i], (unsigned long long)rb.dbgBlasHeads[i],
               rb.dbgBlasHeads[i] == 0 ? "  <-- ZERO" : "");
        }
      }
    }
    LOGE("[BlasHeadScan] ==== dump end ====\n");
  }

  // NV-DXVK: chain the Path B (animated template system) capture - same loss
  if(deviceLostAuxDumpFn())
  {
    deviceLostAuxDumpFn()();
  }
}

void RendererRayTraceClustersLod::deinit(Resources& res)
{
  deinitBasics(res);

  // NV-DXVK: forensic capture teardown
  res.deviceLostDumpFn = nullptr;
  deviceLostQueueDumpFn() = nullptr;
  nvvk::CheckError::getInstance().setCallbackFunction(nvvk::CheckError::Callback());
  for(uint32_t slot = 0; slot < kDbgCaptureSlots; slot++)
  {
    res.m_allocator.destroyBuffer(m_dbgCaptureHost[slot]);
  }
  res.m_allocator.destroyBuffer(m_dbgStampHost);

  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_scratchBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);
  for(uint32_t ringSlot = 0; ringSlot < kBlasRingSlots; ringSlot++)
  {
    // NV-DXVK: [HeadWatch] unregister before destroy
    lodclusters::unregisterWatchedAsPool(m_sceneBlasDataBuffers[ringSlot].buffer);
#if USE_LARGE_BUFFER_BLAS
    res.m_allocator.destroyLargeBuffer(m_sceneBlasDataBuffers[ringSlot]);
#else
    res.m_allocator.destroyBuffer(m_sceneBlasDataBuffers[ringSlot]);
#endif
  }
  res.m_allocator.destroyBuffer(m_sceneGeometryHistogramBuffer);

  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();
  m_resourceReservedUsage = {};
}


bool RendererRayTraceClustersLod::initRayTracingBlas(Resources& res, RenderScene& rscene, const RendererConfig& config, VkDeviceSize& scratchSize)
{
  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.

  m_blasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // Just using m_hiPerGeometryClusters here is problematic, as the intermediate state
  // of a continuous lod can yield higher numbers (especially when streaming may temporarily cause overlapping of different levels).
  // Therefore, we use the highest sum of all clusters across all lod levels.
  m_blasInput.maxClusterCountPerAccelerationStructure = std::min(rscene.scene->m_maxPerGeometryClusters, m_maxRenderClusters);
  m_blasInput.maxTotalClusterCount = m_maxRenderClusters;

  if(config.useBlasSharing && config.useBlasMerging)
  {
    // we are guaranteeing only 2 BLAS per geometry that has multiple instances.
    // one through sharing, one through merging
    // the low-detail is pre-built.
    // NV-DXVK P2.5: sized by geometry capacity so appended geometries fit
    m_maxBlasBuilds = uint32_t(m_maxGeometries * 2);
  }
  else
  {
    m_maxBlasBuilds = m_config.maxRenderInstances;
  }

  if(config.useBlasSharing && config.useBlasCaching)
  {
    // With caching we might build a few extra BLAS per-frame.
    // This value is at maximum `rscene.scene->getActiveGeometryCount()` plus some rounding/alignment.
    m_maxBlasBuilds += rscene.sceneStreaming.getMaxCachedBlasBuilds();
  }

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = m_maxBlasBuilds;
  inputs.opMode                                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel = &m_blasInput;
  inputs.flags                        = config.clusterBlasFlags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);
  LOGI("raytracer: BLAS build scratchsize %d KiB\n", uint32_t((sizesInfo.buildScratchSize + 1023) / 1024));

  m_blasDataSize = sizesInfo.accelerationStructureSize;

  if(config.useBlasSharing && config.useBlasCaching)
  {
    const StreamingConfig& streamingConfig = rscene.sceneStreaming.getStreamingConfig();

    m_blasMoveInput               = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};
    m_blasMoveInput.noMoveOverlap = VK_TRUE;
    m_blasMoveInput.maxMovedBytes = streamingConfig.maxBlasCachingMegaBytes * 1024 * 1024;
    m_blasMoveInput.type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV;

    inputs.maxAccelerationStructureCount = rscene.sceneStreaming.getMaxCachedBlasBuilds();
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    inputs.opInput.pMoveObjects          = &m_blasMoveInput;
    inputs.flags                         = 0;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);
    LOGI("raytracer: BLAS move scratchsize %d KiB\n", uint32_t((sizesInfo.buildScratchSize + 1023) / 1024));
  }

  return true;
}

std::unique_ptr<Renderer> makeRendererRayTraceClustersLod()
{
  return std::make_unique<RendererRayTraceClustersLod>();
}

// NV-DXVK P4: sample function trimmed to the one binding that remains in
// Remix - the HiZ far pyramid consumed by the culling kernels (the sample
// also rewrote its ray tracing depth / DLSS / color targets here).
void RendererRayTraceClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);

  nvvk::WriteSetContainer writeSets;

  writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), &res.m_hizUpdate[0].farImageInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}

}  // namespace lodclusters
