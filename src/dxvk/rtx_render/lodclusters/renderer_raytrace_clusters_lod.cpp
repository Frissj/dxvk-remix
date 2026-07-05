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
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
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

    res.createBuffer(m_sceneDataBuffer, mem.getSize(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
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
  }

  m_frameIndex++;
}

void RendererRayTraceClustersLod::deinit(Resources& res)
{
  deinitBasics(res);

  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_scratchBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);
  for(uint32_t ringSlot = 0; ringSlot < kBlasRingSlots; ringSlot++)
  {
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
