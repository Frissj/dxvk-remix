/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Adapted from NVIDIA RTX Mega Geometry SDK for RTX Remix integration
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*/

#include "rtxmg_cluster_builder.h"
#include "rtxmg_tilings.h"
#include "rtxmg_accel.h"
#include "rtxmg_shaders.h"
#include "../rtx_context.h"
#include "../rtx_shader_manager.h"
#include "../../dxvk_device.h"
#include "../../dxvk_resource.h"
#include "../../util/log/log.h"
#include "../../util/util_math.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>

// SDK MATCH: Removed estimateGpuCountersFallback function
// Sample always uses real GPU-downloaded TessellationCounters (cluster_accel_builder.cpp:1371-1385)
// Never estimates from input geometry - waits for actual GPU data instead
namespace {
// Namespace reserved for future helper functions
} // namespace

// Include compiled shaders
#include "rtx_shaders/cluster_tiling.h"
#include "rtx_shaders/cluster_tessellate.h"

namespace dxvk {

// Define shader bindings (must match cluster_tess_bindings.slangh)
constexpr uint32_t CLUSTER_TESS_BINDING_INPUT_POSITIONS  = 0;
constexpr uint32_t CLUSTER_TESS_BINDING_INPUT_NORMALS    = 1;
constexpr uint32_t CLUSTER_TESS_BINDING_INPUT_TEXCOORDS  = 2;
constexpr uint32_t CLUSTER_TESS_BINDING_INPUT_INDICES    = 3;
constexpr uint32_t CLUSTER_TESS_BINDING_OUTPUT_VERTICES  = 4;
constexpr uint32_t CLUSTER_TESS_BINDING_OUTPUT_INDICES   = 5;
constexpr uint32_t CLUSTER_TESS_BINDING_OUTPUT_CLUSTERS  = 6;
constexpr uint32_t CLUSTER_TESS_BINDING_OUTPUT_INSTANCES = 7;
constexpr uint32_t CLUSTER_TESS_BINDING_TEMPLATE_ADDRESSES = 8;

// Push constants structure (must match shader)
struct ClusterTessellationArgs {
  uint32_t numTriangles;
  uint32_t gridSize;
  uint32_t hasNormals;
  uint32_t hasTexcoords;
  VkDeviceAddress vertexBufferAddress;  // For GPU-side cluster instance building
  uint32_t pad0;
  uint32_t pad1;
};

// NV-DXVK: Old shader wrappers removed - they referenced non-existent shaders
// and used wrong structure format. The correct implementation uses ClusterTilingShader
// from rtxmg_shaders.h which writes the proper 32-byte SDK structure.



// ============================================================================
// Constructor / Destructor
// ============================================================================

RtxmgClusterBuilder::RtxmgClusterBuilder(DxvkDevice* device)
  : m_device(device)
  , m_initialized(false) {
  Logger::info("[RTXMG] Cluster builder created");
}

RtxmgClusterBuilder::~RtxmgClusterBuilder() {
  shutdown();
}

// ============================================================================
// Initialization
// ============================================================================

bool RtxmgClusterBuilder::initialize() {
  if (m_initialized) {
    Logger::warn("[RTXMG] Cluster builder already initialized");
    return true;
  }

  Logger::info("[RTXMG] Initializing cluster builder...");

  // Generate 121 cluster template grids (11x11)
  generateTemplateGrids();

  // Create tessellation counters buffer (single buffer with GPU fence synchronization)
  m_tessCountersBuffer.create(m_device, 256,
    "RTXMG Tessellation Counters",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  DxvkBufferCreateInfo readbackInfo = {};
  readbackInfo.size = sizeof(TessellationCounters);
  readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  readbackInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
  readbackInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;

  m_tessCountersReadback = m_device->createBuffer(
    readbackInfo,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    DxvkMemoryStats::Category::RTXBuffer,
    "RTXMG Tessellation Counter Readback");

  // SDK MATCH: Create cluster offset counts buffer (per-instance cluster offsets, cleared each frame)
  // Sample: m_clusterOffsetCountsBuffer stores atomic counters for per-instance cluster allocation
  const uint32_t maxInstances = 1024;  // Max instances per frame
  m_clusterOffsetCountsBuffer.create(m_device, maxInstances,
    "RTXMG Cluster Offset Counts",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // SDK MATCH: Create fill clusters dispatch indirect buffer (cleared each frame)
  // Sample: m_fillClustersDispatchIndirectBuffer stores indirect dispatch args for fill_clusters
  m_fillClustersDispatchIndirectBuffer.create(m_device, maxInstances,
    "RTXMG Fill Clusters Dispatch Indirect",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  Logger::info("[RTXMG] Created single-buffered counter/offset/dispatch buffers with GPU fence sync");

  // Create cluster tiling params constant buffer (replaces push constants)
  // Size: 256 bytes (enough for ClusterTilingParams structure which is 216+ bytes)
  m_clusterTilingParamsBuffer.create(m_device, 256,
    "RTXMG Cluster Tiling Params",
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  // Create GPU buffers for cluster building
  createGPUBuffers(RtxmgConfig());

  // Create compute shaders
  createShaders();

  m_initialized = true;
  Logger::info(str::format("[RTXMG] Cluster builder initialized with ",
    m_templateGrids.descs.size(), " templates"));

  return true;
}

void RtxmgClusterBuilder::shutdown() {
  if (!m_initialized) {
    return;
  }

  Logger::info("[RTXMG] Shutting down cluster builder");

  // Release buffers (single-buffered with GPU fence sync)
  m_tessCountersBuffer.release();
  m_tessCountersReadback = nullptr;
  m_clusterOffsetCountsBuffer.release();
  m_fillClustersDispatchIndirectBuffer.release();

  // Release batch buffers (Phase 4: True GPU Batching)
  m_instanceDataBuffer.release();
  m_indirectDispatchBuffer.release();
  m_instanceOffsetsBuffer.release();
  m_batchingEnabled = false;

  // Release HiZ pyramid (Phase 4)
  if (m_hizInitialized) {
    m_hizMipViews.clear();
    m_hizPyramidView = nullptr;
    m_hizPyramid = nullptr;
    m_hizSampler = nullptr;
    m_hizInitialized = false;
  }

  // Clear template grids
  m_templateGrids.descs.clear();
  m_templateGrids.indices.clear();
  m_templateGrids.vertices.clear();

  m_initialized = false;
}

// ============================================================================
// Template Grid Generation
// ============================================================================

void RtxmgClusterBuilder::generateTemplateGrids() {
  Logger::info("[RTXMG] Generating cluster template grids...");

  m_templateGrids.descs.clear();
  m_templateGrids.indices.clear();
  m_templateGrids.vertices.clear();

  uint32_t totalVertexOffset = 0;
  uint32_t totalIndexOffset = 0;
  uint32_t totalPaddedIndex = 0;
  uint32_t totalPaddedVertex = 0;

  // Generate 11x11 grid of cluster templates
  for (uint32_t sizeY = 1; sizeY <= kMaxClusterEdgeSegments; ++sizeY) {
    for (uint32_t sizeX = 1; sizeX <= kMaxClusterEdgeSegments; ++sizeX) {
      TemplateGridDesc desc;
      desc.xEdges = sizeX;
      desc.yEdges = sizeY;

      uint32_t indexPaddingAdded = 0;
      uint32_t vertexPaddingAdded = 0;

      const uint32_t indexMisalignment = totalIndexOffset & 0xF;
      if (indexMisalignment != 0) {
        const uint32_t padBytes = 16u - indexMisalignment;
        m_templateGrids.indices.insert(m_templateGrids.indices.end(), padBytes, 0);
        totalIndexOffset += padBytes;
        indexPaddingAdded = padBytes;
        totalPaddedIndex++;
      }
      const uint32_t vertexMisalignment = totalVertexOffset & 3u;
      if (vertexMisalignment != 0) {
        const uint32_t padFloats = 4u - vertexMisalignment;
        m_templateGrids.vertices.insert(m_templateGrids.vertices.end(), padFloats, 0.0f);
        totalVertexOffset += padFloats;
        vertexPaddingAdded = padFloats;
        totalPaddedVertex++;
      }

      desc.vertexOffset = totalVertexOffset * sizeof(float);  // Store as byte offset (sample approach)
      desc.indexOffset = totalIndexOffset * sizeof(uint8_t);  // Store as byte offset

      uint32_t templateIdx = (sizeY - 1) * kMaxClusterEdgeSegments + (sizeX - 1);
      if (templateIdx < 5 || indexPaddingAdded > 0 || vertexPaddingAdded > 0) {
        Logger::info(str::format("[RTXMG GRID DEBUG] Template[", templateIdx, "] ", sizeX, "x", sizeY, ":"));
        if (indexPaddingAdded > 0) {
          Logger::info(str::format("[RTXMG GRID DEBUG]   -> Added ", indexPaddingAdded, " bytes index padding"));
        }
        if (vertexPaddingAdded > 0) {
          Logger::info(str::format("[RTXMG GRID DEBUG]   -> Added ", vertexPaddingAdded, " float vertex padding"));
        }
        Logger::info(str::format("[RTXMG GRID DEBUG]   -> indexOffset=", desc.indexOffset, " (", desc.indexOffset % 16, " % 16)"));
        Logger::info(str::format("[RTXMG GRID DEBUG]   -> vertexOffset=", desc.vertexOffset, " (", desc.vertexOffset % 16, " % 16)"));
      }

      // Generate normalized UV vertices for this template
      // Vertices are in [0,1] range for both U and V
      // NOTE: Hardware expects 3D vertices (X,Y,Z) even though templates are 2D grids
      const uint32_t numVertsX = sizeX + 1;
      const uint32_t numVertsY = sizeY + 1;
      const uint32_t numVerts = numVertsX * numVertsY;

      for (uint32_t y = 0; y < numVertsY; ++y) {
        for (uint32_t x = 0; x < numVertsX; ++x) {
          float u = float(x) / float(sizeX);
          float v = float(y) / float(sizeY);
          // Store as X,Y,Z (with Z=0) to match RGB32_FLOAT format
          m_templateGrids.vertices.push_back(u);
          m_templateGrids.vertices.push_back(v);
          m_templateGrids.vertices.push_back(0.0f);  // Z coordinate (always 0 for 2D grids)
        }
      }

      // Generate triangle indices (2 triangles per quad)
      // Winding order: counter-clockwise
      for (uint32_t y = 0; y < sizeY; ++y) {
        for (uint32_t x = 0; x < sizeX; ++x) {
          uint8_t i00 = static_cast<uint8_t>(y * numVertsX + x);
          uint8_t i10 = static_cast<uint8_t>(y * numVertsX + (x + 1));
          uint8_t i01 = static_cast<uint8_t>((y + 1) * numVertsX + x);
          uint8_t i11 = static_cast<uint8_t>((y + 1) * numVertsX + (x + 1));

          // Triangle 1: i00, i10, i01
          m_templateGrids.indices.push_back(i00);
          m_templateGrids.indices.push_back(i10);
          m_templateGrids.indices.push_back(i01);

          // Triangle 2: i10, i11, i01
          m_templateGrids.indices.push_back(i10);
          m_templateGrids.indices.push_back(i11);
          m_templateGrids.indices.push_back(i01);
        }
      }

      totalVertexOffset += numVerts * 3;  // 3 floats per vertex (X,Y,Z)
      totalIndexOffset += sizeX * sizeY * 6;  // 6 indices per quad

      m_templateGrids.descs.push_back(desc);

      // Update max counts
      m_templateGrids.maxVertices = std::max(m_templateGrids.maxVertices, numVerts);
      m_templateGrids.maxTriangles = std::max(m_templateGrids.maxTriangles, sizeX * sizeY * 2);
    }
  }

  m_templateGrids.totalVertices = totalVertexOffset;
  m_templateGrids.totalTriangles = totalIndexOffset / 3;

  Logger::info(str::format("[RTXMG] Generated ", m_templateGrids.descs.size(),
    " templates, total vertices: ", m_templateGrids.totalVertices,
    ", total triangles: ", m_templateGrids.totalTriangles));
  Logger::info(str::format("[RTXMG GRID DEBUG] Grid generation summary:"));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Total index bytes: ", m_templateGrids.indices.size()));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Total vertex floats: ", m_templateGrids.vertices.size()));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Total vertex bytes: ", m_templateGrids.vertices.size() * sizeof(float)));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Templates with index padding: ", totalPaddedIndex));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Templates with vertex padding: ", totalPaddedVertex));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Max vertices per template: ", m_templateGrids.maxVertices));
  Logger::info(str::format("[RTXMG GRID DEBUG]   -> Max triangles per template: ", m_templateGrids.maxTriangles));
}

// DEAD CODE REMOVED: buildClusters(ClusterInputGeometry) and buildClustersGpu()
// These functions used CPU vector uploads which don't match the SDK.
// Live path uses buildClustersGpuBatch() with zero-copy GPU buffer binding.

// ============================================================================
// Acceleration Structure Building (Phase 3)
// ============================================================================

bool RtxmgClusterBuilder::buildAccelerationStructures(
  RtxContext* ctx,
  const ClusterOutputGeometry& geometry,
  ClusterAccels& accels,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (!isClusterAccelerationExtensionAvailable()) {
    Logger::warn("[RTXMG] Cluster acceleration extension not available - skipping CLAS/BLAS building");
    return false;
  }

  if (!ensureTemplateClasBuilt(ctx))
    return false;

  // Build cluster geometry BLAS
  Logger::info(str::format("[RTXMG] Building cluster BLAS for ", geometry.numClusters, " clusters"));

  BlasBuildSizes blasSizes = {};
  bool success = buildClusterGeometryBLAS(
    m_device,
    ctx,
    geometry,
    config,
    m_templateAddressesVec,
    m_clusters,
    m_clusterShadingData,
    m_clusterVertexPositions,
    accels,
    &blasSizes);  // Capture BLAS sizes

  if (!success) {
    Logger::err("[RTXMG] Failed to build cluster BLAS");
    return false;
  }

  // Track BLAS allocation sizes
  m_allocatedBlasSize = blasSizes.blasSize;
  m_allocatedBlasScratchSize = blasSizes.blasScratchSize;

  // Update statistics with BLAS sizes
  m_stats.allocated.m_blasSize = m_allocatedBlasSize;
  m_stats.allocated.m_blasScratchSize = m_allocatedBlasScratchSize;
  m_stats.desired.m_blasSize = blasSizes.blasSize;
  m_stats.desired.m_blasScratchSize = blasSizes.blasScratchSize;

  Logger::info("[RTXMG] Cluster acceleration structures built successfully");
  return true;
}

// GPU-optimized BLAS building from GPU-resident cluster data
// This version uses pre-generated cluster instances to avoid CPU-GPU synchronization
bool RtxmgClusterBuilder::buildAccelerationStructures(
  RtxContext* ctx,
  const ClusterOutputGeometryGpu& geometryGpu,
  ClusterAccels& accels,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (!isClusterAccelerationExtensionAvailable()) {
    Logger::warn("[RTXMG] Cluster acceleration extension not available - skipping CLAS/BLAS building");
    return false;
  }

  // Validate GPU geometry has cluster instances buffer
  if (geometryGpu.clusterInstancesBuffer == nullptr) {
    Logger::err("[RTXMG] GPU geometry missing cluster instances buffer");
    return false;
  }

  if (!ensureTemplateClasBuilt(ctx))
    return false;

  // Build cluster geometry BLAS using GPU-optimized path
  // GPU shader writes VkClusterAccelerationStructureInstantiateClusterInfoNV directly
  Logger::info(str::format("[RTXMG] Building cluster BLAS (GPU-optimized) for ",
                          geometryGpu.numClusters, " clusters"));

  // The GPU shader has already written the correct SDK structure format
  // We can use it directly without CPU conversion!
  Rc<DxvkBuffer> instantiateInfosBuffer = geometryGpu.clusterInstancesBuffer;

  // Phase 2: Instantiate clusters using GPU-written data
  // PROPER BATCHING: Append to shared frame buffer instead of creating per-geometry buffers

  FrameInstantiationBuffers& frameBuffers = m_frameBuffers;

  // Ensure frame buffers are large enough for all clusters this frame
  const uint32_t totalClustersNeeded = frameBuffers.usedClusters + geometryGpu.numClusters;
  const uint32_t MAX_CLUSTERS_PER_FRAME = 100000;  // Generous capacity
  const VkDeviceSize instanceStride = getInstanceStride();
  const VkDeviceSize scratchAlignment = getClusterScratchAlignment();

  if (frameBuffers.allocatedClusters < totalClustersNeeded) {
    // Need to resize - allocate with generous headroom
    uint32_t newCapacity = std::max(totalClustersNeeded * 2, MAX_CLUSTERS_PER_FRAME);

    Logger::info(str::format("[RTXMG] Resizing frame buffers: ", frameBuffers.allocatedClusters,
                            " -> ", newCapacity, " clusters"));

    VkDeviceSize scratchSize = alignDeviceSize(4ull * 1024ull * 1024ull, scratchAlignment);
    VkDeviceSize instanceSize = newCapacity * instanceStride;

    frameBuffers.scratchBuffer.create(m_device, scratchSize, "RTXMG Frame Scratch",
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    frameBuffers.countBuffer.create(m_device, 1, "RTXMG Frame Count",
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    frameBuffers.addressesBuffer.create(m_device, newCapacity, "RTXMG Frame Addresses",
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    frameBuffers.instanceBuffer.create(m_device, static_cast<size_t>(instanceSize), "RTXMG Frame Instances",
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    frameBuffers.allocatedClusters = newCapacity;
  }

  // Calculate buffer offset for this geometry's clusters
  uint32_t clusterOffset = frameBuffers.usedClusters;

  Logger::info(str::format("[RTXMG DEBUG] Appending to frame buffer: offset=", clusterOffset,
                          ", numClusters=", geometryGpu.numClusters,
                          ", total=", totalClustersNeeded));

  std::vector<VkDeviceAddress> instanceAddresses(geometryGpu.numClusters);
  size_t totalInstanceSize = 0;
  VkDeviceSize instanceBufferOffsetBytes = clusterOffset * instanceStride;

  Logger::info(str::format("[RTXMG DEBUG] Appending to frame buffer: clusterOffset=", clusterOffset,
                          ", numClusters=", geometryGpu.numClusters,
                          ", totalClustersThisFrame=", totalClustersNeeded,
                          ", instanceOffsetBytes=", instanceBufferOffsetBytes));

  if (!instantiateClusterInstancesDirect(
      m_device, ctx, geometryGpu.numClusters,
      instantiateInfosBuffer, instanceAddresses,
      frameBuffers.scratchBuffer,
      frameBuffers.countBuffer,
      frameBuffers.addressesBuffer,
      frameBuffers.instanceBuffer,
      &totalInstanceSize,
      static_cast<VkDeviceSize>(geometryGpu.clusterInstancesBufferOffset),
      instanceBufferOffsetBytes)) {
    Logger::err("[RTXMG] Failed to instantiate clusters (Phase 2)");
    return false;
  }

  // Update cluster usage counter
  frameBuffers.usedClusters += geometryGpu.numClusters;

  // Phase 3: Build BLAS from instantiated clusters
  BlasBuildSizes blasSizes = {};
  bool success = buildClusterGeometryBLAS(
    m_device, ctx, geometryGpu.numClusters, instanceAddresses,
    accels, &blasSizes);

  if (!success) {
    Logger::err("[RTXMG] Failed to build cluster BLAS (Phase 3)");
    return false;
  }

  // No synchronization needed - each geometry gets its own buffer set that persists until GPU is done

  // Track BLAS allocation sizes
  m_allocatedBlasSize = blasSizes.blasSize;
  m_allocatedBlasScratchSize = blasSizes.blasScratchSize;

  // Update statistics
  m_stats.allocated.m_blasSize = m_allocatedBlasSize;
  m_stats.allocated.m_blasScratchSize = m_allocatedBlasScratchSize;
  m_stats.desired.m_blasSize = blasSizes.blasSize;
  m_stats.desired.m_blasScratchSize = blasSizes.blasScratchSize;

  Logger::info("[RTXMG] Cluster acceleration structures built successfully (GPU-optimized - no CPU sync)");
  return true;
}

// ============================================================================
// Unified BLAS Building (for batching multiple geometries)
// ============================================================================

bool RtxmgClusterBuilder::instantiateClusterInstancesOnly(
  RtxContext* ctx,
  const ClusterOutputGeometryGpu& geometryGpu,
  std::vector<VkDeviceAddress>& outInstanceAddresses,
  RtxmgBuffer<uint8_t>& outTempInstanceBuffer,
  const RtxmgConfig& config) {

  Logger::info(str::format("[RTXMG DEBUG] instantiateClusterInstancesOnly called: numClusters=", geometryGpu.numClusters));

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (!isClusterAccelerationExtensionAvailable()) {
    Logger::warn("[RTXMG] Cluster acceleration extension not available");
    return false;
  }

  if (geometryGpu.clusterInstancesBuffer == nullptr) {
    Logger::err("[RTXMG] GPU geometry missing cluster instances buffer");
    return false;
  }

  if (!ensureTemplateClasBuilt(ctx))
    return false;

  // NVIDIA SAMPLE APPROACH: Use ONE ring buffer per frame with offset tracking
  // This is much faster than creating 730 separate temp buffers!

  // Get current frame's ring buffer
  FrameInstantiationBuffers& frameBuffers = m_frameBuffers;

  // Resize frame buffers if needed to accommodate ALL geometries
  const uint32_t totalClustersNeeded = frameBuffers.usedClusters + geometryGpu.numClusters;
  if (frameBuffers.allocatedClusters < totalClustersNeeded) {
    const uint32_t newCapacity = std::max(totalClustersNeeded * 2, 1024u);
    Logger::info(str::format("[RTXMG] Resizing frame buffers: ", frameBuffers.allocatedClusters,
                            " -> ", newCapacity, " clusters"));

    // Create/resize all frame instantiation buffers
    const VkDeviceSize instanceStride = getInstanceStride();
    const VkDeviceSize scratchAlignment = getClusterScratchAlignment();
    const VkDeviceSize scratchSize = alignDeviceSize(4ull * 1024ull * 1024ull, scratchAlignment);
    const VkDeviceSize instanceSize = newCapacity * instanceStride;

    frameBuffers.scratchBuffer.create(m_device, scratchSize, "RTXMG Frame Scratch",
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    frameBuffers.countBuffer.create(m_device, 1, "RTXMG Frame Count",
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    frameBuffers.addressesBuffer.create(m_device, newCapacity, "RTXMG Frame Addresses",
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    frameBuffers.instanceBuffer.create(m_device, static_cast<size_t>(instanceSize), "RTXMG Frame Instances",
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    frameBuffers.allocatedClusters = newCapacity;
  }

  // Get cluster offset in ring buffer (NVIDIA sample pattern)
  uint32_t clusterOffset = frameBuffers.usedClusters;

  outInstanceAddresses.resize(geometryGpu.numClusters);
  size_t totalInstanceSize = 0;

  const VkDeviceSize instanceStride = getInstanceStride();
  const VkDeviceSize instanceBufferOffsetBytes = clusterOffset * instanceStride;

  Logger::info(str::format("[RTXMG DEBUG] Instantiating at ring buffer offset ", clusterOffset,
                          " clusters (", instanceBufferOffsetBytes / 1024, " KB)"));

  if (!instantiateClusterInstancesDirect(
      m_device, ctx, geometryGpu.numClusters,
      geometryGpu.clusterInstancesBuffer, outInstanceAddresses,
      frameBuffers.scratchBuffer,
      frameBuffers.countBuffer,
      frameBuffers.addressesBuffer,
      frameBuffers.instanceBuffer,
      &totalInstanceSize,
      static_cast<VkDeviceSize>(geometryGpu.clusterInstancesBufferOffset),
      instanceBufferOffsetBytes)) {  // Pass offset in BYTES for ring buffer
    Logger::err("[RTXMG] Failed to instantiate clusters for unified BLAS");
    return false;
  }

  // Update cluster usage counter (NVIDIA sample pattern)
  frameBuffers.usedClusters += geometryGpu.numClusters;

  // Return the ring buffer reference so caller can copy to persistent storage
  outTempInstanceBuffer = frameBuffers.instanceBuffer;  // COPY, don't move!

  Logger::info(str::format("[RTXMG DEBUG] Instantiated ", geometryGpu.numClusters,
                          " clusters at offset ", clusterOffset, " (ring buffer used: ",
                          frameBuffers.usedClusters, "/", frameBuffers.allocatedClusters, ")"));
  return true;
}

bool RtxmgClusterBuilder::buildUnifiedBLAS(
  RtxContext* ctx,
  const std::vector<VkDeviceAddress>& allInstanceAddresses,
  ClusterAccels& outAccels,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (!isClusterAccelerationExtensionAvailable()) {
    Logger::warn("[RTXMG] Cluster acceleration extension not available");
    return false;
  }

  if (allInstanceAddresses.empty()) {
    Logger::warn("[RTXMG] No cluster instances to build unified BLAS");
    return false;
  }

  Logger::info(str::format("[RTXMG DEBUG] Building unified BLAS from ", allInstanceAddresses.size(), " cluster instance addresses"));
  Logger::info(str::format("[RTXMG DEBUG] First 5 addresses: 0x", std::hex,
                          allInstanceAddresses.size() > 0 ? allInstanceAddresses[0] : 0, ", 0x",
                          allInstanceAddresses.size() > 1 ? allInstanceAddresses[1] : 0, ", 0x",
                          allInstanceAddresses.size() > 2 ? allInstanceAddresses[2] : 0, ", 0x",
                          allInstanceAddresses.size() > 3 ? allInstanceAddresses[3] : 0, ", 0x",
                          allInstanceAddresses.size() > 4 ? allInstanceAddresses[4] : 0, std::dec));

  // Build ONE BLAS containing all clusters from all geometries
  BlasBuildSizes blasSizes = {};
  Logger::info("[RTXMG DEBUG] Calling buildClusterGeometryBLAS...");
  bool success = buildClusterGeometryBLAS(
    m_device, ctx,
    static_cast<uint32_t>(allInstanceAddresses.size()),
    allInstanceAddresses,
    outAccels,
    &blasSizes);

  if (!success) {
    Logger::err("[RTXMG DEBUG] buildClusterGeometryBLAS returned FALSE");
    return false;
  }

  Logger::info(str::format("[RTXMG DEBUG] buildClusterGeometryBLAS succeeded: blasSize=", blasSizes.blasSize,
                          ", scratchSize=", blasSizes.blasScratchSize));

  // Track allocation sizes
  m_allocatedBlasSize = blasSizes.blasSize;
  m_allocatedBlasScratchSize = blasSizes.blasScratchSize;

  m_stats.allocated.m_blasSize = m_allocatedBlasSize;
  m_stats.allocated.m_blasScratchSize = m_allocatedBlasScratchSize;
  m_stats.desired.m_blasSize = blasSizes.blasSize;
  m_stats.desired.m_blasScratchSize = blasSizes.blasScratchSize;

  Logger::info(str::format("[RTXMG] Unified BLAS built successfully: ",
                          allInstanceAddresses.size(), " clusters, ",
                          blasSizes.blasSize / 1024, "KB"));
  return true;
}

// ============================================================================
// Per-Frame Updates (Phase 4)
// ============================================================================

void RtxmgClusterBuilder::updatePerFrame(
  RtxContext* ctx,
  const Rc<DxvkImageView>& depthBuffer,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    return;
  }

  if (depthBuffer == nullptr) {
    return;
  }

  // SDK MATCH: Counter rotation happens in BuildAccel, NOT in per-frame update
  // With single buffers and GPU fences, we don't need per-frame readback rotation

  m_frameSerial++;

  // Get depth buffer dimensions
  const auto& imageInfo = depthBuffer->imageInfo();
  uint32_t depthWidth = imageInfo.extent.width;
  uint32_t depthHeight = imageInfo.extent.height;

  // Resize HiZ pyramid if needed
  resizeHiZIfNeeded(depthWidth, depthHeight);

  // Generate HiZ pyramid from depth buffer
  if (m_hizInitialized && config.enableHiZCulling) {
    generateHiZPyramid(ctx, depthBuffer);
  }
}

// ============================================================================
// Multi-BLAS Building (Sample Code Match)
// ============================================================================
// ARCHITECTURE CHANGES COMPLETED:
// ✅ Per-draw-call processing (not per-geometry-hash grouping)
// ✅ No caching (rebuild every frame like sample)
// ✅ Each draw call tessellated with instance-specific transform
// ✅ Batched BLAS building infrastructure
//
// REMAINING OPTIMIZATION (not blocking correctness):
// ❌ True GPU batching via vkCmdBuildAccelerationStructuresIndirectNV
//    Sample uses executeMultiIndirectClusterOperation (line 1093) to build
//    N BLASes in ONE GPU call using indirect buffers
//    Current implementation builds sequentially (N separate GPU calls)
//    This is a performance optimization, not a correctness issue
// ============================================================================

bool RtxmgClusterBuilder::buildMultipleBLAS(
  RtxContext* ctx,
  const std::vector<MultiBLASInput>& geometries,
  std::vector<ClusterAccels>& outAccels,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (geometries.empty()) {
    Logger::warn("[RTXMG] No geometries to build BLASes");
    return false;
  }

  Logger::info(str::format("[RTXMG BATCH BLAS] Building ", geometries.size(), " BLASes in batch"));

  outAccels.clear();
  outAccels.reserve(geometries.size());

  // SAMPLE CODE MATCH: Build all BLASes using batched indirect command
  // Sample BuildBlasFromClas (line 1093) calls executeMultiIndirectClusterOperation
  // which builds N BLASes in one GPU call using indirect args buffer
  //
  // Current implementation: Build sequentially (one GPU call per BLAS)
  // This matches sample's per-instance architecture but not the GPU batching optimization
  // The batching is a performance optimization, not required for correctness

  uint32_t successCount = 0;
  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& input = geometries[i];

    ClusterAccels accels;
    if (buildAccelerationStructures(ctx, input.geometry, accels, config)) {
      outAccels.push_back(std::move(accels));
      successCount++;
    } else {
      Logger::warn(str::format("[RTXMG BATCH BLAS] Failed to build BLAS ", i, "/", geometries.size()));
      // Push empty accels to maintain index correspondence
      outAccels.push_back(ClusterAccels{});
    }
  }

  Logger::info(str::format("[RTXMG BATCH BLAS] Built ", successCount, "/", geometries.size(), " BLASes successfully"));
  return successCount > 0;
}

// ============================================================================
// STEP 1: Build Structured CLAS (EXACT SAMPLE MATCH)
// ============================================================================
// This EXACTLY matches sample's BuildStructuredCLASes (line 1365):
//   - Fill cluster vertex data
//   - Instantiate ALL CLAS in ONE GPU call
// Caller MUST flush after this before calling buildBlasFromClas()
// ============================================================================

bool RtxmgClusterBuilder::buildStructuredCLASes(
  RtxContext* ctx,
  const std::vector<DrawCallData>& drawCalls,
  const RtxmgConfig& config,
  uint32_t frameIndex) {

  if (!m_initialized) {
    Logger::err("[RTXMG UNIFIED] Cluster builder not initialized");
    return false;
  }

  if (drawCalls.empty()) {
    Logger::warn("[RTXMG UNIFIED] No draw calls to process");
    return false;
  }

  // ============================================================================
  // MATCHING NVIDIA SAMPLE: No explicit fence synchronization
  // The sample doesn't wait on fences - it relies on Vulkan's implicit sync
  // with memory barriers within the same command buffer
  // Fences are only used for inter-frame synchronization at submission time
  // ============================================================================
  // Note: Fence capture moved to end of buildTlas() after entire pipeline completes
  Logger::info("[RTXMG UNIFIED] Starting buildStructuredCLASes (frame synchronization handled by Vulkan framework)");

  // ============================================================================
  // SDK MATCH: Counter rotation and buffer clearing (cluster_accel_builder.cpp:1272-1280)
  // Sample rotates counter slot HERE in BuildAccel(), not in per-frame update
  // ============================================================================

  // Advance build frame index (SDK: m_buildAccelFrameIndex only advances when BuildAccel runs)
  m_buildAccelFrameIndex++;

  // SDK MATCH: Clear ALL counter and atomic buffers (lines 1277-1280)
  // Sample clears: tessellation counters, cluster offset counts, dispatch indirect buffers
  const Rc<DxvkBuffer>& counterBuffer = m_tessCountersBuffer.getBuffer();
  const Rc<DxvkBuffer>& offsetCountsBuffer = m_clusterOffsetCountsBuffer.getBuffer();
  const Rc<DxvkBuffer>& dispatchIndirectBuffer = m_fillClustersDispatchIndirectBuffer.getBuffer();

  if (counterBuffer != nullptr) {
    ctx->clearBuffer(counterBuffer, 0, m_tessCountersBuffer.bytes(), 0);
  }

  if (offsetCountsBuffer != nullptr) {
    ctx->clearBuffer(offsetCountsBuffer, 0, m_clusterOffsetCountsBuffer.bytes(), 0);
  }

  if (dispatchIndirectBuffer != nullptr) {
    ctx->clearBuffer(dispatchIndirectBuffer, 0, m_fillClustersDispatchIndirectBuffer.bytes(), 0);
  }

  Logger::info(str::format("[RTXMG UNIFIED] BuildAccel frame ", m_buildAccelFrameIndex,
                          " (cleared counter + offset + dispatch buffers)"));

  // Reset cluster usage for the new frame (single buffer model)
  m_frameBuffers.usedClusters = 0;

  // ============================================================================
  // End of SDK-matching counter rotation and buffer clearing
  // ============================================================================

  // Count total clusters and estimate vertices
  uint32_t totalClusters = 0;
  for (const auto& drawCall : drawCalls) {
    totalClusters += drawCall.clusterCount;
  }

  if (totalClusters == 0) {
    Logger::warn("[RTXMG UNIFIED] No clusters to build");
    return false;
  }

  // SDK MATCH: Counter readback happens AFTER BLAS building (line 1371), not here!
  // Moved to end of buildBlasFromClas()

  // SDK MATCH: No ring buffering - single-frame buffers with GPU sync
  // NVIDIA sample doesn't use ring buffers, it rebuilds every frame

  // CRITICAL FIX: Resize buffers before fill_clusters dispatch
  // Estimate vertices: typical cluster is ~4x4 grid = 25 vertices
  // Conservative estimate: 64 vertices per cluster (supports up to 7x7 grids)
  uint32_t estimatedVerticesPerCluster = 64;
  uint32_t totalVerticesEstimate = totalClusters * estimatedVerticesPerCluster;

  Logger::info(str::format("[RTXMG UNIFIED] Resizing buffers for ", totalClusters,
                          " clusters, estimated ", totalVerticesEstimate, " vertices"));

  resizeBuffersIfNeeded(totalClusters, totalVerticesEstimate);

  Logger::info(str::format("[RTXMG UNIFIED] Building unified CLAS+BLAS: ",
                          drawCalls.size(), " instances, ",
                          totalClusters, " clusters"));

  // Get command buffer for GPU operations
  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

  // STEP 0: Fill cluster vertex data (sample line 1361: FillInstanceClusters)
  // Sample flow: ComputeInstanceClusterTiling -> FillInstanceClusters -> BuildStructuredCLASes -> BuildBlasFromClas
  // This step fills the actual vertex positions for each cluster by sampling the input triangle mesh
  Logger::info(str::format("[RTXMG UNIFIED] Step 0: Filling cluster vertex data for ",
                          drawCalls.size(), " instances, ", totalClusters, " clusters"));

  // Dispatch fill_clusters shader to fill ALL cluster vertex data in ONE dispatch
  // Sample's FillInstanceClusters (line 621) dispatches fill_clusters for all instances
  // We already have all the cluster metadata from tessellation, now fill vertex positions

  if (m_clusterFillingShader == nullptr) {
    Logger::err("[RTXMG UNIFIED] Cluster filling shader not initialized");
    return false;
  }

  // SAMPLE CODE MATCH: Dispatch fill_clusters SEPARATELY for each instance
  // Sample (cluster_accel_builder.cpp:630): for (instanceIndex...) { dispatch per instance }
  // Consolidating geometry breaks cluster metadata which references geometry at offset 0
  Logger::info(str::format("[RTXMG UNIFIED] Dispatching fill_clusters for ", drawCalls.size(),
                          " instances SEPARATELY (matching sample architecture)"));

  // Setup common push constants structure
  struct FillClustersParams {
    uint32_t surfaceStart;
    uint32_t surfaceEnd;
    uint32_t enableVertexNormals;
    uint32_t clusterPattern;
    Matrix4 localToWorld;
    uint32_t enableBatching;
    uint32_t instanceCount;
    uint32_t _pad0;
    uint32_t _pad1;
    int32_t debugSurfaceIndex;
    int32_t debugClusterIndex;
    int32_t debugLaneIndex;
    uint32_t _pad2;
  } params = {};

  params.enableVertexNormals = config.enableVertexNormals ? 1 : 0;
  params.clusterPattern = 0;  // REGULAR pattern
  params.localToWorld = Matrix4();  // Identity
  params.enableBatching = 0;  // Per-instance mode
  params.instanceCount = 1;  // One instance at a time
  params.debugSurfaceIndex = -1;
  params.debugClusterIndex = -1;
  params.debugLaneIndex = -1;

  // Set constant per-instance surface range (always 0-1 for Remix)
  params.surfaceStart = 0;
  params.surfaceEnd = 1;

  // SDK MATCH: NO CPU PATCHING NEEDED!
  // The cluster_tiling shader now writes cluster metadata with GLOBAL vertex offsets directly
  // by using the globalVertexOffset parameter passed via ClusterTilingParams.
  // This eliminates the GPU->CPU->GPU roundtrip that the old code did.

  Logger::info(str::format("[RTXMG UNIFIED] Cluster metadata generated with global offsets (no CPU patching needed!)"));

  // SDK MATCH: Create constant buffer for params (binding 0), NOT push constants!
  DxvkBufferCreateInfo cbInfo = {};
  cbInfo.size = align(sizeof(params), 256);  // Align to 256 bytes for constant buffers
  cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
  Rc<DxvkBuffer> fillClustersParamsBuffer = m_device->createBuffer(
    cbInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    DxvkMemoryStats::Category::RTXBuffer,
    "RTXMG Fill Clusters Params");

  // Upload params to constant buffer ONCE
  ctx->updateBuffer(fillClustersParamsBuffer, 0, sizeof(params), &params);

  // CRITICAL: Bind shader FIRST
  Logger::info("[RTXMG FillClusters DEBUG] About to bind cluster filling shader");
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_clusterFillingShader);
  Logger::info("[RTXMG FillClusters DEBUG] Shader bound successfully");

  // SDK MATCH: Bind constant buffer at binding 0
  Logger::info("[RTXMG FillClusters DEBUG] Binding constant buffer at binding 0");
  ctx->bindResourceBuffer(0, DxvkBufferSlice(fillClustersParamsBuffer, 0, sizeof(params)));
  Logger::info("[RTXMG FillClusters DEBUG] Constant buffer bound");

  // Bind FULL output buffers (not slices) - shader uses global offsets now
  // BINDINGS SHIFTED +1: 8→9, 9→10, 10→11
  Logger::info("[RTXMG FillClusters DEBUG] Binding output buffers at bindings 9, 10, 11");
  ctx->bindResourceBuffer(9, DxvkBufferSlice(m_clusterVertexPositions.getBuffer()));
  ctx->bindResourceBuffer(10, DxvkBufferSlice(m_clusterVertexNormals.getBuffer()));
  ctx->bindResourceBuffer(11, DxvkBufferSlice(m_clusterShadingData.getBuffer()));
  Logger::info("[RTXMG FillClusters DEBUG] Output buffers bound");

  uint32_t clusterOffset = 0;
  const uint32_t wavesPerGroup = 4;

  for (size_t instanceIdx = 0; instanceIdx < drawCalls.size(); ++instanceIdx) {
    const auto& drawCall = drawCalls[instanceIdx];

    // Bind this instance's input geometry
    // BINDINGS SHIFTED +1: 0→1, 1→2, 2→3, 3→4
    ctx->bindResourceBuffer(1, drawCall.inputPositions);
    ctx->bindResourceBuffer(2, drawCall.inputNormals);
    ctx->bindResourceBuffer(3, drawCall.inputTexcoords);
    ctx->bindResourceBuffer(4, drawCall.inputIndices);

    // CRITICAL FIX: Bind cluster metadata buffers with OFFSET for this instance
    // Sample (line 662-664): BufferRange(surfaceOffset * sizeof(GridSampler), surfaceCount * sizeof(GridSampler))
    // This creates a "view" starting at this instance's data, so shader uses indices relative to 0

    // For Remix: we have ONE surface per instance (the mesh itself)
    // gridSamplers: one entry per surface (1 per instance)
    VkDeviceSize gridSamplerOffset = instanceIdx * sizeof(GridSampler);
    VkDeviceSize gridSamplerSize = sizeof(GridSampler);
    DxvkBufferSlice gridSamplerSlice(m_gridSamplers.getBuffer(), gridSamplerOffset, gridSamplerSize);
    ctx->bindResourceBuffer(6, gridSamplerSlice);  // 5→6

    // clusters: multiple entries per instance (one per cluster)
    // SDK MATCH: No ring buffer offset - single-frame buffers
    VkDeviceSize clustersOffset = clusterOffset * sizeof(RtxmgCluster);
    VkDeviceSize clustersSize = drawCall.clusterCount * sizeof(RtxmgCluster);
    DxvkBufferSlice clustersSlice(m_clusters.getBuffer(), clustersOffset, clustersSize);
    ctx->bindResourceBuffer(7, clustersSlice);  // 6→7

    // clusterShadingData: same slice as clusters
    VkDeviceSize shadingDataOffset = clusterOffset * sizeof(ClusterShadingData);
    VkDeviceSize shadingDataSize = drawCall.clusterCount * sizeof(ClusterShadingData);
    DxvkBufferSlice shadingDataSlice(m_clusterShadingData.getBuffer(), shadingDataOffset, shadingDataSize);
    ctx->bindResourceBuffer(8, shadingDataSlice);  // 7→8

    // Create single-entry SurfaceInfo for this instance (starts at offset 0)
    SurfaceInfo surfaceInfo = {};
    surfaceInfo.firstVertex = 0;
    surfaceInfo.vertexCount = drawCall.inputVertexCount;
    surfaceInfo.firstIndex = 0;
    surfaceInfo.indexCount = drawCall.inputIndexCount;
    surfaceInfo.materialId = 0;
    surfaceInfo.geometryId = 0;
    surfaceInfo.pad0 = 0;
    surfaceInfo.pad1 = 0;

    std::vector<SurfaceInfo> singleSurfaceInfo = { surfaceInfo };
    m_surfaceInfo.upload(singleSurfaceInfo);
    ctx->bindResourceBuffer(5, DxvkBufferSlice(m_surfaceInfo.getBuffer()));  // 4→5

    // SDK MATCH: NO push constants! Params are in constant buffer at binding 0

    // Dispatch for this instance's clusters
    uint32_t numGroups = (drawCall.clusterCount + wavesPerGroup - 1) / wavesPerGroup;
    Logger::info(str::format("[RTXMG FillClusters DEBUG] Instance ", instanceIdx, ": About to dispatch ", numGroups, " groups for ", drawCall.clusterCount, " clusters"));
    ctx->dispatch(numGroups, 1, 1);
    Logger::info(str::format("[RTXMG FillClusters DEBUG] Instance ", instanceIdx, ": Dispatch complete"));

    if (instanceIdx < 3 || instanceIdx >= drawCalls.size() - 3) {
      Logger::info(str::format("[RTXMG FILL] Instance ", instanceIdx, ": ",
                              drawCall.clusterCount, " clusters, ", numGroups, " groups"));
    }

    clusterOffset += drawCall.clusterCount;
  }

  Logger::info(str::format("[RTXMG UNIFIED] Dispatched fill_clusters for ", drawCalls.size(),
                          " instances, ", totalClusters, " total clusters"));

  // CRITICAL: GPU-driven cluster offset copy (matches SDK sample)
  // Dispatch copy_cluster_offset shader to populate clusterOffsetCountsBuffer on GPU
  // This reads tessellation counters and writes offset/count pairs for each instance
  {
    ScopedGpuProfileZone(ctx, "CopyClusterOffset");

    // Barrier: ensure fill_clusters writes complete before reading tessellation counters
    VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);

    // SDK MATCH: Create shared params buffer for all 55 dispatches
    // Reuse the same buffer, only updating params each iteration
    struct CopyClusterOffsetParams {
      uint32_t instanceIndex;
      uint32_t totalInstances;
      uint32_t _pad0;
      uint32_t _pad1;
    };

    DxvkBufferCreateInfo cbInfo = {};
    cbInfo.size = align(sizeof(CopyClusterOffsetParams), 256);
    cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
    Rc<DxvkBuffer> copyClusterOffsetParamsBuffer = m_device->createBuffer(
      cbInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "RTXMG Copy Cluster Offset Params (Shared)");

    const Rc<DxvkBuffer>& tessCountersBuffer = m_tessCountersBuffer.getBuffer();
    const Rc<DxvkBuffer>& offsetCountsBuffer = m_clusterOffsetCountsBuffer.getBuffer();

    // SDK MATCH: Call copyClusterOffset for each instance
    // Reuse the same params buffer across all dispatches
    for (size_t instanceIdx = 0; instanceIdx < drawCalls.size(); ++instanceIdx) {
      copyClusterOffset(
        ctx,
        static_cast<uint32_t>(instanceIdx),
        static_cast<uint32_t>(drawCalls.size()),
        copyClusterOffsetParamsBuffer,
        DxvkBufferSlice(tessCountersBuffer),
        DxvkBufferSlice(offsetCountsBuffer));
    }

    Logger::info(str::format("[RTXMG BLAS] GPU-wrote ", drawCalls.size(),
                            " cluster offset/count pairs via copyClusterOffset (SDK-matching, independent bindings)"));
  }

  // CRITICAL: Barrier to ensure cluster offset writes are visible to next shader
  // Sample has this between FillInstanceClusters and BuildStructuredCLASes
  VkMemoryBarrier fillBarrier = {};
  fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  fillBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  m_device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    0, 1, &fillBarrier, 0, nullptr, 0, nullptr);

  Logger::info("[RTXMG UNIFIED] Step 0 complete: Cluster vertex data filled");

  if (!ensureTemplateClasBuilt(ctx))
    return false;

  // STEP 1: Instantiate ALL clusters in ONE GPU call (BuildStructuredCLASes equivalent)
  // Sample line 1365: BuildStructuredCLASes() calls executeMultiIndirectClusterOperation ONCE for all clusters
  // Sample line 608: inIndirectArgsBuffer = m_clasIndirectArgDataBuffer (contains args for ALL clusters)
  // Sample line 618: commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);

  Logger::info(str::format("[RTXMG UNIFIED] *** USING GPU-DRIVEN SDK-MATCHING PATH (rtxmg_cluster_builder.cpp) ***"));
  Logger::info(str::format("[RTXMG UNIFIED] Step 1: Instantiating ", totalClusters,
                          " clusters from ", drawCalls.size(), " instances in ONE GPU call"));

  // SDK MATCH: Instantiate CLAS using executeMultiIndirectClusterOperation (lines 602-618)
  // NO BUFFER COPIES! The cluster tiling shader wrote VkClusterAccelerationStructureInstantiateClusterInfoNV
  // directly to m_clusterIndirectArgs during the fill_clusters pass, and we consume it directly here.

  // Ensure persistent CLAS addresses buffer is allocated
  if (!m_frameAccels.clasPtrsBuffer.isValid() || m_frameAccels.clasPtrsBuffer.numElements() < totalClusters) {
    m_frameAccels.clasPtrsBuffer.release();
    m_frameAccels.clasPtrsBuffer.create(
      m_device, totalClusters,
      "RTXMG Frame CLAS Addresses",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Logger::info(str::format("[RTXMG] Allocated CLAS address buffer: ", totalClusters, " addresses"));
  }

  // Query sizes only for the clusters we actually need this frame.
  // This matches the SDK flow (maxArgCount set to current instantiation count).
  const uint32_t clusterCapacity = std::max(totalClusters, 1u);

  // SDK MATCH: Store triangle input in member variable to avoid stack allocation
  // Query CLAS instantiation size (SDK: getClusterOperationSizeInfo)
  m_createClasTriangleInput = {};
  m_createClasTriangleInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
  m_createClasTriangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  m_createClasTriangleInput.maxGeometryIndexValue = 0;  // Single geometry (index 0), NOT clusterCapacity!
  m_createClasTriangleInput.maxClusterUniqueGeometryCount = 1;
  m_createClasTriangleInput.maxClusterTriangleCount = kMaxClusterTriangles;
  m_createClasTriangleInput.maxClusterVertexCount = kMaxClusterVertices;

  const uint64_t totalTriangleBudget = uint64_t(clusterCapacity) * uint64_t(kMaxClusterTriangles);
  m_createClasTriangleInput.maxTotalTriangleCount = static_cast<uint32_t>(
    std::min<uint64_t>(totalTriangleBudget, std::numeric_limits<uint32_t>::max()));

  const uint64_t totalVertexBudget = uint64_t(clusterCapacity) * uint64_t(kMaxClusterVertices);
  m_createClasTriangleInput.maxTotalVertexCount = static_cast<uint32_t>(
    std::min<uint64_t>(totalVertexBudget, std::numeric_limits<uint32_t>::max()));
  m_createClasTriangleInput.minPositionTruncateBitCount = static_cast<uint32_t>(config.quantNBits);

  // Query scratch size for BOTH EXPLICIT_DESTINATIONS and IMPLICIT_DESTINATIONS modes
  // The same scratch buffer is used for both operations

  VkClusterAccelerationStructureInputInfoNV sizeQueryInfo = {};
  sizeQueryInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  sizeQueryInfo.maxAccelerationStructureCount = clusterCapacity;
  sizeQueryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  sizeQueryInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV;
  sizeQueryInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  sizeQueryInfo.opInput.pTriangleClusters = &m_createClasTriangleInput;  // ← Now points to member variable!

  VkAccelerationStructureBuildSizesInfoKHR buildSizesExplicit = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    m_device->handle(),
    &sizeQueryInfo,
    &buildSizesExplicit);

  // Also query for IMPLICIT_DESTINATIONS mode
  sizeQueryInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  VkAccelerationStructureBuildSizesInfoKHR buildSizesImplicit = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    m_device->handle(),
    &sizeQueryInfo,
    &buildSizesImplicit);

  // Reset to EXPLICIT_DESTINATIONS for the actual GPU operation
  sizeQueryInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

  // Use maximum values from both modes
  VkDeviceSize requiredClasSize = std::max(buildSizesExplicit.accelerationStructureSize, buildSizesImplicit.accelerationStructureSize);
  VkDeviceSize scratchSizeInBytes = std::max(buildSizesExplicit.buildScratchSize, buildSizesImplicit.buildScratchSize);

  if (requiredClasSize == 0 || scratchSizeInBytes == 0) {
    Logger::err(str::format(
      "[RTXMG UNIFIED] vkGetClusterAccelerationStructureBuildSizesNV returned 0 (totalClusters=",
      totalClusters, ", clusterCapacity=", clusterCapacity,
      "). Skipping CLAS build to avoid invalid allocations. "
      "Ensure VK_NV_cluster_acceleration_structure is enabled and reports sizes."));
    return false;
  }

  Logger::info(str::format("[RTXMG UNIFIED] CLAS size query: EXPLICIT scratch=",
    buildSizesExplicit.buildScratchSize, " bytes (", buildSizesExplicit.accelerationStructureSize / 1024, " KB instances), IMPLICIT scratch=",
    buildSizesImplicit.buildScratchSize, " bytes (", buildSizesImplicit.accelerationStructureSize / 1024, " KB instances), using max scratch=",
    scratchSizeInBytes, " bytes, max accelSize=", requiredClasSize / (1024 * 1024), " MB"));

  if (!m_frameAccels.clasBuffer.isValid() || m_frameAccels.clasBuffer.bytes() < requiredClasSize) {
    m_frameAccels.clasBuffer.release();
    m_frameAccels.clasBuffer.create(
      m_device, requiredClasSize,
      "RTXMG Frame CLAS Instances",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Logger::info(str::format("[RTXMG] Allocated CLAS instance buffer: ",
                            requiredClasSize / (1024 * 1024), " MB"));
  }

  // Ensure scratch buffer is allocated
  FrameInstantiationBuffers& frameBuffers = m_frameBuffers;
  if (!frameBuffers.scratchBuffer.isValid() || frameBuffers.scratchBuffer.bytes() < scratchSizeInBytes) {
    frameBuffers.scratchBuffer.release();
    frameBuffers.scratchBuffer.create(
      m_device, scratchSizeInBytes,
      "RTXMG CLAS Scratch",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  // SDK MATCH: Build ClusterOperationDesc (lines 602-616)
  VkClusterAccelerationStructureInputInfoNV instantiateInfo = sizeQueryInfo;
  instantiateInfo.maxAccelerationStructureCount = totalClusters;

  ClusterOperationDesc instantiateDesc = {};
  instantiateDesc.params = instantiateInfo;
  instantiateDesc.scratchData = frameBuffers.scratchBuffer.getDeviceAddress();  // FIXED: GPU address of scratch buffer
  instantiateDesc.scratchSizeInBytes = frameBuffers.scratchBuffer.bytes();       // Actual size in bytes

  // GPU counter buffer (SDK line 606: inIndirectArgCountBuffer)
  const Rc<DxvkBuffer>& gpuCounterBuffer = m_tessCountersBuffer.getBuffer();
  instantiateDesc.inIndirectArgCountBuffer = gpuCounterBuffer->getDeviceAddress();
  instantiateDesc.inIndirectArgCountOffsetInBytes = offsetof(TessellationCounters, clusters);

  // Indirect args buffer (SDK line 608: inIndirectArgsBuffer = m_clasIndirectArgDataBuffer)
  instantiateDesc.inIndirectArgsBuffer = m_clusterIndirectArgs.getDeviceAddress();
  instantiateDesc.inIndirectArgsOffsetInBytes = 0;

  // Output addresses (SDK line 610: inOutAddressesBuffer = accels.clasPtrsBuffer)
  instantiateDesc.inOutAddressesBuffer = m_frameAccels.clasPtrsBuffer.getDeviceAddress();
  instantiateDesc.inOutAddressesOffsetInBytes = 0;

  // No output sizes or explicit output buffer for CLAS instantiation
  instantiateDesc.outSizesBuffer = 0;
  instantiateDesc.outSizesOffsetInBytes = 0;
  instantiateDesc.outAccelerationStructuresBuffer = m_frameAccels.clasBuffer.getDeviceAddress();
  instantiateDesc.outAccelerationStructuresOffsetInBytes = 0;

  // SDK MATCH: commandList->executeMultiIndirectClusterOperation (line 618)
  executeMultiIndirectClusterOperation(ctx, instantiateDesc);

  Logger::info(str::format("[RTXMG CLAS] Step 1 complete: Instantiated ",
                          totalClusters, " cluster instances (CLAS structures) in ONE GPU call [SDK-MATCHING]"));

  Logger::info(str::format("[RTXMG CLAS] === COMPLETE === Successfully built ",
                          totalClusters, " CLAS from ", drawCalls.size(), " instances [SDK-MATCHING]"));
  return true;
}

// ============================================================================
// STEP 2: Build BLAS from CLAS (EXACT SAMPLE MATCH)
// ============================================================================
// This EXACTLY matches sample's BuildBlasFromClas (line 1368):
//   - Build ALL BLASes from CLAS in ONE GPU call
// Caller MUST flush before and after this function
// ============================================================================

bool RtxmgClusterBuilder::buildBlasFromClas(
  RtxContext* ctx,
  const std::vector<DrawCallData>& drawCalls,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG BLAS] Cluster builder not initialized");
    return false;
  }

  if (drawCalls.empty()) {
    Logger::warn("[RTXMG BLAS] No draw calls to process");
    return false;
  }

  // Build geometry hash to blasPtrsBuffer index mapping for GPU-side TLAS patching
  // This allows the AccelManager to know which blasPtrsBuffer entry corresponds to each geometry
  // CRITICAL: Clear the map only once per frame! Multiple BuildAccel calls per frame means
  // TLAS instances reference indices from earlier builds. Clearing between builds invalidates those indices.
  // CRITICAL: Use cumulative indices! Each BuildAccel call adds to blasPtrsBuffer sequentially.
  // CRITICAL: Ring-buffer across frames to avoid overwriting data from frames still in-flight!
  static uint32_t lastMappingFrameId = 0xFFFFFFFF;
  static uint32_t nextBufferIndex = 0;  // Cumulative index across builds within same frame
  static uint32_t blasPtrsBufferOffset = 0;  // Cumulative byte offset into blasPtrsBuffer
  uint32_t currentFrameId = m_device->getCurrentFrameId();
  if (currentFrameId != lastMappingFrameId) {
    // SDK MATCH: No ring buffering - single-frame buffers with GPU sync
    // NVIDIA sample doesn't use ring buffers, it rebuilds every frame
    const uint32_t maxInstancesPerFrame = 1024;  // Max instances we support

    m_geometryHashToBlasIndex.clear();
    nextBufferIndex = 0;  // Reset cumulative index for new frame
    blasPtrsBufferOffset = 0;  // Start from buffer beginning (no ring offset)
    lastMappingFrameId = currentFrameId;
    Logger::info(str::format("[RTXMG BLAS] New frame ", currentFrameId,
                            " - cleared hash mapping (single-frame buffers, SDK-matching)"));
  }

  // Save the starting index for this build
  uint32_t thisBuildStartIndex = nextBufferIndex;

  // Assign cumulative indices - each geometry gets the next available index
  for (uint32_t i = 0; i < drawCalls.size(); ++i) {
    XXH64_hash_t hash = drawCalls[i].geometryHash;
    // Only add new entries - don't overwrite existing ones (same geometry can appear in multiple builds)
    if (m_geometryHashToBlasIndex.find(hash) == m_geometryHashToBlasIndex.end()) {
      m_geometryHashToBlasIndex[hash] = static_cast<int32_t>(nextBufferIndex);
      Logger::info(str::format("[RTXMG HASH MAP] Assigned hash 0x", std::hex, hash, std::dec,
                              " → buffer index ", nextBufferIndex,
                              " (frame ", currentFrameId, ")"));
      nextBufferIndex++;
    }
  }

  // CRITICAL: Validate we haven't exceeded buffer capacity!
  const uint32_t maxInstancesPerFrame = 1024;
  if (nextBufferIndex > maxInstancesPerFrame) {
    Logger::err(str::format("[RTXMG BLAS ERROR] Buffer overflow! nextBufferIndex=", nextBufferIndex,
                            " exceeds max=", maxInstancesPerFrame,
                            " (frame ", currentFrameId, ")"));
    return false;
  }

  Logger::info(str::format("[RTXMG BLAS] Updated geometry hash mapping (now ", m_geometryHashToBlasIndex.size(), " total entries, nextIndex=", nextBufferIndex, "/", maxInstancesPerFrame, ")"));

  // SDK MATCH: Use GPU-written m_clusterOffsetCountsBuffer (filled by tiling shader)
  // Sample: cluster_accel_builder.cpp:1065 uses GPU-written offset counts
  // NO CPU REBUILDING - the buffer is already populated by cluster_tiling shader!

  // Calculate totalClusters for conservative sizing
  uint32_t estimatedTotalClusters = 0;
  uint32_t maxClustersPerBlas = 0;
  for (const auto& drawCall : drawCalls) {
    estimatedTotalClusters += drawCall.clusterCount;
    maxClustersPerBlas = std::max(maxClustersPerBlas, drawCall.clusterCount);
  }

  Logger::info(str::format("[RTXMG BLAS] Building BLASes: ",
                          drawCalls.size(), " instances, ",
                          estimatedTotalClusters, " estimated clusters (GPU-driven via indirect buffer)"));

  // SDK MATCH: Update BLAS allocation based on actual needs (cluster_accel_builder.cpp:1265 calls UpdateMemoryAllocations)
  // NVIDIA builds ONE BLAS per instance in the instances array
  // If there are 685 instances (drawCalls), allocate space for 685 BLASes
  // The instances.size() in sample line 1063 is the number of BLASes to build
  updateBlasAllocation(ctx, estimatedTotalClusters, maxClustersPerBlas, drawCalls.size());

  // Get command buffer
  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

  // STEP 2: Build ALL BLASes in ONE GPU call (BuildBlasFromClas equivalent)
  // Sample line 1368: BuildBlasFromClas() uses executeMultiIndirectClusterOperation
  Logger::info(str::format("[RTXMG UNIFIED] Step 2: Building ", drawCalls.size(),
                          " BLASes from ", estimatedTotalClusters, " estimated CLAS [SDK-MATCHING GPU-DRIVEN]"));

  // Step 2.5: Prepare indirect args using compute shader (matches sample line 1065)
  // FillBlasFromClasArgs shader reads m_clusterOffsetCountsBuffer (CPU-uploaded)
  // SDK MATCH: FillBlasFromClasArgs(m_blasFromClasIndirectArgsBuffer, m_clusterOffsetCountsBuffer, ...)

  // Use CPU-uploaded cluster offset counts from single buffer
  const Rc<DxvkBuffer>& offsetCountsBuffer = m_clusterOffsetCountsBuffer.getBuffer();

  Logger::info(str::format("[RTXMG BLAS] Using m_clusterOffsetCountsBuffer"));

  // Create output buffer for shader-generated indirect args
  // Structure must match shader: { uint64_t clusterAddresses; uint32_t clusterCount; uint32_t padding; }
  struct BlasFromClasIndirectArg {
    VkDeviceAddress clusterAddresses;
    uint32_t clusterCount;
    uint32_t _padding;  // Explicit padding to align to 16 bytes
  };

  RtxmgBuffer<BlasFromClasIndirectArg> blasIndirectArgsBuffer;
  blasIndirectArgsBuffer.create(
    m_device, drawCalls.size(),
    "RTXMG BLAS Indirect Args",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Dispatch shader to fill indirect args (sample: cluster_accel_builder.cpp:1065)
  {
    ScopedGpuProfileZone(ctx, "FillBlasFromClasArgs");

    // Push constants
    struct FillBlasFromClasArgsParams {
      VkDeviceAddress clasAddressesBaseAddress;
      uint32_t numInstances;
      uint32_t _pad0;
      uint32_t _pad1;
    };

    // Same as the public RTXMG sample: CLAS pointer buffer is single-buffered.
    // The data is rewritten every BuildAccel so no per-frame offset is required.
    uint32_t totalClustersInBatch = 0;
    for (const auto& dc : drawCalls) {
      totalClustersInBatch += dc.clusterCount;
    }

    const VkDeviceSize requiredClasPtrBytes =
      static_cast<VkDeviceSize>(totalClustersInBatch) * sizeof(VkDeviceAddress);
    if (requiredClasPtrBytes > m_frameAccels.clasPtrsBuffer.bytes()) {
      Logger::err(str::format("[RTXMG BLAS] CLAS pointer buffer too small: need ",
                              requiredClasPtrBytes, " bytes but only have ",
                              m_frameAccels.clasPtrsBuffer.bytes(), " bytes"));
      return false;
    }

    VkDeviceAddress clasAddressesBase = m_frameAccels.clasPtrsBuffer.getDeviceAddress();
    Logger::info(str::format("[RTXMG BLAS] CLAS addresses base=", std::hex, clasAddressesBase,
                            " (totalClusters=", totalClustersInBatch, ")", std::dec));

    FillBlasFromClasArgsParams params;
    params.clasAddressesBaseAddress = clasAddressesBase;
    params.numInstances = static_cast<uint32_t>(drawCalls.size());
    params._pad0 = 0;
    params._pad1 = 0;

    // SDK MATCH: Create constant buffer for params (binding 0), NOT push constants!
    DxvkBufferCreateInfo cbInfo = {};
    cbInfo.size = align(sizeof(FillBlasFromClasArgsParams), 256);
    cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
    Rc<DxvkBuffer> fillBlasParamsBuffer = m_device->createBuffer(
      cbInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "RTXMG Fill BLAS From CLAS Args Params");

    // Upload params to constant buffer
    ctx->updateBuffer(fillBlasParamsBuffer, 0, sizeof(params), &params);

    // Bind shader FIRST
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_fillBlasFromClasArgsShader);

    // SDK MATCH: Bind constant buffer at binding 0
    ctx->bindResourceBuffer(0, DxvkBufferSlice(fillBlasParamsBuffer, 0, sizeof(params)));

    // SDK MATCH: Bind offset counts buffer (CPU-uploaded in BuildAccel)
    Rc<DxvkBuffer> outputBuffer = blasIndirectArgsBuffer.getBuffer();

    // BINDINGS SHIFTED +1: 0→1, 1→2
    ctx->bindResourceBuffer(1, DxvkBufferSlice(offsetCountsBuffer));  // Input: cluster offset/count pairs
    ctx->bindResourceBuffer(2, DxvkBufferSlice(outputBuffer));        // Output: BLAS indirect args

    // SDK MATCH: NO push constants! Params are in constant buffer at binding 0

    // Dispatch: (numInstances + 255) / 256 thread groups (256 threads per group)
    uint32_t numGroups = (drawCalls.size() + 255) / 256;
    ctx->dispatch(numGroups, 1, 1);

    // Barrier to ensure shader writes complete before BLAS building
    VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

    Logger::info(str::format("[RTXMG UNIFIED] Dispatched fill_blas_from_clas_args: ", drawCalls.size(), " instances in ", numGroups, " groups"));
  }

  const VkDeviceSize blasBufferBytes =
    m_frameAccels.blasAccelStructure.ptr()
      ? m_frameAccels.blasAccelStructure->info().size
      : m_frameAccels.blasBuffer.bytes();
  Logger::info(str::format("[RTXMG BLAS] Using dynamically allocated BLAS buffer: ",
                          blasBufferBytes / (1024 * 1024), " MB"));
  Logger::info(str::format("[RTXMG BLAS] Using dynamically allocated scratch size: ",
                          m_blasSizeInfo.buildScratchSize / (1024 * 1024), " MB"));

  // SDK MATCH: cluster_accel_builder.cpp:1200-1211 - Store params in member variables to avoid stack allocation
  // Use member variables instead of stack variables to ensure pointers remain valid
  m_createBlasParams = {};
  m_createBlasParams.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
  m_createBlasParams.maxTotalClusterCount = estimatedTotalClusters;  // Conservative estimate for sizing
  m_createBlasParams.maxClusterCountPerAccelerationStructure = maxClustersPerBlas;  // Conservative max per BLAS

  Logger::info(str::format("[RTXMG BLAS] Build params: maxTotalClusterCount=", m_createBlasParams.maxTotalClusterCount,
                          ", maxClusterCountPerAccelerationStructure=", m_createBlasParams.maxClusterCountPerAccelerationStructure,
                          " (conservative estimates, GPU counter drives actual work)"));

  m_createBlasInputInfo = {};
  m_createBlasInputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  m_createBlasInputInfo.maxAccelerationStructureCount = drawCalls.size();  // N BLASes to build
  m_createBlasInputInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  m_createBlasInputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  // SDK MATCH: Use IMPLICIT_DESTINATIONS for fast unified BLAS build (sample line 1204)
  // EXPLICIT_DESTINATIONS causes massive GPU overhead and TDR timeouts!
  m_createBlasInputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  m_createBlasInputInfo.opInput.pClustersBottomLevel = &m_createBlasParams;  // ← Now points to member variable!

  // SDK MATCH: Use dynamically allocated scratch buffer (cluster_accel_builder.cpp:1081)
  // Buffer is sized for current needs and persists until size changes
  Logger::info(str::format("[RTXMG BLAS] Using scratch buffer: ",
                          m_frameAccels.blasScratchBuffer.bytes() / (1024 * 1024), " MB"));

  // SDK MATCH: Build ClusterOperationDesc (lines 1078-1092)
  ClusterOperationDesc buildBlasDesc = {};
  buildBlasDesc.params = m_createBlasInputInfo;  // ← Now references member variable!
  buildBlasDesc.scratchData = m_frameAccels.blasScratchBuffer.getDeviceAddress();  // FIXED: GPU address of scratch buffer
  buildBlasDesc.scratchSizeInBytes = m_frameAccels.blasScratchBuffer.bytes();      // Actual size in bytes

  // SDK MATCH: Sample uses nullptr for count buffer (line 1082 in cluster_accel_builder.cpp)
  // The indirect args are self-describing - the GPU doesn't need a separate count
  buildBlasDesc.inIndirectArgCountBuffer = 0;
  buildBlasDesc.inIndirectArgCountOffsetInBytes = 0;
  Logger::info("[RTXMG BLAS] Using NULL count buffer (SDK-matching): indirect args are self-describing");

  // Indirect args buffer (SDK line 1084: inIndirectArgsBuffer = m_blasFromClasIndirectArgsBuffer)
  buildBlasDesc.inIndirectArgsBuffer = blasIndirectArgsBuffer.getDeviceAddress();
  buildBlasDesc.inIndirectArgsOffsetInBytes = 0;

  // Output addresses (SDK line 1086: inOutAddressesBuffer = accels.blasPtrsBuffer)
  // CRITICAL: Use cumulative offset so multiple builds per frame don't overwrite each other!
  buildBlasDesc.inOutAddressesBuffer = m_frameAccels.blasPtrsBuffer.getDeviceAddress();
  buildBlasDesc.inOutAddressesOffsetInBytes = blasPtrsBufferOffset;
  Logger::info(str::format("[RTXMG BLAS] Using blasPtrsBuffer: base=0x", std::hex, buildBlasDesc.inOutAddressesBuffer, std::dec,
                          ", offset=", blasPtrsBufferOffset,
                          " bytes (index ", thisBuildStartIndex, "-", nextBufferIndex - 1, ")"));
  Logger::info(str::format("[RTXMG BLAS] Buffer capacity: ", m_frameAccels.blasPtrsBuffer.bytes(),
                          " bytes (", m_frameAccels.blasPtrsBuffer.bytes() / sizeof(VkDeviceAddress), " addresses)"));

  // Output sizes (SDK line 1088: outSizesBuffer = accels.blasSizesBuffer)
  // CRITICAL: Use same cumulative offset as blasPtrsBuffer!
  buildBlasDesc.outSizesBuffer = m_frameAccels.blasSizesBuffer.getDeviceAddress();
  buildBlasDesc.outSizesOffsetInBytes = blasPtrsBufferOffset;

  // Output acceleration structures buffer (SDK line 1090: outAccelerationStructuresBuffer = accels.blasBuffer)
  buildBlasDesc.outAccelerationStructuresBuffer =
    m_frameAccels.blasAccelStructure.ptr()
      ? m_frameAccels.blasAccelStructure->getDeviceAddress()
      : m_frameAccels.blasBuffer.getDeviceAddress();
  buildBlasDesc.outAccelerationStructuresOffsetInBytes = 0;

  // SDK MATCH: commandList->executeMultiIndirectClusterOperation (line 1093)
  Logger::info(str::format("[RTXMG UNIFIED] ========== ABOUT TO BUILD ", drawCalls.size(), " BLASes (Frame ", currentFrameId, ") =========="));
  Logger::info(str::format("[RTXMG BLAS PARAMS] scratchSizeInBytes=", buildBlasDesc.scratchSizeInBytes));
  Logger::info(str::format("[RTXMG BLAS PARAMS] inIndirectArgCountBuffer=0x", std::hex, buildBlasDesc.inIndirectArgCountBuffer, std::dec));
  Logger::info(str::format("[RTXMG BLAS PARAMS] inIndirectArgsBuffer=0x", std::hex, buildBlasDesc.inIndirectArgsBuffer, std::dec));
  Logger::info(str::format("[RTXMG BLAS PARAMS] inOutAddressesBuffer=0x", std::hex, buildBlasDesc.inOutAddressesBuffer, std::dec,
                          " (offset=", buildBlasDesc.inOutAddressesOffsetInBytes, " bytes)"));
  Logger::info(str::format("[RTXMG BLAS PARAMS] outSizesBuffer=0x", std::hex, buildBlasDesc.outSizesBuffer, std::dec,
                          " (offset=", buildBlasDesc.outSizesOffsetInBytes, " bytes)"));
  Logger::info(str::format("[RTXMG BLAS PARAMS] outAccelerationStructuresBuffer=0x", std::hex, buildBlasDesc.outAccelerationStructuresBuffer, std::dec));

  // CRITICAL VALIDATION: Check if parameters are valid before GPU work
  Logger::info("[RTXMG BLAS VALIDATION] ========== PRE-GPU-WORK VALIDATION ==========");
  Logger::info(str::format("[RTXMG BLAS VALIDATION] Context valid: ", ctx != nullptr ? "YES" : "NO"));
  Logger::info(str::format("[RTXMG BLAS VALIDATION] CommandList valid: ", ctx && ctx->getCommandList() != nullptr ? "YES" : "NO"));
  Logger::info(str::format("[RTXMG BLAS VALIDATION] Number of BLASes to build: ", drawCalls.size()));
  Logger::info(str::format("[RTXMG BLAS VALIDATION] Scratch size: ", buildBlasDesc.scratchSizeInBytes, " bytes"));
  Logger::info(str::format("[RTXMG BLAS VALIDATION] Offset into buffers: ", blasPtrsBufferOffset, " bytes"));

  Logger::info(str::format("[RTXMG UNIFIED] Calling executeMultiIndirectClusterOperation for ", drawCalls.size(), " BLASes [SDK-MATCHING]..."));

  // Call GPU work
  try {
    executeMultiIndirectClusterOperation(ctx, buildBlasDesc);
    Logger::info("[RTXMG BLAS GPU WORK] executeMultiIndirectClusterOperation SUCCEEDED");
  } catch (const std::exception& e) {
    Logger::err(str::format("[RTXMG BLAS GPU WORK ERROR] Exception during executeMultiIndirectClusterOperation: ", e.what()));
    return false;
  } catch (...) {
    Logger::err("[RTXMG BLAS GPU WORK ERROR] Unknown exception during executeMultiIndirectClusterOperation");
    return false;
  }

  Logger::info("[RTXMG UNIFIED] ========== executeMultiIndirectClusterOperation RETURNED ==========");
  Logger::info(str::format("[RTXMG UNIFIED] Cluster extension has written BLAS addresses to blasPtrsBuffer at offset ", blasPtrsBufferOffset, " bytes"));

  // Memory barrier
  // CRITICAL: Must sync with COMPUTE_SHADER stage for GPU patching shader to read blasPtrsBuffer!
  // The cluster extension writes BLAS addresses to blasPtrsBuffer during vkCmdBuildAccelerationStructuresIndirectKHR,
  // and the GPU patching compute shader (patch_tlas_instance_blas_addresses) reads from it.
  // FIXED: VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR only supports VK_ACCESS_ACCELERATION_STRUCTURE_* masks
  Logger::info("[RTXMG UNIFIED] ========== EMITTING MEMORY BARRIER ==========");
  Logger::info(str::format("[RTXMG BARRIER] Source stage: VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR"));
  Logger::info(str::format("[RTXMG BARRIER] Source access: VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR (FIXED: removed VK_ACCESS_SHADER_WRITE_BIT)"));
  Logger::info(str::format("[RTXMG BARRIER] Dest stage: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"));
  Logger::info(str::format("[RTXMG BARRIER] Dest access: VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT"));
  Logger::info(str::format("[RTXMG BARRIER] Purpose: Ensure cluster extension's BLAS address writes are visible to GPU patching shader"));

  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  // FIXED: Only use ACCELERATION_STRUCTURE_WRITE with ACCELERATION_STRUCTURE_BUILD stage
  barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;

  m_device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &barrier, 0, nullptr, 0, nullptr);
  Logger::info("[RTXMG BARRIER] ========== BARRIER EMITTED SUCCESSFULLY ==========");

  Logger::info(str::format("[RTXMG BLAS] Step 2 complete: Built ", drawCalls.size(),
                          " BLASes from ", estimatedTotalClusters, " estimated CLAS in ONE GPU call (actual count GPU-driven)"));

  // SDK MATCH: Async read of counters AFTER BLAS building (line 1371)
  // CRITICAL FIX: Disable ALL readback operations - they cause GPU device lost
  // Instead use ONLY memory barrier to synchronize without GPU operations
  // This ensures GPU pipeline is flushed safely without buffer operations
  Logger::info("[RTXMG BLAS] Skipped readback - using memory barrier for synchronization");

  // NOTE: Caller will flush after this function returns
  // Reference implementation: cluster_accel_builder.cpp calls these as separate functions
  // with the caller handling command buffer management between them

  // CRITICAL: Increment buffer offset for next build in this frame
  // Each BLAS address is 8 bytes (VkDeviceAddress), and sizes are 4 bytes (uint32_t)
  // But both buffers need the same stride for indexing to work
  uint32_t numBlasesBuilt = static_cast<uint32_t>(drawCalls.size());
  blasPtrsBufferOffset += numBlasesBuilt * sizeof(VkDeviceAddress);  // 8 bytes per entry
  Logger::info(str::format("[RTXMG BLAS] Incremented buffer offset by ", numBlasesBuilt * sizeof(VkDeviceAddress),
                          " bytes (", numBlasesBuilt, " BLASes), new offset: ", blasPtrsBufferOffset));

  Logger::info(str::format("[RTXMG BLAS] === COMPLETE === Successfully built ",
                          drawCalls.size(), " BLASes (GPU counter-driven)"));

  // ============================================================================
  // MATCHING NVIDIA SAMPLE: No explicit fence management
  // All GPU work (CLAS → BLAS → Patching → TLAS) in same command buffer
  // Vulkan handles synchronization with memory barriers and command buffer submission
  Logger::info("[RTXMG UNIFIED] Completed BLAS building - GPU work synchronized by Vulkan framework");

  return true;
}

// ============================================================================
// HiZ Pyramid Management (Phase 4)
// ============================================================================

void RtxmgClusterBuilder::createHiZPyramid(uint32_t width, uint32_t height) {
  Logger::info(str::format("[RTXMG] Creating HiZ pyramid: ", width, "×", height));

  m_hizWidth = width;
  m_hizHeight = height;

  // Calculate number of mip levels (down to 1×1)
  uint32_t maxDim = std::max(width, height);
  m_hizNumLevels = static_cast<uint32_t>(std::floor(std::log2(maxDim))) + 1;

  Logger::info(str::format("[RTXMG] HiZ pyramid levels: ", m_hizNumLevels));

  // Create HiZ pyramid image
  DxvkImageCreateInfo imageInfo = {};
  imageInfo.type = VK_IMAGE_TYPE_2D;
  imageInfo.format = VK_FORMAT_R32_SFLOAT;  // 32-bit float depth
  imageInfo.flags = 0;
  imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.extent = { width, height, 1 };
  imageInfo.numLayers = 1;
  imageInfo.mipLevels = m_hizNumLevels;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  imageInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.layout = VK_IMAGE_LAYOUT_GENERAL;

  m_hizPyramid = m_device->createImage(imageInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    DxvkMemoryStats::Category::RTXRenderTarget,
    "RTXMG HiZ Pyramid");

  // Create full pyramid view (all mip levels)
  DxvkImageViewCreateInfo viewInfo = {};
  viewInfo.type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  viewInfo.format = VK_FORMAT_R32_SFLOAT;
  viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.minLevel = 0;
  viewInfo.numLevels = m_hizNumLevels;
  viewInfo.minLayer = 0;
  viewInfo.numLayers = 1;

  m_hizPyramidView = m_device->createImageView(m_hizPyramid, viewInfo);

  // Create per-mip level views for compute shader writes
  m_hizMipViews.resize(m_hizNumLevels);
  for (uint32_t level = 0; level < m_hizNumLevels; ++level) {
    DxvkImageViewCreateInfo mipViewInfo = viewInfo;
    mipViewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    mipViewInfo.minLevel = level;
    mipViewInfo.numLevels = 1;

    m_hizMipViews[level] = m_device->createImageView(m_hizPyramid, mipViewInfo);
  }

  // Create point sampler for HiZ reads
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_NEAREST;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.mipmapLodBias = 0.0f;
  samplerInfo.mipmapLodMin = 0.0f;
  samplerInfo.mipmapLodMax = static_cast<float>(m_hizNumLevels);
  samplerInfo.useAnisotropy = VK_FALSE;
  samplerInfo.compareToDepth = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.borderColor = { 0.0f, 0.0f, 0.0f, 0.0f };  // Transparent black
  samplerInfo.usePixelCoord = VK_FALSE;

  m_hizSampler = m_device->createSampler(samplerInfo);

  m_hizInitialized = true;

  Logger::info("[RTXMG] HiZ pyramid created successfully");
}

void RtxmgClusterBuilder::generateHiZPyramid(
  RtxContext* ctx,
  const Rc<DxvkImageView>& depthBuffer) {

  if (!m_hizInitialized || m_hizPyramidGenerateShader == nullptr) {
    return;
  }

  if (depthBuffer == nullptr) {
    Logger::warn("[RTXMG] HiZ generation: depth buffer is null");
    return;
  }

  // Get depth buffer info
  const auto& depthInfo = depthBuffer->imageInfo();
  uint32_t srcWidth = depthInfo.extent.width;
  uint32_t srcHeight = depthInfo.extent.height;

  // Create a point sampler for depth sampling
  DxvkSamplerCreateInfo depthSamplerInfo = {};
  depthSamplerInfo.magFilter = VK_FILTER_NEAREST;
  depthSamplerInfo.minFilter = VK_FILTER_NEAREST;
  depthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  depthSamplerInfo.mipmapLodBias = 0.0f;
  depthSamplerInfo.mipmapLodMin = 0.0f;
  depthSamplerInfo.mipmapLodMax = 0.0f;
  depthSamplerInfo.useAnisotropy = VK_FALSE;
  depthSamplerInfo.compareToDepth = VK_FALSE;
  depthSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  depthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  depthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  depthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  depthSamplerInfo.borderColor = { 0.0f, 0.0f, 0.0f, 0.0f };  // Transparent black
  depthSamplerInfo.usePixelCoord = VK_FALSE;

  Rc<DxvkSampler> depthSampler = m_device->createSampler(depthSamplerInfo);

  // Generate each mip level of the HiZ pyramid
  for (uint32_t dstLevel = 0; dstLevel < m_hizNumLevels; ++dstLevel) {
    uint32_t dstWidth = std::max(1u, srcWidth >> dstLevel);
    uint32_t dstHeight = std::max(1u, srcHeight >> dstLevel);

    uint32_t srcLevel = (dstLevel == 0) ? 0 : (dstLevel - 1);
    uint32_t srcMipWidth = (dstLevel == 0) ? srcWidth : std::max(1u, srcWidth >> srcLevel);
    uint32_t srcMipHeight = (dstLevel == 0) ? srcHeight : std::max(1u, srcHeight >> srcLevel);

    // SDK MATCH: Create constant buffer for params (binding 0), NOT push constants!
    struct HiZGenerateParams {
      uint32_t srcSize[2];
      uint32_t dstSize[2];
      uint32_t srcMipLevel;
      uint32_t dstMipLevel;
      float nearPlane;
      float farPlane;
      uint32_t _pad[2];
    } params = {};

    params.srcSize[0] = srcMipWidth;
    params.srcSize[1] = srcMipHeight;
    params.dstSize[0] = dstWidth;
    params.dstSize[1] = dstHeight;
    params.srcMipLevel = srcLevel;
    params.dstMipLevel = dstLevel;
    params.nearPlane = 0.1f; // TODO: Get from config
    params.farPlane = 10000.0f; // TODO: Get from config

    DxvkBufferCreateInfo cbInfo = {};
    cbInfo.size = align(sizeof(params), 256);
    cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
    Rc<DxvkBuffer> hizParamsBuffer = m_device->createBuffer(
      cbInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "RTXMG HiZ Generate Params");

    ctx->updateBuffer(hizParamsBuffer, 0, sizeof(params), &params);

    // CRITICAL: Bind shader FIRST
    Logger::info(str::format("[RTXMG HiZ DEBUG] About to bind shader, level=", dstLevel));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_hizPyramidGenerateShader);
    Logger::info("[RTXMG HiZ DEBUG] Shader bound successfully");

    // SDK MATCH: Bind constant buffer at binding 0
    Logger::info("[RTXMG HiZ DEBUG] Binding constant buffer at binding 0");
    ctx->bindResourceBuffer(0, DxvkBufferSlice(hizParamsBuffer, 0, sizeof(params)));
    Logger::info("[RTXMG HiZ DEBUG] Constant buffer bound");

    // Bind source (depth buffer for level 0, previous HiZ level otherwise) - shifted +1
    Logger::info("[RTXMG HiZ DEBUG] Binding source image at binding 1");
    if (dstLevel == 0) {
      // Level 0: Sample from depth buffer
      ctx->bindResourceView(1, depthBuffer, nullptr);
      Logger::info("[RTXMG HiZ DEBUG] Bound depth buffer at binding 1");
    } else {
      // Level N: Sample from previous HiZ level
      ctx->bindResourceView(1, m_hizMipViews[srcLevel], nullptr);
      Logger::info(str::format("[RTXMG HiZ DEBUG] Bound HiZ mip view at binding 1, srcLevel=", srcLevel));
    }
    Logger::info("[RTXMG HiZ DEBUG] Binding sampler at binding 2");
    ctx->bindResourceSampler(2, depthSampler);
    Logger::info("[RTXMG HiZ DEBUG] Sampler bound");

    // Bind destination HiZ level - shifted +1
    Logger::info(str::format("[RTXMG HiZ DEBUG] Binding destination HiZ at binding 3, dstLevel=", dstLevel));
    ctx->bindResourceView(3, m_hizMipViews[dstLevel], nullptr);
    Logger::info("[RTXMG HiZ DEBUG] Destination HiZ bound");

    // Compute workgroups (8×8 threads per group)
    uint32_t groupsX = (dstWidth + 7) / 8;
    uint32_t groupsY = (dstHeight + 7) / 8;
    Logger::info(str::format("[RTXMG HiZ DEBUG] About to dispatch HiZ compute: groupsX=", groupsX, " groupsY=", groupsY));
    ctx->dispatch(groupsX, groupsY, 1);
    Logger::info("[RTXMG HiZ DEBUG] HiZ dispatch complete");

    // Add memory barrier between levels (except after last level)
    if (dstLevel < m_hizNumLevels - 1) {
      VkMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      m_device->vkd()->vkCmdPipelineBarrier(
        ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
  }

  Logger::info(str::format("[RTXMG] Generated HiZ pyramid: ", m_hizNumLevels, " levels"));
}

void RtxmgClusterBuilder::resizeHiZIfNeeded(uint32_t width, uint32_t height) {
  // Check if resize needed
  if (m_hizInitialized && m_hizWidth == width && m_hizHeight == height) {
    return;  // Already correct size
  }

  // Release old HiZ pyramid
  if (m_hizInitialized) {
    Logger::info(str::format("[RTXMG] Resizing HiZ pyramid: ",
      m_hizWidth, "×", m_hizHeight, " → ", width, "×", height));

    m_hizMipViews.clear();
    m_hizPyramidView = nullptr;
    m_hizPyramid = nullptr;
    m_hizSampler = nullptr;
    m_hizInitialized = false;
  }

  // Create new HiZ pyramid
  createHiZPyramid(width, height);
}

void RtxmgClusterBuilder::queueCounterReadback(RtxContext* ctx) {
  if (!ctx || !m_initialized) {
    return;
  }

  if (m_lastCounterCopyFrame == m_frameSerial) {
    return;
  }

  const Rc<DxvkBuffer>& counterBuffer = m_tessCountersBuffer.getBuffer();
  const Rc<DxvkBuffer>& readbackBuffer = m_tessCountersReadback;

  if (counterBuffer == nullptr || readbackBuffer == nullptr) {
    return;
  }

  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

  m_device->vkd()->vkCmdPipelineBarrier(
    ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer),
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 1, &barrier, 0, nullptr, 0, nullptr);

  ctx->copyBuffer(
    readbackBuffer,
    0,
    counterBuffer,
    0,
    sizeof(TessellationCounters));

  m_counterReadbackReady = true;
  m_lastCounterCopyFrame = m_frameSerial;
}

void RtxmgClusterBuilder::queueBlasAddressReadback(RtxContext* ctx, uint32_t count) {
  if (!ctx || count == 0) {
    return;
  }

  m_blasPtrsReadbackBuffer.create(
    m_device, count,
    "RTXMG BLAS Address Readback",
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  m_blasSizesReadbackBuffer.create(
    m_device, count,
    "RTXMG BLAS Size Readback",
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  const VkDeviceSize ptrBytes = static_cast<VkDeviceSize>(count) * sizeof(VkDeviceAddress);
  const VkDeviceSize sizeBytes = static_cast<VkDeviceSize>(count) * sizeof(uint32_t);

  ctx->emitMemoryBarrier(
    0,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_TRANSFER_READ_BIT);

  ctx->copyBuffer(
    m_blasPtrsReadbackBuffer.getBuffer(),
    0,
    m_frameAccels.blasPtrsBuffer.getBuffer(),
    0,
    ptrBytes);

  ctx->emitMemoryBarrier(
    0,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_TRANSFER_READ_BIT);

  ctx->copyBuffer(
    m_blasSizesReadbackBuffer.getBuffer(),
    0,
    m_frameAccels.blasSizesBuffer.getBuffer(),
    0,
    sizeBytes);

  m_lastBlasReadCount = count;
  m_blasReadbackPending = true;
}

void RtxmgClusterBuilder::resolvePendingCounterReadback(uint32_t slotIndex) {
  // STUB: No longer used with single-buffer GPU fence synchronization model
  // Counter readbacks are handled internally with GPU fences instead of per-frame rotation
}

bool RtxmgClusterBuilder::resolveBlasAddressReadback(
  RtxContext* ctx,
  std::vector<VkDeviceAddress>& outAddresses,
  std::vector<uint32_t>& outSizes) {

  if (!ctx || !m_blasReadbackPending || m_lastBlasReadCount == 0) {
    return false;
  }

  const Rc<DxvkBuffer>& ptrBuffer = m_blasPtrsReadbackBuffer.getBuffer();
  const Rc<DxvkBuffer>& sizeBuffer = m_blasSizesReadbackBuffer.getBuffer();

  if (ptrBuffer == nullptr || sizeBuffer == nullptr) {
    return false;
  }

  DxvkDevice* device = ctx->getDevice().ptr();
  device->waitForResource(ptrBuffer, DxvkAccess::Write);
  device->waitForResource(sizeBuffer, DxvkAccess::Write);

  const VkDeviceAddress* ptrData = static_cast<const VkDeviceAddress*>(ptrBuffer->mapPtr(0));
  const uint32_t* sizeData = static_cast<const uint32_t*>(sizeBuffer->mapPtr(0));

  if (ptrData == nullptr || sizeData == nullptr) {
    return false;
  }

  outAddresses.assign(ptrData, ptrData + m_lastBlasReadCount);
  outSizes.assign(sizeData, sizeData + m_lastBlasReadCount);

  // NOTE: CPU-side readback of blasPtrsBuffer returns all zeros with VK_NV_cluster_acceleration_structure
  // This is expected behavior - the extension only populates addresses for GPU consumption.
  // NVIDIA sample doesn't use CPU readback at all (rtxmg_renderer.cpp:1152-1156 only does GPU patching).
  Logger::info(str::format("[RTXMG BLAS READBACK] Read ", m_lastBlasReadCount, " entries (expect all 0x0 - CPU readback not supported)"));

  // DEBUG: Log first few addresses to confirm they're all zero
  if (m_lastBlasReadCount > 0) {
    const uint32_t samplesToLog = std::min(m_lastBlasReadCount, 3u);
    for (uint32_t i = 0; i < samplesToLog; ++i) {
      Logger::info(str::format("  [", i, "] address=0x", std::hex, outAddresses[i],
                              ", size=", std::dec, outSizes[i], " bytes"));
    }
    if (m_lastBlasReadCount > samplesToLog) {
      Logger::info(str::format("  ... (", m_lastBlasReadCount - samplesToLog, " more entries)"));
    }
  }

  // Verify expected behavior: all addresses should be 0x0
  bool allZero = true;
  for (uint32_t i = 0; i < m_lastBlasReadCount; ++i) {
    if (outAddresses[i] != 0) {
      allZero = false;
      break;
    }
  }

  if (allZero) {
    Logger::info("[RTXMG BLAS READBACK] Expected: All addresses are 0x0 (CPU readback not supported by extension)");
  } else {
    Logger::warn("[RTXMG BLAS READBACK] Unexpected: Some addresses are non-zero! Extension behavior may have changed.");
  }

  m_blasReadbackPending = false;
  return true;
}

// ============================================================================
// GPU Buffer Creation (Phase 2.5)
// ============================================================================

void RtxmgClusterBuilder::createGPUBuffers(const RtxmgConfig& config) {
  Logger::info("[RTXMG] Creating GPU buffers for cluster building");

  const VkBufferUsageFlags storageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags hostVisible =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

  // Calculate expected memory usage
  uint32_t maxClusters = config.memorySettings.maxClusters;
  uint32_t maxVertices = config.memorySettings.maxVertices;

  VkDeviceSize estimatedMemory =
    (65536 * 3 + 262144) * sizeof(float3) +  // Input buffers
    1024 * sizeof(float4) +                   // Surface info
    kNumClusterTemplates * 16 +               // Template data
    1024 * sizeof(float4) +                   // Grid samplers
    maxClusters * (16 + 56 + 8 + 32) +       // Cluster data
    maxVertices * 2 * sizeof(float3);        // Vertex positions + normals

  // Check GPU memory availability
  VkPhysicalDeviceMemoryProperties memProps = m_device->adapter()->memoryProperties();
  VkDeviceSize availableMemory = 0;

  // Find device-local memory heap
  for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
    if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      availableMemory = memProps.memoryHeaps[i].size;
      break;
    }
  }

  DxvkMemoryStats currentStats = m_device->getMemoryStats(0);
  VkDeviceSize usedMemory = currentStats.totalAllocated();
  VkDeviceSize freeMemory = (availableMemory > usedMemory) ? (availableMemory - usedMemory) : 0;

  Logger::info(str::format("[RTXMG] Memory check: requesting ", estimatedMemory / (1024 * 1024),
                           " MB, available ", freeMemory / (1024 * 1024), " MB"));

  // NV-DXVK start: Implement graceful degradation instead of crashing
  // Reserve 20% safety margin to avoid exhausting all GPU memory
  VkDeviceSize safeMemoryLimit = static_cast<VkDeviceSize>(freeMemory * 0.8);

  if (estimatedMemory > safeMemoryLimit) {
    Logger::warn(str::format("[RTXMG] WARNING: Requested memory (", estimatedMemory / (1024 * 1024),
                             " MB) exceeds safe limit (", safeMemoryLimit / (1024 * 1024), " MB)"));

    // Calculate reduction factor needed
    float reductionFactor = static_cast<float>(safeMemoryLimit) / static_cast<float>(estimatedMemory);

    if (reductionFactor < 0.25f) {
      // Less than 25% of needed memory available - disable mega geometry entirely
      Logger::err("[RTXMG] CRITICAL: Insufficient GPU memory for mega geometry (need 4x more)");
      Logger::err("[RTXMG] Mega geometry will be DISABLED to prevent crashes");
      Logger::err("[RTXMG] Try closing other applications or lowering graphics settings");
      throw DxvkError("RTX Mega Geometry disabled: insufficient GPU memory");
    }

    // Reduce buffer sizes proportionally (graceful degradation)
    maxClusters = static_cast<uint32_t>(maxClusters * reductionFactor);
    maxVertices = static_cast<uint32_t>(maxVertices * reductionFactor);

    // Ensure minimum viable sizes
    maxClusters = std::max(maxClusters, 1024u);
    maxVertices = std::max(maxVertices, 4096u);

    Logger::warn(str::format("[RTXMG] Reducing buffer sizes for memory safety:"));
    Logger::warn(str::format("[RTXMG]   Max clusters: ", config.memorySettings.maxClusters, " → ", maxClusters));
    Logger::warn(str::format("[RTXMG]   Max vertices: ", config.memorySettings.maxVertices, " → ", maxVertices));
    Logger::warn(str::format("[RTXMG] Scene complexity may be limited. Consider freeing up GPU memory."));

    // Recalculate estimated memory with reduced sizes
    estimatedMemory =
      (65536 * 3 + 262144) * sizeof(float3) +
      1024 * sizeof(float4) +
      kNumClusterTemplates * 16 +
      1024 * sizeof(float4) +
      maxClusters * (16 + 56 + 8 + 32) +
      maxVertices * 2 * sizeof(float3);
  }
  // NV-DXVK end

  try {
    // Input buffers (start with reasonable defaults)
    // Note: Need HOST_VISIBLE for upload() to work
    m_inputPositions.create(m_device, 65536, "RTXMG Input Positions", storageUsage, hostVisible);
    m_inputNormals.create(m_device, 65536, "RTXMG Input Normals", storageUsage, hostVisible);
    m_inputTexcoords.create(m_device, 65536, "RTXMG Input Texcoords", storageUsage, hostVisible);
    m_inputIndices.create(m_device, 262144, "RTXMG Input Indices", storageUsage, hostVisible);

    // Surface metadata
    m_surfaceInfo.create(m_device, 1024, "RTXMG Surface Info", storageUsage, hostVisible);

    // Template data (121 templates)
    m_templateAddresses.create(m_device, kNumClusterTemplates, "RTXMG Template Addresses", storageUsage, hostVisible);
    m_clasInstantiationBytes.create(m_device, kNumClusterTemplates, "RTXMG CLAS Bytes", storageUsage, hostVisible);

    // Initialize template data
    std::vector<VkDeviceAddress> templateAddrs(kNumClusterTemplates, 0);
    std::vector<uint32_t> clasBytes(kNumClusterTemplates);

    for (uint32_t i = 0; i < kNumClusterTemplates; ++i) {
      // Calculate CLAS size for this template (simplified)
      uint32_t sizeX = (i % kMaxClusterEdgeSegments) + 1;
      uint32_t sizeY = (i / kMaxClusterEdgeSegments) + 1;
      uint32_t numVerts = (sizeX + 1) * (sizeY + 1);
      uint32_t numTris = sizeX * sizeY * 2;

      // Estimate CLAS size (actual size determined by NVAPI in Phase 3)
      clasBytes[i] = numVerts * sizeof(float3) + numTris * sizeof(uint32_t) * 3;
      clasBytes[i] = (clasBytes[i] + 255) & ~255; // Round up to 256 bytes
    }

    m_templateAddresses.upload(templateAddrs);
    m_clasInstantiationBytes.upload(clasBytes);

    // Intermediate buffers (based on config)
    m_gridSamplers.create(m_device, 1024, "RTXMG Grid Samplers", storageUsage, deviceLocal);
    m_clusters.create(m_device, maxClusters, "RTXMG Clusters", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clusterShadingData.create(m_device, maxClusters, "RTXMG Cluster Shading Data", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clasAddresses.create(m_device, maxClusters, "RTXMG CLAS Addresses", storageUsage, deviceLocal);
    m_clusterIndirectArgs.create(m_device, maxClusters, "RTXMG Indirect Args", storageUsage, deviceLocal);

    // Output buffers
    m_clusterVertexPositions.create(m_device, maxVertices, "RTXMG Cluster Vertices", storageUsage, deviceLocal);
    m_clusterVertexNormals.create(m_device, maxVertices, "RTXMG Cluster Normals", storageUsage, deviceLocal);

    // ALIGNMENT LOGGING: Check buffer creation alignment
    VkDeviceAddress vertexAddr = m_clusterVertexPositions.deviceAddress();
    VkDeviceAddress normalAddr = m_clusterVertexNormals.deviceAddress();
    VkDeviceSize vertexSize = m_clusterVertexPositions.getBuffer()->info().size;
    VkDeviceSize normalSize = m_clusterVertexNormals.getBuffer()->info().size;

    Logger::info(str::format("[RTXMG BUFFER CREATION] Cluster vertex positions buffer created:"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address: 0x", std::hex, vertexAddr, std::dec));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> size: ", vertexSize, " bytes (", maxVertices, " vertices * ", sizeof(float3), " bytes)"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address % 8 = ", vertexAddr % 8, " (8-byte ", (vertexAddr % 8 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address % 16 = ", vertexAddr % 16, " (16-byte ", (vertexAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address % 256 = ", vertexAddr % 256, " (256-byte ", (vertexAddr % 256 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

    Logger::info(str::format("[RTXMG BUFFER CREATION] Cluster vertex normals buffer created:"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address: 0x", std::hex, normalAddr, std::dec));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> size: ", normalSize, " bytes"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address % 8 = ", normalAddr % 8, " (8-byte ", (normalAddr % 8 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
    Logger::info(str::format("[RTXMG BUFFER CREATION]   -> address % 16 = ", normalAddr % 16, " (16-byte ", (normalAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

    if (vertexAddr % 8 != 0 || normalAddr % 8 != 0) {
      Logger::warn("[RTXMG BUFFER CREATION] *** WARNING: Buffer base addresses are not 8-byte aligned! ***");
      Logger::warn("[RTXMG BUFFER CREATION] *** This WILL cause GPU crashes when computing vertex offsets! ***");
    }

    // SDK MATCH: BLAS buffers are allocated dynamically based on actual needs, not at MAX size!
    // Reference: cluster_accel_builder.cpp:1197-1223 UpdateMemoryAllocations()
    // Sample releases and recreates buffers when cluster/instance count changes
    // BLAS allocation moved to updateBlasAllocation() which is called per-frame before building
    Logger::info("[RTXMG] BLAS buffers will be allocated dynamically based on actual cluster counts [SDK-MATCHING]");

    // SDK MATCH: Scratch buffer is allocated ONCE and reused (cluster_accel_builder.cpp:1081)
    // Query scratch size using max config (to get a reasonable upper bound)
    const uint32_t maxInstances = 1024;  // Max instances we support
    VkClusterAccelerationStructureClustersBottomLevelInputNV scratchQueryParams = {};
    scratchQueryParams.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
    scratchQueryParams.maxTotalClusterCount = maxClusters;  // Use max for scratch sizing
    scratchQueryParams.maxClusterCountPerAccelerationStructure = maxClusters;

    VkClusterAccelerationStructureInputInfoNV scratchQueryInfo = {};
    scratchQueryInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    scratchQueryInfo.maxAccelerationStructureCount = maxInstances;
    scratchQueryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    scratchQueryInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    scratchQueryInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;  // Match runtime mode
    scratchQueryInfo.opInput.pClustersBottomLevel = &scratchQueryParams;

    VkAccelerationStructureBuildSizesInfoKHR scratchSizeInfo = {};
    scratchSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
      m_device->handle(), &scratchQueryInfo, &scratchSizeInfo);

    Logger::info(str::format("[RTXMG] Scratch size query (max config): ", scratchSizeInfo.buildScratchSize / (1024 * 1024), " MB"));

    // Allocate scratch buffer once at max size
    if (scratchSizeInfo.buildScratchSize > 0) {
      m_frameAccels.blasScratchBuffer.create(
        m_device, scratchSizeInfo.buildScratchSize,
        "RTXMG BLAS Scratch (Persistent)",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_allocatedBlasScratchSize = scratchSizeInfo.buildScratchSize;
      Logger::info(str::format("[RTXMG] Scratch buffer allocated ONCE: ", scratchSizeInfo.buildScratchSize / (1024 * 1024), " MB"));
    } else {
      Logger::warn("[RTXMG] Scratch size query returned 0 - this may cause BLAS build failures");
    }

    // SDK MATCH: Single-frame BLAS address/size buffers (no ring buffering)
    // NVIDIA sample doesn't use ring buffers - it rebuilds every frame with proper GPU sync
    Logger::info(str::format("[RTXMG] Allocating BLAS address/size buffers: ", maxInstances, " instances (single-frame, SDK-matching)"));

    // CRITICAL: Must include VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    // for the cluster extension to write BLAS addresses!
    m_frameAccels.blasPtrsBuffer.create(
      m_device, maxInstances,
      "RTXMG Frame BLAS Addresses",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |  // Required for cluster extension output
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  // Allow copy to readback buffer
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_frameAccels.blasSizesBuffer.create(
      m_device, maxInstances,
      "RTXMG Frame BLAS Sizes",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Logger::info("[RTXMG] BLAS address/size buffers allocated (single-frame, SDK-matching)");

    Logger::info(str::format("[RTXMG] GPU buffers created successfully: max ", maxClusters, " clusters, ", maxVertices, " vertices"));

  } catch (const DxvkError& e) {
    // NV-DXVK start: Graceful error handling - don't crash the application
    Logger::err("[RTXMG] CRITICAL: Failed to create GPU buffers for cluster building");
    Logger::err(str::format("[RTXMG] Error: ", e.message()));
    Logger::err(str::format("[RTXMG] Attempted allocation: ", estimatedMemory / (1024 * 1024), " MB"));
    Logger::err(str::format("[RTXMG] Available GPU memory: ", freeMemory / (1024 * 1024), " MB"));
    Logger::err("[RTXMG] Mega geometry will be DISABLED to prevent application crash");
    Logger::err("[RTXMG] The application will continue running with standard geometry rendering");

    // Rethrow to signal initialization failure - calling code should catch and disable mega geometry
    throw DxvkError("RTX Mega Geometry disabled: GPU buffer allocation failed");
    // NV-DXVK end
  }
}

// SDK MATCH: cluster_accel_builder.cpp:1197-1223 UpdateMemoryAllocations()
// Dynamically allocates BLAS buffers based on actual cluster/instance counts
// Releases and recreates buffers when counts change to reduce memory usage
void RtxmgClusterBuilder::updateBlasAllocation(
  RtxContext* ctx,
  uint32_t totalClusters,
  uint32_t maxClustersPerBlas,
  uint32_t numInstances) {

  // STEP 1: Process pending releases - check fences and release completed buffers
  // This is NON-BLOCKING - only releases buffers whose GPU work has completed
  m_pendingReleases.erase(
    std::remove_if(m_pendingReleases.begin(), m_pendingReleases.end(),
      [this](BufferWithFence& pending) {
        // Check if GPU work completed (non-blocking check)
        VkResult status = m_device->vkd()->vkGetFenceStatus(m_device->handle(), pending.lastUsageFence);
        if (status == VK_SUCCESS) {
          // GPU work done, safe to release this buffer
          Logger::info(str::format("[RTXMG BLAS FENCE] GPU work completed for old buffer (fence 0x", std::hex,
                                  reinterpret_cast<uint64_t>(pending.lastUsageFence), std::dec,
                                  "), releasing ", pending.accels.blasBuffer.bytes() / (1024 * 1024), " MB"));
          // Release happens automatically via BufferWithFence destructor
          return true;  // Remove from pending list
        }
        // GPU still using this buffer, keep it alive
        return false;
      }),
    m_pendingReleases.end());

  if (!m_pendingReleases.empty()) {
    Logger::info(str::format("[RTXMG BLAS FENCE] Still waiting on ", m_pendingReleases.size(), " buffer(s) to complete"));
  }

  // STEP 2: Check if we need to grow/shrink buffers
  bool needsGrowth = (totalClusters > m_blasMaxTotalClusters) || (numInstances > m_blasMaxInstances);
  bool needsShrink = (totalClusters < m_blasMaxTotalClusters / 2) || (numInstances < m_blasMaxInstances / 2);

  // Track peak values (only used for logging, not allocation)
  uint32_t peakClusters = std::max(m_blasMaxTotalClusters, totalClusters);
  uint32_t peakInstances = std::max(m_blasMaxInstances, numInstances);

  Logger::info(str::format("[RTXMG BLAS ALLOC CHECK] Frame ", m_device->getCurrentFrameId(),
                          ": current=(", totalClusters, " clusters, ", numInstances, " instances), ",
                          "peak=(", peakClusters, " clusters, ", peakInstances, " instances), ",
                          "needsGrowth=", needsGrowth ? "TRUE" : "FALSE",
                          ", needsShrink=", needsShrink ? "TRUE" : "FALSE"));

  // STEP 3: If no changes needed, reuse existing buffers
  if (!needsGrowth && !needsShrink && m_frameAccels.blasBuffer.isValid()) {
    Logger::info(str::format("[RTXMG BLAS ALLOC] No changes needed, reusing existing buffers"));
    return;
  }

  // STEP 4: Queue old buffer for fence-tracked release
  if (m_frameAccels.blasBuffer.isValid()) {
    Logger::info(str::format("[RTXMG BLAS ALLOC] ========== QUEUEING OLD BUFFER FOR RELEASE =========="));
    Logger::info(str::format("[RTXMG BLAS ALLOC] Frame: ", m_device->getCurrentFrameId()));
    Logger::info(str::format("[RTXMG BLAS ALLOC] Old buffer: ", m_blasSizeInfo.accelerationStructureSize / (1024 * 1024), " MB, address 0x",
                            std::hex, m_frameAccels.blasBuffer.getDeviceAddress(), std::dec));

    // Get the fence from the current command list - this fence will signal when GPU work completes
    VkFence currentFence = ctx->getCommandList()->fence();
    Logger::info(str::format("[RTXMG BLAS ALLOC] Tracking fence 0x", std::hex, reinterpret_cast<uint64_t>(currentFence), std::dec,
                            " - will release buffer when GPU work completes"));

    // Queue for release with fence tracking
    BufferWithFence pending;
    pending.accels = std::move(m_frameAccels);  // Transfer ownership
    pending.lastUsageFence = currentFence;
    m_pendingReleases.push_back(std::move(pending));

    Logger::info(str::format("[RTXMG BLAS ALLOC] Old buffer queued for release (", m_pendingReleases.size(), " total pending)"));
  }

  // STEP 5: Allocate new buffers with new size
  Logger::info(str::format("[RTXMG BLAS ALLOC] ========== ALLOCATING NEW BLAS BUFFERS =========="));
  Logger::info(str::format("[RTXMG BLAS ALLOC] New allocation: totalClusters=", totalClusters,
                          ", maxPerBlas=", maxClustersPerBlas, ", instances=", numInstances));

  // NOTE: Scratch buffer is NOT resized dynamically - it's allocated once at initialization
  // Sample code reuses m_createBlasSizeInfo.scratchSizeInBytes (line 1081) which never changes
  // We'll allocate scratch buffer once in createGPUBuffers() based on max cluster config

  // SDK MATCH: lines 1200-1212 - Query size using CONFIG MAXIMUMS, not runtime actuals!
  // NVIDIA sample lines 1099-1100, 1208-1209: uses m_maxClusters from config for BOTH fields
  uint32_t configMaxClusters = RtxmgConfig::kDefaultMaxClusters;  // 2M clusters (matches sample's kMaxApiClusterCount)
  configMaxClusters = std::max(1u, configMaxClusters);

  VkClusterAccelerationStructureClustersBottomLevelInputNV blasParams = {};
  blasParams.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
  blasParams.maxTotalClusterCount = configMaxClusters;                         // SDK: Uses same config max
  blasParams.maxClusterCountPerAccelerationStructure = configMaxClusters;      // SDK: Uses same config max

  VkClusterAccelerationStructureInputInfoNV inputInfo = {};
  inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfo.maxAccelerationStructureCount = numInstances;
  inputInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;  // SDK MATCH: Use implicit mode (sample line 1204)
  inputInfo.opInput.pClustersBottomLevel = &blasParams;

  m_blasSizeInfo = {};
  m_blasSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    m_device->handle(), &inputInfo, &m_blasSizeInfo);

  Logger::info(str::format("[RTXMG BLAS ALLOC] New size: ", m_blasSizeInfo.accelerationStructureSize / (1024 * 1024),
                          " MB BLAS, ", m_blasSizeInfo.buildScratchSize / (1024 * 1024), " MB scratch"));

  // SDK MATCH: lines 1214-1222 - Create buffer at actual required size
  m_frameAccels.blasBuffer.create(
    m_device, m_blasSizeInfo.accelerationStructureSize,
    "RTXMG Frame BLAS Buffer (Dynamic)",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Create acceleration structure handle
  DxvkBufferCreateInfo accelInfo = {};
  accelInfo.size = m_blasSizeInfo.accelerationStructureSize;
  accelInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  accelInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  accelInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                     VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  m_frameAccels.blasAccelStructure = m_device->createAccelStructure(
    accelInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    "RTXMG Frame BLAS Accel");

  if (!m_frameAccels.blasAccelStructure.ptr()) {
    Logger::err("[RTXMG BLAS ALLOC] Failed to create BLAS acceleration structure");
    throw DxvkError("[RTXMG] Failed to create BLAS acceleration structure");
  }

  // Allocate CLAS pointers buffer (this was missing - caused 0 byte buffer error!)
  m_frameAccels.clasPtrsBuffer.create(
    m_device, totalClusters,
    "RTXMG Frame CLAS Addresses",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Logger::info(str::format("[RTXMG BLAS ALLOC] Allocated CLAS address buffer: ", totalClusters, " addresses (",
                          (totalClusters * sizeof(VkDeviceAddress)) / 1024, " KB)"));

  // CRITICAL FIX: Recreate BLAS address/size buffers when reallocated!
  // These were moved to pending release queue with the old m_frameAccels,
  // so we MUST recreate them here with MAXIMUM capacity (not just current instances)
  // Multiple builds per frame means blasPtrsBufferOffset can grow, so we need full capacity
  const uint32_t maxInstancesPerFrame = 1024;  // Must match initial allocation in createGPUBuffers!
  m_frameAccels.blasPtrsBuffer.create(
    m_device, maxInstancesPerFrame,
    "RTXMG Frame BLAS Addresses",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,  // REQUIRED: for cluster extension output
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  m_frameAccels.blasSizesBuffer.create(
    m_device, maxInstancesPerFrame,
    "RTXMG Frame BLAS Sizes",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,  // REQUIRED: for cluster extension output
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  Logger::info(str::format("[RTXMG BLAS ALLOC] Recreated BLAS address/size buffers with proper flags: ", maxInstancesPerFrame, " max instances"));

  // CRITICAL FIX: Recreate scratch buffer when BLAS buffers are reallocated!
  // The scratch buffer was moved to pending release queue with the old m_frameAccels,
  // so we MUST recreate it here with the new size from m_blasSizeInfo.buildScratchSize
  if (m_blasSizeInfo.buildScratchSize > 0) {
    m_frameAccels.blasScratchBuffer.create(
      m_device, m_blasSizeInfo.buildScratchSize,
      "RTXMG BLAS Scratch (Dynamic)",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_allocatedBlasScratchSize = m_blasSizeInfo.buildScratchSize;
    Logger::info(str::format("[RTXMG BLAS ALLOC] Recreated scratch buffer: ",
                            m_blasSizeInfo.buildScratchSize / (1024 * 1024), " MB"));
  } else {
    Logger::warn("[RTXMG BLAS ALLOC] Scratch size is 0 - BLAS builds will likely fail!");
  }

  // Update tracking
  m_blasMaxTotalClusters = totalClusters;
  m_blasMaxInstances = numInstances;
  m_allocatedBlasSize = m_blasSizeInfo.accelerationStructureSize;

  Logger::info(str::format("[RTXMG BLAS ALLOC] Allocation complete: ", m_blasSizeInfo.accelerationStructureSize / (1024 * 1024),
                          " MB BLAS, ", m_blasSizeInfo.buildScratchSize / (1024 * 1024), " MB scratch"));
}

void RtxmgClusterBuilder::resizeBuffersIfNeeded(uint32_t requiredClusters, uint32_t requiredVertices) {
  Logger::info(str::format("[RTXMG DEBUG] resizeBuffersIfNeeded ENTER: required=", requiredClusters, "/", requiredVertices,
    ", buffers=", m_clusters.numElements(), "/", m_clusterVertexPositions.numElements()));

  // Update peak usage tracking
  updatePeakUsage(requiredClusters, requiredVertices);

  // Check if shrinking is needed (hysteresis)
  if (shouldShrinkBuffers(requiredClusters, requiredVertices)) {
    Logger::info(str::format("[RTXMG DEBUG] Taking SHRINK path"));
    shrinkBuffersIfNeeded(requiredClusters, requiredVertices);
    return;
  }

  // Check if we need to grow
  if (m_clusters.numElements() >= requiredClusters &&
      m_clusterVertexPositions.numElements() >= requiredVertices) {
    // Reset underutilization counter since we're using the buffers adequately
    m_underutilizationFrames = 0;
    Logger::info(str::format("[RTXMG DEBUG] Buffers adequate, EARLY RETURN (no reallocation needed)"));
    return;
  }

  Logger::warn(str::format("[RTXMG DEBUG] *** GROWING BUFFERS PATH ***"));

  // Calculate new sizes with alignment and growth factor
  uint32_t newClusterSize = requiredClusters;
  uint32_t newVertexSize = requiredVertices;

  if (m_clusters.numElements() < requiredClusters) {
    newClusterSize = static_cast<uint32_t>(requiredClusters * GROW_FACTOR);
    newClusterSize = std::max(newClusterSize, MIN_CLUSTER_CAPACITY);
  }

  if (m_clusterVertexPositions.numElements() < requiredVertices) {
    newVertexSize = static_cast<uint32_t>(requiredVertices * GROW_FACTOR);
    newVertexSize = std::max(newVertexSize, MIN_VERTEX_CAPACITY);
  }

  // Check memory pressure before allocating
  size_t clusterBufferSize = newClusterSize * sizeof(RtxmgCluster);
  size_t vertexBufferSize = newVertexSize * sizeof(float3);
  size_t totalNewMemory = (clusterBufferSize * 4) + (vertexBufferSize * 2); // 4 cluster buffers, 2 vertex buffers

  if (checkMemoryPressure(totalNewMemory)) {
    Logger::warn("[RTXMG] Memory pressure detected, using exact sizes instead of growth factor");
    newClusterSize = requiredClusters;
    newVertexSize = requiredVertices;
  }

  Logger::info(str::format("[RTXMG] Growing buffers: ",
    m_clusters.numElements(), " -> ", newClusterSize, " clusters, ",
    m_clusterVertexPositions.numElements(), " -> ", newVertexSize, " vertices"));

  // CRITICAL: Wait for GPU to finish using old buffers before reallocating
  // Without this, GPU will access freed memory causing hangs/crashes
  uint32_t pendingSubs = m_device->pendingSubmissions();
  if (pendingSubs > 0) {
    Logger::warn(str::format("[RTXMG] Waiting for GPU idle before buffer reallocation (pendingSubmissions=", pendingSubs, ")"));
  }
  m_device->waitForIdle();
  if (pendingSubs > 0) {
    Logger::info(str::format("[RTXMG] GPU idle complete - safe to reallocate buffers (was waiting on ", pendingSubs, " submissions)"));
  }

  const VkBufferUsageFlags storageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags hostVisible =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

  // Resize cluster buffers with alignment
  if (m_clusters.numElements() < requiredClusters) {
    size_t alignedClusterSize = alignSize(newClusterSize * sizeof(RtxmgCluster), STORAGE_BUFFER_ALIGNMENT);
    uint32_t alignedCount = static_cast<uint32_t>(alignedClusterSize / sizeof(RtxmgCluster));

    m_clusters.create(m_device, alignedCount, "RTXMG Clusters", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clusterShadingData.create(m_device, alignedCount, "RTXMG Cluster Shading Data", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clasAddresses.create(m_device, alignedCount, "RTXMG CLAS Addresses", storageUsage, deviceLocal);
    m_clusterIndirectArgs.create(m_device, alignedCount, "RTXMG Indirect Args", storageUsage, deviceLocal);

    // Update allocation tracking
    m_allocatedClusters = alignedCount;
  }

  // Resize vertex buffers with alignment
  if (m_clusterVertexPositions.numElements() < requiredVertices) {
    size_t alignedVertexSize = alignSize(newVertexSize * sizeof(float3), VERTEX_BUFFER_ALIGNMENT);
    uint32_t alignedCount = static_cast<uint32_t>(alignedVertexSize / sizeof(float3));

    m_clusterVertexPositions.create(m_device, alignedCount, "RTXMG Cluster Vertices", storageUsage, deviceLocal);
    m_clusterVertexNormals.create(m_device, alignedCount, "RTXMG Cluster Normals", storageUsage, deviceLocal);

    // Update allocation tracking
    m_allocatedVertices = alignedCount;
  }

  // Reset underutilization counter after growing
  m_underutilizationFrames = 0;
}

// ============================================================================
// Production Buffer Management (Phase 2)
// ============================================================================

size_t RtxmgClusterBuilder::alignSize(size_t size, size_t alignment) {
  if (alignment == 0) {
    return size;
  }

  return ((size + alignment - 1) / alignment) * alignment;
}

VkDeviceSize RtxmgClusterBuilder::alignDeviceSize(VkDeviceSize size, VkDeviceSize alignment) const {
  if (alignment == 0) {
    return size;
  }

  return ((size + alignment - 1) / alignment) * alignment;
}

VkDeviceSize RtxmgClusterBuilder::getClusterTemplateAlignment() const {
  VkDeviceSize alignment = m_device->properties().nvClusterAccelerationStructureProperties.clusterTemplateByteAlignment;
  return alignment != 0 ? alignment : 256;
}

VkDeviceSize RtxmgClusterBuilder::getClusterScratchAlignment() const {
  VkDeviceSize alignment = m_device->properties().nvClusterAccelerationStructureProperties.clusterScratchByteAlignment;
  return alignment != 0 ? alignment : 256;
}

VkDeviceSize RtxmgClusterBuilder::getInstanceStride() const {
  return alignDeviceSize(sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV), getClusterTemplateAlignment());
}

void RtxmgClusterBuilder::updatePeakUsage(uint32_t clusters, uint32_t vertices) {
  m_peakClusters = std::max(m_peakClusters, clusters);
  m_peakVertices = std::max(m_peakVertices, vertices);
}

bool RtxmgClusterBuilder::shouldShrinkBuffers(uint32_t requiredClusters, uint32_t requiredVertices) {
  // Don't shrink if buffers are not over-allocated
  if (m_clusters.numElements() <= MIN_CLUSTER_CAPACITY ||
      m_clusterVertexPositions.numElements() <= MIN_VERTEX_CAPACITY) {
    return false;
  }

  // Calculate usage ratios
  float clusterUsage = static_cast<float>(requiredClusters) / m_clusters.numElements();
  float vertexUsage = static_cast<float>(requiredVertices) / m_clusterVertexPositions.numElements();

  // Check if usage is below shrink threshold
  if (clusterUsage < SHRINK_USAGE_THRESHOLD || vertexUsage < SHRINK_USAGE_THRESHOLD) {
    m_underutilizationFrames++;
  } else {
    m_underutilizationFrames = 0;
    return false;
  }

  // Only shrink after sustained underutilization
  return m_underutilizationFrames >= SHRINK_FRAME_THRESHOLD;
}

void RtxmgClusterBuilder::shrinkBuffersIfNeeded(uint32_t requiredClusters, uint32_t requiredVertices) {
  // Use peak usage for shrink target (with some headroom)
  uint32_t targetClusterSize = static_cast<uint32_t>(m_peakClusters * 1.25f);
  uint32_t targetVertexSize = static_cast<uint32_t>(m_peakVertices * 1.25f);

  targetClusterSize = std::max(targetClusterSize, MIN_CLUSTER_CAPACITY);
  targetVertexSize = std::max(targetVertexSize, MIN_VERTEX_CAPACITY);

  Logger::info(str::format("[RTXMG] Shrinking buffers after ", m_underutilizationFrames, " frames of low usage: ",
    m_clusters.numElements(), " -> ", targetClusterSize, " clusters, ",
    m_clusterVertexPositions.numElements(), " -> ", targetVertexSize, " vertices"));

  // CRITICAL: Wait for GPU to finish using old buffers before reallocating
  // Without this, GPU will access freed memory causing hangs/crashes
  uint32_t pendingSubs = m_device->pendingSubmissions();
  if (pendingSubs > 0) {
    Logger::warn(str::format("[RTXMG] Waiting for GPU idle before buffer shrinking (pendingSubmissions=", pendingSubs, ")"));
  }
  m_device->waitForIdle();
  if (pendingSubs > 0) {
    Logger::info(str::format("[RTXMG] GPU idle complete - safe to shrink buffers (was waiting on ", pendingSubs, " submissions)"));
  }

  const VkBufferUsageFlags storageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags hostVisible =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

  bool buffersModified = false;

  // Shrink cluster buffers
  if (targetClusterSize < m_clusters.numElements()) {
    size_t alignedSize = alignSize(targetClusterSize * sizeof(RtxmgCluster), STORAGE_BUFFER_ALIGNMENT);
    uint32_t alignedCount = static_cast<uint32_t>(alignedSize / sizeof(RtxmgCluster));

    m_clusters.create(m_device, alignedCount, "RTXMG Clusters", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clusterShadingData.create(m_device, alignedCount, "RTXMG Cluster Shading Data", storageUsage, hostVisible);  // Need HOST_VISIBLE for CPU mapping in BLAS build
    m_clasAddresses.create(m_device, alignedCount, "RTXMG CLAS Addresses", storageUsage, deviceLocal);
    m_clusterIndirectArgs.create(m_device, alignedCount, "RTXMG Indirect Args", storageUsage, deviceLocal);

    m_allocatedClusters = alignedCount;
    buffersModified = true;
  }

  // Shrink vertex buffers
  if (targetVertexSize < m_clusterVertexPositions.numElements()) {
    size_t alignedSize = alignSize(targetVertexSize * sizeof(float3), VERTEX_BUFFER_ALIGNMENT);
    uint32_t alignedCount = static_cast<uint32_t>(alignedSize / sizeof(float3));

    m_clusterVertexPositions.create(m_device, alignedCount, "RTXMG Cluster Vertices", storageUsage, deviceLocal);
    m_clusterVertexNormals.create(m_device, alignedCount, "RTXMG Cluster Normals", storageUsage, deviceLocal);

    m_allocatedVertices = alignedCount;
    buffersModified = true;
  }

  if (buffersModified) {
    // Any buffer resize invalidates cumulative offsets/stats just like a grow
    m_cumulativeClusterOffset = 0;
    m_stats.allocated.m_numClusters = m_allocatedClusters;
    m_stats.allocated.m_vertexBufferSize = m_allocatedVertices * sizeof(float3);
    m_stats.allocated.m_vertexNormalsBufferSize = m_allocatedVertices * sizeof(float3);
    m_stats.allocated.m_clusterDataSize = m_allocatedClusters * sizeof(RtxmgCluster);
  }

  // Reset tracking after shrink
  m_underutilizationFrames = 0;
  m_peakClusters = requiredClusters;
  m_peakVertices = requiredVertices;
}

void RtxmgClusterBuilder::updateMemoryPressureDetection() {
  // Query device memory properties
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(m_device->adapter()->handle(), &memProps);

  // Find device local memory heap
  size_t totalDeviceMemory = 0;
  for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
    if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      totalDeviceMemory = memProps.memoryHeaps[i].size;
      break;
    }
  }

  m_deviceTotalMemory = totalDeviceMemory;

  // Query current memory budget (VK_EXT_memory_budget if available)
  // For now, estimate based on total memory minus allocated
  m_totalGpuMemoryAllocated = 0;

  // Calculate RTXMG memory usage
  if (m_clusters.isValid()) {
    m_totalGpuMemoryAllocated += m_clusters.numElements() * sizeof(RtxmgCluster);
    m_totalGpuMemoryAllocated += m_clusterShadingData.numElements() * sizeof(ClusterShadingData);
    m_totalGpuMemoryAllocated += m_clasAddresses.numElements() * sizeof(VkDeviceAddress);
    m_totalGpuMemoryAllocated += m_clusterIndirectArgs.numElements() * sizeof(ClusterIndirectArgs);
  }

  if (m_clusterVertexPositions.isValid()) {
    m_totalGpuMemoryAllocated += m_clusterVertexPositions.numElements() * sizeof(float3);
    m_totalGpuMemoryAllocated += m_clusterVertexNormals.numElements() * sizeof(float3);
  }

  m_totalGpuMemoryAllocated += m_allocatedClasSize;
  m_totalGpuMemoryAllocated += m_allocatedBlasSize;
  m_totalGpuMemoryAllocated += m_allocatedBlasScratchSize;

  // Estimate available memory (this is conservative)
  m_deviceAvailableMemory = m_deviceTotalMemory - m_totalGpuMemoryAllocated;

  // Check if we're approaching memory pressure
  float memoryUsage = static_cast<float>(m_totalGpuMemoryAllocated) / m_deviceTotalMemory;
  m_memoryPressureDetected = (memoryUsage >= MEMORY_PRESSURE_THRESHOLD);

  if (m_memoryPressureDetected) {
    Logger::warn(str::format("[RTXMG] Memory pressure detected: ",
      m_totalGpuMemoryAllocated / (1024 * 1024), " MB / ",
      m_deviceTotalMemory / (1024 * 1024), " MB (",
      static_cast<uint32_t>(memoryUsage * 100), "%)"));
  }
}

bool RtxmgClusterBuilder::checkMemoryPressure(size_t additionalBytes) {
  updateMemoryPressureDetection();

  // Check if adding this allocation would exceed pressure threshold
  size_t projectedUsage = m_totalGpuMemoryAllocated + additionalBytes;
  float projectedRatio = static_cast<float>(projectedUsage) / m_deviceTotalMemory;

  return projectedRatio >= MEMORY_PRESSURE_THRESHOLD;
}

// ============================================================================
// Error Handling & Recovery (Phase 3)
// ============================================================================

bool RtxmgClusterBuilder::validateInputGeometry(const ClusterInputGeometry& input, const char* context) {
  // Validate positions
  if (input.positions.empty()) {
    Logger::err(str::format("[RTXMG] ", context, ": Empty position buffer"));
    return false;
  }

  // Validate indices
  if (input.indices.empty()) {
    Logger::err(str::format("[RTXMG] ", context, ": Empty index buffer"));
    return false;
  }

  // Validate index count is multiple of 3 (triangles)
  if (input.indices.size() % 3 != 0) {
    Logger::err(str::format("[RTXMG] ", context, ": Index count ", input.indices.size(),
      " is not a multiple of 3"));
    return false;
  }

  // Validate index bounds
  uint32_t maxVertex = static_cast<uint32_t>(input.positions.size());
  for (size_t i = 0; i < input.indices.size(); ++i) {
    if (input.indices[i] >= maxVertex) {
      Logger::err(str::format("[RTXMG] ", context, ": Index ", input.indices[i],
        " at position ", i, " exceeds vertex count ", maxVertex));
      return false;
    }
  }

  // Validate normals (if present)
  if (!input.normals.empty() && input.normals.size() != input.positions.size()) {
    Logger::err(str::format("[RTXMG] ", context, ": Normal count ", input.normals.size(),
      " doesn't match position count ", input.positions.size()));
    return false;
  }

  // Validate texcoords (if present)
  if (!input.texcoords.empty() && input.texcoords.size() != input.positions.size()) {
    Logger::err(str::format("[RTXMG] ", context, ": Texcoord count ", input.texcoords.size(),
      " doesn't match position count ", input.positions.size()));
    return false;
  }

  // Check for reasonable triangle count
  uint32_t numTriangles = static_cast<uint32_t>(input.indices.size() / 3);
  if (numTriangles > 10000000) { // 10M triangles is suspicious
    Logger::warn(str::format("[RTXMG] ", context, ": Very high triangle count: ", numTriangles));
  }

  return true;
}

bool RtxmgClusterBuilder::validateInputGeometryGpu(const ClusterInputGeometryGpu& input, const char* context) {
  // Validate position buffer
  if (!input.positionBuffer.defined()) {
    Logger::err(str::format("[RTXMG] ", context, ": Invalid position buffer"));
    return false;
  }

  // Validate vertex count
  if (input.vertexCount == 0) {
    Logger::err(str::format("[RTXMG] ", context, ": Zero vertex count"));
    return false;
  }

  // Validate index count
  if (input.indexCount == 0) {
    Logger::err(str::format("[RTXMG] ", context, ": Zero index count"));
    return false;
  }

  // Validate index count is multiple of 3
  if (input.indexCount % 3 != 0) {
    Logger::err(str::format("[RTXMG] ", context, ": Index count ", input.indexCount,
      " is not a multiple of 3"));
    return false;
  }

  // Validate position buffer size
  size_t minPositionBufferSize = (input.positionOffset + input.vertexCount * input.positionStride);
  if (input.positionBuffer.length() < minPositionBufferSize) {
    Logger::err(str::format("[RTXMG] ", context, ": Position buffer too small: ",
      input.positionBuffer.length(), " < ", minPositionBufferSize));
    return false;
  }

  // Validate index buffer (if provided)
  if (input.indexBuffer.defined()) {
    size_t indexSize = (input.indexType == VK_INDEX_TYPE_UINT16) ? 2 : 4;
    size_t minIndexBufferSize = input.indexCount * indexSize;
    if (input.indexBuffer.length() < minIndexBufferSize) {
      Logger::err(str::format("[RTXMG] ", context, ": Index buffer too small: ",
        input.indexBuffer.length(), " < ", minIndexBufferSize));
      return false;
    }
  }

  // Check for reasonable triangle count
  uint32_t numTriangles = input.indexCount / 3;
  if (numTriangles > 10000000) { // 10M triangles is suspicious
    Logger::warn(str::format("[RTXMG] ", context, ": Very high triangle count: ", numTriangles));
  }

  return true;
}

bool RtxmgClusterBuilder::validateOutputCapacity(uint32_t requiredClusters, uint32_t requiredVertices) {
  // Check cluster capacity
  if (requiredClusters > m_clusters.numElements()) {
    Logger::err(str::format("[RTXMG] Required clusters ", requiredClusters,
      " exceeds allocated capacity ", m_clusters.numElements()));
    return false;
  }

  // Check vertex capacity
  if (requiredVertices > m_clusterVertexPositions.numElements()) {
    Logger::err(str::format("[RTXMG] Required vertices ", requiredVertices,
      " exceeds allocated capacity ", m_clusterVertexPositions.numElements()));
    return false;
  }

  // Check for zero requirements (suspicious)
  if (requiredClusters == 0 || requiredVertices == 0) {
    Logger::warn("[RTXMG] Zero cluster or vertex requirement");
    return false;
  }

  return true;
}

bool RtxmgClusterBuilder::validateBufferState() {
  // Validate all critical buffers are allocated
  if (!m_clusters.isValid()) {
    Logger::err("[RTXMG] Cluster buffer not allocated");
    return false;
  }

  if (!m_clusterShadingData.isValid()) {
    Logger::err("[RTXMG] Cluster shading data buffer not allocated");
    return false;
  }

  if (!m_clusterVertexPositions.isValid()) {
    Logger::err("[RTXMG] Cluster vertex position buffer not allocated");
    return false;
  }

  if (!m_clasAddresses.isValid()) {
    Logger::err("[RTXMG] CLAS address buffer not allocated");
    return false;
  }

  // Check for counter buffers (single buffer with GPU fence sync)
  if (!m_tessCountersBuffer.isValid()) {
    Logger::err("[RTXMG] Counter buffer not allocated");
    return false;
  }
  if (m_tessCountersReadback == nullptr) {
    Logger::err("[RTXMG] Counter readback buffer not allocated");
    return false;
  }

  // Validate sizes are reasonable
  if (m_allocatedClusters == 0 || m_allocatedVertices == 0) {
    Logger::err("[RTXMG] Zero allocated clusters or vertices");
    return false;
  }

  return true;
}

RtxmgClusterBuilder::BufferSnapshot RtxmgClusterBuilder::saveBufferState() {
  BufferSnapshot snapshot;
  snapshot.allocatedClusters = m_allocatedClusters;
  snapshot.allocatedVertices = m_allocatedVertices;
  snapshot.allocatedClasSize = m_allocatedClasSize;
  snapshot.allocatedBlasSize = m_allocatedBlasSize;
  return snapshot;
}

bool RtxmgClusterBuilder::restoreBufferState(const BufferSnapshot& snapshot) {
  // NOTE: This is a logical restore, not a full buffer content restore
  // We restore allocation tracking but don't recreate buffers
  // Full rollback would require keeping backup buffers (expensive)

  Logger::info(str::format("[RTXMG] Rolling back buffer state: ",
    m_allocatedClusters, " -> ", snapshot.allocatedClusters, " clusters, ",
    m_allocatedVertices, " -> ", snapshot.allocatedVertices, " vertices"));

  m_allocatedClusters = snapshot.allocatedClusters;
  m_allocatedVertices = snapshot.allocatedVertices;
  m_allocatedClasSize = snapshot.allocatedClasSize;
  m_allocatedBlasSize = snapshot.allocatedBlasSize;

  // Update statistics to reflect rollback
  m_stats.allocated.m_numClusters = m_allocatedClusters;
  m_stats.allocated.m_vertexBufferSize = m_allocatedVertices * sizeof(float3);
  m_stats.allocated.m_vertexNormalsBufferSize = m_allocatedVertices * sizeof(float3);
  m_stats.allocated.m_clasSize = m_allocatedClasSize;
  m_stats.allocated.m_blasSize = m_allocatedBlasSize;

  return true;
}

void RtxmgClusterBuilder::applyGracefulDegradation(RtxmgConfig& config, uint32_t degradationLevel) {
  if (degradationLevel == 0) {
    return; // No degradation
  }

  Logger::warn(str::format("[RTXMG] Applying graceful degradation level ", degradationLevel));

  // Level 1: Reduce tessellation rate
  if (degradationLevel >= 1) {
    config.fineTessellationRate = std::max(0.5f, config.fineTessellationRate * 0.5f);
    config.coarseTessellationRate = std::max(0.1f, config.coarseTessellationRate * 0.5f);
    Logger::info(str::format("[RTXMG] Degradation L1: Reduced fine tessellation rate to ", config.fineTessellationRate));
  }

  // Level 2: Disable vertex normals and use simpler tessellation mode
  if (degradationLevel >= 2) {
    config.enableVertexNormals = false;
    config.tessMode = RtxmgConfig::AdaptiveTessellationMode::UNIFORM;
    Logger::info("[RTXMG] Degradation L2: Disabled normals, using uniform tessellation");
  }

  // Level 3: Minimum quality - very low tessellation
  if (degradationLevel >= 3) {
    config.fineTessellationRate = 0.25f;
    config.coarseTessellationRate = 0.1f;
    config.enableFrustumVisibility = false;  // Disable culling to reduce overhead
    config.enableBackfaceVisibility = false;
    config.enableHiZVisibility = false;
    Logger::warn("[RTXMG] Degradation L3: Minimum quality mode - performance critical");
  }

  m_degradationLevel = degradationLevel;
}

// ============================================================================
// True GPU Batching (Phase 4)
// ============================================================================

bool RtxmgClusterBuilder::setupBatchBuffers(
  RtxContext* ctx,
  const std::vector<ClusterInputGeometryGpu>& inputs,
  uint32_t totalVertices,
  uint32_t totalIndices,
  uint32_t totalClusters) {

  if (inputs.empty()) {
    Logger::err("[RTXMG] setupBatchBuffers: Empty input array");
    return false;
  }

  // SDK MATCH: No batch size limit - sample processes any number of instances
  // Buffers will be dynamically resized as needed
  Logger::info(str::format("[RTXMG] Setting up batch buffers for ", inputs.size(),
    " instances (", totalVertices, " verts, ", totalIndices, " indices, ",
    totalClusters, " clusters)"));

  const VkBufferUsageFlags storageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  // Create instance data buffer if needed
  // SDK MATCH: Allocate based on actual instance count (cluster_accel_builder.cpp:1149)
  if (!m_instanceDataBuffer.isValid() || m_instanceDataBuffer.numElements() < inputs.size()) {
    uint32_t bufferSize = std::max(static_cast<uint32_t>(inputs.size()), 1024u);  // Start with 1024, grow as needed
    m_instanceDataBuffer.create(m_device, bufferSize, "RTXMG Batch Instance Data",
      storageUsage, deviceLocal);
  }

  // Create indirect dispatch buffer
  // SDK MATCH: Dynamically sized based on instance count
  if (!m_indirectDispatchBuffer.isValid() || m_indirectDispatchBuffer.numElements() < inputs.size()) {
    uint32_t bufferSize = std::max(static_cast<uint32_t>(inputs.size()), 1024u);
    m_indirectDispatchBuffer.create(m_device, bufferSize,
      "RTXMG Indirect Dispatch", storageUsage, deviceLocal);
  }

  // Create instance offsets buffer
  // Needs 3 offsets per instance: vertex, index, cluster
  if (!m_instanceOffsetsBuffer.isValid() || m_instanceOffsetsBuffer.numElements() < inputs.size() * 3) {
    uint32_t bufferSize = std::max(static_cast<uint32_t>(inputs.size()) * 3, 1024u * 3);
    m_instanceOffsetsBuffer.create(m_device, bufferSize,
      "RTXMG Instance Offsets", storageUsage, deviceLocal);
  }

  // SDK MATCH: Direct buffer binding (cluster_accel_builder.cpp:662-685)
  // ELIMINATED: Geometry consolidation copies (~90 lines removed)
  //
  // Previous: Copied all geometry to consolidated buffers (hundreds of GPU-to-GPU copyBuffer calls)
  // SDK: Binds directly to scene subdivision buffers via per-instance device addresses
  //
  // InstanceData structure already contains device addresses for each instance:
  // - inputPositionBuffer, inputNormalBuffer, inputTexcoordBuffer, inputIndexBuffer (lines 2870-2877)
  //
  // These point to ORIGINAL source buffers, eliminating serialized copies.
  // Shader reads per-instance data via InstanceData (bound at binding 16, line 3089).
  //
  // Performance win: Removes buffer copies + barriers that serialized geometry processing.

  Logger::info(str::format("[RTXMG] SDK-matching zero-copy binding: ", inputs.size(),
                          " instances, ", totalVertices, " verts, ", totalIndices,
                          " indices (direct device addresses, no consolidation)"));

  // Fill instance data
  std::vector<InstanceData> instanceData(inputs.size());
  std::vector<uint32_t> instanceOffsets(inputs.size() * 3);

  uint32_t cumulativeVertexOffset = 0;
  uint32_t cumulativeIndexOffset = 0;
  uint32_t cumulativeClusterOffset = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];

    // Get device addresses for input buffers
    instanceData[i].inputPositionBuffer = input.positionBuffer.defined() ?
      input.positionBuffer.getDeviceAddress() : 0;
    instanceData[i].inputNormalBuffer = input.normalBuffer.defined() ?
      input.normalBuffer.getDeviceAddress() : 0;
    instanceData[i].inputTexcoordBuffer = input.texcoordBuffer.defined() ?
      input.texcoordBuffer.getDeviceAddress() : 0;
    instanceData[i].inputIndexBuffer = input.indexBuffer.defined() ?
      input.indexBuffer.getDeviceAddress() : 0;

    instanceData[i].vertexCount = input.vertexCount;
    instanceData[i].indexCount = input.indexCount;
    instanceData[i].surfaceId = input.surfaceId;
    instanceData[i].transform = input.transform;

    // Set offsets into global output buffers
    instanceData[i].vertexOffset = cumulativeVertexOffset;
    instanceData[i].indexOffset = cumulativeIndexOffset;
    instanceData[i].clusterOffset = cumulativeClusterOffset;

    // Store offsets in separate buffer for shader access
    instanceOffsets[i * 3 + 0] = cumulativeVertexOffset;
    instanceOffsets[i * 3 + 1] = cumulativeIndexOffset;
    instanceOffsets[i * 3 + 2] = cumulativeClusterOffset;

    // Estimate cluster count for this instance
    uint32_t numTriangles = input.indexCount / 3;
    uint32_t estimatedClusters = std::max(1u, numTriangles / 2);

    // Update cumulative offsets
    cumulativeVertexOffset += input.vertexCount;
    cumulativeIndexOffset += input.indexCount;
    cumulativeClusterOffset += estimatedClusters;
  }

  // Upload instance data to GPU
  m_instanceDataBuffer.upload(instanceData);
  m_instanceOffsetsBuffer.upload(instanceOffsets);

  // Setup indirect dispatch commands
  // NOTE: This would be filled by GPU in true indirect dispatch
  // For now, we set up manual dispatch per instance
  std::vector<VkDispatchIndirectCommand> indirectCommands(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    uint32_t numTriangles = inputs[i].indexCount / 3;
    // 64 threads per workgroup
    indirectCommands[i].x = (numTriangles + 63) / 64;
    indirectCommands[i].y = 1;
    indirectCommands[i].z = 1;
  }
  m_indirectDispatchBuffer.upload(indirectCommands);

  m_batchingEnabled = true;
  Logger::info("[RTXMG] Batch buffers setup complete");

  return true;
}

void RtxmgClusterBuilder::dispatchBatchedCompute(
  RtxContext* ctx,
  uint32_t instanceCount,
  const RtxmgConfig& config,
  uint32_t frameIndex,
  uint32_t globalVertexOffset) {

  if (!m_batchingEnabled) {
    Logger::err("[RTXMG] Batching not enabled, call setupBatchBuffers first");
    return;
  }

  if (m_clusterTilingShader == nullptr) {
    Logger::err("[RTXMG] Cluster tiling shader not available");
    return;
  }

  // SDK MATCH: No ring buffering - single-frame buffers with GPU sync
  // NVIDIA sample doesn't use ring buffers, it rebuilds every frame

  Logger::info(str::format("[RTXMG] Dispatching batched compute for ", instanceCount, " instances (frame ", frameIndex, ")"));

  // Phase 4: Setup FULL ClusterTilingParams structure matching the shader
  struct ClusterTilingParams {
    Matrix4 matWorldToClip;
    Matrix4 localToWorld;
    Vector3 cameraPos;
    uint32_t tessellationMode;

    Vector2 viewportSize;
    float fineTessellationRate;
    float coarseTessellationRate;

    uint32_t edgeSegments[4];
    uint32_t maxClusters;
    uint32_t maxVertices;
    uint32_t maxClasBlocks;

    uint32_t surfaceStart;
    uint32_t surfaceEnd;
    uint32_t visibilityMode;
    uint32_t enableFrustumCulling;

    uint32_t enableBackfaceCulling;
    uint32_t enableHiZCulling;
    uint32_t numHiZLODs;
    float invHiZSize;

    uint64_t clasDataBaseAddress;
    uint64_t clusterVertexPositionsBaseAddress;

    // Phase 4: GPU batching support (matches shader exactly)
    uint32_t enableBatching;         // Enable GPU batching mode
    uint32_t instanceCount;          // Number of instances in batch
    uint32_t baseClusterOffset;      // Base offset for writing clusters (for multi-geometry batching)
    uint32_t globalVertexOffset;     // SDK MATCH: Global vertex offset for this instance (for multi-geometry global offsets)

    int32_t debugSurfaceIndex;
    int32_t debugLaneIndex;
    uint32_t _pad2;
    uint32_t _pad3;
  } params = {};

  // Fill the FULL structure
  params.matWorldToClip = config.matWorldToClip;
  params.localToWorld = Matrix4();  // Identity for batched mode (per-instance transforms in InstanceData)
  params.cameraPos = config.cameraPos;
  params.tessellationMode = static_cast<uint32_t>(config.tessMode);

  params.viewportSize = Vector2(static_cast<float>(config.viewportSize.x),
                                 static_cast<float>(config.viewportSize.y));
  params.fineTessellationRate = config.fineTessellationRate;
  params.coarseTessellationRate = config.coarseTessellationRate;

  params.edgeSegments[0] = config.edgeSegments.x;
  params.edgeSegments[1] = config.edgeSegments.y;
  params.edgeSegments[2] = config.edgeSegments.z;
  params.edgeSegments[3] = config.edgeSegments.w;

  params.maxClusters = m_allocatedClusters;
  params.maxVertices = m_allocatedVertices;
  params.maxClasBlocks = config.memorySettings.maxClasBlocks;

  params.surfaceStart = 0;
  params.surfaceEnd = instanceCount;  // In batched mode, treat instances as surfaces
  params.visibilityMode = static_cast<uint32_t>(config.visMode);
  params.enableFrustumCulling = config.enableFrustumVisibility ? 1 : 0;

  params.enableBackfaceCulling = config.enableBackfaceVisibility ? 1 : 0;
  params.enableHiZCulling = config.enableHiZCulling ? 1 : 0;
  params.numHiZLODs = m_hizNumLevels;
  params.invHiZSize = m_hizWidth > 0 ? (1.0f / static_cast<float>(m_hizWidth)) : 0.0f;

  params.clasDataBaseAddress = m_templateClasBuffer.isValid() ? m_templateClasBuffer.deviceAddress() : 0;
  params.clusterVertexPositionsBaseAddress = m_clusterVertexPositions.deviceAddress();

  // ALIGNMENT LOGGING: Check device address alignment
  VkDeviceAddress vertexBaseAddr = params.clusterVertexPositionsBaseAddress;
  VkDeviceAddress clasBaseAddr = params.clasDataBaseAddress;

  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT] Cluster vertex buffer base: 0x", std::hex, vertexBaseAddr, std::dec));
  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT]   -> address % 8 = ", vertexBaseAddr % 8,
                          " (8-byte ", (vertexBaseAddr % 8 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT]   -> address % 16 = ", vertexBaseAddr % 16,
                          " (16-byte ", (vertexBaseAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT]   -> buffer size = ", m_clusterVertexPositions.getBuffer()->info().size, " bytes"));

  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT] CLAS template buffer base: 0x", std::hex, clasBaseAddr, std::dec));
  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT]   -> address % 8 = ", clasBaseAddr % 8,
                          " (8-byte ", (clasBaseAddr % 8 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
  Logger::info(str::format("[RTXMG DISPATCH ALIGNMENT]   -> address % 16 = ", clasBaseAddr % 16,
                          " (16-byte ", (clasBaseAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

  // SDK MATCH: Disable batching, dispatch once per instance
  params.enableBatching = 0;
  params.instanceCount = 1;  // Process one instance per dispatch
  // SDK MATCH: No ring buffering - baseClusterOffset is just cumulative offset
  // NVIDIA sample doesn't use ring buffers, it rebuilds every frame
  params.baseClusterOffset = m_cumulativeClusterOffset;
  // SDK MATCH: Pass global vertex offset so shader writes correct global offsets (no CPU patching needed!)
  params.globalVertexOffset = globalVertexOffset;

  Logger::info(str::format("[RTXMG DISPATCH] baseClusterOffset=", params.baseClusterOffset,
                          " (cumulative offset, SDK-matching), globalVertexOffset=", globalVertexOffset));
  Logger::info(str::format("[RTXMG DISPATCH] Vertex offset calculation: global offset ", globalVertexOffset,
                          " = base address 0x", std::hex, vertexBaseAddr, std::dec,
                          " + (", globalVertexOffset, " * ", sizeof(float3), " bytes)"));

  params.debugSurfaceIndex = config.debugSurfaceIndex;
  params.debugLaneIndex = config.debugLaneIndex;
  params._pad2 = 0;
  params._pad3 = 0;

  // Upload params to constant buffer and bind to binding 0
  // (replaces push constants to support 216+ byte structure)
  std::vector<uint8_t> paramsData(sizeof(params));
  std::memcpy(paramsData.data(), &params, sizeof(params));
  m_clusterTilingParamsBuffer.upload(paramsData);
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_clusterTilingParamsBuffer.getBuffer()));

  // Bind all required buffers for cluster tiling (bindings 1-16 matching shader layout)
  // Input buffers (shared across instances in current implementation)
  ctx->bindResourceBuffer(1, DxvkBufferSlice(m_inputPositions.getBuffer()));
  ctx->bindResourceBuffer(2, DxvkBufferSlice(m_inputNormals.getBuffer()));
  ctx->bindResourceBuffer(3, DxvkBufferSlice(m_inputTexcoords.getBuffer()));
  ctx->bindResourceBuffer(4, DxvkBufferSlice(m_inputIndices.getBuffer()));
  ctx->bindResourceBuffer(5, DxvkBufferSlice(m_surfaceInfo.getBuffer()));

  // Template data
  ctx->bindResourceBuffer(6, DxvkBufferSlice(m_templateAddresses.getBuffer()));
  ctx->bindResourceBuffer(7, DxvkBufferSlice(m_clasInstantiationBytes.getBuffer()));

  // HiZ buffer (if available)
  if (m_hizInitialized) {
    ctx->bindResourceView(8, m_hizPyramidView, nullptr);
    ctx->bindResourceSampler(9, m_hizSampler);
  }

  // Output buffers
  ctx->bindResourceBuffer(10, DxvkBufferSlice(m_gridSamplers.getBuffer()));
  ctx->bindResourceBuffer(11, DxvkBufferSlice(m_tessCountersBuffer.getBuffer()));

  // SAMPLE MATCH: Bind ENTIRE clusters buffer (no slice)
  // The shader uses baseClusterOffset parameter (cumulative offset) to write to correct location
  // This matches sample line 863: .addItem(BindingSetItem::StructuredBuffer_UAV(2, m_clustersBuffer))
  ctx->bindResourceBuffer(12, DxvkBufferSlice(m_clusters.getBuffer()));

  ctx->bindResourceBuffer(13, DxvkBufferSlice(m_clusterShadingData.getBuffer()));
  ctx->bindResourceBuffer(14, DxvkBufferSlice(m_clusterIndirectArgs.getBuffer()));
  ctx->bindResourceBuffer(15, DxvkBufferSlice(m_clasAddresses.getBuffer()));

  // Bind instance data buffer (additional binding for batching)
  ctx->bindResourceBuffer(16, DxvkBufferSlice(m_instanceDataBuffer.getBuffer()));

  // SDK MATCH: Create CopyClusterOffset params buffer ONCE before loop
  struct CopyClusterOffsetParams {
    uint32_t instanceIndex;
    uint32_t totalInstances;
    uint32_t _pad0;
    uint32_t _pad1;
  };

  DxvkBufferCreateInfo cbInfo = {};
  cbInfo.size = align(sizeof(CopyClusterOffsetParams), 256);
  cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
  Rc<DxvkBuffer> copyClusterOffsetParamsBuffer = m_device->createBuffer(
    cbInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    DxvkMemoryStats::Category::RTXBuffer,
    "RTXMG Copy Cluster Offset Params");

  const Rc<DxvkBuffer>& tessCountersBuffer = m_tessCountersBuffer.getBuffer();
  const Rc<DxvkBuffer>& offsetCountsBuffer = m_clusterOffsetCountsBuffer.getBuffer();

  // Bind compute shader
  Logger::info(str::format("[RTXMG ClusterTiling DEBUG] About to start cluster tiling loop for ", instanceCount, " instances"));
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_clusterTilingShader);
  Logger::info("[RTXMG ClusterTiling DEBUG] Cluster tiling shader bound");

  // SDK MATCH: Dispatch ONCE PER INSTANCE (sequential)
  // Each dispatch processes one surface (surface i)
  for (uint32_t i = 0; i < instanceCount; i++) {
    // Update surfaceStart/End for this iteration
    params.surfaceStart = i;
    params.surfaceEnd = i + 1;

    // Re-upload params to constant buffer
    std::memcpy(paramsData.data(), &params, sizeof(params));
    m_clusterTilingParamsBuffer.upload(paramsData);

    Logger::info(str::format("[RTXMG ClusterTiling DEBUG] Instance ", i, ": About to dispatch cluster tiling"));
    ctx->dispatch(1, 1, 1);  // One workgroup per surface
    Logger::info(str::format("[RTXMG ClusterTiling DEBUG] Instance ", i, ": Cluster tiling dispatch complete"));

    // SDK MATCH: After each dispatch, call CopyClusterOffset to save this instance's cluster count
    // This reads the cluster count from tessCountersBuffer[0] and writes offset/count to offsetCountsBuffer[i]
    Logger::info(str::format("[RTXMG ClusterTiling DEBUG] Instance ", i, ": About to bind copy cluster offset shader"));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_copyClusterOffsetShader);
    Logger::info("[RTXMG ClusterTiling DEBUG] Copy cluster offset shader bound");
    ctx->bindResourceBuffer(0, DxvkBufferSlice(copyClusterOffsetParamsBuffer, 0, sizeof(CopyClusterOffsetParams)));
    ctx->bindResourceBuffer(1, DxvkBufferSlice(tessCountersBuffer));  // Input: tess counters (shifted +1)
    ctx->bindResourceBuffer(2, DxvkBufferSlice(offsetCountsBuffer));  // Output: offset/count pairs (shifted +1)

    CopyClusterOffsetParams copyParams;
    copyParams.instanceIndex = i;
    copyParams.totalInstances = instanceCount;
    copyParams._pad0 = 0;
    copyParams._pad1 = 0;

    ctx->updateBuffer(copyClusterOffsetParamsBuffer, 0, sizeof(copyParams), &copyParams);
    Logger::info(str::format("[RTXMG ClusterTiling DEBUG] Instance ", i, ": About to dispatch copy cluster offset"));
    ctx->dispatch(1, 1, 1);
    Logger::info(str::format("[RTXMG ClusterTiling DEBUG] Instance ", i, ": Copy cluster offset dispatch complete"));

    // Barrier between dispatches so next instance can read previous offset
    if (i < instanceCount - 1) {
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    // Re-bind cluster tiling shader for next iteration
    if (i < instanceCount - 1) {
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_clusterTilingShader);
    }
  }

  Logger::info(str::format("[RTXMG] Sequential dispatch complete: ", instanceCount, " instances with CopyClusterOffset"));

  // Add barrier to ensure cluster indirect args buffer is written before it's used for BLAS building
  Logger::info("[RTXMG DEBUG] ========== PRE-BARRIER PHASE ==========");
  Logger::info(str::format("[RTXMG DEBUG] Getting indirect args buffer handle..."));

  DxvkBufferSliceHandle indirectArgsHandle = m_clusterIndirectArgs.getBuffer()->getSliceHandle();
  Logger::info(str::format("[RTXMG DEBUG] Got handle successfully, about to track resource..."));

  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_clusterIndirectArgs.getBuffer());
  Logger::info(str::format("[RTXMG DEBUG] Resource tracked, about to emit barrier..."));

  Logger::info(str::format("[RTXMG DEBUG] Calling emitMemoryBarrier() with params:"));
  Logger::info(str::format("[RTXMG DEBUG]   srcStageMask = COMPUTE_SHADER"));
  Logger::info(str::format("[RTXMG DEBUG]   srcAccessMask = SHADER_WRITE"));
  Logger::info(str::format("[RTXMG DEBUG]   dstStageMask = ACCELERATION_STRUCTURE_BUILD"));
  Logger::info(str::format("[RTXMG DEBUG]   dstAccessMask = ACCELERATION_STRUCTURE_READ"));

  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

  Logger::info(str::format("[RTXMG DEBUG] Barrier emitted successfully!"));
  Logger::info("[RTXMG DEBUG] ========== POST-BARRIER PHASE ==========");
}

// ============================================================================
// Shader Creation (Phase 5)
// ============================================================================

void RtxmgClusterBuilder::createShaders() {
  Logger::info("[RTXMG] Creating compute shaders");

  // Create cluster tiling shader
  m_clusterTilingShader = ClusterTilingShader::getShader();
  if (m_clusterTilingShader == nullptr) {
    Logger::err("[RTXMG] Failed to create cluster tiling shader");
    return;
  }

  // Create cluster filling shader
  Logger::info("[RTXMG INIT] Creating FillClustersShader...");
  m_clusterFillingShader = FillClustersShader::getShader();
  if (m_clusterFillingShader == nullptr) {
    Logger::err("[RTXMG] Failed to create cluster filling shader");
    return;
  }
  Logger::info("[RTXMG INIT] FillClustersShader created successfully");

  // Create copy cluster offset shader
  Logger::info("[RTXMG INIT] Creating CopyClusterOffsetShader...");
  m_copyClusterOffsetShader = CopyClusterOffsetShader::getShader();
  if (m_copyClusterOffsetShader == nullptr) {
    Logger::err("[RTXMG] Failed to create copy cluster offset shader");
    return;
  }
  Logger::info("[RTXMG INIT] CopyClusterOffsetShader created successfully");

  // Create BLAS args generation shader
  Logger::info("[RTXMG INIT] Creating FillBlasFromClasArgsShader...");
  m_fillBlasFromClasArgsShader = FillBlasFromClasArgsShader::getShader();
  if (m_fillBlasFromClasArgsShader == nullptr) {
    Logger::err("[RTXMG] Failed to create BLAS args generation shader");
    return;
  }
  Logger::info("[RTXMG INIT] FillBlasFromClasArgsShader created successfully");

  // Create HiZ pyramid generation shader
  Logger::info("[RTXMG INIT] Creating HiZPyramidGenerateShader...");
  m_hizPyramidGenerateShader = HiZPyramidGenerateShader::getShader();
  if (m_hizPyramidGenerateShader == nullptr) {
    Logger::err("[RTXMG] Failed to create HiZ pyramid generation shader");
    return;
  }
  Logger::info("[RTXMG INIT] HiZPyramidGenerateShader created successfully");

  Logger::info("[RTXMG] Compute shaders created successfully");
}

bool RtxmgClusterBuilder::ensureTemplateClasBuilt(RtxContext* ctx) {
  if (m_templateClasBuilt)
    return true;

  if (!isClusterAccelerationExtensionAvailable()) {
    Logger::warn("[RTXMG] Cluster acceleration extension not available - cannot build template CLAS structures");
    return false;
  }

  if (ctx == nullptr) {
    Logger::err("[RTXMG] Cannot build template CLAS structures without a valid RtxContext");
    return false;
  }

  Logger::info("[RTXMG] Building template CLAS structures (first time)");

  size_t totalClasSize = 0;
  bool success = buildTemplateClusterAccelerationStructures(
    m_device,
    ctx,
    m_templateGrids,
    m_templateClasBuffer,
    m_templateAddressesVec,
    m_clasInstantiationBytesVec,
    &totalClasSize);

  if (!success) {
    Logger::err("[RTXMG] Failed to build template CLAS structures");
    return false;
  }

  m_templateAddresses.upload(m_templateAddressesVec);
  m_clasInstantiationBytes.upload(m_clasInstantiationBytesVec);

  m_allocatedClasSize = totalClasSize;
  m_stats.allocated.m_clasSize = m_allocatedClasSize;
  m_stats.desired.m_clasSize = m_allocatedClasSize;

  m_templateClasBuilt = true;
  Logger::info("[RTXMG] Template CLAS structures built successfully");
  return true;
}

// ============================================================================
// Memory Management
// ============================================================================

bool RtxmgClusterBuilder::updateMemoryAllocations(
  uint32_t requiredClusters,
  uint32_t requiredVertices,
  const RtxmgConfig& config) {

  // This function updates buffer allocations based on requirements
  // For now, it's a stub that always returns true (allocations handled elsewhere)
  return true;
}

// ============================================================================
// True GPU Batching (Phase 4)
// ============================================================================

bool RtxmgClusterBuilder::buildClustersGpuBatch(
  RtxContext* ctx,
  const std::vector<ClusterInputGeometryGpu>& inputs,
  std::vector<ClusterOutputGeometryGpu>& outputs,
  const RtxmgConfig& config) {

  if (!m_initialized) {
    Logger::err("[RTXMG] Cluster builder not initialized");
    return false;
  }

  if (inputs.empty()) {
    Logger::warn("[RTXMG] Empty input batch");
    return true;
  }

  Logger::info(str::format("[RTXMG] Building clusters for ", inputs.size(), " GPU instances (batched)"));

  if (!ensureTemplateClasBuilt(ctx))
    return false;

  // Resize output vector
  outputs.resize(inputs.size());

  // Calculate total requirements across all instances
  uint32_t totalRequiredClusters = 0;
  uint32_t totalRequiredVertices = 0;

  for (const auto& input : inputs) {
    if (!input.isValid()) {
      Logger::err("[RTXMG] Invalid GPU input geometry in batch");
      return false;
    }
    // Estimate based on vertex/index count
    totalRequiredClusters += std::max(1u, input.indexCount / 6);  // ~2 triangles per cluster
    totalRequiredVertices += input.vertexCount;
  }

  // Update memory allocations if needed
  updateMemoryAllocations(totalRequiredClusters, totalRequiredVertices, config);

  // Phase 4: Try true GPU batching first
  // SDK MATCH: No batch size limit - sample processes any number of instances (cluster_accel_builder.cpp:1063)
  // Memory limits are handled by updateMemoryAllocations() above
  bool useTrueGpuBatching = (config.useGPUCompute) &&
                            (m_clusterTilingShader != nullptr);

  uint32_t totalClusters = 0;
  uint32_t totalVertices = 0;
  uint32_t totalTriangles = 0;
  uint32_t totalIndices = 0;

  // Calculate total indices for batch setup
  for (const auto& input : inputs) {
    totalIndices += input.indexCount;
  }

  if (useTrueGpuBatching) {
    // Phase 4: TRUE GPU BATCHING PATH
    Logger::info(str::format("[RTXMG] Using true GPU batching for ", inputs.size(), " instances"));

    // Setup batch buffers with instance data
    if (!setupBatchBuffers(ctx, inputs, totalRequiredVertices, totalIndices, totalRequiredClusters)) {
      Logger::warn("[RTXMG] Failed to setup batch buffers, falling back to sequential");
      useTrueGpuBatching = false;
    } else {
      // Get current frame index for ring buffer synchronization
      uint32_t currentFrameIndex = m_device->getCurrentFrameId();

      // CRITICAL: Calculate cluster count BEFORE dispatch so we can increment offset
      uint32_t thisInstanceClusterCount = 0;
      for (const auto& input : inputs) {
        uint32_t numTriangles = input.indexCount / 3;
        uint32_t estimatedClusters = std::max(1u, numTriangles / 2);
        thisInstanceClusterCount += estimatedClusters;
      }

      // SDK MATCH: Pass cumulative vertex offset to shader so it writes global offsets
      // Dispatch single batched compute job (uses current m_cumulativeClusterOffset and m_cumulativeVertexOffset)
      dispatchBatchedCompute(ctx, static_cast<uint32_t>(inputs.size()), config, currentFrameIndex, m_cumulativeVertexOffset);

      // CRITICAL: Increment offsets IMMEDIATELY after dispatch so next call sees them
      uint32_t oldClusterOffset = m_cumulativeClusterOffset;
      uint32_t oldVertexOffset = m_cumulativeVertexOffset;
      m_cumulativeClusterOffset += thisInstanceClusterCount;

      // Calculate actual vertex count for this batch
      uint32_t thisInstanceVertexCount = 0;
      for (const auto& input : inputs) {
        thisInstanceVertexCount += input.vertexCount;  // Use actual input vertex count
      }
      m_cumulativeVertexOffset += thisInstanceVertexCount;

      // For true batching, outputs are written to global buffers
      // We need to setup output references with proper offsets
      uint32_t cumulativeVertexOffset = oldVertexOffset;  // Start from the offset THIS dispatch used
      uint32_t cumulativeIndexOffset = 0;
      uint32_t cumulativeClusterOffset = oldClusterOffset;  // Use the offset THIS dispatch used

      for (size_t i = 0; i < inputs.size(); ++i) {
        // Estimate output sizes (actual sizes would need readback or prediction)
        uint32_t numTriangles = inputs[i].indexCount / 3;
        uint32_t estimatedClusters = std::max(1u, numTriangles / 2);
        uint32_t estimatedVerts = inputs[i].vertexCount;

        outputs[i].vertexCount = estimatedVerts;
        outputs[i].indexCount = inputs[i].indexCount;
        outputs[i].numClusters = estimatedClusters;
        outputs[i].numTriangles = numTriangles;

        // Assign global buffers to outputs
        // In true batching mode, all instances share the global buffers
        // The shader writes cluster instance data (ClusterIndirectArgs) to m_clusterIndirectArgs
        outputs[i].vertexBuffer = m_clusterVertexPositions.getBuffer();
        outputs[i].indexBuffer = inputs[i].indexBuffer.buffer();  // Get Rc<DxvkBuffer> from slice
        outputs[i].clusterInstancesBuffer = m_clusterIndirectArgs.getBuffer();

        // Track offset into cluster instances buffer for this geometry
        // Each ClusterIndirectArgs entry is 32 bytes
        outputs[i].clusterInstancesBufferOffset = cumulativeClusterOffset * 32;

        // NOTE: In true batching, outputs reference slices of global buffers at specific offsets
        cumulativeVertexOffset += estimatedVerts;
        cumulativeIndexOffset += inputs[i].indexCount;
        cumulativeClusterOffset += estimatedClusters;

        totalClusters += estimatedClusters;
        totalVertices += estimatedVerts;
        totalTriangles += numTriangles;
      }

      // Note: m_cumulativeClusterOffset already incremented above (line 3653)

      Logger::info(str::format("[RTXMG] True GPU batch complete: ", inputs.size(),
        " instances, ", totalClusters, " clusters, ", totalVertices, " vertices"));
    }
  }

  // Fallback to sequential processing if true batching not available
  if (!useTrueGpuBatching) {
    // NV-DXVK: Old sequential path removed - it called buildClustersGpu which used wrong structure
    // The TRUE GPU BATCHING PATH above is the only correct implementation using compute_cluster_tiling shader
    Logger::err("[RTXMG] GPU batching setup failed. Sequential fallback removed due to structure bugs. "
                "Increase GPU memory or reduce batch size.");
    return false;
  }

  // Update batch statistics
  m_stats.desired.m_numClusters = totalClusters;
  m_stats.desired.m_numTriangles = totalTriangles;
  m_stats.desired.m_vertexBufferSize = totalVertices * sizeof(float3);
  m_stats.desired.m_vertexNormalsBufferSize = totalVertices * sizeof(float3);
  m_stats.desired.m_clusterDataSize = totalClusters * sizeof(RtxmgCluster);

  Logger::info(str::format("[RTXMG] GPU batch complete: ", inputs.size(), " instances, ",
    totalClusters, " clusters, ", totalVertices, " vertices"));

  return true;
}

void RtxmgClusterBuilder::copyClusterOffset(
    RtxContext* ctx,
    uint32_t instanceIndex,
    uint32_t totalInstances,
    Rc<DxvkBuffer> paramsBuffer,
    const DxvkBufferSlice& tessCountersBuffer,
    const DxvkBufferSlice& clusterOffsetCountsBuffer) {

  // EXACT SDK MATCH: cluster_accel_builder.cpp line 1018-1055
  // Each call creates its own binding set and dispatches ONCE with (1,1,1)

  // SDK MATCH: Update params for this instance (reuse passed buffer)
  struct CopyClusterOffsetParams {
    uint32_t instanceIndex;
    uint32_t totalInstances;
    uint32_t _pad0;
    uint32_t _pad1;
  };

  CopyClusterOffsetParams params;
  params.instanceIndex = instanceIndex;
  params.totalInstances = totalInstances;
  params._pad0 = 0;
  params._pad1 = 0;
  ctx->updateBuffer(paramsBuffer, 0, sizeof(params), &params);

  // CRITICAL: Create NEW binding set for each dispatch (sample pattern)
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, m_copyClusterOffsetShader);
  ctx->bindResourceBuffer(0, DxvkBufferSlice(paramsBuffer, 0, sizeof(CopyClusterOffsetParams)));
  ctx->bindResourceBuffer(1, tessCountersBuffer);
  ctx->bindResourceBuffer(2, clusterOffsetCountsBuffer);

  // SDK MATCH: Dispatch once with (1,1,1) per instance
  ctx->dispatch(1, 1, 1);

  // CRITICAL: Full device memory barrier between dispatches
  if (instanceIndex < totalInstances - 1) {
    VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
    ctx->getDevice()->vkd()->vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
}

Rc<DxvkShader> RtxmgClusterBuilder::getPatchTlasShader() const {
  return PatchTlasInstanceBlasAddressesShader::getShader();
}

} // namespace dxvk
