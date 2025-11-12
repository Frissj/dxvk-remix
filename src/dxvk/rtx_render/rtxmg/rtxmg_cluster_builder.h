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

#pragma once

#include "rtxmg_config.h"
#include "rtxmg_accel.h"
#include "rtxmg_counters.h"
#include "rtxmg_cluster.h"
#include "rtxmg_buffer.h"
#include "vk_nv_cluster_acceleration_structure.h"
#include "../../dxvk_device.h"
#include <vector>
#include <memory>

namespace dxvk {

class DxvkContext;
class RtxContext;

// Input geometry for cluster tessellation
struct ClusterInputGeometry {
  // Input triangle mesh
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float2> texcoords;
  std::vector<uint32_t> indices;

  // Material/surface ID
  uint32_t surfaceId = 0;

  // Optional transform
  Matrix4 transform = Matrix4();

  bool isValid() const {
    return !positions.empty() && !indices.empty();
  }
};

// GPU-based input using buffer slices (zero-copy)
struct ClusterInputGeometryGpu {
  // GPU buffer slices - for binding to compute shader
  DxvkBufferSlice positionBuffer;
  DxvkBufferSlice normalBuffer;
  DxvkBufferSlice texcoordBuffer;
  DxvkBufferSlice indexBuffer;

  // Buffer metadata
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t positionStride = 0;
  uint32_t normalStride = 0;
  uint32_t texcoordStride = 0;
  uint32_t positionOffset = 0;
  uint32_t normalOffset = 0;
  uint32_t texcoordOffset = 0;

  // Index format
  VkIndexType indexType = VK_INDEX_TYPE_UINT16;

  // Material/surface ID
  uint32_t surfaceId = 0;

  // Optional transform
  Matrix4 transform = Matrix4();

  bool isValid() const {
    return positionBuffer.defined() && vertexCount > 0;
  }
};

// Output tessellated cluster geometry
struct ClusterOutputGeometry {
  // Tessellated mesh
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float2> texcoords;
  std::vector<uint32_t> indices;

  // Cluster metadata
  std::vector<RtxmgCluster> clusters;
  uint32_t numClusters = 0;

  // Statistics
  uint32_t numVertices = 0;
  uint32_t numTriangles = 0;

  void clear() {
    positions.clear();
    normals.clear();
    texcoords.clear();
    indices.clear();
    clusters.clear();
    numClusters = 0;
    numVertices = 0;
    numTriangles = 0;
  }
};

// GPU-based output (buffers remain on GPU)
struct ClusterOutputGeometryGpu {
  // GPU buffer containing tessellated vertices (interleaved: pos, norm, uv)
  Rc<DxvkBuffer> vertexBuffer;
  uint32_t vertexBufferStride = 32; // Position(12) + Normal(12) + TexCoord(8)

  // GPU buffer containing tessellated indices
  Rc<DxvkBuffer> indexBuffer;

  // GPU buffer containing cluster instance data (written by cluster tiling shader)
  // This buffer is used directly for BLAS building without CPU-GPU synchronization
  // Uses ClusterIndirectArgs format (32 bytes, SDK-compatible)
  Rc<DxvkBuffer> clusterInstancesBuffer;

  // Offset into the cluster instances buffer for this geometry (in bytes)
  uint32_t clusterInstancesBufferOffset = 0;

  // Counts
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t numClusters = 0;
  uint32_t numTriangles = 0;  // Number of triangles (indexCount / 3)

  bool isValid() const {
    return vertexBuffer != nullptr && indexBuffer != nullptr && vertexCount > 0;
  }

  void clear() {
    vertexBuffer = nullptr;
    indexBuffer = nullptr;
    clusterInstancesBuffer = nullptr;
    vertexCount = 0;
    indexCount = 0;
    numClusters = 0;
    numTriangles = 0;
  }
};

// RTX Mega Geometry cluster builder
// Simplified interface for RTX Remix integration without NVRHI dependencies
class RtxmgClusterBuilder {
public:
  RtxmgClusterBuilder(DxvkDevice* device);
  ~RtxmgClusterBuilder();

  // Initialize cluster builder
  // Sets up template grids and compute pipelines
  bool initialize();

  // Shutdown and release resources
  void shutdown();

  // Build cluster tessellation for multiple GPU instances (batched processing)
  // Pure GPU path with zero-copy, processes multiple instances in single batch
  // Returns true if successful, false otherwise
  bool buildClustersGpuBatch(
    RtxContext* ctx,
    const std::vector<ClusterInputGeometryGpu>& inputs,
    std::vector<ClusterOutputGeometryGpu>& outputs,
    const RtxmgConfig& config);

  // Build cluster acceleration structures (CLAS + BLAS)
  // Uses VK_NV_cluster_acceleration_structure extension
  bool buildAccelerationStructures(
    RtxContext* ctx,
    const ClusterOutputGeometry& geometry,
    ClusterAccels& accels,
    const RtxmgConfig& config);

  // GPU-optimized BLAS building from GPU-resident cluster data
  // Uses pre-generated cluster instances buffer to avoid CPU-GPU synchronization
  bool buildAccelerationStructures(
    RtxContext* ctx,
    const ClusterOutputGeometryGpu& geometryGpu,
    ClusterAccels& accels,
    const RtxmgConfig& config);

  // Instantiate cluster instances without building BLAS (for per-geometry BLAS)
  // Returns instance addresses and the temporary instance buffer (must be kept alive!)
  // The caller MUST keep outTempInstanceBuffer alive until GPU commands complete
  bool instantiateClusterInstancesOnly(
    RtxContext* ctx,
    const ClusterOutputGeometryGpu& geometryGpu,
    std::vector<VkDeviceAddress>& outInstanceAddresses,
    RtxmgBuffer<uint8_t>& outTempInstanceBuffer,
    const RtxmgConfig& config);

  // Build unified BLAS from collected cluster instance addresses
  // This builds ONE BLAS for all clusters across multiple geometries
  bool buildUnifiedBLAS(
    RtxContext* ctx,
    const std::vector<VkDeviceAddress>& allInstanceAddresses,
    ClusterAccels& outAccels,
    const RtxmgConfig& config);

  // SAMPLE CODE MATCH: Build multiple BLASes at once (one per geometry)
  // This matches sample's BuildBlasFromClas which builds N BLASes in one GPU call
  // Each geometry gets its own BLAS built in parallel
  struct MultiBLASInput {
    ClusterOutputGeometryGpu geometry;
    XXH64_hash_t geometryHash;
  };

  bool buildMultipleBLAS(
    RtxContext* ctx,
    const std::vector<MultiBLASInput>& geometries,
    std::vector<ClusterAccels>& outAccels,
    const RtxmgConfig& config);

  // EXACT SAMPLE MATCH: BuildStructuredCLASes + BuildBlasFromClas
  // Sample lines 1365-1368: Build ALL CLAS in one GPU call, then ALL BLAS in one GPU call
  // This is the ONLY correct way that matches NVIDIA's architecture
  struct DrawCallData {
    XXH64_hash_t geometryHash;
    Matrix4 transform;
    uint32_t drawCallIndex;
    uint32_t clusterCount;
    ClusterOutputGeometryGpu output;

    // CRITICAL FIX: Input geometry for fill_clusters shader reconstruction
    DxvkBufferSlice inputPositions;
    DxvkBufferSlice inputNormals;
    DxvkBufferSlice inputTexcoords;
    DxvkBufferSlice inputIndices;
    uint32_t inputVertexCount = 0;
    uint32_t inputIndexCount = 0;
    uint32_t positionOffset = 0;
    uint32_t normalOffset = 0;
    uint32_t texcoordOffset = 0;

    // CRITICAL: CPU-side cluster metadata for patching vertex offsets
    // Sample generates metadata with global offsets via GPU tiling
    // We generate with local offsets via CPU tessellation, so we need to patch
    std::vector<RtxmgCluster> clusters;  // CPU-side cluster metadata

    VkDeviceAddress clusterBlasAddress = 0;
    uint32_t clusterBlasSize = 0;
  };

  struct ClusterOffsetCount {
    uint32_t offset;
    uint32_t count;
  };

  // Split CLAS and BLAS building to match reference implementation
  // Reference: cluster_accel_builder.cpp:1365-1368
  // Line 1365: BuildStructuredCLASes() - builds all CLAS
  // Line 1368: BuildBlasFromClas() - builds all BLAS from CLAS
  bool buildStructuredCLASes(
    RtxContext* ctx,
    const std::vector<DrawCallData>& drawCalls,
    const RtxmgConfig& config,
    uint32_t frameIndex);

  bool buildBlasFromClas(
    RtxContext* ctx,
    const std::vector<DrawCallData>& drawCalls,
    const RtxmgConfig& config);

  const ClusterAccels& getFrameAccels() const { return m_frameAccels; }

  // Get BLAS addresses buffer for GPU-side TLAS patching (NVIDIA sample method)
  Rc<DxvkBuffer> getBlasAddressesBuffer() const {
    return m_frameAccels.blasPtrsBuffer.getBuffer();
  }

  // Get blasPtrsBuffer index for a geometry hash (-1 if not found)
  // Used for GPU patching of TLAS instance descriptors
  int32_t getBlasBufferIndex(XXH64_hash_t geometryHash) const {
    auto it = m_geometryHashToBlasIndex.find(geometryHash);
    return (it != m_geometryHashToBlasIndex.end()) ? it->second : -1;
  }

  uint32_t getBlasCount() const {
    return static_cast<uint32_t>(m_geometryHashToBlasIndex.size());
  }

  // Get GPU patching shader (defined in .cpp to avoid header include conflicts)
  Rc<DxvkShader> getPatchTlasShader() const;

  bool resolveBlasAddressReadback(
    RtxContext* ctx,
    std::vector<VkDeviceAddress>& outAddresses,
    std::vector<uint32_t>& outSizes);

  // Reset cumulative offsets for ring-buffered writes (SDK MATCH: reset at frame start)
  void resetCumulativeOffsets() {
    m_cumulativeClusterOffset = 0;
    m_cumulativeVertexOffset = 0;
  }

  // Update per-frame (HiZ buffer, visibility culling, etc.)
  void updatePerFrame(
    RtxContext* ctx,
    const Rc<DxvkImageView>& depthBuffer,
    const RtxmgConfig& config);

  // Get statistics
  const RtxmgStatistics& getStatistics() const { return m_stats; }

  // Get template grids (for debugging)
  const TemplateGrids& getTemplateGrids() const { return m_templateGrids; }

  // Update memory allocations based on requirements
  // Returns true if allocations changed, false if no change needed
  bool updateMemoryAllocations(
    uint32_t requiredClusters,
    uint32_t requiredVertices,
    const RtxmgConfig& config);

  // Get current frame's instance buffer (for copying to persistent storage)
  Rc<DxvkBuffer> getRingBufferInstanceBuffer() {
    return m_frameBuffers.instanceBuffer.getBuffer();
  }

private:
  // Generate cluster template grids (called once during init)
  void generateTemplateGrids();

  // Create shader objects and pipelines
  void createShaders();

  // Ensure CLAS templates are resident before GPU work
  bool ensureTemplateClasBuilt(RtxContext* ctx);


  // Buffer management
  void createGPUBuffers(const RtxmgConfig& config);
  void resizeBuffersIfNeeded(uint32_t requiredClusters, uint32_t requiredVertices);
  void queueBlasAddressReadback(RtxContext* ctx, uint32_t count);

  // Dynamic BLAS allocation (SDK MATCH: cluster_accel_builder.cpp:1197-1223)
  // Releases and recreates BLAS buffers when cluster/instance count changes significantly
  // Takes RtxContext for fence tracking to safely release old buffers
  void updateBlasAllocation(RtxContext* ctx, uint32_t totalClusters, uint32_t maxClustersPerBlas, uint32_t numInstances);

  // Production buffer management (Phase 2)
  void updateMemoryPressureDetection();
  bool checkMemoryPressure(size_t additionalBytes);
  size_t alignSize(size_t size, size_t alignment);
  void updatePeakUsage(uint32_t clusters, uint32_t vertices);
  bool shouldShrinkBuffers(uint32_t requiredClusters, uint32_t requiredVertices);
  void shrinkBuffersIfNeeded(uint32_t requiredClusters, uint32_t requiredVertices);
  VkDeviceSize alignDeviceSize(VkDeviceSize size, VkDeviceSize alignment) const;
  VkDeviceSize getClusterTemplateAlignment() const;
  VkDeviceSize getClusterScratchAlignment() const;
  VkDeviceSize getInstanceStride() const;
  // Removed ring buffer indexing - using single buffers with GPU sync like NVIDIA sample

  // Error handling & recovery (Phase 3)
  bool validateInputGeometry(const ClusterInputGeometry& input, const char* context);
  bool validateInputGeometryGpu(const ClusterInputGeometryGpu& input, const char* context);
  bool validateOutputCapacity(uint32_t requiredClusters, uint32_t requiredVertices);
  bool validateBufferState();
  void applyGracefulDegradation(RtxmgConfig& config, uint32_t degradationLevel);
  struct BufferSnapshot {
    uint32_t allocatedClusters;
    uint32_t allocatedVertices;
    size_t allocatedClasSize;
    size_t allocatedBlasSize;
  };
  BufferSnapshot saveBufferState();
  bool restoreBufferState(const BufferSnapshot& snapshot);

  // True GPU batching (Phase 4)
  struct InstanceData {
    VkDeviceAddress inputPositionBuffer;
    VkDeviceAddress inputNormalBuffer;
    VkDeviceAddress inputTexcoordBuffer;
    VkDeviceAddress inputIndexBuffer;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t vertexOffset;  // Offset into global output buffers
    uint32_t indexOffset;   // Offset into global output buffers
    uint32_t clusterOffset; // Offset into global cluster buffers
    uint32_t surfaceId;
    Matrix4 transform;
  };
  bool setupBatchBuffers(RtxContext* ctx, const std::vector<ClusterInputGeometryGpu>& inputs, uint32_t totalVertices, uint32_t totalIndices, uint32_t totalClusters);
  void dispatchBatchedCompute(RtxContext* ctx, uint32_t instanceCount, const RtxmgConfig& config, uint32_t frameIndex, uint32_t globalVertexOffset);

  // HiZ pyramid management (Phase 4)
  void createHiZPyramid(uint32_t width, uint32_t height);
  void generateHiZPyramid(RtxContext* ctx, const Rc<DxvkImageView>& depthBuffer);

  // Copy cluster offset for a single instance (GPU-driven offsets)
  // Matches NVIDIA sample pattern: cluster_accel_builder.cpp line 1018-1055
  void copyClusterOffset(
    RtxContext* ctx,
    uint32_t instanceIndex,
    uint32_t totalInstances,
    Rc<DxvkBuffer> paramsBuffer,
    const DxvkBufferSlice& tessCountersBuffer,
    const DxvkBufferSlice& clusterOffsetCountsBuffer);
  void resizeHiZIfNeeded(uint32_t width, uint32_t height);
  void queueCounterReadback(RtxContext* ctx);
  void resolvePendingCounterReadback(uint32_t slotIndex);

private:
  DxvkDevice* m_device = nullptr;
  bool m_initialized = false;

  // Template grids (121 pre-generated cluster templates)
  TemplateGrids m_templateGrids;

  // Statistics
  RtxmgStatistics m_stats;

  // Allocation tracking
  uint32_t m_allocatedClusters = 0;
  uint32_t m_allocatedVertices = 0;
  size_t m_allocatedClasSize = 0;
  size_t m_allocatedBlasSize = 0;
  size_t m_allocatedBlasScratchSize = 0;

  // BLAS allocation tracking (for dynamic sizing - SDK MATCH: sample releases/recreates as needed)
  // Reference: cluster_accel_builder.cpp:1197-1223 UpdateMemoryAllocations()
  uint32_t m_blasMaxTotalClusters = 0;       // Last allocated max total clusters
  uint32_t m_blasMaxInstances = 0;           // Last allocated max instance count
  VkAccelerationStructureBuildSizesInfoKHR m_blasSizeInfo = {}; // Current BLAS size (not MAX!)

  // Buffer management - hysteresis tracking
  uint32_t m_peakClusters = 0;           // Peak cluster usage
  uint32_t m_peakVertices = 0;           // Peak vertex usage
  uint32_t m_underutilizationFrames = 0; // Frames below shrink threshold
  static constexpr uint32_t SHRINK_FRAME_THRESHOLD = 120;   // 2 seconds at 60 FPS
  static constexpr float SHRINK_USAGE_THRESHOLD = 0.25f;    // Shrink if usage < 25%
  static constexpr float GROW_FACTOR = 1.5f;                // Grow by 1.5× (not 2×)
  static constexpr uint32_t MIN_CLUSTER_CAPACITY = 1024;    // Minimum buffer size
  static constexpr uint32_t MIN_VERTEX_CAPACITY = 16384;    // Minimum buffer size

  // Memory pressure detection
  size_t m_totalGpuMemoryAllocated = 0;  // Total GPU memory used by RTXMG
  size_t m_deviceTotalMemory = 0;        // Total device memory available
  size_t m_deviceAvailableMemory = 0;    // Available device memory
  bool m_memoryPressureDetected = false; // Memory pressure flag
  static constexpr float MEMORY_PRESSURE_THRESHOLD = 0.90f; // 90% usage

  // Error handling & graceful degradation (Phase 3)
  uint32_t m_degradationLevel = 0;       // Current degradation level (0 = none)
  uint32_t m_consecutiveFailures = 0;    // Track consecutive allocation failures
  static constexpr uint32_t MAX_DEGRADATION_LEVEL = 3;
  static constexpr uint32_t FAILURE_THRESHOLD = 3; // Failures before degradation

  // True GPU batching infrastructure (Phase 4)
  RtxmgBuffer<InstanceData> m_instanceDataBuffer;  // Per-instance data for batching
  RtxmgBuffer<VkDispatchIndirectCommand> m_indirectDispatchBuffer;  // Indirect dispatch commands
  RtxmgBuffer<uint32_t> m_instanceOffsetsBuffer;  // Cumulative offsets for each instance
  bool m_batchingEnabled = false;  // Whether true GPU batching is available
  // SDK MATCH: No MAX_BATCH_INSTANCES limit - sample processes any number of instances
  uint32_t m_cumulativeClusterOffset = 0;  // Cumulative cluster count across batches (reset each frame)
  uint32_t m_cumulativeVertexOffset = 0;   // SDK MATCH: Cumulative vertex count across geometries (reset each frame)

  // Alignment requirements
  static constexpr size_t ACCEL_STRUCT_ALIGNMENT = 256;  // VK spec for AS
  static constexpr size_t STORAGE_BUFFER_ALIGNMENT = 16; // Common SSBO alignment
  static constexpr size_t VERTEX_BUFFER_ALIGNMENT = 4;   // Float alignment

  // Tessellation counters buffer (SINGLE buffer with GPU sync - SDK MATCH)
  RtxmgBuffer<TessellationCounters> m_tessCountersBuffer;
  Rc<DxvkBuffer> m_tessCountersReadback;
  bool m_counterReadbackReady = false;
  TessellationCounters m_lastCompletedCounters = {};
  bool m_lastCompletedCountersValid = false;
  uint64_t m_frameSerial = 0;
  uint64_t m_lastCounterCopyFrame = 0;

  // SDK-matching indirect buffers for GPU-driven cluster pipeline (SINGLE buffers - SDK MATCH)
  // Sample: m_clusterOffsetCountsBuffer stores per-instance cluster offsets and counts
  // Sample: m_fillClustersDispatchIndirectBuffer stores indirect dispatch args for fill_clusters
  RtxmgBuffer<ClusterOffsetCount> m_clusterOffsetCountsBuffer;     // Per-instance cluster offsets (cleared via GPU each frame)
  RtxmgBuffer<VkDispatchIndirectCommand> m_fillClustersDispatchIndirectBuffer;  // Indirect dispatch args (cleared via GPU each frame)

  // Frame index for BuildAccel (SDK: m_buildAccelFrameIndex)
  // Only advances when buildStructuredCLASes runs, NOT every updatePerFrame
  uint64_t m_buildAccelFrameIndex = 0;

  // Cluster tiling params constant buffer (replaces push constants to support 216+ byte structure)
  RtxmgBuffer<uint8_t> m_clusterTilingParamsBuffer;

  // GPU buffers for cluster building
  // Input buffers
  RtxmgBuffer<float3> m_inputPositions;
  RtxmgBuffer<float3> m_inputNormals;
  RtxmgBuffer<float2> m_inputTexcoords;
  RtxmgBuffer<uint32_t> m_inputIndices;

  // Surface metadata
  struct SurfaceInfo {
    uint32_t firstVertex;
    uint32_t vertexCount;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialId;
    uint32_t geometryId;
    uint32_t pad0;
    uint32_t pad1;
  };
  RtxmgBuffer<SurfaceInfo> m_surfaceInfo;

  // Template data
  RtxmgBuffer<VkDeviceAddress> m_templateAddresses;
  RtxmgBuffer<uint32_t> m_clasInstantiationBytes;

  // Template CLAS buffer (Phase 3)
  RtxmgBuffer<uint8_t> m_templateClasBuffer;
  std::vector<VkDeviceAddress> m_templateAddressesVec;  // CPU-side copy
  std::vector<uint32_t> m_clasInstantiationBytesVec;    // CPU-side copy
  bool m_templateClasBuilt = false;  // Track if template CLAS has been built

  // Intermediate buffers
  RtxmgBuffer<GridSampler> m_gridSamplers;
  RtxmgBuffer<RtxmgCluster> m_clusters;
  RtxmgBuffer<RtxmgClusterShadingData> m_clusterShadingData;
  RtxmgBuffer<VkDeviceAddress> m_clasAddresses;

  // Output buffers
  RtxmgBuffer<float3> m_clusterVertexPositions;
  RtxmgBuffer<float3> m_clusterVertexNormals;

  // Cluster indirect args (matches VkClusterAccelerationStructureInstantiateClusterInfoNV)
  // SDK Format: clusterIdOffset (4) + geometryIndexAndReserved (4) + clusterTemplate (8) + vertexBuffer.startAddress (8) + vertexBuffer.strideInBytes (8) = 32 bytes
  struct ClusterIndirectArgs {
    uint32_t clusterIdOffset;                  // Offset 0, 4 bytes
    uint32_t geometryIndexAndReserved;         // Offset 4, 4 bytes (24-bit geomIndex + 8-bit reserved, packed)
    VkDeviceAddress clusterTemplate;           // Offset 8, 8 bytes
    VkDeviceAddress vertexBufferStartAddress;  // Offset 16, 8 bytes (VkStridedDeviceAddressNV.startAddress)
    uint64_t vertexBufferStrideInBytes;        // Offset 24, 8 bytes (VkStridedDeviceAddressNV.strideInBytes)
    // Total: 32 bytes
  };
  RtxmgBuffer<ClusterIndirectArgs> m_clusterIndirectArgs;

  // Frame-wide instantiation buffers (SINGLE buffers with GPU sync - SDK MATCH)
  // PROPER BATCHING: All geometries in a frame use the same buffers, instantiated in a single GPU call
  struct FrameInstantiationBuffers {
    RtxmgBuffer<uint8_t> scratchBuffer;
    RtxmgBuffer<uint32_t> countBuffer;
    RtxmgBuffer<VkDeviceAddress> addressesBuffer;
    RtxmgBuffer<uint8_t> instanceBuffer;
    uint32_t usedClusters = 0;       // Track how many clusters used this frame
    uint32_t allocatedClusters = 0;  // Total capacity
    VkFence lastUsedFence = VK_NULL_HANDLE;  // GPU sync for safe reuse
  };
  FrameInstantiationBuffers m_frameBuffers;  // SINGLE buffer, no ring buffering

  // Frame-wide acceleration structures (persistent across frames, matches sample's ClusterAccels)
  ClusterAccels m_frameAccels;
  RtxmgBuffer<VkDeviceAddress> m_blasPtrsReadbackBuffer;
  RtxmgBuffer<uint32_t> m_blasSizesReadbackBuffer;
  uint32_t m_lastBlasReadCount = 0;
  bool m_blasReadbackPending = false;

  // Fence-tracked buffer release system (precise GPU work tracking)
  struct BufferWithFence {
    ClusterAccels accels;
    VkFence lastUsageFence;  // Fence from the command buffer that last used this buffer
  };
  std::vector<BufferWithFence> m_pendingReleases;

  // Geometry hash to blasPtrsBuffer index mapping for GPU-side TLAS patching
  // Maps geometry hash → index in blasPtrsBuffer
  // Updated each frame when building multiple BLASes
  std::unordered_map<XXH64_hash_t, int32_t> m_geometryHashToBlasIndex;

  // BLAS build parameters (stored as member to avoid stack allocation issues)
  // SDK MATCH: cluster_accel_builder.h:273 - stores OperationParams as member
  VkClusterAccelerationStructureClustersBottomLevelInputNV m_createBlasParams;
  VkClusterAccelerationStructureInputInfoNV m_createBlasInputInfo;
  // Note: m_maxBlasSizeInfo already declared at line 388

  // CLAS instantiation parameters (stored as member to avoid stack allocation issues)
  VkClusterAccelerationStructureTriangleClusterInputNV m_createClasTriangleInput;
  VkClusterAccelerationStructureInputInfoNV m_createClasInputInfo;

  // Compute shaders (Phase 2.5)
  Rc<DxvkShader> m_clusterTilingShader;
  Rc<DxvkShader> m_clusterFillingShader;
  Rc<DxvkShader> m_copyClusterOffsetShader;
  Rc<DxvkShader> m_fillBlasFromClasArgsShader;

  // HiZ pyramid generation shader (Phase 4)
  Rc<DxvkShader> m_hizPyramidGenerateShader;

  // HiZ buffer system (Phase 4)
  Rc<DxvkImage> m_hizPyramid;              // HiZ pyramid image (2D array with mip levels)
  Rc<DxvkImageView> m_hizPyramidView;      // Full pyramid view
  std::vector<Rc<DxvkImageView>> m_hizMipViews;  // Per-mip level views
  Rc<DxvkSampler> m_hizSampler;            // Point sampler for HiZ
  uint32_t m_hizWidth = 0;                 // HiZ pyramid width (base level)
  uint32_t m_hizHeight = 0;                // HiZ pyramid height (base level)
  uint32_t m_hizNumLevels = 0;             // Number of mip levels
  bool m_hizInitialized = false;           // HiZ system ready
};

} // namespace dxvk
