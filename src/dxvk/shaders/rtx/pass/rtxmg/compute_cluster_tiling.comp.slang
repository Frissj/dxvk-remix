/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Compute Cluster Tiling - Main shader
*
* This compute shader determines the cluster layout for input surfaces.
* It evaluates tessellation rates, performs visibility culling, and
* allocates clusters from a global pool.
*
* Simplified for triangle mesh tessellation (no subdivision surfaces).
*/

#include "rtx/pass/rtxmg/rtxmg_cluster_tiling.slangh"
#include "rtx/utility/common.slangh"

// Constant buffer (b0 - NOT push constant due to 216+ byte size exceeding 128 byte limit)
[[vk::binding(0, 0)]] ConstantBuffer<ClusterTilingParams> cb;

// Input buffers (all in set 0, starting from binding 1)
[[vk::binding(1, 0)]] StructuredBuffer<float3> inputPositions;
[[vk::binding(2, 0)]] StructuredBuffer<float3> inputNormals;
// Use ByteAddressBuffer for potentially misaligned game buffers (only requires 4-byte alignment)
[[vk::binding(3, 0)]] ByteAddressBuffer inputTexcoords;
[[vk::binding(4, 0)]] ByteAddressBuffer inputIndices;
[[vk::binding(5, 0)]] StructuredBuffer<SurfaceInfo> surfaceInfo;

// Template data
[[vk::binding(6, 0)]] StructuredBuffer<uint64_t> templateAddresses;
[[vk::binding(7, 0)]] StructuredBuffer<uint> clasInstantiationBytes;

// HiZ buffer (array of LODs)
[[vk::binding(8, 0)]] Texture2DArray<float> hizBuffer;
[[vk::binding(9, 0)]] SamplerState hizSampler;

// Output buffers
[[vk::binding(10, 0)]] RWStructuredBuffer<GridSampler> gridSamplersOut;
[[vk::binding(11, 0)]] RWStructuredBuffer<TessellationCounters> tessCountersOut;
[[vk::binding(12, 0)]] RWStructuredBuffer<Cluster> clustersOut;
[[vk::binding(13, 0)]] RWStructuredBuffer<ClusterShadingData> shadingDataOut;
[[vk::binding(14, 0)]] RWStructuredBuffer<ClusterIndirectArgs> indirectArgsOut;
[[vk::binding(15, 0)]] RWStructuredBuffer<uint64_t> clasAddressesOut;

// Phase 4: GPU batching support - Instance data buffer (optional)
[[vk::binding(16, 0)]] StructuredBuffer<InstanceData> instanceDataBuffer;

// Process a single surface
void processSurface(uint surfaceId, uint laneId, uint waveId) {
  // Get surface info
  SurfaceInfo surface = surfaceInfo[surfaceId];

  // Early out if no geometry
  if (surface.indexCount == 0) {
    return;
  }

  // Frustum culling (lane 0 only)
  bool visible = true;
  if (cb.enableFrustumCulling && laneId == 0) {
    visible = frustumCullSurface(
      cb,
      inputPositions,
      inputIndices,
      surfaceId,
      surface.firstIndex,
      surface.indexCount,
      laneId
    );
  }
  visible = WaveReadLaneFirst(visible);

  if (!visible) {
    return;
  }

  // Evaluate edge segments based on tessellation mode
  uint4 edgeSegments;
  if (laneId == 0) {
    evaluateEdgeSegments(
      cb,
      inputPositions,
      inputIndices,
      surfaceId,
      surface.firstIndex,
      laneId,
      edgeSegments
    );
  }

  // Broadcast edge segments to all lanes
  edgeSegments.x = WaveReadLaneFirst(edgeSegments.x);
  edgeSegments.y = WaveReadLaneFirst(edgeSegments.y);
  edgeSegments.z = WaveReadLaneFirst(edgeSegments.z);
  edgeSegments.w = WaveReadLaneFirst(edgeSegments.w);

  // Create grid sampler
  GridSampler gridSampler;
  gridSampler.edgeSegments[0] = (uint16_t)edgeSegments.x;
  gridSampler.edgeSegments[1] = (uint16_t)edgeSegments.y;
  gridSampler.edgeSegments[2] = (uint16_t)edgeSegments.z;
  gridSampler.edgeSegments[3] = (uint16_t)edgeSegments.w;

  if (laneId == 0) {
    gridSamplersOut[surfaceId] = gridSampler;
  }

  // Calculate surface size
  uint16_t surfaceSizeX = (uint16_t)max(edgeSegments.x, edgeSegments.z);
  uint16_t surfaceSizeY = (uint16_t)max(edgeSegments.y, edgeSegments.w);

  // Create surface tiling (simplified - assuming regular grid for now)
  // Full implementation in Phase 2.5 will use MakeSurfaceTiling algorithm

  // For now, use a single regular tiling
  uint targetEdgeSegments = 8;
  uint numClustersX = (surfaceSizeX + targetEdgeSegments - 1) / targetEdgeSegments;
  uint numClustersY = (surfaceSizeY + targetEdgeSegments - 1) / targetEdgeSegments;

  uint clusterCount = numClustersX * numClustersY;
  uint verticesPerCluster = (targetEdgeSegments + 1) * (targetEdgeSegments + 1);
  uint vertexCount = clusterCount * verticesPerCluster;
  uint triangleCount = clusterCount * targetEdgeSegments * targetEdgeSegments * 2;

  // Calculate CLAS blocks needed
  uint templateIndex = (targetEdgeSegments - 1) * 11 + (targetEdgeSegments - 1);
  uint clasBytes = clasInstantiationBytes[templateIndex];
  uint clasBlocks = (clasBytes + 255) / 256; // Round up to 256-byte alignment
  uint totalClasBlocks = clasBlocks * clusterCount;

  // Allocate from global pool (lane 0 only)
  uint surfaceClusterOffset = 0;
  uint surfaceVertexOffset = 0;
  uint clasBlocksOffset = 0;
  bool allocationSucceeded = false;

  if (laneId == 0) {
    // Use group shared memory to coalesce atomics
    uint waveClusterOffset, waveVertexOffset, waveClasBlocksOffset;

    InterlockedAdd(s_clusters, clusterCount, waveClusterOffset);
    InterlockedAdd(s_vertices, vertexCount, waveVertexOffset);
    InterlockedAdd(s_clasBlocks, totalClasBlocks, waveClasBlocksOffset);
    InterlockedAdd(s_triangles, triangleCount);

    GroupMemoryBarrierWithGroupSync();

    // One global atomic per thread group
    // SDK MATCH: Write to GLOBAL counter [0]
    // SDK dispatches once per instance, each dispatch writes its cluster count to [0]
    // Then CopyClusterOffset reads [0] after each dispatch to calculate cumulative offsets
    if (waveId == 0) {
      InterlockedAdd(tessCountersOut[0].desiredClasBlocks, s_clasBlocks, s_groupClasBlocksOffset);
      InterlockedAdd(tessCountersOut[0].desiredClusters, s_clusters, s_groupClusterOffset);
      InterlockedAdd(tessCountersOut[0].desiredVertices, s_vertices, s_groupVertexOffset);
      InterlockedAdd(tessCountersOut[0].desiredTriangles, s_triangles);

      allocationSucceeded =
        ((s_groupClasBlocksOffset + s_clasBlocks) <= cb.maxClasBlocks) &&
        ((s_groupClusterOffset + s_clusters) <= cb.maxClusters) &&
        ((s_groupVertexOffset + s_vertices) <= cb.maxVertices);

      if (allocationSucceeded) {
        InterlockedAdd(tessCountersOut[0].clusters, s_clusters, s_groupClusterOffset);
      }

      s_allocationSucceeded = allocationSucceeded;
    }

    GroupMemoryBarrierWithGroupSync();

    surfaceClusterOffset = s_groupClusterOffset + waveClusterOffset;
    surfaceVertexOffset = s_groupVertexOffset + waveVertexOffset;
    clasBlocksOffset = s_groupClasBlocksOffset + waveClasBlocksOffset;
    allocationSucceeded = s_allocationSucceeded;
  }

  // Broadcast allocation results
  allocationSucceeded = WaveReadLaneFirst(allocationSucceeded);
  if (!allocationSucceeded) {
    return;
  }

  surfaceClusterOffset = WaveReadLaneFirst(surfaceClusterOffset);
  surfaceVertexOffset = WaveReadLaneFirst(surfaceVertexOffset);
  clasBlocksOffset = WaveReadLaneFirst(clasBlocksOffset);

  // Write clusters (parallel across lanes)
  for (uint iCluster = laneId; iCluster < clusterCount; iCluster += kThreadsPerWave) {
    uint clusterX = iCluster % numClustersX;
    uint clusterY = iCluster / numClustersX;

    uint2 clusterOffset = uint2(clusterX * targetEdgeSegments, clusterY * targetEdgeSegments);
    uint2 clusterSize = uint2(targetEdgeSegments, targetEdgeSegments);

    // Apply base cluster offset for multi-geometry batching
    uint clusterIndex = cb.baseClusterOffset + surfaceClusterOffset + iCluster;
    uint vertexOffset = surfaceVertexOffset + iCluster * verticesPerCluster;

    writeCluster(
      clustersOut,
      shadingDataOut,
      indirectArgsOut,
      clasAddressesOut,
      templateAddresses,
      clasInstantiationBytes,
      cb,
      clusterIndex,
      surfaceId,
      vertexOffset,
      clusterOffset,
      clusterSize,
      edgeSegments,
      surface.geometryId,
      uint2(numClustersX, numClustersY)  // NV-DXVK: Total grid dimensions for UV-based stable cluster IDs
    );
  }
}

// Phase 4: Process a single instance in batched mode
void processInstance(uint instanceId, uint laneId, uint waveId) {
  // Early out if beyond instance count
  if (instanceId >= cb.instanceCount) {
    return;
  }

  // Read instance data
  InstanceData instance = instanceDataBuffer[instanceId];

  // Apply per-instance output offsets
  // In batched mode, each instance writes to its allocated region
  // The processSurface function would need instance-specific offsets
  // For now, we process the instance's surface normally
  // Future enhancement: Full device address support for per-instance input buffers

  // Process the instance's surface
  // Note: Current implementation uses shared input buffers
  // Full batching would require device address support for per-instance inputs
  processSurface(instance.surfaceId, laneId, waveId);
}

// Main entry point
[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(
  uint3 threadIdx : SV_GroupThreadID,
  uint3 groupIdx : SV_GroupID,
  uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint laneId = threadIdx.x % kThreadsPerWave;
  uint waveId = threadIdx.x / kThreadsPerWave;

  // Initialize group shared memory
  if (laneId == 0 && waveId == 0) {
    s_clusters = 0;
    s_vertices = 0;
    s_clasBlocks = 0;
    s_triangles = 0;
    s_allocationSucceeded = false;
  }

  GroupMemoryBarrierWithGroupSync();

  // Phase 4: Check if batching mode is enabled
  if (cb.enableBatching != 0) {
    // GPU Batching Mode: Process instances from instance data buffer
    // Each thread processes one instance
    uint instanceId = dispatchThreadId.x;

    if (instanceId < cb.instanceCount) {
      processInstance(instanceId, laneId, waveId);
    }
  }
  else {
    // Sequential Mode: Process surfaces in range
    uint surfaceId = groupIdx.x * kWavesPerGroup + waveId + cb.surfaceStart;

    // Early out if beyond surface range
    if (surfaceId >= cb.surfaceEnd) {
      return;
    }

    // Process this surface
    processSurface(surfaceId, laneId, waveId);
  }
}
