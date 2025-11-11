/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Copy Cluster Offset - Compute shader to populate cluster offset/count buffer
*
* This shader reads GPU-written tessellation counters and writes per-instance
* cluster offset/count pairs for use by fill_blas_from_clas_args.
* Ported from NVIDIA RTX Mega Geometry SDK for RTX Remix integration.
*/

#include "rtx/pass/rtxmg/rtxmg_bindings.slangh"

// Constants
static const uint kThreadsPerGroup = 1;

// Parameters
struct CopyClusterOffsetParams {
  uint instanceIndex;
  uint totalInstances;
  uint _pad0;
  uint _pad1;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0)]] ConstantBuffer<CopyClusterOffsetParams> cb;

// Input: tessellation counters (written by compute_cluster_tiling) - shifted +1
[[vk::binding(1)]] StructuredBuffer<TessellationCounters> tessCountersIn;

// Output: cluster offset/count pairs - shifted +1
[[vk::binding(2)]] RWStructuredBuffer<uint2> clusterOffsetCountsOut;

[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint instanceIndex = cb.instanceIndex;
  if (instanceIndex >= cb.totalInstances)
    return;

  // Read TOTAL cluster count from GLOBAL counter [0] (SDK MATCH)
  // All instances accumulated into [0], this shader calculates per-instance offsets sequentially
  uint totalClusterCount = tessCountersIn[0].clusters;

  // Calculate this instance's cluster count based on cumulative offset
  // SDK pattern: dispatchClusterCount = totalClusterCount - instanceOffset
  uint dispatchClusterCount = 0;
  uint clusterOffset = 0;

  if (instanceIndex == 0) {
    // First instance gets all clusters starting at offset 0
    dispatchClusterCount = totalClusterCount;
    clusterOffset = 0;
  } else {
    // Read previous instance's data to calculate cumulative offset
    uint2 prevOffsetCount = clusterOffsetCountsOut[instanceIndex - 1];
    clusterOffset = prevOffsetCount.x + prevOffsetCount.y;
    dispatchClusterCount = totalClusterCount - clusterOffset;
  }

  // Write offset and count for this instance
  clusterOffsetCountsOut[instanceIndex] = uint2(clusterOffset, dispatchClusterCount);
}
