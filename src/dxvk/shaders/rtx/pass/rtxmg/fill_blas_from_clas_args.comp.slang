/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Fill BLAS From CLAS Args - Compute shader to generate indirect args for BLAS building
*
* This shader generates the indirect argument buffer needed for building BLASes from CLAS structures.
* Ported from NVIDIA RTX Mega Geometry SDK for RTX Remix integration.
*/

#include "rtx/pass/rtxmg/rtxmg_bindings.slangh"

// Constants
static const uint kThreadsPerGroup = 256;

// Parameters
struct FillBlasFromClasArgsParams {
  uint64_t clasAddressesBaseAddress;  // Base address of CLAS addresses array
  uint numInstances;                   // Number of BLASes to build
  uint _pad0;
  uint _pad1;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0, 0)]] ConstantBuffer<FillBlasFromClasArgsParams> cb;

// Input: cluster offset/count pairs per instance - shifted +1
[[vk::binding(1, 0)]] StructuredBuffer<uint2> clusterOffsetCounts;

// Output: indirect args for BLAS building - shifted +1
// Must match VK_NV_cluster_acceleration_structure indirect args layout
// Structure: address first (8 bytes), then count (4 bytes), then padding (4 bytes)
struct BlasFromClasIndirectArg {
  uint64_t clusterAddresses;
  uint clusterCount;
  uint _padding;  // Explicit padding to align to 16 bytes
};

[[vk::binding(2, 0)]] RWStructuredBuffer<BlasFromClasIndirectArg> blasFromClasArgsOut;

[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint instanceIndex = threadIdx.x;
  if (instanceIndex >= cb.numInstances)
    return;

  // Read cluster offset and count for this instance
  // Format: x = offset into CLAS addresses array, y = number of clusters
  uint2 offsetCount = clusterOffsetCounts[instanceIndex];

  // Fill indirect args
  BlasFromClasIndirectArg args;
  args.clusterAddresses = cb.clasAddressesBaseAddress + (sizeof(uint64_t) * offsetCount.x);
  args.clusterCount = offsetCount.y;
  args._padding = 0;

  blasFromClasArgsOut[instanceIndex] = args;
}
