/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Patch TLAS Instance BLAS Addresses - Compute shader to patch TLAS instance descriptors
*
* This shader patches TLAS instance descriptors with BLAS addresses from the cluster extension.
* This implements NVIDIA's GPU-side patching method from the RTX Mega Geometry SDK,
* avoiding CPU readback of BLAS addresses entirely.
*
* Based on fill_instance_descs.hlsl from NVIDIA RTX Mega Geometry sample.
*/

// Constants
static const uint kThreadsPerGroup = 256;

// Parameters
struct PatchTlasInstanceParams {
  uint numInstances;  // Total number of TLAS instances to process
  uint _pad1;
  uint _pad2;
  uint _pad3;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0, 0)]] ConstantBuffer<PatchTlasInstanceParams> cb;

// Input: BLAS addresses (one per instance, pre-populated with correct addresses)
// Matching NVIDIA RTXMG sample: blasAddresses[instanceIndex] = correct BLAS address
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> blasAddresses;

// Output: TLAS instance descriptors to patch
// VkAccelerationStructureInstanceKHR structure:
// - float transform[12] (48 bytes)
// - uint instanceCustomIndex : 24
// - uint mask : 8
// - uint instanceShaderBindingTableRecordOffset : 24
// - uint flags : 8
// - uint64_t accelerationStructureReference (8 bytes) <-- This is what we patch at offset 56
[[vk::binding(2, 0)]] RWByteAddressBuffer instanceDescsBuffer;

// Offset to accelerationStructureReference within VkAccelerationStructureInstanceKHR
static const uint kBlasAddressOffset = 56;
static const uint kInstanceDescSize = 64;

[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint instanceIndex = threadIdx.x;
  if (instanceIndex >= cb.numInstances)
    return;

  // NVIDIA RTXMG Sample Pattern: Simple and direct
  // Read BLAS address for this instance (pre-populated with correct address)
  uint64_t blasAddress = blasAddresses[instanceIndex];

  // Patch the accelerationStructureReference field in the instance descriptor
  uint byteOffset = instanceIndex * kInstanceDescSize + kBlasAddressOffset;

  // Write the 64-bit BLAS address as two 32-bit words
  uint low = uint(blasAddress & 0xFFFFFFFF);
  uint high = uint(blasAddress >> 32);

  instanceDescsBuffer.Store(byteOffset, low);
  instanceDescsBuffer.Store(byteOffset + 4, high);
}
