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
  uint numPatchEntries;         // Number of (tlasIndex, blasBufferIndex) pairs to patch
  uint blasAddressesBaseIndex;  // Ring buffer base index for current frame (frameIndex % 4 * maxInstancesPerFrame)
  uint _pad1;
  uint _pad2;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0)]] ConstantBuffer<PatchTlasInstanceParams> cb;

// Patch entry: maps TLAS instance index to blasPtrsBuffer index
struct PatchEntry {
  uint tlasInstanceIndex;      // Index in TLAS instance buffer
  uint blasBufferIndex;        // Index in blasAddresses buffer
};

// Input: Patch entries (TLAS index → blasPtrsBuffer index mapping) - shifted +1
[[vk::binding(1)]] StructuredBuffer<PatchEntry> patchEntries;

// Input: BLAS addresses from cluster extension (sparse - only cluster BLASes) - shifted +1
[[vk::binding(2)]] StructuredBuffer<uint64_t> blasAddresses;

// Output: TLAS instance descriptors to patch - shifted +1
// VkAccelerationStructureInstanceKHR structure:
// - float transform[12] (48 bytes)
// - uint instanceCustomIndex : 24
// - uint mask : 8
// - uint instanceShaderBindingTableRecordOffset : 24
// - uint flags : 8
// - uint64_t accelerationStructureReference (8 bytes) <-- This is what we patch
[[vk::binding(3)]] RWByteAddressBuffer instanceDescsBuffer;

// Offset to accelerationStructureReference within VkAccelerationStructureInstanceKHR
// Structure layout: 48 bytes (transform) + 4 bytes (packed fields) + 4 bytes (packed fields) = 56 bytes offset
static const uint kBlasAddressOffset = 56;

// Size of VkAccelerationStructureInstanceKHR
static const uint kInstanceDescSize = 64;

[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint patchIndex = threadIdx.x;
  if (patchIndex >= cb.numPatchEntries)
    return;

  // Read the patch entry
  PatchEntry entry = patchEntries[patchIndex];

  // CRITICAL: Add ring buffer base index to get correct address for current frame
  // The blasAddresses buffer is ring-buffered (4 copies for multi-frame-in-flight),
  // so we must add the base index for this frame's slot to read from the correct location.
  uint actualBlasIndex = cb.blasAddressesBaseIndex + entry.blasBufferIndex;

  // Read BLAS address from cluster extension's output buffer (ring-buffered)
  uint64_t blasAddress = blasAddresses[actualBlasIndex];

  // Patch the accelerationStructureReference field in the TLAS instance descriptor
  uint byteOffset = entry.tlasInstanceIndex * kInstanceDescSize + kBlasAddressOffset;

  // Write the 64-bit BLAS address as two 32-bit words (Slang RWByteAddressBuffer requirement)
  uint low = uint(blasAddress & 0xFFFFFFFF);
  uint high = uint(blasAddress >> 32);

  instanceDescsBuffer.Store(byteOffset, low);
  instanceDescsBuffer.Store(byteOffset + 4, high);
}
