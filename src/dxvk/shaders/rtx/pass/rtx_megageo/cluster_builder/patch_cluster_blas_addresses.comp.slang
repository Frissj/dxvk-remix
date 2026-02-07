/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

// Shader to patch cluster BLAS addresses into RTX Remix's instance buffer
// This is needed because cluster BLAS addresses are only available on GPU (in blasPtrsBuffer)
// RTX Remix uses VkAccelerationStructureInstanceKHR which has the BLAS address at offset 56

//!variant patch_cluster_blas_addresses.comp
//!end-variants

#include "rtx/pass/rtx_megageo/cluster_builder/fill_instance_descs_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/patch_cluster_blas_addresses_binding_indices.h"

// VkAccelerationStructureInstanceKHR layout (64 bytes total):
// - transform: VkTransformMatrixKHR (3x4 float matrix = 48 bytes)
// - instanceCustomIndex_mask: uint32 (24-bit index + 8-bit mask)
// - instanceShaderBindingTableRecordOffset_flags: uint32 (24-bit offset + 8-bit flags)
// - accelerationStructureReference: uint64 (BLAS device address at offset 56)
struct VkInstanceRaw
{
    float4 transform0;    // 0-16
    float4 transform1;    // 16-32
    float4 transform2;    // 32-48
    uint instanceCustomIndex_mask;  // 48-52
    uint instanceSBTOffset_flags;   // 52-56
    uint2 blasAddress;    // 56-64 (uint64 as two uint32s)
};

layout(binding = PATCH_CLUSTER_BLAS_ADDRESSES_PARAMS)
ConstantBuffer<PatchClusterBlasAddressParams> cb;

// Mapping buffer: each entry maps RTX Remix instance index -> RTXMG instance index
layout(binding = PATCH_CLUSTER_BLAS_ADDRESSES_MAPPINGS_INPUT)
StructuredBuffer<ClusterInstanceMapping> InMappings;

// BLAS addresses from cluster build (one per RTXMG instance)
layout(binding = PATCH_CLUSTER_BLAS_ADDRESSES_BLAS_ADDRESSES_INPUT)
StructuredBuffer<uint2> InBlasAddresses;  // uint64 as uint2

// RTX Remix instance buffer (VkAccelerationStructureInstanceKHR array)
layout(binding = PATCH_CLUSTER_BLAS_ADDRESSES_INSTANCE_BUFFER_OUTPUT)
RWStructuredBuffer<VkInstanceRaw> InOutInstanceBuffer;

[shader("compute")]
[numthreads(kFillInstanceDescsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint mappingIndex = threadIdx.x;
    if (mappingIndex >= cb.numMappings)
        return;

    // Get the mapping for this cluster instance
    ClusterInstanceMapping mapping = InMappings[mappingIndex];

    // Get the BLAS address from the cluster build output
    uint2 blasAddress = InBlasAddresses[mapping.rtxmgInstanceIndex];

    // If BLAS address is 0 (not built yet, overflow, or cleared), make instance invisible
    // to ray tracing by zeroing the mask. This prevents GPU fault from traversing address 0.
    if (blasAddress.x == 0 && blasAddress.y == 0)
    {
        // Clear mask bits (top 8 bits of instanceCustomIndex_mask) to hide from all ray types
        InOutInstanceBuffer[mapping.remixInstanceIndex].instanceCustomIndex_mask &= 0x00FFFFFFu;
        return;
    }

    // Patch the BLAS address in the instance buffer
    InOutInstanceBuffer[mapping.remixInstanceIndex].blasAddress = blasAddress;
}
