//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "../utils/box3.h"

static const uint32_t kComputeClusterTilingWaves = 4;

struct ComputeClusterTilingParams
{
    uint32_t surfaceStart; //inclusive
    uint32_t surfaceEnd; //exclusive
    uint32_t debugSurfaceIndex;
    uint32_t debugLaneIndex;

    float4x4 matWorldToClip;
    float4 localToWorld[3];  // 3 rows of float4 (row-major 3x4 matrix, 48 bytes)

    float4 cameraPos;  // xyz = camera position, w = padding (C++ float3 is 16 bytes, breaks alignment)

    Box3 aabb;

    uint4 edgeSegments;
    
    uint firstGeometryIndex;
    uint isolationLevel;
    float fineTessellationRate;
    float coarseTessellationRate;

    float2 viewportSize;
    float2 invHiZSize;

    int enableFrustumVisibility;
    int enableBackfaceVisibility;
    int enableHiZVisibility;
    int numHiZLODs;
    
    float globalDisplacementScale;
    uint maxClusters;
    uint maxVertices;
    uint maxClasBlocks;

    uint instanceIndex;  // Instance index for debug logging
    uint pad3;
    uint pad4;
    uint pad5;

    nvrhi::GpuVirtualAddress clasDataBaseAddress;
    nvrhi::GpuVirtualAddress clusterVertexPositionsBaseAddress;
};

#if defined(__cplusplus)
static_assert(sizeof(ComputeClusterTilingParams) % 16 == 0);
#elif defined(TARGET_D3D12)
_Static_assert(sizeof(ComputeClusterTilingParams) % 16 == 0, "Must be 16 byte aligned");
#endif
