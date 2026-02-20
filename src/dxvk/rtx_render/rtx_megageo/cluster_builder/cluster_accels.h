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

#include "../nvrhi_adapter/nvrhi_types.h"
#include "../utils/buffer.h"
#include "tessellator_config.h"

#include <memory>
#include <span>

struct ClusterAccels
{
    RTXMGBuffer<uint8_t> blasBuffer;
    RTXMGBuffer<uint8_t> clasBuffer;
    // Note: scratch buffer is allocated on-demand by nvrhi adapter, not stored here (matching sample)

    RTXMGBuffer<nvrhi::GpuVirtualAddress> clasPtrsBuffer;  // address of each CLAS header in clasBuffer
    RTXMGBuffer<nvrhi::GpuVirtualAddress> blasPtrsBuffer;  // handles in device memory
    RTXMGBuffer<uint32_t> blasSizesBuffer;

    // -------------------------------------------------------------------------
    // Cluster data buffer for shading information
    //
    RTXMGBuffer<ClusterShadingData> clusterShadingDataBuffer;

    // -------------------------------------------------------------------------
    // Vertex Position buffer that we stage into before creating CLASes
    //
    // IMPORTANT: C++ float3 (MathLib) = 16 bytes (SSE __m128 padded), but GPU
    // stride is 12 bytes (Slang float3 with scalar layout). Buffer is allocated
    // with 12-byte stride (see kGpuFloat3Stride in cluster_accel_builder.cpp).
    // The RTXMGBuffer<float3> template param affects Download() element counting
    // but NOT the actual buffer size or GPU layout. Use raw byte readback with
    // 12-byte stride for correct vertex interpretation.
    // Sample uses donut::math::vector<float,3> which IS 12 bytes — no mismatch there.
    RTXMGBuffer<float3> clusterVertexPositionsBuffer;

    // -------------------------------------------------------------------------
    // Vertex Normal buffer (optional - only allocated when vertex normals are enabled)
    // Same float3 stride caveat as clusterVertexPositionsBuffer above.
    //
    RTXMGBuffer<float3> clusterVertexNormalsBuffer;
};

struct ClusterStatistics
{
    struct BufferStatistics
    {
        uint32_t m_numClusters = 0;
        uint32_t m_numTriangles = 0;
        size_t m_blasScratchSize = 0;
        size_t m_blasSize = 0;
        size_t m_vertexBufferSize = 0;
        size_t m_vertexNormalsBufferSize = 0;
        size_t m_clasSize = 0;
        size_t m_clusterDataSize = 0;
    };

    BufferStatistics desired;
    BufferStatistics allocated;

    // Cluster accel sampler statistics
    struct ClusterAccelSamplerStats {
        uint32_t count = 0;
        size_t totalSize = 0;
    };
    ClusterAccelSamplerStats clusterAccelSamplers;
};
