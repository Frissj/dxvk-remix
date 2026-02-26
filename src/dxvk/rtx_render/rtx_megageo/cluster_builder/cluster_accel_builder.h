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

#include "cluster.h"
#include "cluster_accels.h"
#include "tessellation_counters.h"
#include "copy_cluster_offset_params.h"

#include "../nvrhi_adapter/nvrhi_types.h"
#include "../utils/shader_debug.h"

#include <deque>
#include <memory>
#include <vector>

#include "cluster_lod_builder.h"
#include "meshlet_template_builder.h"

// Forward declarations
namespace dxvk {
  class NvrhiDxvkDevice;
  class RtxContext;
}


class ClusterAccelBuilder
{
public:
    ClusterAccelBuilder(
        nvrhi::DeviceHandle device,
        dxvk::RtxContext* rtxContext);

    // Meshlet-based BuildAccel: CPU LOD select + CPU buffer fill + CLAS instantiate + BLAS build
    // This replaces the GPU tessellation pipeline for meshoptimizer-clusterized meshes.
    struct MeshletInstance {
        uint32_t meshIndex;               // Index into meshletTemplateSets
        const MeshletTemplateSet* templates; // Per-mesh template set
        const ClusterLODData* lodData;     // LOD DAG for cluster selection
        float localToWorld[12];            // 3x4 row-major transform
        uint32_t surfaceId;                // For shading data
    };
    void BuildAccelMeshlet(
        const std::vector<MeshletInstance>& instances,
        const float cameraPos[3],
        float cameraProj,
        float cameraNear,
        float errorThreshold,
        ClusterAccels& accels,
        ClusterStatistics& stats,
        uint32_t frameIndex,
        nvrhi::ICommandList* commandList);

    RTXMGBuffer<ShaderDebugElement>& GetDebugBuffer() { return m_debugBuffer; }

    // Get shader factory for creating additional compute shaders
    donut::engine::ShaderFactory& getShaderFactory() { return m_shaderFactory; }

    // Initialize templates for the meshlet path (per-meshlet templates)
    void EnsureMeshletTemplatesInitialized(
        const std::vector<MeshletInstance>& instances,
        uint32_t maxGeometryIndex,
        nvrhi::ICommandList* commandList);

protected:
    void UpdateMemoryAllocationsMeshlet(ClusterAccels& accels, uint32_t numInstances, uint32_t maxClusters, uint32_t maxVertices, size_t maxClasBytes);

protected:
    nvrhi::DeviceHandle m_device;
    dxvk::RtxContext* m_rtxContext;

    // Donut adapters for compatibility with sample code
    donut::engine::ShaderFactory m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

    RTXMGBuffer<TessellationCounters> m_tessellationCountersBuffer;
    uint32_t m_buildAccelFrameIndex = 0;

    nvrhi::rt::cluster::OperationParams m_createBlasParams;
    nvrhi::rt::cluster::OperationSizeInfo m_createBlasSizeInfo;

    RTXMGBuffer<Cluster> m_clustersBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::IndirectArgs> m_blasFromClasIndirectArgsBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> m_clasIndirectArgDataBuffer;
    RTXMGBuffer<uint3> m_fillClustersDispatchIndirectBuffer;
    RTXMGBuffer<uint2> m_clusterOffsetCountsBuffer;

    uint32_t m_numInstances = 0;
    uint32_t m_instanceCapacity = 0;
    uint32_t m_maxClusters = 0;
    uint32_t m_maxVertices = 0;
    uint64_t m_maxClasBytes = 0;

    // Instance buffer growth headroom
    static constexpr uint32_t kInstanceGrowHeadroom = 32;

    // Deferred buffer release: old buffers kept alive for N frames after resize
    static constexpr uint32_t kDeferredReleaseFrames = 4;
    struct DeferredBufferRelease {
        uint32_t frameIndex;
        std::vector<nvrhi::BufferHandle> buffers;
    };
    std::deque<DeferredBufferRelease> m_deferredReleases;

    RTXMGBuffer<ShaderDebugElement> m_debugBuffer;

    // Current frame index tracking
    uint32_t m_currentFrameIndex = 0;
};
