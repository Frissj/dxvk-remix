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

// clang-format off
#include "cluster.h"
#include "cluster_accels.h"
#include "tessellator_config.h"
#include "tessellation_counters.h"
#include "copy_cluster_offset_params.h"
#include "fill_blas_from_clas_args_params.h"

#include "../nvrhi_adapter/nvrhi_types.h"
#include "../utils/shader_debug.h"
#include "../hiz/hiz_buffer_constants.h"

#include <deque>
#include <memory>
#include <vector>
// clang-format on

// Forward declarations
namespace dxvk {
  class NvrhiDxvkDevice;
  class RtxContext;
}

class RTXMGScene;
class SubdivisionSurface;
class ZBuffer;
struct TopologyMap;
struct Instance;

struct TemplateGridDesc
{
    uint32_t xEdges = 0;
    uint32_t yEdges = 0;
    uint32_t indexOffset = 0;
    uint32_t vertexOffset = 0;

    uint32_t getXVerts() const { return xEdges + 1; }
    uint32_t getYVerts() const { return yEdges + 1; }
    uint32_t getNumTriangles() const { return xEdges * yEdges * 2; }
    uint32_t getNumVerts() const { return getXVerts() * getYVerts(); }
};

struct TemplateGrids
{
    typedef uint8_t IndexType;

    std::vector<TemplateGridDesc> descs;
    std::vector<IndexType> indices;
    std::vector<float> vertices;

    uint32_t maxVertices = 0;
    uint32_t maxTriangles = 0;
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
};

enum class ShaderPermutationSurfaceType : uint32_t
{
    PureBSpline,
    RegularBSpline,
    Limit,
    All,
    Count
};

// Permutation definitions
class ComputeClusterTilingPermutation
{
public:
    static constexpr uint32_t kTessModeBitCount = 2;
    static constexpr uint32_t kVisibilityBitCount = 1;
    static constexpr uint32_t kSurfaceTypeBitCount = 2;
    static_assert(uint32_t(ShaderPermutationSurfaceType::Count) <= (1u << kSurfaceTypeBitCount));

    static_assert(uint32_t(TessellatorConfig::AdaptiveTessellationMode::COUNT) <= (1u << kTessModeBitCount));
    static_assert(uint32_t(TessellatorConfig::VisibilityMode::COUNT) <= (1u << kVisibilityBitCount));

    enum BitIndices : uint32_t
    {
        DisplacementMaps,
        FrustumVisibility,
        TessMode,
        VisibilityMode = TessMode + kTessModeBitCount,
        SurfaceTypeStartBit = VisibilityMode + kVisibilityBitCount,
        Count = SurfaceTypeStartBit + kSurfaceTypeBitCount
    };

    static constexpr size_t kCount = 1u << BitIndices::Count;

    ComputeClusterTilingPermutation(bool enableDisplacement,
        bool enableFrustumVisibility,
        TessellatorConfig::AdaptiveTessellationMode tessMode,
        TessellatorConfig::VisibilityMode visMode,
        ShaderPermutationSurfaceType surfaceType)
        : m_bits
        ((enableDisplacement ? (1u << BitIndices::DisplacementMaps) : 0u) 
         | (enableFrustumVisibility ? (1u << BitIndices::FrustumVisibility) : 0u)
         | (uint32_t(tessMode) << BitIndices::TessMode)
         | (uint32_t(visMode) << BitIndices::VisibilityMode)
         | (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit))
    {}

    bool isDisplacementEnabled() const { return m_bits & (1u << BitIndices::DisplacementMaps); }
    bool isFrustumVisibilityEnabled() const { return m_bits & (1u << BitIndices::FrustumVisibility); }

    TessellatorConfig::AdaptiveTessellationMode tessellationMode() const
    {
        constexpr uint32_t kBitMask = (1 << kTessModeBitCount) - 1;
        return TessellatorConfig::AdaptiveTessellationMode((m_bits >> BitIndices::TessMode) & kBitMask);
    }

    TessellatorConfig::VisibilityMode visibilityMode() const
    {
        constexpr uint32_t kBitMask = (1 << kVisibilityBitCount) - 1;
        return TessellatorConfig::VisibilityMode((m_bits >> BitIndices::VisibilityMode) & kBitMask);
    }

    ShaderPermutationSurfaceType surfaceType() const
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        return ShaderPermutationSurfaceType((m_bits >> BitIndices::SurfaceTypeStartBit) & kBitMask);
    }
    void setSurfaceType(ShaderPermutationSurfaceType surfaceType)
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        m_bits &= ~(kBitMask << BitIndices::SurfaceTypeStartBit);
        m_bits |= (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit);
    }

    uint32_t index() const { return m_bits; }

private:
    uint32_t m_bits = 0;
};

class FillClustersPermutation
{
public:
    static constexpr uint32_t kSurfaceTypeBitCount = 2;
    static_assert(uint32_t(ShaderPermutationSurfaceType::Count) <= (1u << kSurfaceTypeBitCount));

    enum BitIndices : uint32_t
    {
        DisplacementMaps = 0,
        VertexNormals,
        SurfaceTypeStartBit,
        Count = SurfaceTypeStartBit + kSurfaceTypeBitCount
    };
    static constexpr size_t kCount = 1u << BitIndices::Count;
    uint32_t index() const { return m_bits; }

    FillClustersPermutation(bool enableDisplacement,
        bool enableVertexNormals,
        ShaderPermutationSurfaceType surfaceType)
        : m_bits((enableDisplacement ? (1u << BitIndices::DisplacementMaps) : 0u)
         | (enableVertexNormals ? (1u << BitIndices::VertexNormals) : 0u)
         | (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit))
    {}

    bool isDisplacementEnabled() const { return m_bits & (1u << BitIndices::DisplacementMaps); }
    bool isVertexNormalsEnabled() const { return m_bits & (1u << BitIndices::VertexNormals); }
    ShaderPermutationSurfaceType surfaceType() const
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        return ShaderPermutationSurfaceType((m_bits >> BitIndices::SurfaceTypeStartBit) & kBitMask);
    }

private:
    uint32_t m_bits = 0;
};


class ClusterAccelBuilder
{
public:
    ClusterAccelBuilder(
        nvrhi::DeviceHandle device,
        dxvk::RtxContext* rtxContext);

    void BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config, 
        ClusterAccels& accels, ClusterStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList);
    
    RTXMGBuffer<ShaderDebugElement>& GetDebugBuffer() { return m_debugBuffer; }

    // Get shader factory for creating additional compute shaders
    donut::engine::ShaderFactory& getShaderFactory() { return m_shaderFactory; }

    // Initialize cluster templates early (before any image views are bound)
    // This MUST be called before updateHiZBuffer to avoid destroying bound resources
    // when the sync Downloads in template init close/reopen the command list
    void EnsureTemplatesInitialized(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList);

    // Diagnostic: dump all key buffer contents to log for debugging garbled geometry
    // WARNING: This does sync GPU readbacks (close/reopen command list). Only call
    // when no image views are bound (same restriction as template init).
    void DumpDiagnosticData(ClusterAccels& accels, nvrhi::ICommandList* commandList);

protected:
    void UpdateMemoryAllocations(ClusterAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches);

    nvrhi::BufferHandle GenerateStructuredClusterTemplateArgs(const TemplateGrids& grids, nvrhi::ICommandList* commandList);
    void InitStructuredClusterTemplates(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList);
    void BuildStructuredCLASes(ClusterAccels& accels, uint32_t maxGeometryCountPerMesh, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList);
    void BuildBlasFromClas(ClusterAccels& accels, const Instance* instances, size_t instanceCount, nvrhi::ICommandList* commandList);

    void FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList);
    void FillInstanceClusters(const RTXMGScene& scene, ClusterAccels& accels, nvrhi::ICommandList* commandList);
    void FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets, 
        nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList);

    // Calculates the cluster layout based off of various visibility metrics
    // A cluster tiling is the number of clusters and cluster sizes that are used to cover a surface.
    // Outputs cluster headers, shading data, and addresses
    void ComputeInstanceClusterTiling(ClusterAccels& accels,
        const RTXMGScene& scene,
        uint32_t instanceIndex,
        uint32_t surfaceOffset,
        uint32_t surfaceCount,
        const nvrhi::BufferRange& tessCounterRange,
        nvrhi::ICommandList* commandList);
    void CopyClusterOffset(uint32_t instanceIndex, ClusterDispatchType dispatchType,
        const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList);

protected:
    TessellatorConfig m_tessellatorConfig;
    nvrhi::DeviceHandle m_device;
    dxvk::RtxContext* m_rtxContext;

    // Donut adapters for compatibility with sample code
    donut::engine::ShaderFactory m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    // Descriptor table for bindless resources (displacement textures, etc.)
    // This is a binding set created from the bindless layout that satisfies
    // pipeline binding requirements even when no actual bindless resources are used.
    nvrhi::BindingSetHandle m_descriptorTable;
    
    RTXMGBuffer<TessellationCounters> m_tessellationCountersBuffer;
    uint32_t m_buildAccelFrameIndex = 0; // substition for frameIndex since we don't necessarily build every frame

    // Pipeline descs
    nvrhi::BindingLayoutHandle m_bindlessBL;

    nvrhi::BindingLayoutHandle m_fillInstantiateTemplateBL;
    nvrhi::ComputePipelineHandle m_fillInstantiateTemplatePSO;

    nvrhi::BindingLayoutHandle m_fillBlasFromClasArgsBL;
    nvrhi::ComputePipelineHandle m_fillBlasFromClasArgsPSO;

    nvrhi::BindingLayoutHandle m_copyClusterOffsetBL;
    nvrhi::ComputePipelineHandle m_copyClusterOffsetPSO;

    nvrhi::BindingLayoutHandle m_fillClustersBL;
    nvrhi::ComputePipelineHandle m_fillClustersPSOs[FillClustersPermutation::kCount];
    nvrhi::ComputePipelineHandle m_fillClustersTexcoordsPSO;

    nvrhi::BindingLayoutHandle m_computeClusterTilingBL;
    nvrhi::BindingLayoutHandle m_computeClusterTilingHizBL;  // Separate HiZ binding layout (space 1)
    nvrhi::BindingSetHandle m_dummyHizBindingSet;           // Dummy HiZ binding set when zbuffer is null
    nvrhi::BindingSetHandle m_cachedHizBindingSet;          // Cached HiZ binding set when zbuffer available
    const ZBuffer* m_cachedHizBuffer = nullptr;              // Track which zbuffer the cached set was created for
    uint32_t m_cachedHizFrame = UINT32_MAX;                  // Track which frame the cached set was created for
    nvrhi::ComputePipelineHandle m_computeClusterTilingPSOs[ComputeClusterTilingPermutation::kCount];

    // Dummy HiZ textures for when HiZ culling is disabled
    // The shader expects HIZ_MAX_LODS textures, so we need to bind something even when HiZ is disabled
    nvrhi::TextureHandle m_dummyHiZTextures[HIZ_MAX_LODS];
    bool m_dummyHiZTexturesInitialized = false;
    bool m_hizInitialized = false;  // Track if real HiZ buffer has been cleared/initialized

    RTXMGBuffer<uint3> m_fillClustersDispatchIndirectBuffer; // number of thread groups per each instance
    RTXMGBuffer<uint2> m_clusterOffsetCountsBuffer; // offset+count per each instance
    
    nvrhi::rt::cluster::OperationParams m_createBlasParams;
    nvrhi::rt::cluster::OperationSizeInfo m_createBlasSizeInfo;

    // Per input surface patch 
    RTXMGBuffer<GridSampler> m_gridSamplersBuffer;
    
    RTXMGBuffer<Cluster> m_clustersBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::IndirectArgs> m_blasFromClasIndirectArgsBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> m_clasIndirectArgDataBuffer;

    uint32_t m_numInstances = 0;        // Actual current instance count (for build operations)
    uint32_t m_instanceCapacity = 0;    // Allocated buffer capacity (>= m_numInstances)
    uint32_t m_sceneSubdPatches = 0;
    uint32_t m_gridSamplersCapacity = 0; // Current allocated capacity for gridSamplersBuffer
    uint32_t m_gridSamplersShrinkCounter = 0;   // Consecutive frames below shrink threshold
    uint32_t m_gridSamplersResizeCooldown = 0;  // Frames remaining in cooldown after resize
    uint32_t m_maxClusters = 0;
    uint32_t m_maxVertices = 0;
    uint64_t m_maxClasBytes = 0;

    // Smart instance buffer scaling (hysteresis + sustained check + cooldown)
    // Inspired by dynamic worker pool pattern: grow fast, shrink slow, prevent oscillation
    static constexpr uint32_t kInstanceGrowHeadroom = 32;       // Extra capacity when growing (prevents micro-reallocations)
    static constexpr float kInstanceShrinkThreshold = 0.4f;     // Only consider shrinking when using < 40% of capacity
    static constexpr uint32_t kInstanceShrinkSustainedFrames = 300; // ~5 seconds at 60fps of sustained low usage before shrinking
    static constexpr uint32_t kInstanceResizeCooldownFrames = 120;  // ~2 seconds cooldown after any resize
    uint32_t m_instanceShrinkCounter = 0;   // Consecutive frames below shrink threshold
    uint32_t m_instanceResizeCooldown = 0;  // Frames remaining in cooldown after a resize

    // Deferred buffer release: old buffers kept alive for N frames after resize
    // so in-flight GPU work can finish before memory is freed.
    // This replaces flushCommandList+waitForIdle, avoiding mid-frame command buffer splits.
    static constexpr uint32_t kDeferredReleaseFrames = 4;
    struct DeferredBufferRelease {
        uint32_t frameIndex;
        std::vector<nvrhi::BufferHandle> buffers;
    };
    std::deque<DeferredBufferRelease> m_deferredReleases;

    struct TemplateBuffers
    {
        uint32_t                                maxGeometryCountPerMesh = 0;
        uint32_t                                quantNBits = 0;
        nvrhi::BufferHandle                     dataBuffer; // Holds the template data
        RTXMGBuffer<nvrhi::GpuVirtualAddress>   addressesBuffer; // Array of addresses within dataBuffer, one per template
        RTXMGBuffer<uint32_t>                   instantiationSizesBuffer; // Size to instanstiate each template
        RTXMGBuffer<uint32_t>                   sizesBuffer; // CRITICAL: Keep alive - GPU caches address references (ClusterTemplateSizes)
        std::vector<uint32_t>                   instantiationSizes;
        std::vector<nvrhi::GpuVirtualAddress>   addresses; // CPU-side copy of template addresses for CPU fill
        // CRITICAL: Keep index/vertex buffers alive - their GPU addresses are referenced by cluster template args
        nvrhi::BufferHandle                     indexBuffer;
        nvrhi::BufferHandle                     vertexBuffer;
    };
    TemplateBuffers m_templateBuffers; // Buffers used to Create templates. They are created once but need to be persistent throughout the app's run time.
    
    nvrhi::BufferHandle m_fillInstantiateTemplateArgsParamsBuffer; // constant buffer for filling indirect args for getting template sizes
    nvrhi::BufferHandle m_computeClusterTilingParamsBuffer; // constant buffer for compute cluster tiling
    nvrhi::BufferHandle m_copyClusterOffsetParamsBuffer; // constant buffer for copying cluster offsets
    nvrhi::BufferHandle m_fillClustersParamsBuffer; // constant buffer for fill clusters
    nvrhi::BufferHandle m_fillBlasFromClasArgsParamsBuffer; // constant buffer for filling indirect args to initialize blas from clas

    RTXMGBuffer<ShaderDebugElement> m_debugBuffer;

    // Current frame index tracking
    uint32_t m_currentFrameIndex = 0;
};
