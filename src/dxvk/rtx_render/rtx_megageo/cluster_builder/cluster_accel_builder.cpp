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

// Enable chrono timing for performance profiling (set to 1 to enable)
#define RTXMG_CHRONO_TIMING 0

// Helper to align buffer sizes to 4 bytes (required for Vulkan vkCmdUpdateBuffer)
inline size_t alignBufferSize(size_t size) {
    return (size + 3) & ~3;
}

// STL includes
#include <algorithm>
#include <limits>
#include <chrono>
#include <set>

// RTX Remix includes
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"
#include "../nvrhi_adapter/nvrhi_types.h"
#include "../nvrhi_adapter/nvrhi_dxvk_device.h"
#include "../nvrhi_adapter/nvrhi_dxvk_command_list.h"
#include "../nvrhi_adapter/nvrhi_dxvk_buffer.h"
#include "../nvrhi_adapter/nvrhi_dxvk_texture.h"
#include "../../rtx_shader_manager.h"
#include "../../rtx_context.h"

// RTX MG shader includes
#include <rtx_shaders/copy_cluster_offset.h>
#include <rtx_shaders/fill_instantiate_template_args.h>
#include <rtx_shaders/fill_blas_from_clas_args.h>
#include <rtx_shaders/fill_instance_descs.h>

// RTX MG shader binding indices
#include <rtx/pass/rtx_megageo/cluster_builder/copy_cluster_offset_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_clusters_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_instantiate_template_args_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_blas_from_clas_args_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_instance_descs_binding_indices.h>

#include <map>
#include <fstream>

// RTX MG includes - updated paths
#include "cluster_accels.h"
#include "cluster_accel_builder.h"
#include "fill_clusters_params.h"
#include "copy_cluster_offset_params.h"
#include "fill_blas_from_clas_args_params.h"
#include "fill_instantiate_template_args_params.h"
#include "compute_cluster_tiling_params.h"
#include "tessellation_counters.h"
#include "tessellator_config.h"
#include "../scene/rtxmg_scene.h"
#include "../scene/instance.h"
#include "../scene/camera.h"

using namespace dxvk;
#include "tessellator_constants.h"

#include "../utils/buffer.h"
#include "../hiz/zbuffer.h"
#include "../hiz/hiz_buffer_constants.h"
#include "../profiler/profiler_stub.h"  // Lightweight profiler stub for RTX Remix

#include "../subdivision/subdivision_surface.h"
#include "../subdivision/shape.h"
#include "../subdivision/topology_map.h"

#include "../rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

using namespace donut;
using namespace nvrhi::rt;

// Global debug flags shared between BuildAccel (clear) and FillInstanceClusters (readback)
bool g_megageoDbgGotData = false;
bool g_megageoDbgGotLimitData = false;

constexpr uint32_t kNumTemplates = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments;
constexpr uint32_t kClusterMaxTriangles = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments * 2;
constexpr uint32_t kClusterMaxVertices = (kMaxClusterEdgeSegments + 1) * (kMaxClusterEdgeSegments + 1);
constexpr uint32_t kFrameCount = 4;

ClusterAccelBuilder::ClusterAccelBuilder(
    nvrhi::DeviceHandle device,
    dxvk::RtxContext* rtxContext)
    : m_device(device)
    , m_rtxContext(rtxContext)
    , m_shaderFactory(rtxContext)
    , m_commonPasses(std::make_shared<donut::engine::CommonRenderPasses>(device))
{
    // CRITICAL: tessellation counters buffer is used as srcInfosCount for CLAS operations,
    // which requires VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    {
        size_t byteSize = kFrameCount * sizeof(TessellationCounters);
        size_t alignedByteSize = (byteSize + 3) & ~3;  // Round up to multiple of 4
        nvrhi::BufferDesc tessCounterDesc = {
            .byteSize = alignedByteSize,
            .debugName = "tessellation counters",
            .structStride = sizeof(TessellationCounters),
            .canHaveUAVs = true,
            .canHaveTypedViews = true,
            .canHaveRawViews = true,
            .isAccelStructBuildInput = true,  // Required for srcInfosCount in CLAS operations
            .initialState = nvrhi::ResourceStates::UnorderedAccess,
            .keepInitialState = true
        };
        m_tessellationCountersBuffer.Create(tessCounterDesc, m_device.Get());
    }
    m_debugBuffer.Create(512, "ClusterAccelDebug", m_device.Get());

    //////////////////////////////////////////////////
    // Parameter buffers for shaders
    //////////////////////////////////////////////////
    m_fillInstantiateTemplateArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillInstantiateTemplateArgsParams), "FillInstantiateTemplateArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    // Per-instance CBs: NON-volatile with vkCmdUpdateBuffer.
    // The sample's NVRHI Vulkan backend uses VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC for volatile CBs,
    // with per-slot version tracking and fence-based slot recycling. Our DXVK adapter's volatile path
    // uses a simple ring buffer (host memcpy, no GPU commands) that wraps after maxVersions writes.
    // With 200+ instances per frame, the 16-slot ring buffer wraps and the GPU reads overwritten data.
    // Non-volatile CBs use vkCmdUpdateBuffer (GPU command) which is properly serialized in the
    // command buffer — each dispatch reads the correct data regardless of instance count.
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = sizeof(ComputeClusterTilingParams);
        desc.debugName = "ComputeClusterTilingParams";
        desc.isConstantBuffer = true;
        desc.isVolatile = false;
        desc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        desc.keepInitialState = true;
        m_computeClusterTilingParamsBuffer = m_device->createBuffer(desc);
    }
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = sizeof(FillClustersParams);
        desc.debugName = "FillClustersParams";
        desc.isConstantBuffer = true;
        desc.isVolatile = false;
        desc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        desc.keepInitialState = true;
        m_fillClustersParamsBuffer = m_device->createBuffer(desc);
    }

    m_fillBlasFromClasArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillBlasFromClasArgsParams), "FillBlasFromClasArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    // m_copyClusterOffsetParamsBuffer: created as volatile in UpdateMemoryAllocations (matching sample).
    // Volatile CB uses a ring buffer so each CopyClusterOffset dispatch per instance gets its own version,
    // preventing data races when the same buffer is written+read across multiple dispatches per frame.

    //////////////////////////////////////////////////
    // Create common bindless binding layout and descriptor table
    //////////////////////////////////////////////////
    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv;
    m_bindlessBL = m_device->createBindlessLayout(bindlessLayoutDesc);

    // Create descriptor table (empty binding set) for the bindless layout.
    // This satisfies pipeline binding requirements even when no displacement maps are used.
    // When displacement is enabled, this would need to be populated with texture descriptors.
    m_descriptorTable = m_device->createDescriptorTable(m_bindlessBL);

    //////////////////////////////////////////////////
    // Create dummy HiZ textures for when HiZ culling is disabled
    // The shader expects HIZ_MAX_LODS textures at binding set 1, so we need
    // to bind valid textures even when HiZ is disabled to avoid validation errors
    //////////////////////////////////////////////////
    nvrhi::TextureDesc dummyHiZDesc;
    dummyHiZDesc.width = 1;
    dummyHiZDesc.height = 1;
    dummyHiZDesc.format = nvrhi::Format::R32_FLOAT;
    dummyHiZDesc.isUAV = true;  // Must be UAV so image is in GENERAL layout (matches bindHiZDescriptorSet expectations)
    dummyHiZDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    dummyHiZDesc.keepInitialState = true;

    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
    {
        std::string debugName = "DummyHiZ_Level_" + std::to_string(i);
        dummyHiZDesc.debugName = debugName.c_str();
        m_dummyHiZTextures[i] = m_device->createTexture(dummyHiZDesc);
    }
}

// Must match shader defines in compute_cluster_tiling.hlsl
inline char const* toString(TessellatorConfig::AdaptiveTessellationMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::AdaptiveTessellationMode::UNIFORM: return "TESS_MODE_UNIFORM";
    case TessellatorConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH: return "TESS_MODE_WORLD_SPACE_EDGE_LENGTH";
    case TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION: return "TESS_MODE_SPHERICAL_PROJECTION";
    default: return "UNKNOWN";
    }
}

inline char const* toString(TessellatorConfig::VisibilityMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::VisibilityMode::VIS_SURFACE: return "VIS_MODE_SURFACE";
    case TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES: return "VIS_MODE_LIMIT_EDGES";
    default: return "UNKNOWN";
    }
}

constexpr auto kSurfaceTypeDefines = std::to_array<const char*>(
{
    "SURFACE_TYPE_PUREBSPLINE",
    "SURFACE_TYPE_REGULARBSPLINE",
    "SURFACE_TYPE_LIMIT",
    "SURFACE_TYPE_ALL"
});
static_assert(kSurfaceTypeDefines.size() == size_t(ShaderPermutationSurfaceType::Count));

inline char const* toString(ShaderPermutationSurfaceType surfaceType)
{
    return kSurfaceTypeDefines[uint32_t(surfaceType)];
}

void ClusterAccelBuilder::FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList)
{
    FillInstantiateTemplateArgsParams params = {};
    params.numTemplates = numTemplates;
    params.pad = uint3();

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillInstantiateTemplateArgs");
    commandList->writeBuffer(m_fillInstantiateTemplateArgsParamsBuffer, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, templateAddresses))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillInstantiateTemplateArgsParamsBuffer));

    // Create layout once, then reuse for all binding sets (avoids CreateBindingSetAndLayout overhead)
    if (!m_fillInstantiateTemplateBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
        m_fillInstantiateTemplateBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillInstantiateTemplateBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for fill_instantiate_template_args.hlsl");
    }

    if (!m_fillInstantiateTemplatePSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_instantiate_template_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillInstantiateTemplateBL);

        m_fillInstantiateTemplatePSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillInstantiateTemplatePSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numTemplates, kFillInstantiateTemplateArgsThreads), 1, 1);
}

void ClusterAccelBuilder::FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets,
    nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList)
{
    FillBlasFromClasArgsParams params = {};
    params.clasAddressesBaseAddress = clasPtrsBaseAddress;
    params.numInstances = numInstances;

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillBlasFromClasArgs");
    commandList->writeBuffer(m_fillBlasFromClasArgsParamsBuffer, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, clusterOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillBlasFromClasArgsParamsBuffer));

    // Create layout once, then reuse for all binding sets
    if (!m_fillBlasFromClasArgsBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
        m_fillBlasFromClasArgsBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillBlasFromClasArgsBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for fill_blas_from_clas_args.hlsl");
    }

    if (!m_fillBlasFromClasArgsPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_blas_from_clas_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillBlasFromClasArgsBL);

        m_fillBlasFromClasArgsPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillBlasFromClasArgsPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numInstances, kFillBlasFromClasArgsThreads), 1, 1);
}

static TemplateGrids GenerateTemplateGrids()
{
    TemplateGrids result;

    // Offsets per template
    result.descs.resize(kNumTemplates);
    result.indices.reserve(kNumTemplates * kClusterMaxTriangles * 3);
    result.vertices.reserve(kNumTemplates * kClusterMaxVertices * 3);

    // Generate cluster topologies for 11x11 grid
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        assert(i % kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());
        assert(i / kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());

        TemplateGridDesc& gridDesc = result.descs[i];
        gridDesc =
        {
            .xEdges = i % kMaxClusterEdgeSegments + 1,
            .yEdges = i / kMaxClusterEdgeSegments + 1,
            .indexOffset = static_cast<uint32_t>(result.indices.size() * sizeof(result.indices[0])),
            .vertexOffset = static_cast<uint32_t>(result.vertices.size() * sizeof(result.vertices[0]))
        };

        // x, y = lower - left vertex of quad
        // s: 0 is the first triangle, (left vertical edge), 1 is the second triangle (right vertical edge)
        auto TriIndices = [&gridDesc](uint32_t x, uint32_t y, uint32_t s)->std::array<uint32_t, 3>
        {
            uint32_t vs = gridDesc.getXVerts();  // vertex stride  (same as xVerts)
            uint32_t vid = y * vs + x;          // lower-left vertex id
            bool diag03 = ((x & 1) == (y & 1)); // is this triangle a 0-3 diagonal (true) or a 1-2 diagonal (false)

            assert(vid + vs + 1 < std::numeric_limits<TemplateGrids::IndexType>::max());

            // Example output for (xEdges = 3, yEdges = 1, x = {0..3}, y = 0)
            //      4_____5_____6_____7
            //      |    /|\    |    /|
            //      | a / | \ d | e / |
            //      |  /  |  \  |  /  |
            //      | / b | c \ | / f |
            //      |/____|____\|/____|
            //      0     1     2     3
            //
            //    (x,y,s)
            // a  (0,0,0) diag03 = (0, 5, 4) 
            // b  (0,0,1) diag03 = (0, 1, 5)
            // c  (1,0,0)        = (1, 2, 5)
            // d  (1,0,1)        = (2, 6, 5)
            // e  (2,0,0) diag03 = (2, 7, 6)
            // f  (2,0,1) diag03 = (2, 3, 7)

            if (diag03)
            {
                if (s == 0) return { vid, vid + 1 + vs, vid + vs };
                else        return { vid, vid + 1     , vid + 1 + vs };
            }
            else
            {
                if (s == 0) return { vid    , vid + 1     , vid + vs };
                else        return { vid + 1u, vid + 1 + vs, vid + vs };
            }
        };

        float xScale = 1.0f / gridDesc.xEdges;
        float yScale = 1.0f / gridDesc.yEdges;

        uint32_t xVerts = gridDesc.getXVerts();
        uint32_t yVerts = gridDesc.getYVerts();

        for (uint32_t y = 0; y < yVerts; y++)
        {
            for (uint32_t x = 0; x < xVerts; x++)
            {
                // Add triangles to index buffer
                if (x < gridDesc.xEdges && y < gridDesc.yEdges)
                {
                    for (uint32_t s = 0; s < 2; s++)
                    {
                        std::array<uint32_t, 3> triIndices = TriIndices(x, y, s);
                        std::transform(triIndices.begin(), triIndices.end(), std::back_inserter(result.indices), [](uint32_t e)
                            {
                                assert(e < std::numeric_limits<TemplateGrids::IndexType>::max());
                                return static_cast<TemplateGrids::IndexType>(e);
                            });
                    }
                }

                // Add verts
                result.vertices.push_back(x * xScale);
                result.vertices.push_back(y * yScale);
                result.vertices.push_back(0.0f);
            }
        }

        result.maxTriangles = std::max(result.maxTriangles, gridDesc.getNumTriangles());
        result.totalTriangles += gridDesc.getNumTriangles();

        result.maxVertices = std::max(result.maxVertices, gridDesc.getNumVerts());
        result.totalVertices += gridDesc.getNumVerts();
    }

    assert(result.maxVertices == kClusterMaxVertices);
    assert(result.maxTriangles == kClusterMaxTriangles);

    return result;
}

nvrhi::BufferHandle ClusterAccelBuilder::GenerateStructuredClusterTemplateArgs(const TemplateGrids &grids, nvrhi::ICommandList* commandList)
{
    // Align buffer size to 4 bytes for Vulkan vkCmdUpdateBuffer compatibility
    size_t indexDataSize = grids.indices.size() * sizeof(grids.indices[0]);
    nvrhi::BufferDesc indexBufferDesc = {
        .byteSize = alignBufferSize(indexDataSize),
        .debugName = "ClusterTemplateIndices",
        .structStride = sizeof(grids.indices[0]),
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };

    nvrhi::BufferHandle indexBuffer = CreateBuffer(indexBufferDesc, m_device.Get());
    if (grids.indices.size() > 0)
    {
        // writeBuffer uploads data to indexBuffer (use original data size, buffer is aligned)
        commandList->writeBuffer(indexBuffer, grids.indices.data(), indexDataSize);
    }
    // CRITICAL: Store in m_templateBuffers to keep alive - GPU addresses in cluster args reference this buffer
    m_templateBuffers.indexBuffer = indexBuffer;

    nvrhi::BufferDesc vertexBufferDesc = {
        .byteSize = grids.vertices.size() * sizeof(grids.vertices[0]),
        .debugName = "ClusterTemplateVertices",
        .format = nvrhi::Format::RGB32_FLOAT,
        .isVertexBuffer = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle vertexBuffer = CreateBuffer(vertexBufferDesc, m_device.Get());
    if (grids.vertices.size() > 0)
    {
        // writeBuffer uploads data to vertexBuffer
        commandList->writeBuffer(vertexBuffer, grids.vertices.data(), grids.vertices.size() * sizeof(grids.vertices[0]));
    }
    // CRITICAL: Store in m_templateBuffers to keep alive - GPU addresses in cluster args reference this buffer
    m_templateBuffers.vertexBuffer = vertexBuffer;

    nvrhi::GpuVirtualAddress indexBufferAddress = indexBuffer->getGpuVirtualAddress();
    nvrhi::GpuVirtualAddress vertexBufferAddress = vertexBuffer->getGpuVirtualAddress();

    RTXMG_LOG(str::format("RTX MegaGeo: Base indexBufferAddress=", std::hex, indexBufferAddress,
                             " vertexBufferAddress=", std::hex, vertexBufferAddress));
    if (indexBufferAddress == 0) {
        Logger::err("RTX MegaGeo: indexBufferAddress is NULL!");
    }
    if (vertexBufferAddress == 0) {
        Logger::err("RTX MegaGeo: vertexBufferAddress is NULL!");
    }
    RTXMG_LOG(str::format("RTX MegaGeo: indexBuffer desc: byteSize=", indexBuffer->getDesc().byteSize,
                             " structStride=", indexBuffer->getDesc().structStride,
                             " isAccelStructBuildInput=", indexBuffer->getDesc().isAccelStructBuildInput));
    RTXMG_LOG(str::format("RTX MegaGeo: vertexBuffer desc: byteSize=", vertexBuffer->getDesc().byteSize,
                             " structStride=", vertexBuffer->getDesc().structStride,
                             " isVertexBuffer=", vertexBuffer->getDesc().isVertexBuffer,
                             " isAccelStructBuildInput=", vertexBuffer->getDesc().isAccelStructBuildInput));

    uint32_t indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat32bit);
    switch (sizeof(TemplateGrids::IndexType))
    {
    case 1: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat8bit); break;
    case 2: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat16bit); break;
    case 4: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat32bit); break;
    default: assert(false);
    }

    // Use the correct NVRHI cluster::IndirectTriangleTemplateArgs structure
    // This matches the Vulkan VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV
    std::vector<cluster::IndirectTriangleTemplateArgs> createTemplateArgData(grids.descs.size());
    for (uint32_t i = 0; i < createTemplateArgData.size(); i++)
    {
        const TemplateGridDesc& grid = grids.descs[i];

        // Zero-initialize unused bit fields
        createTemplateArgData[i] = { };
        createTemplateArgData[i] = cluster::IndirectTriangleTemplateArgs
        {
            .clusterId = 0,
            .clusterFlags = 0,
            .triangleCount = grid.getNumTriangles(),
            .vertexCount = grid.getNumVerts(),
            .positionTruncateBitCount = 0,
            .indexFormat = indexFormat,
            .opacityMicromapIndexFormat = 0,
            .baseGeometryIndexAndFlags = 0,
            .indexBufferStride = static_cast<uint16_t>(sizeof(grids.indices[0])),
            .vertexBufferStride = static_cast<uint16_t>(sizeof(grids.vertices[0]) * 3),
            .geometryIndexAndFlagsBufferStride = 0,
            .opacityMicromapIndexBufferStride = 0,
            .indexBuffer = indexBufferAddress + grid.indexOffset,
            .vertexBuffer = vertexBufferAddress + grid.vertexOffset,
            .geometryIndexAndFlagsBuffer = 0,
            .opacityMicromapArray = 0,
            .opacityMicromapIndexBuffer = 0,
            .instantiationBoundingBoxLimit = 0
        };

        // DEBUG: Log first few indirect args
        if (i < 3) {
            RTXMG_LOG(str::format("RTX MegaGeo: Template[", i, "] triCount=", createTemplateArgData[i].triangleCount,
                                     " vertCount=", createTemplateArgData[i].vertexCount, " indexOffset=", grid.indexOffset,
                                     " vertexOffset=", grid.vertexOffset));
            RTXMG_LOG(str::format("RTX MegaGeo: Template[", i, "] indexBuffer=", createTemplateArgData[i].indexBuffer,
                                     " vertexBuffer=", createTemplateArgData[i].vertexBuffer));
        }
    }

    nvrhi::BufferDesc clusterTemplateArgsDesc =
    {
        .byteSize = createTemplateArgData.size() * sizeof(createTemplateArgData[0]),
        .debugName = "ClusterTemplateArgs",
        .structStride = sizeof(createTemplateArgData[0]),
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    return CreateAndUploadBuffer(createTemplateArgData, clusterTemplateArgsDesc, commandList);
}

void ClusterAccelBuilder::InitStructuredClusterTemplates(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList)
{
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates called, maxGeometryCountPerMesh=", maxGeometryCountPerMesh));
    RTXMG_LOG(str::format("RTX MegaGeo: Current buffer state: indexBuffer=", (void*)m_templateBuffers.indexBuffer.Get(),
                             " vertexBuffer=", (void*)m_templateBuffers.vertexBuffer.Get(),
                             " dataBuffer=", (void*)m_templateBuffers.dataBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: Template settings: stored maxGeo=", m_templateBuffers.maxGeometryCountPerMesh,
                             " stored quantNBits=", m_templateBuffers.quantNBits,
                             " config quantNBits=", m_tessellatorConfig.quantNBits));

    // maxGeometryIndex must match between template creation and CLAS instantiation.
    // CRITICAL: Use maxGeometryCountPerMesh (the number of surfaces per mesh), NOT m_maxClusters.
    // The compute shader writes geometryIndexOffset = localGeometryIndex (= t_SurfaceToGeometryIndex[iSurface]),
    // which is a small per-mesh surface index (0..numSurfaces-1), NOT the global cluster index.
    // Declaring maxGeometryIndex = m_maxClusters-1 tells the driver to support 17-bit geometry indices
    // while only 6-bit indices are actually written, causing CLAS internal layout mismatches.
    uint32_t maxGeometryIndex = maxGeometryCountPerMesh > 0 ? maxGeometryCountPerMesh - 1 : 0;

    Logger::warn(str::format("RTX MegaGeo: InitStructuredClusterTemplates GEOMETRY INDEX CHECK:"
        " maxGeometryCountPerMesh(param)=", maxGeometryCountPerMesh,
        " maxGeometryIndex(used)=", maxGeometryIndex,
        " (FIXED: now uses maxGeometryCountPerMesh-1, not m_maxClusters-1)"));

    // only initialize if maxGeometryIndex or quantNBits changes
    if (m_templateBuffers.dataBuffer.Get() != 0 &&
        m_templateBuffers.maxGeometryCountPerMesh == maxGeometryIndex &&
        m_templateBuffers.quantNBits == m_tessellatorConfig.quantNBits) {
        RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - early return, templates already initialized");
        return;
    }
    RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - building new templates");

    nvrhi::utils::ScopedMarker marker(commandList, "InitStructuredClusterTemplates");
    m_templateBuffers.maxGeometryCountPerMesh = maxGeometryIndex;
    m_templateBuffers.quantNBits = m_tessellatorConfig.quantNBits;

    TemplateGrids grids = GenerateTemplateGrids();
    
    // First compute the size of each template so we can build the address buffer
    // this will also act as the settings for further operations below.
    cluster::OperationParams operationParams =
    {
        .maxArgCount = kNumTemplates,
        .type = cluster::OperationType::ClasBuildTemplates,
        .mode = cluster::OperationMode::GetSizes,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxGeometryIndex = maxGeometryIndex,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = grids.totalTriangles,
            .maxTotalVertexCount = grids.totalVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };
    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(operationParams);

    nvrhi::BufferHandle clusterTemplateArgsBuffer = GenerateStructuredClusterTemplateArgs(grids, commandList);

    // CRITICAL: The template build reads indexBuffer and vertexBuffer via device address
    // (embedded in clusterTemplateArgsBuffer). writeBuffer() uploads data via DXVK's
    // updateBuffer() which records in m_execBarriers, but those barriers are only flushed
    // by DXVK's own dispatch/draw commands. Our vkCmdBuildClusterAccelerationStructureIndirectNV
    // call bypasses DXVK entirely, so the pending transfer barriers never get flushed.
    // Without these explicit barriers, the template build can read stale/uninitialized data
    // from the index/vertex buffers, producing templates with wrong triangle connectivity.
    commandList->bufferBarrier(m_templateBuffers.indexBuffer.Get(), nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(m_templateBuffers.vertexBuffer.Get(), nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::AccelStructBuildInput);

    // DIAG: Readback template index/vertex buffers AFTER barriers to confirm data survived upload.
    // If data mismatches CPU-side grids, the barrier or writeBuffer is not working correctly.
    {
        // Readback index buffer
        size_t indexBytes = grids.indices.size() * sizeof(grids.indices[0]);
        auto indexBuf = m_templateBuffers.indexBuffer;
        // Use Download-style readback via staging buffer
        nvrhi::BufferDesc stagingDesc;
        stagingDesc.byteSize = indexBytes;
        stagingDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        stagingDesc.debugName = "TemplateIndexReadback";
        auto indexStaging = m_device->createBuffer(stagingDesc);

        nvrhi::BufferDesc vtxStagingDesc;
        size_t vtxBytes = grids.vertices.size() * sizeof(grids.vertices[0]);
        vtxStagingDesc.byteSize = vtxBytes;
        vtxStagingDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        vtxStagingDesc.debugName = "TemplateVertexReadback";
        auto vtxStaging = m_device->createBuffer(vtxStagingDesc);

        commandList->copyBuffer(indexStaging.Get(), 0, indexBuf.Get(), 0, indexBytes);
        commandList->copyBuffer(vtxStaging.Get(), 0, m_templateBuffers.vertexBuffer.Get(), 0, vtxBytes);

        // Sync readback
        commandList->close();
        m_device->executeCommandList(commandList);
        m_device->waitForIdle();

        // Compare index data
        void* idxMapped = m_device->mapBuffer(indexStaging.Get(), nvrhi::CpuAccessMode::Read);
        const uint8_t* gpuIndices = reinterpret_cast<const uint8_t*>(idxMapped);
        uint32_t idxMismatches = 0;
        for (size_t i = 0; i < grids.indices.size(); ++i) {
            if (gpuIndices[i] != grids.indices[i]) idxMismatches++;
        }
        // Compute simple checksum
        uint32_t cpuIdxSum = 0, gpuIdxSum = 0;
        for (size_t i = 0; i < grids.indices.size(); ++i) {
            cpuIdxSum += grids.indices[i];
            gpuIdxSum += gpuIndices[i];
        }
        Logger::warn(str::format("DIAG TEMPLATE-INDEX-READBACK: count=", grids.indices.size(),
            " mismatches=", idxMismatches, " cpuSum=", cpuIdxSum, " gpuSum=", gpuIdxSum));

        // Log first 30 indices for manual inspection
        std::string idxStr = "  CPU indices: ";
        std::string gpuStr = "  GPU indices: ";
        for (size_t i = 0; i < std::min(grids.indices.size(), (size_t)30); ++i) {
            idxStr += str::format((uint32_t)grids.indices[i], " ");
            gpuStr += str::format((uint32_t)gpuIndices[i], " ");
        }
        Logger::warn(idxStr);
        Logger::warn(gpuStr);
        m_device->unmapBuffer(indexStaging.Get());

        // Compare vertex data
        void* vtxMapped = m_device->mapBuffer(vtxStaging.Get(), nvrhi::CpuAccessMode::Read);
        const float* gpuVerts = reinterpret_cast<const float*>(vtxMapped);
        uint32_t vtxMismatches = 0;
        for (size_t i = 0; i < grids.vertices.size(); ++i) {
            if (gpuVerts[i] != grids.vertices[i]) vtxMismatches++;
        }
        Logger::warn(str::format("DIAG TEMPLATE-VERTEX-READBACK: count=", grids.vertices.size(),
            " mismatches=", vtxMismatches));

        // Log first 18 floats (6 vertices * xyz) for manual inspection
        std::string cpuVtxStr = "  CPU verts: ";
        std::string gpuVtxStr = "  GPU verts: ";
        for (size_t i = 0; i < std::min(grids.vertices.size(), (size_t)18); ++i) {
            cpuVtxStr += str::format(grids.vertices[i], " ");
            gpuVtxStr += str::format(gpuVerts[i], " ");
        }
        Logger::warn(cpuVtxStr);
        Logger::warn(gpuVtxStr);
        m_device->unmapBuffer(vtxStaging.Get());

        // Reopen command list for subsequent operations
        commandList->open();
    }

    // CRITICAL: Use member variable to keep buffer alive - GPU caches address references
    m_templateBuffers.sizesBuffer.Create(kNumTemplates, "ClusterTemplateSizes", m_device.Get());

    cluster::OperationDesc templateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = m_templateBuffers.sizesBuffer.Get(),
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(templateGetSizesDesc);

    // readback templateSizes
    std::vector<uint32_t> templateSizes = m_templateBuffers.sizesBuffer.Download(commandList);

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    m_templateBuffers.sizesBuffer.Log(commandList);
#endif

    // Match sample: tight packing, no alignment padding between templates
    size_t totalTemplateSize = 0;
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        totalTemplateSize += templateSizes[i];
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Template sizes - total=", totalTemplateSize));

    // Create template data buffer based off of totalSize of all templates
    nvrhi::BufferDesc destDataDesc = {
        .byteSize = totalTemplateSize,
        .debugName = "ClusterTemplateData",
        .canHaveUAVs = true,
        .isAccelStructStorage = true,
        .initialState = nvrhi::ResourceStates::AccelStructWrite,
        .keepInitialState = true,
    };
    m_templateBuffers.dataBuffer = CreateBuffer(destDataDesc, m_device.Get());

    // Explicit Destination mode, calculate the address offset for each template to get a tight fit
    operationParams.type = cluster::OperationType::ClasBuildTemplates;
    operationParams.mode = cluster::OperationMode::ExplicitDestinations;

    nvrhi::GpuVirtualAddress baseAddress = m_templateBuffers.dataBuffer->getGpuVirtualAddress();
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - dataBuffer baseAddress=", std::hex, baseAddress));
    if (baseAddress == 0) {
        Logger::err("RTX MegaGeo: InitStructuredClusterTemplates - dataBuffer baseAddress is NULL!");
    }

    // Match sample exactly: tight packing from base address
    std::vector<nvrhi::GpuVirtualAddress> addresses(kNumTemplates);
    totalTemplateSize = 0;
    for (size_t i = 0; i < addresses.size(); i++)
    {
        addresses[i] = baseAddress + totalTemplateSize;
        totalTemplateSize += templateSizes[i];
    }
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - computed ", addresses.size(), " template addresses"));
#if RTXMG_VERBOSE_LOGGING
    if (!addresses.empty()) {
        RTXMG_LOG(str::format("RTX MegaGeo: Template First address=", std::hex, addresses[0], " last=", addresses.back()));
    }
#endif

    m_templateBuffers.addressesBuffer.Create(kNumTemplates, "ClusterTemplateDestAddressData", m_device.Get());
    m_templateBuffers.addressesBuffer.Upload(addresses, commandList);
    m_templateBuffers.addresses = addresses; // Store CPU-side copy for FillInstantiateTemplateArgs
    m_templateBuffers.instantiationSizesBuffer.Create(kNumTemplates, "ClusterTemplateInstantiationSizes", m_device.Get());

    // Log all addresses before cluster template build
    RTXMG_LOG(str::format("RTX MegaGeo: Before createClusterTemplateDesc:"));
    RTXMG_LOG(str::format("  clusterTemplateArgsBuffer ptr=", (void*)clusterTemplateArgsBuffer.Get()));
    RTXMG_LOG(str::format("  clusterTemplateArgsBuffer addr=", std::hex, clusterTemplateArgsBuffer ? clusterTemplateArgsBuffer->getGpuVirtualAddress() : 0));
    RTXMG_LOG(str::format("  addressesBuffer ptr=", (void*)m_templateBuffers.addressesBuffer.Get()));
    RTXMG_LOG(str::format("  addressesBuffer addr=", std::hex, m_templateBuffers.addressesBuffer.GetGpuVirtualAddress()));

    cluster::OperationDesc createClusterTemplateDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = m_templateBuffers.addressesBuffer.Get(),
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = 0,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };
    RTXMG_LOG("RTX MegaGeo: Calling executeMultiIndirectClusterOperation for createClusterTemplateDesc");
    commandList->executeMultiIndirectClusterOperation(createClusterTemplateDesc);
    RTXMG_LOG("RTX MegaGeo: createClusterTemplateDesc complete");

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    m_templateBuffers.addressesBuffer.Log(commandList);
#endif

    // Create and fill out the instantiate args buffer from addressesBuffer
    // Align structStride to 16 bytes for Vulkan minStorageBufferOffsetAlignment
    uint32_t instantiateArgElementSize = sizeof(cluster::IndirectInstantiateTemplateArgs);
    uint32_t instantiateArgAlignedStride = (instantiateArgElementSize + 15) & ~15;
    nvrhi::BufferDesc instantiateTemplateArgsDesc =
    {
        .byteSize = instantiateArgAlignedStride * kNumTemplates,
        .debugName = "InstantiateTemplateArgs",
        .structStride = instantiateArgAlignedStride,
        .canHaveUAVs = true,
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    RTXMGBuffer<cluster::IndirectInstantiateTemplateArgs> instantiateTemplateArgsBuffer(instantiateTemplateArgsDesc, m_device.Get());
    FillInstantiateTemplateArgs(instantiateTemplateArgsBuffer, m_templateBuffers.addressesBuffer, kNumTemplates, commandList);

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    instantiateTemplateArgsBuffer.Log(commandList, [](std::ostream& ss, auto e)
        {
            ss << "{ct: " << std::hex << e.clusterTemplate <<
                " | vb: " << std::hex << e.vertexBuffer.startAddress << "}";
            return true;
        });
#endif

    // Execute GetSizes mode to fill out destSizes
    operationParams.type = cluster::OperationType::ClasInstantiateTemplates;
    operationParams.mode = cluster::OperationMode::GetSizes;
    
    cluster::OperationDesc instantiateTemplateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = instantiateTemplateArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = m_templateBuffers.instantiationSizesBuffer,
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(instantiateTemplateGetSizesDesc);

    m_templateBuffers.instantiationSizes = m_templateBuffers.instantiationSizesBuffer.Download(commandList);

    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - Download complete, size=",
        m_templateBuffers.instantiationSizes.size()));
    if (!m_templateBuffers.instantiationSizes.empty()) {
        RTXMG_LOG(str::format("RTX MegaGeo: First 3 instantiation sizes: [0]=", m_templateBuffers.instantiationSizes[0],
            " [1]=", m_templateBuffers.instantiationSizes.size() > 1 ? m_templateBuffers.instantiationSizes[1] : 0,
            " [2]=", m_templateBuffers.instantiationSizes.size() > 2 ? m_templateBuffers.instantiationSizes[2] : 0));
    }

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    m_templateBuffers.instantiationSizesBuffer.Log(commandList, { .wrap = false });
#endif
    // DIAGNOSTIC: Log key template build results
    {
        Logger::warn(str::format("DIAG TEMPLATE-BUILD: maxGeometryIndex=", maxGeometryIndex,
            " m_maxClusters=", m_maxClusters, " kNumTemplates=", kNumTemplates,
            " quantNBits=", m_tessellatorConfig.quantNBits));

        // Log template addresses
        nvrhi::GpuVirtualAddress dataBase = m_templateBuffers.dataBuffer->getGpuVirtualAddress();
        size_t dataSize = m_templateBuffers.dataBuffer->getDesc().byteSize;
        Logger::warn(str::format("DIAG TEMPLATE-DATA: buffer=[0x", std::hex, dataBase, "-0x", dataBase + dataSize,
            "] size=", std::dec, dataSize));

        for (size_t i = 0; i < m_templateBuffers.addresses.size() && i < 5; ++i) {
            int64_t offset = (int64_t)(m_templateBuffers.addresses[i] - dataBase);
            Logger::warn(str::format("  templateAddr[", i, "]=0x", std::hex, m_templateBuffers.addresses[i],
                " offset=", std::dec, offset,
                " aligned(128)=", (m_templateBuffers.addresses[i] % 128 == 0) ? "Y" : "N",
                " instSize=", i < m_templateBuffers.instantiationSizes.size() ? m_templateBuffers.instantiationSizes[i] : 0));
        }
    }

    RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - complete");
}

void ClusterAccelBuilder::BuildStructuredCLASes(ClusterAccels& accels, uint32_t maxGeometryCountPerMesh,
    const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildStructuredCLASes");

    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - m_maxClusters=", m_maxClusters));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasPtrsBuffer ptr=", (void*)accels.clasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasBuffer ptr=", (void*)accels.clasBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - m_clasIndirectArgDataBuffer ptr=", (void*)m_clasIndirectArgDataBuffer.Get()));

    nvrhi::GpuVirtualAddress clasPtrsAddr = accels.clasPtrsBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress clasBufferAddr = accels.clasBuffer.GetBuffer() ? accels.clasBuffer.GetBuffer()->getGpuVirtualAddress() : 0;
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasPtrsAddr=", std::hex, clasPtrsAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasBufferAddr=", std::hex, clasBufferAddr));

    if (clasPtrsAddr == 0) Logger::err("RTX MegaGeo: BuildStructuredCLASes - clasPtrsAddr is NULL!");
    if (clasBufferAddr == 0) Logger::err("RTX MegaGeo: BuildStructuredCLASes - clasBufferAddr is NULL!");

    // CRITICAL: Must use the same maxGeometryIndex as InitStructuredClusterTemplates.
    // Use the stored value (which now correctly equals maxGeometryCountPerMesh-1).
    uint32_t instantiateMaxGeometryIndex = m_templateBuffers.maxGeometryCountPerMesh;
    Logger::warn(str::format("RTX MegaGeo: BuildStructuredCLASes GEOMETRY INDEX CHECK:"
        " maxGeometryCountPerMesh(param)=", maxGeometryCountPerMesh,
        " instantiateMaxGeometryIndex=", instantiateMaxGeometryIndex,
        " templateMaxGeometryIndex(stored)=", m_templateBuffers.maxGeometryCountPerMesh,
        " (FIXED: uses stored value from InitStructured, not m_maxClusters-1)"));

    cluster::OperationParams instantiateClasParams =
    {
        .maxArgCount = m_maxClusters,
        .type = cluster::OperationType::ClasInstantiateTemplates,
        .mode = cluster::OperationMode::ExplicitDestinations,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            // The compute_cluster_tiling shader writes localGeometryIndex (per-mesh surface index)
            // into geometryIndexOffset. So maxGeometryIndex = maxGeometryCountPerMesh - 1,
            // matching the sample exactly.
            .maxGeometryIndex = instantiateMaxGeometryIndex,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = m_maxClusters * kClusterMaxTriangles,
            .maxTotalVertexCount = m_maxVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };

    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(instantiateClasParams);
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - scratchSizeInBytes=", sizeInfo.scratchSizeInBytes));

    uint64_t countBufferOffset = tessCounterRange.byteOffset + kClusterCountByteOffset;
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - countBuffer offset calculation:",
        " tessCounterRange.byteOffset=", tessCounterRange.byteOffset,
        " kClusterCountByteOffset=", kClusterCountByteOffset,
        " TOTAL offset=", countBufferOffset));

    cluster::OperationDesc instantiateClasDesc =
    {
        .params = instantiateClasParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = m_tessellationCountersBuffer,
        .inIndirectArgCountOffsetInBytes = countBufferOffset,
        .inIndirectArgsBuffer = m_clasIndirectArgDataBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.clasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = nullptr,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };

    // DIAG: Download and log CLAS indirect args before building
    {
        auto clasIndirectArgs = m_clasIndirectArgDataBuffer.Download(commandList);
        uint32_t numActive = 0;
        uint32_t wrongStride = 0, zeroTemplate = 0, zeroVertex = 0, badTemplateAddr = 0;
        // Build set of known template addresses for validation
        std::set<nvrhi::GpuVirtualAddress> knownTemplateAddrs(
            m_templateBuffers.addresses.begin(), m_templateBuffers.addresses.end());

        for (uint32_t i = 0; i < (uint32_t)clasIndirectArgs.size(); ++i) {
            const auto& arg = clasIndirectArgs[i];
            if (arg.clusterTemplate == 0 && arg.vertexBuffer.startAddress == 0) continue;
            numActive++;
            if (arg.vertexBuffer.strideInBytes != 12) wrongStride++;
            if (arg.clusterTemplate == 0) zeroTemplate++;
            if (arg.vertexBuffer.startAddress == 0) zeroVertex++;
            if (arg.clusterTemplate != 0 && knownTemplateAddrs.find(arg.clusterTemplate) == knownTemplateAddrs.end())
                badTemplateAddr++;
        }
        Logger::warn(str::format("DIAG CLAS-ARGS SUMMARY: active=", numActive, "/", clasIndirectArgs.size(),
            " wrongStride=", wrongStride, " zeroTemplate=", zeroTemplate, " zeroVertex=", zeroVertex,
            " unknownTemplateAddr=", badTemplateAddr, " knownTemplates=", knownTemplateAddrs.size()));

        // Log first 5 args with detail
        for (uint32_t i = 0; i < std::min((uint32_t)clasIndirectArgs.size(), 5u); ++i) {
            const auto& arg = clasIndirectArgs[i];
            if (arg.clusterTemplate == 0 && arg.vertexBuffer.startAddress == 0) continue;
            bool templateKnown = knownTemplateAddrs.count(arg.clusterTemplate) > 0;
            Logger::warn(str::format("  CLAS-ARG[", i, "] clusterIdOff=", arg.clusterIdOffset,
                " geomIdx=", (arg.geometryIndexOffsetPacked & 0xFFFFFF),
                " template=0x", std::hex, arg.clusterTemplate, "(known=", templateKnown ? "Y" : "N", ")",
                " vtxAddr=0x", arg.vertexBuffer.startAddress, std::dec,
                " stride=", arg.vertexBuffer.strideInBytes));
        }
    }

    // DIAGNOSTIC: Validate CLAS destination addresses before build
    {
        auto clasPtrs = accels.clasPtrsBuffer.Download(commandList);
        nvrhi::GpuVirtualAddress clasBase = accels.clasBuffer.GetGpuVirtualAddress();
        size_t clasSize = accels.clasBuffer.GetBytes();
        nvrhi::GpuVirtualAddress clasEnd = clasBase + clasSize;

        uint32_t numActive = 0, outOfRange = 0, misaligned = 0, overlapping = 0;
        std::vector<std::pair<nvrhi::GpuVirtualAddress, uint32_t>> activeAddrs; // addr, index

        for (uint32_t i = 0; i < (uint32_t)clasPtrs.size(); ++i) {
            if (clasPtrs[i] == 0) continue;
            numActive++;
            if (clasPtrs[i] < clasBase || clasPtrs[i] >= clasEnd) outOfRange++;
            if (clasPtrs[i] % cluster::kClasByteAlignment != 0) misaligned++;
            activeAddrs.push_back({clasPtrs[i], i});
        }

        // Sort by address to check for overlaps
        std::sort(activeAddrs.begin(), activeAddrs.end());
        for (size_t i = 1; i < activeAddrs.size(); ++i) {
            if (activeAddrs[i].first == activeAddrs[i-1].first) overlapping++;
        }

        Logger::warn(str::format("DIAG PRE-CLAS-BUILD: clasPtrs active=", numActive, "/", clasPtrs.size(),
            " outOfRange=", outOfRange, " misaligned(128)=", misaligned, " duplicateAddr=", overlapping,
            " clasBuffer=[0x", std::hex, clasBase, "-0x", clasEnd, "] size=", std::dec, clasSize));

        // Log first few active addresses
        uint32_t logged = 0;
        for (uint32_t i = 0; i < (uint32_t)clasPtrs.size() && logged < 10; ++i) {
            if (clasPtrs[i] == 0) continue;
            int64_t offsetFromBase = (int64_t)(clasPtrs[i] - clasBase);
            Logger::warn(str::format("  clasPtrs[", i, "]=0x", std::hex, clasPtrs[i],
                " offset=", std::dec, offsetFromBase,
                " aligned=", (clasPtrs[i] % 128 == 0) ? "Y" : "N",
                " inRange=", (clasPtrs[i] >= clasBase && clasPtrs[i] < clasEnd) ? "Y" : "N"));
            logged++;
        }

        // Log instantiation sizes for reference
        if (!m_templateBuffers.instantiationSizes.empty()) {
            std::string sizesStr = "DIAG INSTANTIATION-SIZES:";
            for (size_t i = 0; i < m_templateBuffers.instantiationSizes.size() && i < 10; ++i) {
                sizesStr += str::format(" [", i, "]=", m_templateBuffers.instantiationSizes[i]);
            }
            Logger::warn(sizesStr);
        }
    }

    // DIAG: Verify vertex positions are valid right before CLAS instantiation
    // NOTE: GPU float3 stride = 12 bytes (Slang scalar layout), NOT sizeof(MathLib::float3) = 16.
    // RTXMGBuffer<float3>::Download() interprets with 16-byte stride, giving wrong element count.
    // Must use raw byte readback with 12-byte stride for correct vertex data.
    {
        static bool s_loggedPreClasVerts = false;
        if (!s_loggedPreClasVerts) {
            s_loggedPreClasVerts = true;
            // Raw byte readback — bypass RTXMGBuffer<float3>::Download() which uses 16-byte stride
            size_t bufferBytes = accels.clusterVertexPositionsBuffer.GetBytes();
            static constexpr uint32_t kGpuFloat3Stride = 12; // 3 * sizeof(float)
            uint32_t numVerts = (uint32_t)(bufferBytes / kGpuFloat3Stride);
            // Create staging buffer for raw byte download
            auto stagingDesc = nvrhi::BufferDesc{ .byteSize = bufferBytes, .cpuAccess = nvrhi::CpuAccessMode::Read };
            auto staging = commandList->getDevice()->createBuffer(stagingDesc);
            commandList->copyBuffer(staging.Get(), 0, accels.clusterVertexPositionsBuffer, 0, bufferBytes);
            commandList->close();
            commandList->getDevice()->executeCommandList(commandList);
            commandList->getDevice()->waitForIdle();
            void* mapped = commandList->getDevice()->mapBuffer(staging.Get(), nvrhi::CpuAccessMode::Read);
            const float* fp = reinterpret_cast<const float*>(mapped);
            uint32_t numFloats = (uint32_t)(bufferBytes / sizeof(float));
            uint32_t zeroVerts = 0, nanVerts = 0;
            for (uint32_t i = 0; i < numVerts; ++i) {
                size_t fi = i * 3; // 12-byte stride = 3 floats
                if (fi + 2 >= numFloats) break;
                bool isZero = (fp[fi] == 0 && fp[fi+1] == 0 && fp[fi+2] == 0);
                bool isNan = (std::isnan(fp[fi]) || std::isnan(fp[fi+1]) || std::isnan(fp[fi+2]));
                if (isZero) zeroVerts++;
                if (isNan) nanVerts++;
            }
            // Log first 5 non-zero vertex positions
            uint32_t logged = 0;
            std::string posStr;
            for (uint32_t i = 0; i < numVerts && logged < 5; ++i) {
                size_t fi = i * 3;
                if (fi + 2 >= numFloats) break;
                if (fp[fi] == 0 && fp[fi+1] == 0 && fp[fi+2] == 0) continue;
                posStr += str::format(" [", i, "]=(", fp[fi], ",", fp[fi+1], ",", fp[fi+2], ")");
                logged++;
            }
            commandList->getDevice()->unmapBuffer(staging.Get());
            commandList->open();
            Logger::warn(str::format("DIAG PRE-CLAS-VERTS: total=", numVerts,
                " zero=", zeroVerts, " nan=", nanVerts, " nonZero=", numVerts - zeroVerts,
                " bufBytes=", bufferBytes, " gpuStride=", kGpuFloat3Stride,
                " cppFloat3Size=", sizeof(float3)));
            Logger::warn(str::format("DIAG PRE-CLAS-VERTS first5:", posStr));
        }
    }

    // DIAG: Log CLAS instantiation sizes, cluster count, and CLAS spacing before the build
    {
        static bool s_loggedPreBuild = false;
        if (!s_loggedPreBuild) {
            s_loggedPreBuild = true;

            // 1. Log all instantiation sizes per template
            Logger::warn(str::format("DIAG INST-SIZES: kNumTemplates=", kNumTemplates,
                " stored count=", m_templateBuffers.instantiationSizes.size()));
            for (uint32_t i = 0; i < std::min((uint32_t)m_templateBuffers.instantiationSizes.size(), 10u); ++i) {
                Logger::warn(str::format("  instSize[", i, "]=", m_templateBuffers.instantiationSizes[i],
                    " aligned128=", (m_templateBuffers.instantiationSizes[i] % 128 == 0) ? "Y" : "N"));
            }

            // 2. Read the actual cluster count from tessellation counters
            {
                auto countersData = m_tessellationCountersBuffer.Download(commandList);
                if (!countersData.empty()) {
                    // kClusterCountByteOffset is the offset within TessellationCounters struct
                    const uint32_t* rawData = reinterpret_cast<const uint32_t*>(countersData.data());
                    uint32_t numCounterWords = (uint32_t)(countersData.size() * sizeof(countersData[0]) / sizeof(uint32_t));
                    Logger::warn(str::format("DIAG TESS-COUNTERS: buffer words=", numCounterWords));
                    for (uint32_t i = 0; i < std::min(numCounterWords, 8u); ++i) {
                        Logger::warn(str::format("  counter[", i, "]=", rawData[i]));
                    }
                    // The cluster count is at kClusterCountByteOffset within tessCounterRange
                    uint32_t clusterCountWordIdx = (uint32_t)(kClusterCountByteOffset / sizeof(uint32_t));
                    if (clusterCountWordIdx < numCounterWords) {
                        Logger::warn(str::format("DIAG CLUSTER-COUNT: clusterCountWordIdx=", clusterCountWordIdx,
                            " value=", rawData[clusterCountWordIdx],
                            " m_maxClusters=", m_maxClusters));
                    }
                }
            }

            // 3. Check CLAS destination spacing — are CLASes overlapping?
            {
                auto clasPtrs = accels.clasPtrsBuffer.Download(commandList);
                auto clasArgs = m_clasIndirectArgDataBuffer.Download(commandList);
                // Collect active CLAS ptrs with their template info
                struct ClasInfo { uint32_t idx; nvrhi::GpuVirtualAddress addr; uint32_t templateInstSize; };
                std::vector<ClasInfo> activeClases;
                for (uint32_t i = 0; i < (uint32_t)clasPtrs.size() && i < (uint32_t)clasArgs.size(); ++i) {
                    if (clasPtrs[i] == 0) continue;
                    // Find template index for this cluster
                    nvrhi::GpuVirtualAddress tmplAddr = clasArgs[i].clusterTemplate;
                    uint32_t tmplIdx = 0;
                    for (uint32_t t = 0; t < (uint32_t)m_templateBuffers.addresses.size(); ++t) {
                        if (m_templateBuffers.addresses[t] == tmplAddr) { tmplIdx = t; break; }
                    }
                    uint32_t instSize = (tmplIdx < m_templateBuffers.instantiationSizes.size()) ?
                        m_templateBuffers.instantiationSizes[tmplIdx] : 0;
                    activeClases.push_back({i, clasPtrs[i], instSize});
                }
                // Sort by address and check for overlaps
                std::sort(activeClases.begin(), activeClases.end(),
                    [](const ClasInfo& a, const ClasInfo& b) { return a.addr < b.addr; });
                uint32_t overlaps = 0;
                for (uint32_t i = 1; i < (uint32_t)activeClases.size(); ++i) {
                    uint64_t gap = activeClases[i].addr - activeClases[i-1].addr;
                    if (gap < activeClases[i-1].templateInstSize) overlaps++;
                }
                Logger::warn(str::format("DIAG CLAS-SPACING: activeCLASes=", activeClases.size(),
                    " overlaps=", overlaps));
                // Log first 5 sorted entries
                for (uint32_t i = 0; i < std::min((uint32_t)activeClases.size(), 5u); ++i) {
                    uint64_t gap = (i + 1 < activeClases.size()) ?
                        activeClases[i+1].addr - activeClases[i].addr : 0;
                    Logger::warn(str::format("  sorted[", i, "] idx=", activeClases[i].idx,
                        " addr=0x", std::hex, activeClases[i].addr, std::dec,
                        " instSize=", activeClases[i].templateInstSize,
                        " gapToNext=", gap));
                }
            }
        }
    }

    RTXMG_LOG("RTX MegaGeo: BuildStructuredCLASes - calling executeMultiIndirectClusterOperation");
    commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);

    // DIAG: Download and log CLAS pointers after instantiation
    {
        auto clasPtrs = accels.clasPtrsBuffer.Download(commandList);
        uint32_t numToLog = std::min(uint32_t(clasPtrs.size()), 20u);
        uint32_t zeroPtrs = 0, nonZeroPtrs = 0;
        for (size_t i = 0; i < clasPtrs.size(); ++i) {
            if (clasPtrs[i] == 0) zeroPtrs++;
            else nonZeroPtrs++;
        }
        Logger::warn(str::format("DIAG POST-CLAS: total=", clasPtrs.size(),
            " nonZero=", nonZeroPtrs, " zero=", zeroPtrs));
    }

    // DIAG: Read back raw CLAS data at FIRST ACTIVE CLAS address
    {
        static bool s_loggedClasData = false;
        if (!s_loggedClasData) {
            s_loggedClasData = true;
            // Find first active CLAS address
            auto clasPtrsForData = accels.clasPtrsBuffer.Download(commandList);
            nvrhi::GpuVirtualAddress clasBase = accels.clasBuffer.GetGpuVirtualAddress();
            size_t clasBytes = accels.clasBuffer.GetBytes();
            nvrhi::GpuVirtualAddress firstClasAddr = 0;
            uint32_t firstClasIdx = UINT32_MAX;
            for (uint32_t i = 0; i < (uint32_t)clasPtrsForData.size(); ++i) {
                if (clasPtrsForData[i] != 0) { firstClasAddr = clasPtrsForData[i]; firstClasIdx = i; break; }
            }
            if (firstClasAddr != 0 && firstClasAddr >= clasBase) {
                uint64_t offsetInBuffer = firstClasAddr - clasBase;
                // Read 512 bytes from the first active CLAS
                size_t readSize = std::min((size_t)512, clasBytes - (size_t)offsetInBuffer);
                auto stagingDesc = nvrhi::BufferDesc{ .byteSize = readSize, .cpuAccess = nvrhi::CpuAccessMode::Read };
                auto staging = commandList->getDevice()->createBuffer(stagingDesc);
                commandList->copyBuffer(staging.Get(), 0, accels.clasBuffer, offsetInBuffer, readSize);
                commandList->close();
                commandList->getDevice()->executeCommandList(commandList);
                commandList->getDevice()->waitForIdle();
                void* mapped = commandList->getDevice()->mapBuffer(staging.Get(), nvrhi::CpuAccessMode::Read);
                const uint32_t* data = reinterpret_cast<const uint32_t*>(mapped);
                uint32_t numWords = (uint32_t)(readSize / sizeof(uint32_t));
                uint32_t zeroWords = 0;
                for (uint32_t i = 0; i < numWords; ++i) {
                    if (data[i] == 0) zeroWords++;
                }
                Logger::warn(str::format("DIAG POST-CLAS-DATA: firstClasIdx=", firstClasIdx,
                    " addr=0x", std::hex, firstClasAddr, std::dec,
                    " bufferOffset=", offsetInBuffer,
                    " read=", readSize, " words=", numWords,
                    " zeroWords=", zeroWords, " nonZeroWords=", numWords - zeroWords));
                // Dump first 32 dwords of the actual CLAS
                std::string hexDump;
                for (uint32_t i = 0; i < std::min(numWords, 32u); ++i) {
                    hexDump += str::format(" ", std::hex, data[i]);
                }
                Logger::warn(str::format("DIAG POST-CLAS-DATA raw:", hexDump));
                // Also dump as floats to see if vertex positions are recognizable
                const float* fp = reinterpret_cast<const float*>(mapped);
                std::string floatDump;
                for (uint32_t i = 0; i < std::min(numWords, 12u); ++i) {
                    floatDump += str::format(" ", fp[i]);
                }
                Logger::warn(str::format("DIAG POST-CLAS-DATA floats:", floatDump));
                commandList->getDevice()->unmapBuffer(staging.Get());
                commandList->open();
            } else {
                Logger::warn("DIAG POST-CLAS-DATA: no active CLAS found!");
            }
        }
    }

    RTXMG_LOG("RTX MegaGeo: BuildStructuredCLASes - complete");
}

void ClusterAccelBuilder::FillInstanceClusters(const RTXMGScene& scene, ClusterAccels& accels, nvrhi::ICommandList* commandList)
{
    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - instances.size()=", instances.size(),
        " m_numInstances=", m_numInstances, " buffer size=", m_fillClustersDispatchIndirectBuffer.GetBytes()));

    nvrhi::utils::ScopedMarker marker(commandList, "FillInstanceClusters");
    stats::clusterAccelSamplers.fillClustersTime.Start(commandList);


#if RTXMG_CHRONO_TIMING
    auto fillStart = std::chrono::high_resolution_clock::now();
    float setupTimeMs = 0.0f;
    float bindingTimeMs = 0.0f;
    float dispatchTimeMs = 0.0f;
#endif

    uint32_t surfaceOffset{ 0 };
    // Limit loop to m_numInstances to avoid buffer overflow on indirect dispatch buffer
    uint32_t maxInstances = std::min(uint32_t(instances.size()), m_numInstances);
    if (instances.size() > m_numInstances) {
        dxvk::Logger::warn(dxvk::str::format("RTX MegaGeo: FillInstanceClusters - instances.size()=", instances.size(),
            " > m_numInstances=", m_numInstances, ", limiting to ", m_numInstances));
    }
    for (uint32_t instanceIndex = 0; instanceIndex < maxInstances; ++instanceIndex)
    {
#if RTXMG_CHRONO_TIMING
        auto instStart = std::chrono::high_resolution_clock::now();
#endif
        const auto& instance = instances[instanceIndex];

        // Bounds check to prevent crash - skip instances with invalid meshID
        if (instance.meshID >= subdMeshes.size()) {
            dxvk::Logger::warn(dxvk::str::format("RTX MegaGeo: FillInstanceClusters - meshID ", instance.meshID,
                " out of bounds (subdMeshes.size()=", subdMeshes.size(), "), skipping instance ", instanceIndex));
            continue;
        }

        assert(instance.meshInstance.get());
        const auto& donutMeshInfo = instance.meshInstance->GetMesh();
        assert(donutMeshInfo.get());
        uint32_t firstGeometryIndex = donutMeshInfo.geometries[0]->globalGeometryIndex;

        const auto& subd = *subdMeshes[instance.meshID];

        const uint32_t surfaceCount = subd.SurfaceCount();

        // DEBUG: Log instance 0's mesh AABB and positions buffer size
        {
            static bool s_loggedInst0 = false;
            if (!s_loggedInst0 && instanceIndex == 0) {
                const auto& aabb = subd.m_aabb;
                uint32_t posBufBytes = subd.m_positionsBuffer ? (uint32_t)subd.m_positionsBuffer->getDesc().byteSize : 0;
                s_loggedInst0 = true;
            }
        }

        // DEBUG: Validate all buffers before binding
        {
            bool hasNullBuffer = false;
            if (!subd.m_positionsBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_positionsBuffer is NULL!"); hasNullBuffer = true; }
            if (!subd.m_vertexDeviceData.surfaceDescriptors) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_vertexDeviceData.surfaceDescriptors is NULL!"); hasNullBuffer = true; }
            if (!subd.m_vertexDeviceData.controlPointIndices) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_vertexDeviceData.controlPointIndices is NULL!"); hasNullBuffer = true; }
            if (!subd.m_vertexDeviceData.patchPointsOffsets) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_vertexDeviceData.patchPointsOffsets is NULL!"); hasNullBuffer = true; }
            if (!subd.m_vertexDeviceData.patchPoints) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_vertexDeviceData.patchPoints is NULL!"); hasNullBuffer = true; }
            if (!subd.GetTopologyMap()) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.GetTopologyMap() is NULL!"); hasNullBuffer = true; }
            else {
                if (!subd.GetTopologyMap()->plansBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: TopologyMap->plansBuffer is NULL!"); hasNullBuffer = true; }
                if (!subd.GetTopologyMap()->subpatchTreesArraysBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: TopologyMap->subpatchTreesArraysBuffer is NULL!"); hasNullBuffer = true; }
                if (!subd.GetTopologyMap()->patchPointIndicesArraysBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: TopologyMap->patchPointIndicesArraysBuffer is NULL!"); hasNullBuffer = true; }
                if (!subd.GetTopologyMap()->stencilMatrixArraysBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: TopologyMap->stencilMatrixArraysBuffer is NULL!"); hasNullBuffer = true; }
            }
            if (!subd.m_surfaceToGeometryIndexBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_surfaceToGeometryIndexBuffer is NULL!"); hasNullBuffer = true; }
            if (!subd.m_texcoordDeviceData.surfaceDescriptors) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_texcoordDeviceData.surfaceDescriptors is NULL!"); hasNullBuffer = true; }
            if (!subd.m_texcoordDeviceData.controlPointIndices) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_texcoordDeviceData.controlPointIndices is NULL!"); hasNullBuffer = true; }
            if (!subd.m_texcoordDeviceData.patchPointsOffsets) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_texcoordDeviceData.patchPointsOffsets is NULL!"); hasNullBuffer = true; }
            if (!subd.m_texcoordDeviceData.patchPoints) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_texcoordDeviceData.patchPoints is NULL!"); hasNullBuffer = true; }
            if (!subd.m_texcoordsBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: subd.m_texcoordsBuffer is NULL!"); hasNullBuffer = true; }
            if (!accels.clusterVertexPositionsBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: accels.clusterVertexPositionsBuffer is NULL!"); hasNullBuffer = true; }
            if (!accels.clusterShadingDataBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: accels.clusterShadingDataBuffer is NULL!"); hasNullBuffer = true; }
            if (!accels.clusterVertexNormalsBuffer) { dxvk::Logger::err("RTX MegaGeo DEBUG: accels.clusterVertexNormalsBuffer is NULL!"); hasNullBuffer = true; }

            if (hasNullBuffer) {
                dxvk::Logger::err(dxvk::str::format("RTX MegaGeo DEBUG: Skipping instance ", instanceIndex, " due to NULL buffers"));
                continue;
            }

            // Log buffer info for first instance only
            if (instanceIndex == 0) {
                RTXMG_LOG(dxvk::str::format("RTX MegaGeo DEBUG: Instance 0 buffers OK, surfaceCount=", surfaceCount,
                    " surfaceOffset=", surfaceOffset, " gridSamplerStride=", m_gridSamplersBuffer.GetElementBytes()));
            }
        }

        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            commandList->clearBufferUInt(m_debugBuffer.Get(), 0);
        }

        // Debug buffer cleared before tiling loop in BuildAccel - no need to clear here

        FillClustersParams params = {};
        params.instanceIndex = instanceIndex;
        params.quantNBits = m_tessellatorConfig.quantNBits;
        params.isolationLevel = m_tessellatorConfig.isolationLevel;
        params.globalDisplacementScale = m_tessellatorConfig.displacementScale;
        params.clusterPattern = uint32_t(m_tessellatorConfig.clusterPattern);
        params.firstGeometryIndex = firstGeometryIndex;
        params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
        params.debugClusterIndex = uint32_t(m_tessellatorConfig.debugClusterIndex);
        params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
        commandList->writeBuffer(m_fillClustersParamsBuffer, &params, sizeof(FillClustersParams));

        // DIAGNOSTIC: Log FillClustersParams and all per-mesh buffer metadata for first instance
        {
            static bool s_loggedFillDiag = false;
            if (!s_loggedFillDiag && instanceIndex == 0) {
                s_loggedFillDiag = true;
                auto bufInfo = [](const char* name, nvrhi::IBuffer* buf) {
                    if (!buf) return dxvk::str::format("  ", name, ": NULL");
                    auto& d = buf->getDesc();
                    return dxvk::str::format("  ", name, ": byteSize=", d.byteSize,
                        " structStride=", d.structStride, " elems=", d.structStride ? d.byteSize / d.structStride : 0);
                };
                Logger::warn("=== FillInstanceClusters DIAGNOSTIC (instance 0) ===");
                Logger::warn(str::format("  FillClustersParams: instanceIndex=", params.instanceIndex,
                    " quantNBits=", params.quantNBits, " isolationLevel=", params.isolationLevel,
                    " firstGeomIdx=", params.firstGeometryIndex));
                Logger::warn(str::format("  FillClustersParams: displacementScale=", params.globalDisplacementScale,
                    " clusterPattern=", params.clusterPattern));
                Logger::warn(str::format("  sizeof(FillClustersParams)=", sizeof(FillClustersParams),
                    " sizeof(SurfaceDescriptor)=", sizeof(OpenSubdiv::Tmr::SurfaceDescriptor)));
                Logger::warn(str::format("  surfaceOffset=", surfaceOffset, " surfaceCount=", surfaceCount));
                Logger::warn(bufInfo("t3:positionsBuffer", subd.m_positionsBuffer.Get()));
                Logger::warn(bufInfo("t4:surfaceDescriptors", subd.m_vertexDeviceData.surfaceDescriptors.Get()));
                Logger::warn(bufInfo("t5:controlPointIndices", subd.m_vertexDeviceData.controlPointIndices.Get()));
                Logger::warn(bufInfo("t6:patchPointsOffsets", subd.m_vertexDeviceData.patchPointsOffsets.Get()));
                Logger::warn(bufInfo("t7:plansBuffer", subd.GetTopologyMap()->plansBuffer.Get()));
                Logger::warn(bufInfo("t8:subpatchTreesArrays", subd.GetTopologyMap()->subpatchTreesArraysBuffer.Get()));
                Logger::warn(bufInfo("t9:patchPointIndicesArrays", subd.GetTopologyMap()->patchPointIndicesArraysBuffer.Get()));
                Logger::warn(bufInfo("t10:stencilMatrixArrays", subd.GetTopologyMap()->stencilMatrixArraysBuffer.Get()));
                Logger::warn(bufInfo("t11:patchPoints", subd.m_vertexDeviceData.patchPoints.Get()));
                Logger::warn(bufInfo("t14:surfaceToGeomIdx", subd.m_surfaceToGeometryIndexBuffer.Get()));
                Logger::warn(bufInfo("u0:clusterVertexPositions", accels.clusterVertexPositionsBuffer.Get()));
                Logger::warn(bufInfo("CB:fillClustersParams", m_fillClustersParamsBuffer.Get()));
                // Surface offsets (PureBSpline/RegularBSpline/Limit/NoLimit)
                Logger::warn(str::format("  surfaceOffsets: PureBSpline=", subd.m_surfaceOffsets[0],
                    " RegularBSpline=", subd.m_surfaceOffsets[1],
                    " Limit=", subd.m_surfaceOffsets[2], " NoLimit=", subd.m_surfaceOffsets[3]));
                Logger::warn("=== END FillInstanceClusters DIAGNOSTIC ===");

                // CPU-side readback of surfaceDescriptors and controlPointIndices
                {
                    Logger::warn("=== GPU Buffer Readback (instance 0) ===");

                    // Download all three buffers in one batch
                    auto descBuf = subd.m_vertexDeviceData.surfaceDescriptors;
                    auto cpiBuf = subd.m_vertexDeviceData.controlPointIndices;
                    auto posBuf = subd.m_positionsBuffer;
                    uint32_t descBytes = (uint32_t)descBuf->getDesc().byteSize;
                    uint32_t cpiBytes = (uint32_t)cpiBuf->getDesc().byteSize;
                    uint32_t posBytes = (uint32_t)posBuf->getDesc().byteSize;
                    uint32_t numDescs = descBytes / 8;
                    uint32_t numCPIs = cpiBytes / 4;
                    uint32_t numPos = posBytes / 12;

                    auto descReadback = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                        .byteSize = descBytes, .debugName = "descReadback",
                        .cpuAccess = nvrhi::CpuAccessMode::Read,
                        .initialState = nvrhi::ResourceStates::CopyDest, .keepInitialState = true });
                    auto cpiReadback = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                        .byteSize = cpiBytes, .debugName = "cpiReadback",
                        .cpuAccess = nvrhi::CpuAccessMode::Read,
                        .initialState = nvrhi::ResourceStates::CopyDest, .keepInitialState = true });
                    auto posReadback = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                        .byteSize = posBytes, .debugName = "posReadback",
                        .cpuAccess = nvrhi::CpuAccessMode::Read,
                        .initialState = nvrhi::ResourceStates::CopyDest, .keepInitialState = true });
                    commandList->copyBuffer(descReadback.Get(), 0, descBuf.Get(), 0, descBytes);
                    commandList->copyBuffer(cpiReadback.Get(), 0, cpiBuf.Get(), 0, cpiBytes);
                    commandList->copyBuffer(posReadback.Get(), 0, posBuf.Get(), 0, posBytes);
                    commandList->close();
                    commandList->getDevice()->executeCommandList(commandList);
                    commandList->getDevice()->waitForIdle();

                    uint32_t* descData = (uint32_t*)commandList->getDevice()->mapBuffer(descReadback.Get(), nvrhi::CpuAccessMode::Read);
                    int32_t* cpiData = (int32_t*)commandList->getDevice()->mapBuffer(cpiReadback.Get(), nvrhi::CpuAccessMode::Read);
                    float* posData = (float*)commandList->getDevice()->mapBuffer(posReadback.Get(), nvrhi::CpuAccessMode::Read);

                    // --- Surface descriptor bounds check ---
                    if (descData) {
                        uint32_t descOOB = 0, cpiOOBcount = 0;
                        uint32_t minFirstCP = UINT32_MAX, maxFirstCP = 0;
                        // Log first 10 and last 5 descriptors, plus any OOB
                        for (uint32_t i = 0; i < numDescs; i++) {
                            uint32_t field0 = descData[i * 2];
                            uint32_t firstCP = descData[i * 2 + 1];
                            uint32_t planIdx = (field0 >> 12) & 0xFFFFF;
                            minFirstCP = std::min(minFirstCP, firstCP);
                            maxFirstCP = std::max(maxFirstCP, firstCP);

                            // Compute CPI range for this surface
                            uint32_t nextFirstCP = (i + 1 < numDescs) ? descData[(i + 1) * 2 + 1] : numCPIs;
                            uint32_t numCPForSurf = (nextFirstCP > firstCP) ? (nextFirstCP - firstCP) : 0;
                            bool firstCPOOB = (firstCP >= numCPIs);
                            bool rangeOOB = (firstCP + numCPForSurf > numCPIs);

                            if (firstCPOOB || rangeOOB) descOOB++;

                            // Log first 10, last 5, and any OOB
                            if (i < 10 || i >= numDescs - 5 || firstCPOOB || rangeOOB) {
                                Logger::warn(str::format("  desc[", i, "] field0=0x", std::hex, field0, std::dec,
                                    " firstCP=", firstCP, " numCPForSurf=", numCPForSurf,
                                    " planIdx=", planIdx, " hasLimit=", (field0 & 1),
                                    firstCPOOB ? " **FIRSTCP_OOB**" : "",
                                    rangeOOB ? " **RANGE_OOB**" : ""));
                            }
                            if (i == 10 && numDescs > 15) {
                                Logger::warn(str::format("  ... (", numDescs - 15, " more descriptors) ..."));
                            }

                            // Check CPI values within this surface's range for position OOB
                            if (cpiData && !firstCPOOB && firstCP + numCPForSurf <= numCPIs) {
                                for (uint32_t j = firstCP; j < firstCP + numCPForSurf; j++) {
                                    if (cpiData[j] < 0 || (uint32_t)cpiData[j] >= numPos) {
                                        if (cpiOOBcount < 20) {
                                            Logger::warn(str::format("  **CPI_OOB**: desc[", i, "] CPI[", j, "]=", cpiData[j],
                                                " >= numPos=", numPos));
                                        }
                                        cpiOOBcount++;
                                    }
                                }
                            }
                        }
                        Logger::warn(str::format("  BOUNDS SUMMARY: numDescs=", numDescs, " numCPIs=", numCPIs, " numPos=", numPos,
                            " firstCP range=[", minFirstCP, "..", maxFirstCP, "]",
                            " descOOB=", descOOB, " cpiOOB=", cpiOOBcount));
                    }

                    // --- Position sample ---
                    if (posData) {
                        uint32_t logCount = std::min(numPos, 10u);
                        for (uint32_t i = 0; i < logCount; i++) {
                            Logger::warn(str::format("  pos[", i, "] = (", posData[i*3], ", ", posData[i*3+1], ", ", posData[i*3+2], ")"));
                        }
                        // Check for NaN/Inf positions
                        uint32_t nanCount = 0, infCount = 0, zeroCount = 0;
                        for (uint32_t i = 0; i < numPos; i++) {
                            float x = posData[i*3], y = posData[i*3+1], z = posData[i*3+2];
                            if (std::isnan(x) || std::isnan(y) || std::isnan(z)) nanCount++;
                            if (std::isinf(x) || std::isinf(y) || std::isinf(z)) infCount++;
                            if (x == 0 && y == 0 && z == 0) zeroCount++;
                        }
                        Logger::warn(str::format("  POS SUMMARY: total=", numPos, " nan=", nanCount, " inf=", infCount, " zero=", zeroCount));
                    }

                    // --- CPU-side bilinear check: compute what the shader SHOULD produce ---
                    if (descData && cpiData && posData) {
                        Logger::warn("  === CPU-Side Bilinear Check (first 10 surfaces) ===");
                        uint32_t checkCount = std::min(numDescs, 10u);
                        for (uint32_t s = 0; s < checkCount; s++) {
                            uint32_t field0 = descData[s * 2];
                            uint32_t firstCP = descData[s * 2 + 1];
                            // Read 4 control point indices
                            if (firstCP + 3 >= numCPIs) {
                                Logger::warn(str::format("  surf[", s, "] firstCP=", firstCP, " OOB (numCPIs=", numCPIs, ")"));
                                continue;
                            }
                            int32_t cpi0 = cpiData[firstCP + 0];
                            int32_t cpi1 = cpiData[firstCP + 1];
                            int32_t cpi2 = cpiData[firstCP + 2];
                            int32_t cpi3 = cpiData[firstCP + 3];
                            // Read 4 positions
                            auto getPos = [&](int32_t idx) -> std::tuple<float,float,float> {
                                if (idx < 0 || (uint32_t)idx >= numPos) return {0.f, 0.f, 0.f};
                                return {posData[idx*3], posData[idx*3+1], posData[idx*3+2]};
                            };
                            auto [x0,y0,z0] = getPos(cpi0);
                            auto [x1,y1,z1] = getPos(cpi1);
                            auto [x2,y2,z2] = getPos(cpi2);
                            auto [x3,y3,z3] = getPos(cpi3);
                            // Bilinear at UV=(0.5,0.5) = average of 4 corners
                            float cx = 0.25f*(x0+x1+x2+x3);
                            float cy = 0.25f*(y0+y1+y2+y3);
                            float cz = 0.25f*(z0+z1+z2+z3);
                            Logger::warn(str::format("  surf[", s, "] firstCP=", firstCP,
                                " cpi=[", cpi0, ",", cpi1, ",", cpi2, ",", cpi3, "]",
                                " p00=(", x0, ",", y0, ",", z0, ")",
                                " p10=(", x1, ",", y1, ",", z1, ")",
                                " p11=(", x2, ",", y2, ",", z2, ")",
                                " p01=(", x3, ",", y3, ",", z3, ")",
                                " center=(", cx, ",", cy, ",", cz, ")"));
                        }

                        // Compare GPU positions buffer with CPU-side Shape::verts
                        Logger::warn("  === GPU vs CPU Position Comparison ===");
                        auto& cpuVerts = subd.GetShape()->verts;
                        uint32_t cmpCount = std::min((uint32_t)cpuVerts.size(), std::min(numPos, 10u));
                        uint32_t mismatchCount = 0;
                        for (uint32_t i = 0; i < std::min((uint32_t)cpuVerts.size(), numPos); i++) {
                            float gx = posData[i*3], gy = posData[i*3+1], gz = posData[i*3+2];
                            float cx = cpuVerts[i].x, cy = cpuVerts[i].y, cz = cpuVerts[i].z;
                            bool match = (gx == cx && gy == cy && gz == cz);
                            if (!match) mismatchCount++;
                            if (i < cmpCount) {
                                Logger::warn(str::format("  vert[", i, "] GPU=(", gx, ",", gy, ",", gz,
                                    ") CPU=(", cx, ",", cy, ",", cz, ")", match ? "" : " **MISMATCH**"));
                            }
                        }
                        Logger::warn(str::format("  VERT MATCH: compared=", std::min((uint32_t)cpuVerts.size(), numPos),
                            " mismatches=", mismatchCount,
                            " cpuCount=", cpuVerts.size(), " gpuCount=", numPos));
                    }

                    if (descData) commandList->getDevice()->unmapBuffer(descReadback.Get());
                    if (cpiData) commandList->getDevice()->unmapBuffer(cpiReadback.Get());
                    if (posData) commandList->getDevice()->unmapBuffer(posReadback.Get());

                    Logger::warn("=== END GPU Buffer Readback ===");
                    commandList->open();
                }
            }
        }

        // Bindings matching sample style (separate namespaces per resource type)
        // Order must match [[vk::binding]] in fill_clusters.comp.slang
        size_t gridSamplerStride = m_gridSamplersBuffer.GetElementBytes();
        auto bindingSetDesc = nvrhi::BindingSetDesc()
            // SRVs (0-19)
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_gridSamplersBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(surfaceOffset * gridSamplerStride, surfaceCount * gridSamplerStride)))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterOffsetCountsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_clustersBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subd.m_positionsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subd.m_vertexDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subd.m_vertexDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subd.m_vertexDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subd.GetTopologyMap()->plansBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subd.GetTopologyMap()->subpatchTreesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subd.GetTopologyMap()->patchPointIndicesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subd.GetTopologyMap()->stencilMatrixArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, subd.m_vertexDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, scene.GetGeometryBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, scene.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subd.m_surfaceToGeometryIndexBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subd.m_texcoordDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subd.m_texcoordDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(17, subd.m_texcoordDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(18, subd.m_texcoordDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(19, subd.m_texcoordsBuffer))
            // Sampler (20)
            .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))
            // UAVs (21-24)
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, accels.clusterVertexPositionsBuffer));

        // Verify fill_clusters uses the same buffer as compute_cluster_tiling
        auto fillClustersBufAddr = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
        RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - binding clusterVertexPositions UAV, gpuAddr=", std::hex, fillClustersBufAddr));

        bindingSetDesc
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, accels.clusterShadingDataBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_debugBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterVertexNormalsBuffer))
            // Constant buffer (25)
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillClustersParamsBuffer));

        // Create layout once, then reuse for all binding sets
        if (!m_fillClustersBL)
        {
            auto layoutDesc = nvrhi::BindingLayoutDesc()
                .setVisibility(nvrhi::ShaderType::Compute);
            // SRVs 0-19
            for (uint32_t i = 0; i < 20; ++i)
                layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(i));
            // Sampler 0
            layoutDesc.addItem(nvrhi::BindingLayoutItem::Sampler(0));
            // UAVs 0-3
            for (uint32_t i = 0; i < 4; ++i)
                layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(i));
            // Constant buffer 0
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            m_fillClustersBL = m_device->createBindingLayout(layoutDesc);
        }

        nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillClustersBL);
        if (!bindingSet)
        {
            Logger::err("Failed to create binding set for fill_clusters.hlsl");
        }
#if RTXMG_CHRONO_TIMING
        auto afterBinding = std::chrono::high_resolution_clock::now();
        bindingTimeMs += std::chrono::duration_cast<std::chrono::microseconds>(afterBinding - instStart).count() * 0.001f;
#endif

        auto GetFillClustersPSO = [this](const FillClustersPermutation& shaderPermutation)
            {
                if (!m_fillClustersPSOs[shaderPermutation.index()])
                {
                    std::vector<donut::engine::ShaderMacro> fillClustersMacros;
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("VERTEX_NORMALS", shaderPermutation.isVertexNormalsEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                    nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersMain", &fillClustersMacros, nvrhi::ShaderType::Compute);

                    auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                        .setComputeShader(shader)
                        .addBindingLayout(m_fillClustersBL)
                        .addBindingLayout(m_bindlessBL);

                    m_fillClustersPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                }
                return m_fillClustersPSOs[shaderPermutation.index()];
            };
        
        if (!m_fillClustersTexcoordsPSO)
        {
            nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersTexcoordsMain", nullptr, nvrhi::ShaderType::Compute);

            auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_fillClustersBL)
                .addBindingLayout(m_bindlessBL);

            m_fillClustersTexcoordsPSO = m_device->createComputePipeline(computePipelineDesc);
        }

        RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - instance ", instanceIndex, " creating compute state"));
        auto state = nvrhi::ComputeState()
            .addBindingSet(bindingSet)
            .addBindingSet(m_descriptorTable)  // Bindless descriptor table for displacement textures
            .setIndirectParams(m_fillClustersDispatchIndirectBuffer);

        // Pre-compute buffer bounds for all dispatch checks
        uint32_t fillBufferSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetBytes());
        uint32_t fillElementSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetElementBytes());

        if (m_tessellatorConfig.enableMonolithicClusterBuild)
        {
            RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - monolithic mode");
            // TEMPORARY: Force disable displacement - geometry/material buffers are not populated in RTX Remix
            FillClustersPermutation shaderPermutation = { false /*subd.m_hasDisplacementMaterial*/, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType::All };
            state.setPipeline(GetFillClustersPSO(shaderPermutation));
            commandList->setComputeState(state);
            // Dispatch slot design (matching sample code behavior):
            //   - CopyClusterOffset(AllTypes) writes VERTEX dispatch args to the Limit slot (2)
            //     and TEXCOORD dispatch args to the AllTypes slot (3).
            //     See copy_cluster_offset.comp.slang line 92: when dispatchTypeIndex==All,
            //     it writes InOutFillClustersIndirectArgs[..+ClusterDispatchType::Limit].
            //   - The fill_clusters shader compiled with SURFACE_TYPE_ALL reads
            //     t_ClusterOffsetCounts from slot 3 (ClusterDispatchType::All) to get
            //     the total cluster offset+count for this instance.
            //   - So: dispatch from slot 2 (Limit) for vertex workgroups, shader reads slot 3 (All) for cluster data.
            //     Both are populated by CopyClusterOffset(AllTypes). This is intentional, not a bug.
            uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::Limit) * fillElementSize;
            RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - monolithic dispatchIndirect offset=", dispatchIndirectArgsOffset,
                " bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
            if (dispatchIndirectArgsOffset + fillElementSize > fillBufferSize) {
                Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! monolithic offset=", dispatchIndirectArgsOffset,
                    " + elementSize=", fillElementSize, " > bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
            } else {
                commandList->dispatchIndirect(dispatchIndirectArgsOffset);
            }
        }
        else
        {
            RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - permutation mode");
            for (uint32_t i = 0; i <= uint32_t(ShaderPermutationSurfaceType::Limit); i++)
            {
                // TEMPORARY: Force disable displacement - geometry/material buffers are not populated in RTX Remix
                FillClustersPermutation shaderPermutation = { false /*subd.m_hasDisplacementMaterial*/, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType(i) };
                RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - GetFillClustersPSO permutation ", i));
                state.setPipeline(GetFillClustersPSO(shaderPermutation));
                commandList->setComputeState(state);
                uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType(i)) * fillElementSize;
                RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - permutation dispatchIndirect i=", i, " offset=", dispatchIndirectArgsOffset));
                if (dispatchIndirectArgsOffset + fillElementSize > fillBufferSize) {
                    Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! permutation i=", i, " offset=", dispatchIndirectArgsOffset,
                        " + elementSize=", fillElementSize, " > bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
                } else {
                    commandList->dispatchIndirect(dispatchIndirectArgsOffset);
                }
            }
        }

        RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - texcoords dispatch");
        state.setPipeline(m_fillClustersTexcoordsPSO);
        commandList->setComputeState(state);
        uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::AllTypes) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        uint32_t bufferSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetBytes());
        uint32_t elementSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - texcoords dispatchIndirect offset=", dispatchIndirectArgsOffset,
            " bufferSize=", bufferSize, " instanceIndex=", instanceIndex, " m_numInstances=", m_numInstances));
        // Bounds check: ensure offset + element size <= buffer size
        if (dispatchIndirectArgsOffset + elementSize > bufferSize) {
            Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! texcoords dispatchIndirect offset=", dispatchIndirectArgsOffset,
                " + elementSize=", elementSize, " = ", dispatchIndirectArgsOffset + elementSize,
                " > bufferSize=", bufferSize, " instanceIndex=", instanceIndex, " m_numInstances=", m_numInstances));
            continue; // Skip this dispatch to avoid crash
        }
        commandList->dispatchIndirect(dispatchIndirectArgsOffset);
        RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - instance complete");

        surfaceOffset += surfaceCount;

        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            RTXMG_LOG(str::format("Fill Clusters Debug Instance:", instanceIndex, " Mesh:", donutMeshInfo.name, " (Surface:", m_tessellatorConfig.debugSurfaceIndex,
                " Cluster:", m_tessellatorConfig.debugClusterIndex, " Lane:", m_tessellatorConfig.debugLaneIndex, ")"));

            auto debugOutput = m_debugBuffer.Download(commandList);
            uint numElements = debugOutput.front().payloadType;
            vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
        }

        // Unconditional debug readback: tiling debug (slots 0-7) + fill debug (slots 8-15)
        {
            static uint32_t s_fillDbgAttempts = 0;
            if (!g_megageoDbgGotData && instanceIndex == 0) {
                auto debugOutput = m_debugBuffer.Download(commandList);
                s_fillDbgAttempts++;
                bool hasData = false;
                if (debugOutput.size() > 9) {
                    // Check for our SPECIFIC payloadType markers (100+ range to avoid ShaderDebug conflicts)
                    // payloadType=103: compute_cluster_tiling wrote it
                    // payloadType=102: fill_clusters offsetCount
                    // payloadType=104: fill_clusters cluster metadata
                    for (uint32_t i = 0; i < 8 && i < debugOutput.size(); ++i) {
                        if (debugOutput[i].payloadType == 103) { hasData = true; break; }
                    }
                    for (uint32_t i = 8; i <= 15 && i < debugOutput.size(); ++i) {
                        if (debugOutput[i].payloadType == 102 || debugOutput[i].payloadType == 104) { hasData = true; break; }
                    }
                }
                if (hasData) {
                    g_megageoDbgGotData = true;

                    // Slots 0-7: compute_cluster_tiling debug (payloadType=103)
                    // RAW DUMP first - show all data regardless of payloadType
                    Logger::warn("  === compute_cluster_tiling debug (RAW) ===");
                    for (uint32_t i = 0; i < 8 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        Logger::warn(str::format("  tiling[", i, "] RAW: uint=(",
                            e.uintData.x, ",", e.uintData.y, ",", e.uintData.z, ",", e.uintData.w,
                            ") float=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ",", e.floatData.w,
                            ") payloadType=", e.payloadType, " lineNumber=", e.lineNumber));
                        if (e.payloadType == 103) {
                            if (e.lineNumber == 1) {
                                Logger::warn(str::format("    -> ENTRY: surfRange=[", e.uintData.y,
                                    ",", e.uintData.z, ") instanceIndex=", e.uintData.w));
                            } else {
                                uint64_t vtxAddr = uint64_t(e.uintData.z) | (uint64_t(e.uintData.w) << 32);
                                uint32_t clusterIdx, templateLo, templateHi, sizeofF3;
                                memcpy(&clusterIdx, &e.floatData.x, 4);
                                memcpy(&templateLo, &e.floatData.y, 4);
                                memcpy(&templateHi, &e.floatData.z, 4);
                                memcpy(&sizeofF3, &e.floatData.w, 4);
                                uint64_t templateAddr = uint64_t(templateLo) | (uint64_t(templateHi) << 32);
                                Logger::warn(str::format("    -> iSurface=", e.uintData.x,
                                    " vtxOff=", e.uintData.y, " vtxAddr=0x", std::hex, vtxAddr, std::dec,
                                    " clusterIdx=", clusterIdx, " templateAddr=0x", std::hex, templateAddr, std::dec,
                                    " sizeof(float3)=", sizeofF3));
                            }
                        }
                    }

                    // Slot 8: fill_clusters offsetCount (payloadType=102)
                    Logger::warn("  === fill_clusters (what was READ from clusters buffer) ===");
                    if (debugOutput.size() > 8 && debugOutput[8].payloadType == 102) {
                        const auto& e = debugOutput[8];
                        Logger::warn(str::format("  fill[8] offsetCount: offset=", e.uintData.x, " count=", e.uintData.y,
                            " instanceIndex=", e.uintData.z, " dispatchType=", e.uintData.w,
                            " firstPos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                    }

                    // Slots 9-15: fill_clusters cluster metadata (payloadType=104)
                    for (uint32_t i = 9; i <= 15 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        if (e.payloadType == 104) {
                            uint32_t sizeX = e.uintData.z & 0xFFFF;
                            uint32_t sizeY = (e.uintData.z >> 16) & 0xFFFF;
                            Logger::warn(str::format("  fill[", i, "] cluster[", e.uintData.w, "] iSurface=", e.uintData.x,
                                " vtxOff=", e.uintData.y, " size=", sizeX, "x", sizeY,
                                " pos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                        } else if (e.payloadType != 0) {
                            Logger::warn(str::format("  fill[", i, "] payloadType=", e.payloadType));
                        } else {
                            Logger::warn(str::format("  fill[", i, "] EMPTY"));
                        }
                    }

                    // CPU readback of CLAS indirect args buffer to verify what compute_cluster_tiling wrote
                    Logger::warn("  === CLAS IndirectArgs readback (CPU) ===");
                    auto vtxBaseAddr = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
                    Logger::warn(str::format("  clusterVertexPositionsBaseAddress=0x", std::hex, vtxBaseAddr));

                    // Download GPU-side template addresses to compare with CPU copy
                    auto gpuTemplateAddrs = m_templateBuffers.addressesBuffer.Download(commandList);
                    Logger::warn(str::format("  templateAddresses (CPU=", m_templateBuffers.addresses.size(),
                        " GPU=", gpuTemplateAddrs.size(), " templates):"));
                    // Only log templates around index 84 (8x8 cluster) and a few others for context
                    for (uint32_t i = 0; i < gpuTemplateAddrs.size(); ++i) {
                        bool cpuMatch = (i < m_templateBuffers.addresses.size() && m_templateBuffers.addresses[i] == gpuTemplateAddrs[i]);
                        if (!cpuMatch || i >= 80) {
                            Logger::warn(str::format("    template[", i, "] GPU=0x", std::hex, gpuTemplateAddrs[i],
                                " CPU=0x", (i < m_templateBuffers.addresses.size() ? m_templateBuffers.addresses[i] : 0),
                                std::dec, " instSize=", (i < m_templateBuffers.instantiationSizes.size() ? m_templateBuffers.instantiationSizes[i] : 0),
                                cpuMatch ? "" : " *** MISMATCH ***"));
                        }
                    }
                    auto clasArgs = m_clasIndirectArgDataBuffer.Download(commandList);
                    uint32_t numToLog = std::min(uint32_t(clasArgs.size()), uint32_t(8));
                    for (uint32_t i = 0; i < numToLog; ++i) {
                        const auto& a = clasArgs[i];
                        // Find which GPU template index this address matches
                        int gpuTemplateIdx = -1;
                        for (uint32_t t = 0; t < gpuTemplateAddrs.size(); ++t) {
                            if (gpuTemplateAddrs[t] == a.clusterTemplate) { gpuTemplateIdx = (int)t; break; }
                        }
                        int cpuTemplateIdx = -1;
                        for (uint32_t t = 0; t < m_templateBuffers.addresses.size(); ++t) {
                            if (m_templateBuffers.addresses[t] == a.clusterTemplate) { cpuTemplateIdx = (int)t; break; }
                        }
                        Logger::warn(str::format("  clasArg[", i, "] clusterId=", a.clusterIdOffset,
                            " geomIdx=", a.geometryIndexOffsetPacked,
                            " template=0x", std::hex, a.clusterTemplate,
                            " vtxAddr=0x", a.vertexBuffer.startAddress,
                            " stride=", std::dec, a.vertexBuffer.strideInBytes,
                            " gpuTemplateIdx=", gpuTemplateIdx, " cpuTemplateIdx=", cpuTemplateIdx));
                    }

                    // Also readback clusters buffer to cross-reference
                    Logger::warn("  === Clusters buffer readback (CPU) ===");
                    auto clusters = m_clustersBuffer.Download(commandList);
                    numToLog = std::min(uint32_t(clusters.size()), uint32_t(8));
                    for (uint32_t i = 0; i < numToLog; ++i) {
                        const auto& c = clusters[i];
                        uint64_t expectedVtxAddr = vtxBaseAddr + uint64_t(c.nVertexOffset) * 12; // GPU stride=12, not C++ sizeof(float3)=16
                        Logger::warn(str::format("  cluster[", i, "] iSurface=", c.iSurface,
                            " vtxOff=", c.nVertexOffset, " offset=(", c.offset.x, ",", c.offset.y, ")",
                            " size=", uint32_t(c.sizeX), "x", uint32_t(c.sizeY),
                            " expectedVtxAddr=0x", std::hex, expectedVtxAddr));
                    }

                    // Slots 16-23: Control point diagnostic data (payloadType=105)
                    Logger::warn("  === Control Point Chain (GPU shader readback) ===");
                    for (uint32_t i = 16; i <= 23 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        if (e.payloadType != 105) {
                            Logger::warn(str::format("  cp[", i, "] NOT WRITTEN (payloadType=", e.payloadType, ")"));
                            continue;
                        }
                        switch (e.lineNumber) {
                            case 1: // Surface descriptor
                                Logger::warn(str::format("  cp[16] SurfaceDesc: iSurface=", e.uintData.x,
                                    " field0=0x", std::hex, e.uintData.y, std::dec,
                                    " firstControlPoint=", e.uintData.z));
                                break;
                            case 2: // Control point indices
                            {
                                float firstCP;
                                memcpy(&firstCP, &e.floatData.x, 4);
                                uint32_t firstCPu;
                                memcpy(&firstCPu, &firstCP, 4);
                                Logger::warn(str::format("  cp[17] CPIndices: cpi[0]=", e.uintData.x,
                                    " cpi[1]=", e.uintData.y, " cpi[2]=", e.uintData.z,
                                    " cpi[3]=", e.uintData.w, " (firstCP=", firstCPu, ")"));
                                break;
                            }
                            case 3: case 4: case 5: case 6: // Control point positions
                                Logger::warn(str::format("  cp[", i, "] pos[cpi", e.lineNumber-3, "=", e.uintData.x,
                                    "] = (", e.floatData.x, ", ", e.floatData.y, ", ", e.floatData.z, ")"));
                                break;
                            case 7: // Buffer dimensions
                            {
                                uint32_t cpiStride, posStride;
                                memcpy(&cpiStride, &e.floatData.x, 4);
                                memcpy(&posStride, &e.floatData.y, 4);
                                Logger::warn(str::format("  cp[22] BufferDims: numDescriptors=", e.uintData.x,
                                    " numCPIndices=", e.uintData.y, " numPositions=", e.uintData.z,
                                    " descStride=", e.uintData.w, " cpiStride=", cpiStride, " posStride=", posStride));
                                break;
                            }
                            case 8: // Evaluated limit position
                                Logger::warn(str::format("  cp[23] EvalLimit: pos=(", e.floatData.x,
                                    ", ", e.floatData.y, ", ", e.floatData.z, ")"));
                                break;
                            default:
                                Logger::warn(str::format("  cp[", i, "] lineNumber=", e.lineNumber));
                                break;
                        }
                    }

                    // Slots 24-30: Bilinear vs Evaluate() comparison (payloadType=106)
                    Logger::warn("  === Bilinear vs Evaluate Comparison (GPU shader) ===");
                    for (uint32_t i = 24; i <= 30 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        if (e.payloadType != 106) {
                            Logger::warn(str::format("  cmp[", i, "] NOT WRITTEN (payloadType=", e.payloadType, ")"));
                            continue;
                        }
                        switch (e.lineNumber) {
                            case 1: // Slot 24: path info + bilinear pos + diff at UV=(0.5, 0.5)
                            {
                                const char* pathNames[] = {"PureBSpline", "RegularBSpline", "Limit"};
                                uint32_t path = e.uintData.x < 3 ? e.uintData.x : 2;
                                float diffX, diffY;
                                memcpy(&diffX, &e.uintData.z, 4);
                                memcpy(&diffY, &e.uintData.w, 4);
                                Logger::warn(str::format("  cmp[24] path=", pathNames[path],
                                    " planIdx=", e.uintData.y,
                                    " bilin@(0.5,0.5)=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")",
                                    " diff_xy=(", diffX, ",", diffY, ")"));
                                break;
                            }
                            case 2: // Slot 25: Evaluate result at UV=(0.5, 0.5)
                            {
                                float diffZ;
                                memcpy(&diffZ, &e.uintData.x, 4);
                                Logger::warn(str::format("  cmp[25] subd@(0.5,0.5)=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")",
                                    " diff_z=", diffZ));
                                break;
                            }
                            case 3: // Slot 26: bilinear pos + diff at UV=(0.25, 0.75)
                            {
                                float diffX, diffY, diffZ;
                                memcpy(&diffX, &e.uintData.x, 4);
                                memcpy(&diffY, &e.uintData.y, 4);
                                memcpy(&diffZ, &e.uintData.z, 4);
                                Logger::warn(str::format("  cmp[26] bilin@(0.25,0.75)=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")",
                                    " diff_xyz=(", diffX, ",", diffY, ",", diffZ, ")"));
                                break;
                            }
                            case 4: // Slot 27: Evaluate at UV=(0.25, 0.75)
                            {
                                Logger::warn(str::format("  cmp[27] subd@(0.25,0.75)=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                                break;
                            }
                            case 5: // Slot 28: Surface 0 comparison
                            {
                                const char* pathNames[] = {"PureBSpline", "RegularBSpline", "Limit"};
                                uint32_t path0 = e.uintData.x < 3 ? e.uintData.x : 2;
                                Logger::warn(str::format("  cmp[28] surf0: path=", pathNames[path0],
                                    " planIdx=", e.uintData.y,
                                    " firstCP=", e.uintData.z,
                                    " bilin-subd=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                                break;
                            }
                            case 6: // Slot 29: Sharpness info
                            {
                                Logger::warn(str::format("  cmp[29] hasSharp=", e.uintData.x,
                                    " bndMask=0x", std::hex, e.uintData.y, std::dec,
                                    " numCP=", e.uintData.z,
                                    " evalPath=", e.uintData.w,
                                    " sharp=", e.floatData.x,
                                    " computedSharp=", e.floatData.y));
                                break;
                            }
                            case 7: // Slot 30: BSpline weights at UV=(0.5,0.5)
                            {
                                float outerSum, wP0, wP3, wP15;
                                memcpy(&outerSum, &e.uintData.x, 4);
                                memcpy(&wP0, &e.uintData.y, 4);
                                memcpy(&wP3, &e.uintData.z, 4);
                                memcpy(&wP15, &e.uintData.w, 4);
                                Logger::warn(str::format("  cmp[30] wP[5,6,9,10]=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ",", e.floatData.w, ")",
                                    " outerSum=", outerSum,
                                    " wP[0]=", wP0, " wP[3]=", wP3, " wP[15]=", wP15));
                                break;
                            }
                        }
                    }
                    // Slots 31-38: Per-vertex UV and position for first 8 vertices of cluster 0 (payloadType=107)
                    Logger::warn("  === Per-Vertex UV + Position (first 8 vertices, cluster 0) ===");
                    for (uint32_t i = 31; i <= 38 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        if (e.payloadType != 107) {
                            Logger::warn(str::format("  vtx[", i-31, "] NOT WRITTEN (payloadType=", e.payloadType, ")"));
                            continue;
                        }
                        float uvX, uvY;
                        memcpy(&uvX, &e.uintData.x, 4);
                        memcpy(&uvY, &e.uintData.y, 4);
                        uint32_t sizeX = e.uintData.w & 0xFFFF;
                        uint32_t sizeY = (e.uintData.w >> 16) & 0xFFFF;
                        Logger::warn(str::format("  vtx[", e.uintData.z, "] uv=(", uvX, ",", uvY,
                            ") pos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z,
                            ") clusterSize=", sizeX, "x", sizeY));
                    }

                    // Slots 40-55: Limit surface diagnostics — now read post-loop (after all instances)
                    Logger::warn("  === LIMIT Surface Diagnostics: deferred to post-loop readback ===");

                    // === Output Vertex Positions Readback ===
                    // Read back the actual output of fill_clusters (clusterVertexPositions buffer)
                    Logger::warn("  === Output Vertex Positions (clusterVertexPositionsBuffer) ===");
                    {
                        size_t vtxBufBytes = accels.clusterVertexPositionsBuffer.GetBytes();
                        uint32_t vtxBufCount = (uint32_t)(vtxBufBytes / 12); // stride=12 on GPU (float3)
                        Logger::warn(str::format("  vtxBuf: bytes=", vtxBufBytes, " count=", vtxBufCount));

                        // Create readback buffer and copy
                        auto readbackBuf = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                            .byteSize = vtxBufBytes,
                            .cpuAccess = nvrhi::CpuAccessMode::Read,
                        });

                        commandList->close();
                        commandList->getDevice()->executeCommandList(commandList);
                        commandList->getDevice()->waitForIdle();

                        commandList->open();
                        commandList->copyBuffer(readbackBuf.Get(), 0, accels.clusterVertexPositionsBuffer.Get(), 0, vtxBufBytes);
                        commandList->close();
                        commandList->getDevice()->executeCommandList(commandList);
                        commandList->getDevice()->waitForIdle();

                        const float* vtxData = static_cast<const float*>(
                            commandList->getDevice()->mapBuffer(readbackBuf.Get(), nvrhi::CpuAccessMode::Read));
                        if (vtxData) {
                            // Log first cluster's corner vertices (9x9 grid for 8x8 cluster)
                            // Row-major: vertex(row, col) = row * 9 + col
                            uint32_t sampleIndices[] = {0, 8, 72, 80, 1, 9, 40, 4, 36, 44};
                            const char* sampleNames[] = {"(r0,c0)", "(r0,c8)", "(r8,c0)", "(r8,c8)", "(r0,c1)", "(r1,c0)", "(r4,c4)", "(r0,c4)", "(r4,c0)", "(r4,c8)"};
                            for (int ci = 0; ci < 10; ci++) {
                                uint32_t vi = sampleIndices[ci];
                                if (vi < vtxBufCount) {
                                    Logger::warn(str::format("  outVtx[", vi, "] ", sampleNames[ci],
                                        " = (", vtxData[vi*3], ", ", vtxData[vi*3+1], ", ", vtxData[vi*3+2], ")"));
                                }
                            }

                            // Log first 3 vertices of cluster[1] (vtxOff=81)
                            for (uint32_t vi = 81; vi < 84 && vi < vtxBufCount; vi++) {
                                Logger::warn(str::format("  outVtx[", vi, "] cluster1 = (", vtxData[vi*3], ", ", vtxData[vi*3+1], ", ", vtxData[vi*3+2], ")"));
                            }

                            // Check for all-same positions in cluster 0
                            bool allSame = true;
                            for (uint32_t vi = 1; vi < std::min(vtxBufCount, 81u); vi++) {
                                if (vtxData[vi*3] != vtxData[0] || vtxData[vi*3+1] != vtxData[1] || vtxData[vi*3+2] != vtxData[2]) {
                                    allSame = false;
                                    break;
                                }
                            }
                            Logger::warn(str::format("  cluster0 allSame=", allSame ? "YES (BUG!)" : "no (vertices vary)"));

                            // Compute expected bilinear at corners by re-reading the input buffers
                            if (clusters.size() > 0) {
                                uint32_t iSurf = clusters[0].iSurface;
                                auto descBuf2 = subd.m_vertexDeviceData.surfaceDescriptors;
                                auto cpiBuf2 = subd.m_vertexDeviceData.controlPointIndices;
                                auto posBuf2 = subd.m_positionsBuffer;
                                uint32_t dBytes = (uint32_t)descBuf2->getDesc().byteSize;
                                uint32_t cBytes = (uint32_t)cpiBuf2->getDesc().byteSize;
                                uint32_t pBytes = (uint32_t)posBuf2->getDesc().byteSize;
                                uint32_t nDescs = dBytes / 8, nCPIs = cBytes / 4, nPos = pBytes / 12;

                                auto dr = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = dBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                                auto cr = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = cBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                                auto pr = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = pBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});

                                commandList->open();
                                commandList->copyBuffer(dr.Get(), 0, descBuf2.Get(), 0, dBytes);
                                commandList->copyBuffer(cr.Get(), 0, cpiBuf2.Get(), 0, cBytes);
                                commandList->copyBuffer(pr.Get(), 0, posBuf2.Get(), 0, pBytes);
                                commandList->close();
                                commandList->getDevice()->executeCommandList(commandList);
                                commandList->getDevice()->waitForIdle();

                                auto dd = static_cast<const uint32_t*>(commandList->getDevice()->mapBuffer(dr.Get(), nvrhi::CpuAccessMode::Read));
                                auto cd = static_cast<const int32_t*>(commandList->getDevice()->mapBuffer(cr.Get(), nvrhi::CpuAccessMode::Read));
                                auto pd = static_cast<const float*>(commandList->getDevice()->mapBuffer(pr.Get(), nvrhi::CpuAccessMode::Read));
                                if (dd && cd && pd && iSurf < nDescs) {
                                    uint32_t firstCP = dd[iSurf * 2 + 1];
                                    if (firstCP + 3 < nCPIs) {
                                        int32_t cpi[4] = {cd[firstCP], cd[firstCP+1], cd[firstCP+2], cd[firstCP+3]};
                                        float cp[4][3];
                                        for (int c = 0; c < 4; c++) {
                                            if (cpi[c] >= 0 && (uint32_t)cpi[c] < nPos) {
                                                cp[c][0] = pd[cpi[c]*3]; cp[c][1] = pd[cpi[c]*3+1]; cp[c][2] = pd[cpi[c]*3+2];
                                            } else { cp[c][0] = cp[c][1] = cp[c][2] = 0; }
                                        }
                                        Logger::warn(str::format("  CPU bilinear for cluster0 iSurface=", iSurf,
                                            " firstCP=", firstCP, " cpi=[", cpi[0], ",", cpi[1], ",", cpi[2], ",", cpi[3], "]"));
                                        Logger::warn(str::format("    p00=(", cp[0][0], ",", cp[0][1], ",", cp[0][2], ")"));
                                        Logger::warn(str::format("    p10=(", cp[1][0], ",", cp[1][1], ",", cp[1][2], ")"));
                                        Logger::warn(str::format("    p11=(", cp[2][0], ",", cp[2][1], ",", cp[2][2], ")"));
                                        Logger::warn(str::format("    p01=(", cp[3][0], ",", cp[3][1], ",", cp[3][2], ")"));
                                        float center[3];
                                        for (int d = 0; d < 3; d++)
                                            center[d] = 0.25f * (cp[0][d] + cp[1][d] + cp[2][d] + cp[3][d]);
                                        Logger::warn(str::format("    expected center=(", center[0], ",", center[1], ",", center[2], ")"));

                                        // Compare GPU outVtx[0] with CPU p00
                                        if (vtxBufCount > 0) {
                                            float dx = vtxData[0]-cp[0][0], dy = vtxData[1]-cp[0][1], dz = vtxData[2]-cp[0][2];
                                            Logger::warn(str::format("    outVtx[0] vs p00: diff=(", dx, ",", dy, ",", dz, ")",
                                                (dx==0 && dy==0 && dz==0) ? " MATCH" : " MISMATCH"));
                                        }
                                        // Compare GPU outVtx[80] with CPU bilinear at UV=(1,1) = p11
                                        if (vtxBufCount > 80) {
                                            float dx = vtxData[80*3]-cp[2][0], dy = vtxData[80*3+1]-cp[2][1], dz = vtxData[80*3+2]-cp[2][2];
                                            Logger::warn(str::format("    outVtx[80] vs p11: diff=(", dx, ",", dy, ",", dz, ")",
                                                (dx==0 && dy==0 && dz==0) ? " MATCH" : " MISMATCH"));
                                        }
                                    }
                                }
                                if (dd) commandList->getDevice()->unmapBuffer(dr.Get());
                                if (cd) commandList->getDevice()->unmapBuffer(cr.Get());
                                if (pd) commandList->getDevice()->unmapBuffer(pr.Get());
                            }

                            commandList->getDevice()->unmapBuffer(readbackBuf.Get());
                        } else {
                            Logger::warn("  FAILED to map vtx readback buffer");
                        }
                        commandList->open();
                    }
                } else if (s_fillDbgAttempts % 50 == 0) {
                }
            }
        }

#if RTXMG_CHRONO_TIMING
        auto instEnd = std::chrono::high_resolution_clock::now();
        dispatchTimeMs += std::chrono::duration_cast<std::chrono::microseconds>(instEnd - afterBinding).count() * 0.001f;
#endif
    }

    stats::clusterAccelSamplers.fillClustersTime.Stop();

    // Post-loop: Survey ALL Limit surfaces + read output vertices for Limit clusters
    {
        static bool s_limitSurveyDone = false;
        if (!s_limitSurveyDone) {
            s_limitSurveyDone = true;

            // 1) GPU shader debug readback (slots 40-55) for one detailed Limit surface
            auto debugOutput = m_debugBuffer.Download(commandList);
            if (debugOutput.size() > 55 && debugOutput[40].payloadType == 110) {
                g_megageoDbgGotLimitData = true;
                Logger::warn("  === LIMIT Surface Detailed Diagnostic (shader slots 40-55) ===");
                for (uint32_t i = 40; i <= 55 && i < debugOutput.size(); ++i) {
                    const auto& e = debugOutput[i];
                    if (e.payloadType != 110) {
                        Logger::warn(str::format("  limit[", i, "] EMPTY (payloadType=", e.payloadType, ")"));
                        continue;
                    }
                    switch (e.lineNumber) {
                        case 0:
                            Logger::warn(str::format("  limit[40] iSurface=", e.uintData.x,
                                " planIdx=", e.uintData.y, " numCP=", e.uintData.z,
                                " numPP=", e.uintData.w,
                                " evalPos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                            break;
                        case 1:
                        {
                            uint32_t sOff, sSize, tOff;
                            memcpy(&sOff, &e.floatData.x, 4);
                            memcpy(&sSize, &e.floatData.y, 4);
                            memcpy(&tOff, &e.floatData.z, 4);
                            Logger::warn(str::format("  limit[41] field0=0x", std::hex, e.uintData.x, std::dec,
                                " firstCP=", e.uintData.y, " ppOff=", e.uintData.z,
                                " ppOffNext=", e.uintData.w,
                                " stencilOff=", sOff, " stencilSize=", sSize, " treeOff=", tOff));
                            break;
                        }
                        case 2:
                        {
                            uint32_t offX, offY;
                            memcpy(&offX, &e.floatData.y, 4);
                            memcpy(&offY, &e.floatData.z, 4);
                            Logger::warn(str::format("  limit[42] clusterIdx=", e.uintData.x,
                                " vtxOff=", e.uintData.y,
                                " size=", (e.uintData.z & 0xFFFF), "x", ((e.uintData.z >> 16) & 0xFFFF),
                                " groupClusterIdx=", e.uintData.w));
                            break;
                        }
                        default:
                            if (e.lineNumber >= 3 && e.lineNumber <= 6)
                                Logger::warn(str::format("  limit[", i, "] pp[", e.uintData.y,
                                    "] idx=", e.uintData.x,
                                    " pos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")"));
                            else if (e.lineNumber >= 7 && e.lineNumber <= 10)
                                Logger::warn(str::format("  limit[", i, "] CP[", e.uintData.y,
                                    "] cpi=", e.uintData.x,
                                    " pos=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")",
                                    " numCP=", e.uintData.w));
                            else if (e.lineNumber >= 11 && e.lineNumber <= 14)
                                Logger::warn(str::format("  limit[", i, "] stencil[", e.uintData.y,
                                    "] val=", e.floatData.x, " idx=", e.uintData.x,
                                    " numCP=", e.uintData.z));
                            else if (e.lineNumber == 15)
                                Logger::warn(str::format("  limit[55] EvalLimit@(0.5,0.5)=(", e.floatData.x, ",", e.floatData.y, ",", e.floatData.z, ")",
                                    " iSurface=", e.uintData.x, " numPP=", e.uintData.y,
                                    " ppOff=", e.uintData.z, " instIdx=", e.uintData.w));
                            break;
                    }
                }
            }

            // 2) CPU-side survey: ALL Limit surfaces across ALL instances
            Logger::warn("  === ALL LIMIT SURFACES SURVEY ===");
            const auto& subdMeshes = scene.GetSubdMeshes();
            const auto& instances = scene.GetSubdMeshInstances();
            uint32_t maxInst = std::min((uint32_t)instances.size(), m_instanceCapacity);
            uint32_t totalLimitSurfaces = 0;
            std::vector<uint32_t> surveyedMeshes; // avoid re-surveying same mesh

            // Read back the output vertex positions buffer (once, shared across all instances)
            size_t vtxBufBytes = accels.clusterVertexPositionsBuffer.GetBytes();
            uint32_t vtxBufCount = (uint32_t)(vtxBufBytes / 12);
            auto vtxReadback = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                .byteSize = vtxBufBytes, .cpuAccess = nvrhi::CpuAccessMode::Read });
            commandList->close();
            commandList->getDevice()->executeCommandList(commandList);
            commandList->getDevice()->waitForIdle();
            commandList->open();
            commandList->copyBuffer(vtxReadback.Get(), 0, accels.clusterVertexPositionsBuffer.Get(), 0, vtxBufBytes);
            commandList->close();
            commandList->getDevice()->executeCommandList(commandList);
            commandList->getDevice()->waitForIdle();
            const float* vtxData = static_cast<const float*>(
                commandList->getDevice()->mapBuffer(vtxReadback.Get(), nvrhi::CpuAccessMode::Read));

            // Read back clusters buffer (Cluster struct = 16 bytes: iSurface, nVertexOffset, offset, sizeX|sizeY)
            size_t clusterBufBytes = m_clustersBuffer.GetBytes();
            uint32_t clusterEntries = (uint32_t)(clusterBufBytes / 16);
            auto clusterReadback = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                .byteSize = clusterBufBytes, .cpuAccess = nvrhi::CpuAccessMode::Read });
            commandList->open();
            commandList->copyBuffer(clusterReadback.Get(), 0, m_clustersBuffer.Get(), 0, clusterBufBytes);
            commandList->close();
            commandList->getDevice()->executeCommandList(commandList);
            commandList->getDevice()->waitForIdle();
            const uint32_t* clusterRaw = static_cast<const uint32_t*>(
                commandList->getDevice()->mapBuffer(clusterReadback.Get(), nvrhi::CpuAccessMode::Read));

            for (uint32_t inst = 0; inst < maxInst; inst++) {
                const auto& instance = instances[inst];
                const auto& subd = *subdMeshes[instance.meshID];

                uint32_t limitStart = subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::Limit)];
                uint32_t limitEnd = subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
                uint32_t limitCount = limitEnd - limitStart;
                if (limitCount == 0) continue;

                bool newMesh = std::find(surveyedMeshes.begin(), surveyedMeshes.end(), instance.meshID) == surveyedMeshes.end();
                if (!newMesh) continue; // already surveyed this mesh
                surveyedMeshes.push_back(instance.meshID);

                totalLimitSurfaces += limitCount;

                // Read back this mesh's descriptor buffer
                auto descBuf = subd.m_vertexDeviceData.surfaceDescriptors;
                uint32_t descBytes = (uint32_t)descBuf->getDesc().byteSize;
                uint32_t numDescs = descBytes / 8;
                auto descRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                    .byteSize = descBytes, .cpuAccess = nvrhi::CpuAccessMode::Read });
                commandList->open();
                commandList->copyBuffer(descRB.Get(), 0, descBuf.Get(), 0, descBytes);
                commandList->close();
                commandList->getDevice()->executeCommandList(commandList);
                commandList->getDevice()->waitForIdle();
                const uint32_t* descData = static_cast<const uint32_t*>(
                    commandList->getDevice()->mapBuffer(descRB.Get(), nvrhi::CpuAccessMode::Read));

                // Read back plans buffer to get numCP per planIdx
                auto plansBuf = subd.GetTopologyMap()->plansBuffer;
                uint32_t plansBytes = (uint32_t)plansBuf->getDesc().byteSize;
                auto plansRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                    .byteSize = plansBytes, .cpuAccess = nvrhi::CpuAccessMode::Read });
                commandList->open();
                commandList->copyBuffer(plansRB.Get(), 0, plansBuf.Get(), 0, plansBytes);
                commandList->close();
                commandList->getDevice()->executeCommandList(commandList);
                commandList->getDevice()->waitForIdle();
                const uint8_t* plansData = static_cast<const uint8_t*>(
                    commandList->getDevice()->mapBuffer(plansRB.Get(), nvrhi::CpuAccessMode::Read));
                uint32_t planStride = (uint32_t)(plansBuf->getDesc().structStride);
                if (planStride == 0) planStride = 40; // fallback: sizeof(SubdivisionPlanHLSL)

                Logger::warn(str::format("  inst[", inst, "] meshID=", instance.meshID,
                    " surfaces=", subd.m_surfaceCount,
                    " limitRange=[", limitStart, "..", limitEnd, ") count=", limitCount,
                    " planStride=", planStride));

                // Log each Limit surface
                for (uint32_t s = limitStart; s < limitEnd; s++) {
                    if (s >= numDescs) break;
                    uint32_t field0 = descData[s * 2];
                    uint32_t firstCP = descData[s * 2 + 1];
                    uint32_t planIdx = (field0 >> 12) & 0xFFFFF;

                    // Read numCP from plans buffer
                    uint16_t numCP = 0;
                    if (planIdx * planStride + 2 <= plansBytes) {
                        memcpy(&numCP, plansData + planIdx * planStride, 2);
                    }

                    Logger::warn(str::format("    limitSurf[", s, "] planIdx=", planIdx,
                        " numCP=", numCP, " firstCP=", firstCP,
                        " field0=0x", std::hex, field0, std::dec));
                }

                if (descData) commandList->getDevice()->unmapBuffer(descRB.Get());
                if (plansData) commandList->getDevice()->unmapBuffer(plansRB.Get());
            }

            Logger::warn(str::format("  TOTAL: ", totalLimitSurfaces, " Limit surfaces across ",
                surveyedMeshes.size(), " unique meshes"));

            // 3) Proper per-instance cluster -> mesh mapping using clusterOffsetCounts
            if (vtxData && clusterRaw) {
                Logger::warn("  === CLUSTER CORNER + CP CHECK (using per-instance offsets) ===");

                // Read back clusterOffsetCounts to know which instance owns which clusters
                // Layout: instanceIndex * NumTypes(4) + dispatchType, uint2(offset, count)
                size_t offsetBufBytes = m_clusterOffsetCountsBuffer.GetBytes();
                auto offsetRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{
                    .byteSize = offsetBufBytes, .cpuAccess = nvrhi::CpuAccessMode::Read });
                commandList->open();
                commandList->copyBuffer(offsetRB.Get(), 0, m_clusterOffsetCountsBuffer.Get(), 0, offsetBufBytes);
                commandList->close();
                commandList->getDevice()->executeCommandList(commandList);
                commandList->getDevice()->waitForIdle();
                const uint32_t* offsetData = static_cast<const uint32_t*>(
                    commandList->getDevice()->mapBuffer(offsetRB.Get(), nvrhi::CpuAccessMode::Read));

                // Build instance -> cluster range mapping
                struct InstRange {
                    uint32_t instIdx, meshID, clusterStart, clusterCount;
                    uint32_t limitStart, limitEnd; // surface indices for Limit range
                };
                std::vector<InstRange> instRanges;
                for (uint32_t inst = 0; inst < maxInst; inst++) {
                    // All slot = inst * 4 + 3, each entry = 2 uint32s (offset, count)
                    uint32_t allIdx = inst * 4 + 3;
                    uint32_t offset = offsetData[allIdx * 2];
                    uint32_t count = offsetData[allIdx * 2 + 1];
                    if (count == 0) continue;
                    const auto& ii = instances[inst];
                    const auto& ss = *subdMeshes[ii.meshID];
                    uint32_t ls = ss.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::Limit)];
                    uint32_t le = ss.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
                    instRanges.push_back({inst, ii.meshID, offset, count, ls, le});
                }

                // Per-mesh readback cache
                struct MeshData {
                    uint32_t meshID;
                    const uint32_t* descData = nullptr;
                    const uint32_t* cpiData = nullptr;
                    const float* posData = nullptr;
                    const uint8_t* plansData = nullptr;
                    uint32_t numDescs = 0, numCPIs = 0, numPositions = 0, planStride = 0;
                    nvrhi::BufferHandle descRB, cpiRB, posRB, plansRB;
                };
                std::vector<MeshData> meshCache;

                auto getMeshData = [&](uint32_t meshID) -> MeshData* {
                    for (auto& m : meshCache) if (m.meshID == meshID) return &m;
                    const auto& ss = *subdMeshes[meshID];
                    MeshData md;
                    md.meshID = meshID;
                    // Descriptors
                    auto descBuf = ss.m_vertexDeviceData.surfaceDescriptors;
                    uint32_t descBytes = (uint32_t)descBuf->getDesc().byteSize;
                    md.numDescs = descBytes / 8;
                    md.descRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = descBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                    commandList->open(); commandList->copyBuffer(md.descRB.Get(), 0, descBuf.Get(), 0, descBytes);
                    commandList->close(); commandList->getDevice()->executeCommandList(commandList); commandList->getDevice()->waitForIdle();
                    md.descData = static_cast<const uint32_t*>(commandList->getDevice()->mapBuffer(md.descRB.Get(), nvrhi::CpuAccessMode::Read));
                    // CPI
                    auto cpiBuf = ss.m_vertexDeviceData.controlPointIndices;
                    uint32_t cpiBytes = (uint32_t)cpiBuf->getDesc().byteSize;
                    md.numCPIs = cpiBytes / 4;
                    md.cpiRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = cpiBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                    commandList->open(); commandList->copyBuffer(md.cpiRB.Get(), 0, cpiBuf.Get(), 0, cpiBytes);
                    commandList->close(); commandList->getDevice()->executeCommandList(commandList); commandList->getDevice()->waitForIdle();
                    md.cpiData = static_cast<const uint32_t*>(commandList->getDevice()->mapBuffer(md.cpiRB.Get(), nvrhi::CpuAccessMode::Read));
                    // Positions
                    auto posBuf = ss.m_positionsBuffer;
                    uint32_t posBytes = (uint32_t)posBuf->getDesc().byteSize;
                    md.numPositions = posBytes / 12;
                    md.posRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = posBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                    commandList->open(); commandList->copyBuffer(md.posRB.Get(), 0, posBuf.Get(), 0, posBytes);
                    commandList->close(); commandList->getDevice()->executeCommandList(commandList); commandList->getDevice()->waitForIdle();
                    md.posData = static_cast<const float*>(commandList->getDevice()->mapBuffer(md.posRB.Get(), nvrhi::CpuAccessMode::Read));
                    // Plans
                    auto plansBuf = ss.GetTopologyMap()->plansBuffer;
                    uint32_t plansBytes = (uint32_t)plansBuf->getDesc().byteSize;
                    md.planStride = (uint32_t)(plansBuf->getDesc().structStride);
                    if (md.planStride == 0) md.planStride = 40;
                    md.plansRB = commandList->getDevice()->createBuffer(nvrhi::BufferDesc{.byteSize = plansBytes, .cpuAccess = nvrhi::CpuAccessMode::Read});
                    commandList->open(); commandList->copyBuffer(md.plansRB.Get(), 0, plansBuf.Get(), 0, plansBytes);
                    commandList->close(); commandList->getDevice()->executeCommandList(commandList); commandList->getDevice()->waitForIdle();
                    md.plansData = static_cast<const uint8_t*>(commandList->getDevice()->mapBuffer(md.plansRB.Get(), nvrhi::CpuAccessMode::Read));
                    meshCache.push_back(md);
                    return &meshCache.back();
                };

                // Helper to log cluster details
                auto logCluster = [&](uint32_t c, uint32_t instIdx, uint32_t meshID, uint32_t limitStart, uint32_t limitEnd, const char* label) {
                    uint32_t iSurf = clusterRaw[c * 4 + 0];
                    uint32_t vtxOff = clusterRaw[c * 4 + 1];
                    uint32_t sizePacked = clusterRaw[c * 4 + 3];
                    uint32_t sizeX = sizePacked & 0xFF;
                    uint32_t sizeY = (sizePacked >> 8) & 0xFF;
                    if (sizeX == 0 || sizeY == 0) return;

                    bool isLimit = (iSurf >= limitStart && iSurf < limitEnd);
                    uint32_t vertsX = sizeX + 1, vertsY = sizeY + 1;
                    uint32_t vertsPerCluster = vertsX * vertsY;

                    auto getVtx = [&](uint32_t idx) -> std::string {
                        uint32_t vi = vtxOff + idx;
                        if (vi >= vtxBufCount) return "(OOB)";
                        float x = vtxData[vi*3], y = vtxData[vi*3+1], z = vtxData[vi*3+2];
                        return str::format("(", x, ",", y, ",", z, ")");
                    };

                    Logger::warn(str::format("  ", label, " cluster[", c, "] inst=", instIdx,
                        " meshID=", meshID, " iSurf=", iSurf,
                        " type=", (isLimit ? "LIMIT" : "RegBSpline"),
                        " vtxOff=", vtxOff, " size=", sizeX, "x", sizeY));
                    Logger::warn(str::format("    corners: (0,0)=", getVtx(0),
                        " (1,0)=", getVtx(sizeX),
                        " (0,1)=", getVtx(sizeY * vertsX),
                        " (1,1)=", getVtx(vertsPerCluster - 1)));

                    MeshData* md = getMeshData(meshID);
                    if (!md || iSurf >= md->numDescs) return;

                    uint32_t field0 = md->descData[iSurf * 2];
                    uint32_t firstCP = md->descData[iSurf * 2 + 1];
                    uint32_t planIdx = (field0 >> 12) & 0xFFFFF;

                    uint16_t numCP = 0, coarseFaceSize = 0;
                    int16_t coarseFaceQuadrant = 0;
                    if (md->plansData) {
                        const uint8_t* pp = md->plansData + planIdx * md->planStride;
                        memcpy(&numCP, pp, 2);
                        memcpy(&coarseFaceSize, pp + 2, 2);
                        memcpy(&coarseFaceQuadrant, pp + 4, 2);
                    }

                    Logger::warn(str::format("    desc: planIdx=", planIdx, " firstCP=", firstCP,
                        " numCP=", numCP, " coarseFaceSize=", coarseFaceSize,
                        " quadrant=", coarseFaceQuadrant));

                    float cpCX = 0, cpCY = 0, cpCZ = 0;
                    uint32_t vc = 0;
                    for (uint32_t cp = 0; cp < numCP && cp < 16; cp++) {
                        uint32_t cpiIdx = firstCP + cp;
                        if (cpiIdx < md->numCPIs) {
                            uint32_t posIdx = md->cpiData[cpiIdx];
                            if (posIdx < md->numPositions) {
                                float px = md->posData[posIdx*3], py = md->posData[posIdx*3+1], pz = md->posData[posIdx*3+2];
                                Logger::warn(str::format("    CP[", cp, "] posIdx=", posIdx,
                                    " pos=(", px, ",", py, ",", pz, ")"));
                                cpCX += px; cpCY += py; cpCZ += pz; vc++;
                            }
                        }
                    }
                    if (vc > 0) {
                        cpCX /= vc; cpCY /= vc; cpCZ /= vc;
                        float mnX=1e30f, mxX=-1e30f, mnY=1e30f, mxY=-1e30f, mnZ=1e30f, mxZ=-1e30f;
                        for (uint32_t v = 0; v < vertsPerCluster && (vtxOff+v) < vtxBufCount; v++) {
                            float x = vtxData[(vtxOff+v)*3], y = vtxData[(vtxOff+v)*3+1], z = vtxData[(vtxOff+v)*3+2];
                            mnX = std::min(mnX, x); mxX = std::max(mxX, x);
                            mnY = std::min(mnY, y); mxY = std::max(mxY, y);
                            mnZ = std::min(mnZ, z); mxZ = std::max(mxZ, z);
                        }
                        float bcX=(mnX+mxX)/2, bcY=(mnY+mxY)/2, bcZ=(mnZ+mxZ)/2;
                        float dist = sqrtf((cpCX-bcX)*(cpCX-bcX)+(cpCY-bcY)*(cpCY-bcY)+(cpCZ-bcZ)*(cpCZ-bcZ));
                        Logger::warn(str::format("    CP centroid=(", cpCX, ",", cpCY, ",", cpCZ,
                            ") bbox center=(", bcX, ",", bcY, ",", bcZ, ") dist=", dist));
                    }

                    // BILINEAR FLATNESS CHECK: With infinite sharpness, all surfaces should be
                    // perfectly flat (bilinear interpolation of 4 corners). Check if center vertex
                    // matches bilinear midpoint of corners.
                    if (vertsPerCluster >= 4 && (vtxOff + vertsPerCluster - 1) < vtxBufCount) {
                        // Corner indices in 9x9 grid (sizeX=8, sizeY=8):
                        // v[0] = grid(0,0), v[sizeX] = grid(sizeX,0)
                        // v[sizeY*vertsX] = grid(0,sizeY), v[vertsPerCluster-1] = grid(sizeX,sizeY)
                        uint32_t i00 = 0, i10 = sizeX, i01 = sizeY * vertsX, i11 = vertsPerCluster - 1;
                        uint32_t iMid = (sizeY/2) * vertsX + (sizeX/2); // center vertex

                        auto getV3 = [&](uint32_t localIdx, float& ox, float& oy, float& oz) {
                            uint32_t vi = vtxOff + localIdx;
                            ox = vtxData[vi*3]; oy = vtxData[vi*3+1]; oz = vtxData[vi*3+2];
                        };

                        float c00x,c00y,c00z, c10x,c10y,c10z, c01x,c01y,c01z, c11x,c11y,c11z;
                        float midx,midy,midz;
                        getV3(i00, c00x,c00y,c00z);
                        getV3(i10, c10x,c10y,c10z);
                        getV3(i01, c01x,c01y,c01z);
                        getV3(i11, c11x,c11y,c11z);
                        getV3(iMid, midx,midy,midz);

                        // Expected bilinear center = average of 4 corners
                        float expX = (c00x+c10x+c01x+c11x)/4.0f;
                        float expY = (c00y+c10y+c01y+c11y)/4.0f;
                        float expZ = (c00z+c10z+c01z+c11z)/4.0f;

                        float bilinErr = sqrtf((midx-expX)*(midx-expX)+(midy-expY)*(midy-expY)+(midz-expZ)*(midz-expZ));

                        // Also check max deviation of ALL interior vertices from bilinear
                        float maxBilinErr = 0;
                        uint32_t maxErrVtx = 0;
                        for (uint32_t vy = 0; vy < vertsY; vy++) {
                            for (uint32_t vx = 0; vx < vertsX; vx++) {
                                uint32_t vi = vtxOff + vy * vertsX + vx;
                                if (vi >= vtxBufCount) continue;
                                float ax = vtxData[vi*3], ay = vtxData[vi*3+1], az = vtxData[vi*3+2];
                                // Expected bilinear position
                                float u = (float)vx / (float)sizeX;
                                float v = (float)vy / (float)sizeY;
                                float ex = (1-u)*(1-v)*c00x + u*(1-v)*c10x + u*v*c11x + (1-u)*v*c01x;
                                float ey = (1-u)*(1-v)*c00y + u*(1-v)*c10y + u*v*c11y + (1-u)*v*c01y;
                                float ez = (1-u)*(1-v)*c00z + u*(1-v)*c10z + u*v*c11z + (1-u)*v*c01z;
                                float err = sqrtf((ax-ex)*(ax-ex)+(ay-ey)*(ay-ey)+(az-ez)*(az-ez));
                                if (err > maxBilinErr) { maxBilinErr = err; maxErrVtx = vy * vertsX + vx; }
                            }
                        }

                        // PLANARITY CHECK: Compute plane from first 3 corners, check max distance
                        float ax = vtxData[(vtxOff+i00)*3], ay = vtxData[(vtxOff+i00)*3+1], az = vtxData[(vtxOff+i00)*3+2];
                        float bx = vtxData[(vtxOff+i10)*3] - ax, by = vtxData[(vtxOff+i10)*3+1] - ay, bz = vtxData[(vtxOff+i10)*3+2] - az;
                        float cx = vtxData[(vtxOff+i01)*3] - ax, cy = vtxData[(vtxOff+i01)*3+1] - ay, cz = vtxData[(vtxOff+i01)*3+2] - az;
                        // Plane normal = cross(b, c)
                        float nx = by*cz - bz*cy, ny = bz*cx - bx*cz, nz = bx*cy - by*cx;
                        float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
                        float maxPlaneDist = 0;
                        uint32_t maxPlaneVtx = 0;
                        if (nlen > 1e-12f) {
                            nx /= nlen; ny /= nlen; nz /= nlen;
                            for (uint32_t v = 0; v < vertsPerCluster && (vtxOff+v) < vtxBufCount; v++) {
                                float px = vtxData[(vtxOff+v)*3] - ax;
                                float py = vtxData[(vtxOff+v)*3+1] - ay;
                                float pz = vtxData[(vtxOff+v)*3+2] - az;
                                float d = fabsf(px*nx + py*ny + pz*nz);
                                if (d > maxPlaneDist) { maxPlaneDist = d; maxPlaneVtx = v; }
                            }
                        }

                        Logger::warn(str::format("    FLATNESS: centerErr=", bilinErr,
                            " maxBilinErr=", maxBilinErr, " @vtx=", maxErrVtx,
                            " PLANARITY: maxDist=", maxPlaneDist, " @vtx=", maxPlaneVtx,
                            " corners: (0,0)=", getVtx(i00), " (1,0)=", getVtx(i10),
                            " (0,1)=", getVtx(i01), " (1,1)=", getVtx(i11)));
                    }
                };

                // For each instance, check first 2 Limit clusters + first 2 RegBSpline clusters
                uint32_t totalLimitChecked = 0, totalRegChecked = 0;
                for (auto& ir : instRanges) {
                    if (totalLimitChecked >= 10 && totalRegChecked >= 5) break;
                    uint32_t limitChecked = 0, regChecked = 0;
                    for (uint32_t c = ir.clusterStart; c < ir.clusterStart + ir.clusterCount; c++) {
                        if (c >= clusterEntries) break;
                        uint32_t iSurf = clusterRaw[c * 4 + 0];
                        uint32_t sizePacked = clusterRaw[c * 4 + 3];
                        if ((sizePacked & 0xFF) == 0) continue;
                        bool isLimit = (iSurf >= ir.limitStart && iSurf < ir.limitEnd);
                        if (isLimit && limitChecked < 2 && totalLimitChecked < 10) {
                            logCluster(c, ir.instIdx, ir.meshID, ir.limitStart, ir.limitEnd, "LIMIT");
                            limitChecked++; totalLimitChecked++;
                        } else if (!isLimit && regChecked < 2 && totalRegChecked < 5) {
                            logCluster(c, ir.instIdx, ir.meshID, ir.limitStart, ir.limitEnd, "REG");
                            regChecked++; totalRegChecked++;
                        }
                        if (limitChecked >= 2 && regChecked >= 2) break;
                    }
                }
                Logger::warn(str::format("  Total checked: ", totalLimitChecked, " Limit + ", totalRegChecked, " RegBSpline"));

                // TRIANGLE RECONSTRUCTION CHECK: For first instance with Limit surfaces,
                // find 3 consecutive Limit surfaces (same triangle face, quadrants 0,1,2)
                // and verify their corners form a valid triangle tiling.
                // Each triangle face produces 3 sub-quads with quadrant 0, 1, 2.
                // The sub-quads should share edges and reconstruct the original triangle.
                for (auto& ir : instRanges) {
                    if (ir.limitStart >= ir.limitEnd) continue; // No Limit surfaces
                    MeshData* md = getMeshData(ir.meshID);
                    if (!md || !md->plansData) continue;

                    // Find groups of 3 consecutive Limit surfaces from same triangle
                    // They'll have coarseFaceSize=3 and quadrant 0,1,2
                    struct LimitSurf {
                        uint32_t iSurf, planIdx;
                        int16_t quadrant;
                        float cpPos[3][3]; // 3 control points × (x,y,z)
                    };
                    std::vector<LimitSurf> limitSurfs;
                    for (uint32_t s = ir.limitStart; s < ir.limitEnd && s < md->numDescs; s++) {
                        uint32_t field0 = md->descData[s * 2];
                        uint32_t firstCP = md->descData[s * 2 + 1];
                        uint32_t planIdx = (field0 >> 12) & 0xFFFFF;
                        const uint8_t* pp = md->plansData + planIdx * md->planStride;
                        uint16_t numCP = 0; int16_t quadrant = 0; uint16_t coarseFaceSize = 0;
                        memcpy(&numCP, pp, 2);
                        memcpy(&coarseFaceSize, pp + 2, 2);
                        memcpy(&quadrant, pp + 4, 2);
                        if (coarseFaceSize != 3 || numCP != 3) continue;
                        LimitSurf ls;
                        ls.iSurf = s; ls.planIdx = planIdx; ls.quadrant = quadrant;
                        for (uint32_t cp = 0; cp < 3; cp++) {
                            uint32_t cpiIdx = firstCP + cp;
                            uint32_t posIdx = (cpiIdx < md->numCPIs) ? md->cpiData[cpiIdx] : 0;
                            if (posIdx < md->numPositions) {
                                ls.cpPos[cp][0] = md->posData[posIdx*3];
                                ls.cpPos[cp][1] = md->posData[posIdx*3+1];
                                ls.cpPos[cp][2] = md->posData[posIdx*3+2];
                            } else {
                                ls.cpPos[cp][0] = ls.cpPos[cp][1] = ls.cpPos[cp][2] = 0;
                            }
                        }
                        limitSurfs.push_back(ls);
                    }

                    // Find first group of 3 with matching CPs (same triangle face)
                    // Surfaces from the same triangle share the same 3 control points
                    uint32_t groupsFound = 0;
                    for (uint32_t i = 0; i + 2 < limitSurfs.size() && groupsFound < 2; i++) {
                        // Check if surfaces i, i+1, i+2 share the same 3 CPs
                        // (Same triangle → same control points, just different quadrants)
                        auto sameCP = [](const LimitSurf& a, const LimitSurf& b) {
                            // Check if all 3 CPs match (any order)
                            for (int j = 0; j < 3; j++) {
                                if (fabsf(a.cpPos[j][0] - b.cpPos[j][0]) > 0.001f ||
                                    fabsf(a.cpPos[j][1] - b.cpPos[j][1]) > 0.001f ||
                                    fabsf(a.cpPos[j][2] - b.cpPos[j][2]) > 0.001f)
                                    return false;
                            }
                            return true;
                        };

                        if (!sameCP(limitSurfs[i], limitSurfs[i+1]) || !sameCP(limitSurfs[i], limitSurfs[i+2]))
                            continue;

                        auto& s0 = limitSurfs[i], &s1 = limitSurfs[i+1], &s2 = limitSurfs[i+2];
                        Logger::warn(str::format("  TRIANGLE GROUP inst=", ir.instIdx, " meshID=", ir.meshID,
                            " surfaces=[", s0.iSurf, ",", s1.iSurf, ",", s2.iSurf,
                            "] quadrants=[", s0.quadrant, ",", s1.quadrant, ",", s2.quadrant, "]"));
                        Logger::warn(str::format("    CP[0]=(", s0.cpPos[0][0], ",", s0.cpPos[0][1], ",", s0.cpPos[0][2], ")"));
                        Logger::warn(str::format("    CP[1]=(", s0.cpPos[1][0], ",", s0.cpPos[1][1], ",", s0.cpPos[1][2], ")"));
                        Logger::warn(str::format("    CP[2]=(", s0.cpPos[2][0], ",", s0.cpPos[2][1], ",", s0.cpPos[2][2], ")"));

                        // Expected: centroid = (CP0 + CP1 + CP2) / 3
                        float centX = (s0.cpPos[0][0]+s0.cpPos[1][0]+s0.cpPos[2][0])/3.0f;
                        float centY = (s0.cpPos[0][1]+s0.cpPos[1][1]+s0.cpPos[2][1])/3.0f;
                        float centZ = (s0.cpPos[0][2]+s0.cpPos[1][2]+s0.cpPos[2][2])/3.0f;

                        // For each sub-quad cluster, check if the corner (1,1) is near centroid
                        // (All 3 sub-quads share the centroid at one corner)
                        for (auto* sp : { &s0, &s1, &s2 }) {
                            // Find cluster for this surface
                            for (uint32_t c = ir.clusterStart; c < ir.clusterStart + ir.clusterCount; c++) {
                                if (c >= clusterEntries) break;
                                if (clusterRaw[c * 4] != sp->iSurf) continue;
                                uint32_t vtxOff2 = clusterRaw[c * 4 + 1];
                                uint32_t sp2 = clusterRaw[c * 4 + 3];
                                uint32_t sx2 = sp2 & 0xFF, sy2 = (sp2 >> 8) & 0xFF;
                                if (sx2 == 0 || sy2 == 0) break;
                                uint32_t vx2 = sx2 + 1, vy2 = sy2 + 1;
                                uint32_t vpc = vx2 * vy2;
                                if (vtxOff2 + vpc > vtxBufCount) break;

                                // Get all 4 corners
                                auto gv = [&](uint32_t li) -> std::string {
                                    float x = vtxData[(vtxOff2+li)*3], y = vtxData[(vtxOff2+li)*3+1], z = vtxData[(vtxOff2+li)*3+2];
                                    return str::format("(", x, ",", y, ",", z, ")");
                                };
                                float v00x = vtxData[(vtxOff2)*3], v00y = vtxData[(vtxOff2)*3+1], v00z = vtxData[(vtxOff2)*3+2];
                                float v10x = vtxData[(vtxOff2+sx2)*3], v10y = vtxData[(vtxOff2+sx2)*3+1], v10z = vtxData[(vtxOff2+sx2)*3+2];
                                float v01x = vtxData[(vtxOff2+sy2*vx2)*3], v01y = vtxData[(vtxOff2+sy2*vx2)*3+1], v01z = vtxData[(vtxOff2+sy2*vx2)*3+2];
                                float v11x = vtxData[(vtxOff2+vpc-1)*3], v11y = vtxData[(vtxOff2+vpc-1)*3+1], v11z = vtxData[(vtxOff2+vpc-1)*3+2];

                                // Distance from each corner to centroid and to each CP
                                auto dist3 = [](float ax2, float ay2, float az2, float bx, float by, float bz) {
                                    return sqrtf((ax2-bx)*(ax2-bx)+(ay2-by)*(ay2-by)+(az2-bz)*(az2-bz));
                                };
                                float d00cent = dist3(v00x,v00y,v00z, centX,centY,centZ);
                                float d10cent = dist3(v10x,v10y,v10z, centX,centY,centZ);
                                float d01cent = dist3(v01x,v01y,v01z, centX,centY,centZ);
                                float d11cent = dist3(v11x,v11y,v11z, centX,centY,centZ);

                                // Find which corner is closest to centroid
                                float minCentDist = std::min({d00cent, d10cent, d01cent, d11cent});
                                const char* centCorner = (d00cent == minCentDist) ? "(0,0)" :
                                    (d10cent == minCentDist) ? "(1,0)" :
                                    (d01cent == minCentDist) ? "(0,1)" : "(1,1)";

                                // Check if any corner matches any CP exactly
                                float minCP0 = std::min({dist3(v00x,v00y,v00z, sp->cpPos[0][0],sp->cpPos[0][1],sp->cpPos[0][2]),
                                                         dist3(v10x,v10y,v10z, sp->cpPos[0][0],sp->cpPos[0][1],sp->cpPos[0][2]),
                                                         dist3(v01x,v01y,v01z, sp->cpPos[0][0],sp->cpPos[0][1],sp->cpPos[0][2]),
                                                         dist3(v11x,v11y,v11z, sp->cpPos[0][0],sp->cpPos[0][1],sp->cpPos[0][2])});
                                float minCP1 = std::min({dist3(v00x,v00y,v00z, sp->cpPos[1][0],sp->cpPos[1][1],sp->cpPos[1][2]),
                                                         dist3(v10x,v10y,v10z, sp->cpPos[1][0],sp->cpPos[1][1],sp->cpPos[1][2]),
                                                         dist3(v01x,v01y,v01z, sp->cpPos[1][0],sp->cpPos[1][1],sp->cpPos[1][2]),
                                                         dist3(v11x,v11y,v11z, sp->cpPos[1][0],sp->cpPos[1][1],sp->cpPos[1][2])});
                                float minCP2 = std::min({dist3(v00x,v00y,v00z, sp->cpPos[2][0],sp->cpPos[2][1],sp->cpPos[2][2]),
                                                         dist3(v10x,v10y,v10z, sp->cpPos[2][0],sp->cpPos[2][1],sp->cpPos[2][2]),
                                                         dist3(v01x,v01y,v01z, sp->cpPos[2][0],sp->cpPos[2][1],sp->cpPos[2][2]),
                                                         dist3(v11x,v11y,v11z, sp->cpPos[2][0],sp->cpPos[2][1],sp->cpPos[2][2])});

                                Logger::warn(str::format("    surf[", sp->iSurf, "] q=", sp->quadrant,
                                    " corners: (0,0)=", gv(0), " (1,0)=", gv(sx2),
                                    " (0,1)=", gv(sy2*vx2), " (1,1)=", gv(vpc-1)));
                                Logger::warn(str::format("      centroid=(", centX, ",", centY, ",", centZ,
                                    ") nearest@", centCorner, " dist=", minCentDist,
                                    " | CP0dist=", minCP0, " CP1dist=", minCP1, " CP2dist=", minCP2));
                                break;
                            }
                        }
                        groupsFound++;
                    }
                    if (groupsFound > 0) break; // Only check first instance with Limit surfaces
                }

                // Cleanup
                if (offsetData) commandList->getDevice()->unmapBuffer(offsetRB.Get());
                for (auto& md : meshCache) {
                    if (md.descData) commandList->getDevice()->unmapBuffer(md.descRB.Get());
                    if (md.cpiData) commandList->getDevice()->unmapBuffer(md.cpiRB.Get());
                    if (md.posData) commandList->getDevice()->unmapBuffer(md.posRB.Get());
                    if (md.plansData) commandList->getDevice()->unmapBuffer(md.plansRB.Get());
                }
            }

            if (vtxData) commandList->getDevice()->unmapBuffer(vtxReadback.Get());
            if (clusterRaw) commandList->getDevice()->unmapBuffer(clusterReadback.Get());
            commandList->open();
        }
    }

#if RTXMG_CHRONO_TIMING
    auto fillEnd = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration_cast<std::chrono::microseconds>(fillEnd - fillStart).count() * 0.001f;
    RTXMG_LOG(str::format(">>> RTXMG CHRONO: FillInstanceClusters TOTAL=", totalMs, "ms binding=", bindingTimeMs, "ms dispatch=", dispatchTimeMs, "ms instances=", maxInstances));
#endif
}

void ClusterAccelBuilder::ComputeInstanceClusterTiling(ClusterAccels& accels,
    const RTXMGScene& scene,
    uint32_t instanceIndex,
    uint32_t surfaceOffset,
    uint32_t surfaceCount,
    const nvrhi::BufferRange& tessCounterRange,
    nvrhi::ICommandList* commandList)
{
    using namespace dxvk;
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling entry, instanceIndex=", instanceIndex, " surfaceOffset=", surfaceOffset, " surfaceCount=", surfaceCount));

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instance = scene.GetSubdMeshInstances()[instanceIndex];
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - got instance");

    const SubdivisionSurface& subdivisionSurface = *subdMeshes[instance.meshID];
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - got subdivisionSurface, meshID=", instance.meshID));

    assert(instance.meshInstance.get());
    const auto& donutMeshInfo = instance.meshInstance->GetMesh();
    assert(donutMeshInfo.get());
    uint32_t firstGeometryIndex = donutMeshInfo.geometries[0]->globalGeometryIndex;
    const auto& localToWorld = instance.localToWorld;
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - firstGeometryIndex=", firstGeometryIndex));

    // Only clear debug buffer when debugging is enabled (matching sample behavior)
    if (m_tessellatorConfig.debugSurfaceIndex >= 0 && m_tessellatorConfig.debugLaneIndex >= 0)
    {
        commandList->clearBufferUInt(m_debugBuffer.Get(), 0);
        RTXMG_LOG("RTX MegaGeo: Debug buffer cleared for debugging");
    }

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating params");
    ComputeClusterTilingParams params = {};

    // Debug: Log struct layout - trace all fields to find alignment mismatch
#if RTXMG_VERBOSE_LOGGING
    RTXMG_LOG(str::format("RTX MegaGeo: STRUCT LAYOUT - sizeof=", sizeof(ComputeClusterTilingParams)));
    RTXMG_LOG(str::format("  offset(surfaceStart)=", offsetof(ComputeClusterTilingParams, surfaceStart),
        " offset(matWorldToClip)=", offsetof(ComputeClusterTilingParams, matWorldToClip),
        " offset(localToWorld)=", offsetof(ComputeClusterTilingParams, localToWorld)));
    RTXMG_LOG(str::format("  offset(cameraPos)=", offsetof(ComputeClusterTilingParams, cameraPos),
        " offset(aabb)=", offsetof(ComputeClusterTilingParams, aabb),
        " offset(edgeSegments)=", offsetof(ComputeClusterTilingParams, edgeSegments)));
    RTXMG_LOG(str::format("  offset(firstGeometryIndex)=", offsetof(ComputeClusterTilingParams, firstGeometryIndex),
        " offset(fineTessellationRate)=", offsetof(ComputeClusterTilingParams, fineTessellationRate),
        " offset(viewportSize)=", offsetof(ComputeClusterTilingParams, viewportSize)));
    RTXMG_LOG(str::format("  sizeof(float4)=", sizeof(float4),
        " sizeof(float3)=", sizeof(float3),
        " sizeof(Box3)=", sizeof(Box3),
        " sizeof(float4x4)=", sizeof(float4x4)));
#endif

    params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
    params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
    RTXMG_LOG(str::format("RTX MegaGeo: params - camera ptr=", (void*)m_tessellatorConfig.camera));

    // Get ViewProjection matrix using Camera's standard multiply (P * V)
    // Camera now returns column-vector convention matrices matching sample:
    //   mul(VP, worldPos) in shader = VP * worldPos
    // NOTE: Do NOT use projMatrix * viewMatrix here — DXVK's operator* reverses
    // the multiplication order (A*B = B*A in standard math), giving V*P instead of P*V.
    RTXMG_LOG("RTX MegaGeo: params - getting ViewProjection matrix (standard P*V multiply)");
    auto viewProj = m_tessellatorConfig.camera->GetViewProjectionMatrix();
    // Slang compiles with -matrix-layout-column-major, so float4x4 in the CB is read column-major.
    // DXVK Matrix4 is row-major in memory (data[row][col]). Transpose so that the shader
    // reads the correct matrix: memory columns become shader columns.
    auto viewProjForShader = transpose(viewProj);
    memcpy(&params.matWorldToClip, &viewProjForShader.data[0][0], sizeof(float) * 16);

    // viewProj matrix logging (verbose only)
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row0=(", viewProj.data[0][0], ",", viewProj.data[0][1], ",", viewProj.data[0][2], ",", viewProj.data[0][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row1=(", viewProj.data[1][0], ",", viewProj.data[1][1], ",", viewProj.data[1][2], ",", viewProj.data[1][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row2=(", viewProj.data[2][0], ",", viewProj.data[2][1], ",", viewProj.data[2][2], ",", viewProj.data[2][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row3=(", viewProj.data[3][0], ",", viewProj.data[3][1], ",", viewProj.data[3][2], ",", viewProj.data[3][3], ")"));

    RTXMG_LOG("RTX MegaGeo: params - copying localToWorld (column-vector convention)");
    // DXVK Matrix4 is row-vector convention: pos * M, translation in row 3 (data[3]).
    // Shader uses column-vector convention: dot(localToWorld[i], (pos, 1)) = M * pos.
    // So we transpose the 3x3 part (read columns, not rows) and include translation from row 3.
    // This matches sample's affineToColumnMajor() which transposes linear part + adds translation.
    params.localToWorld[0] = float4(localToWorld.data[0][0], localToWorld.data[1][0], localToWorld.data[2][0], localToWorld.data[3][0]);
    params.localToWorld[1] = float4(localToWorld.data[0][1], localToWorld.data[1][1], localToWorld.data[2][1], localToWorld.data[3][1]);
    params.localToWorld[2] = float4(localToWorld.data[0][2], localToWorld.data[1][2], localToWorld.data[2][2], localToWorld.data[3][2]);

    // Log localToWorld values sent to shader
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[0]=(", params.localToWorld[0].x, ",", params.localToWorld[0].y, ",", params.localToWorld[0].z, ",", params.localToWorld[0].w, ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[1]=(", params.localToWorld[1].x, ",", params.localToWorld[1].y, ",", params.localToWorld[1].z, ",", params.localToWorld[1].w, ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[2]=(", params.localToWorld[2].x, ",", params.localToWorld[2].y, ",", params.localToWorld[2].z, ",", params.localToWorld[2].w, ")"));

    params.viewportSize.x = float(m_tessellatorConfig.viewportSize.x);
    params.viewportSize.y = float(m_tessellatorConfig.viewportSize.y);

    // Update stats renderSize for profiler display
    stats::clusterAccelSamplers.renderSize.x = static_cast<int>(m_tessellatorConfig.viewportSize.x);
    stats::clusterAccelSamplers.renderSize.y = static_cast<int>(m_tessellatorConfig.viewportSize.y);

    params.firstGeometryIndex = firstGeometryIndex;
    params.isolationLevel = m_tessellatorConfig.isolationLevel;
    params.coarseTessellationRate = m_tessellatorConfig.coarseTessellationRate;
    params.fineTessellationRate = m_tessellatorConfig.fineTessellationRate;
    RTXMG_LOG(str::format("RTX MegaGeo: tessRates - coarse=", params.coarseTessellationRate,
        " fine=", params.fineTessellationRate,
        " tessFactor=", params.coarseTessellationRate / params.fineTessellationRate));

    // Log key parameters once per frame (instance 0 only) for debugging screenspace tessellation
    if (instanceIndex == 0) {
        RTXMG_LOG(str::format(">>> RTXMG PARAMS: viewport=(", params.viewportSize.x, ",", params.viewportSize.y,
            ") tessRate=", params.fineTessellationRate, " isolation=", params.isolationLevel));
    }

    RTXMG_LOG("RTX MegaGeo: params - getting camera eye");

    // Convert dxvk Vector3 to float4 (w = padding, C++ float3 is 16 bytes breaking alignment)
    // NOTE: cameraPos is NOT used by SPHERICAL_PROJECTION anymore - we use clip.w instead
    auto eyePos = m_tessellatorConfig.camera->GetEye();
    params.cameraPos = float4(eyePos.x, eyePos.y, eyePos.z, 0.0f);
    RTXMG_LOG(str::format("RTX MegaGeo: cameraPos=(", params.cameraPos.x, ",", params.cameraPos.y, ",", params.cameraPos.z, ") [NOT USED - using clip.w]"));

    // Transform aabb from local space to world space using localToWorld matrix
    // This matches the sample's: params.aabb = subdivisionSurface.m_aabb * localToWorld;
    // DXVK Matrix4 row-vector convention: translation in row 3 (data[3][0..2]),
    // worldPos = localPos * M → world[j] = sum_i local[i] * M[i][j] + M[3][j]
    {
        auto& aabb = subdivisionSurface.m_aabb;

        // Start with translation from row 3 (DXVK row-vector convention)
        float4 translation = float4(localToWorld.data[3][0], localToWorld.data[3][1], localToWorld.data[3][2], 0.0f);
        params.aabb.m_min = translation;
        params.aabb.m_max = translation;

        // Apply the linear transform to bounds using the standard AABB transform algorithm
        // Row-vector: world[j] = sum_i local[i] * M[i][j]
        // For output axis j, accumulate contributions from input axes i
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                float m = localToWorld.data[i][j];  // M[i][j] for row-vector: local[i] contributes to world[j]
                float minVal = aabb.m_mins[i];
                float maxVal = aabb.m_maxs[i];

                float e = m * minVal;
                float f = m * maxVal;

                if (j == 0) {
                    params.aabb.m_min.x += std::min(e, f);
                    params.aabb.m_max.x += std::max(e, f);
                } else if (j == 1) {
                    params.aabb.m_min.y += std::min(e, f);
                    params.aabb.m_max.y += std::max(e, f);
                } else {
                    params.aabb.m_min.z += std::min(e, f);
                    params.aabb.m_max.z += std::max(e, f);
                }
            }
        }

        // Debug: Log transformed aabb values
#if RTXMG_VERBOSE_LOGGING
        float3 extent = float3(params.aabb.m_max.x - params.aabb.m_min.x,
                               params.aabb.m_max.y - params.aabb.m_min.y,
                               params.aabb.m_max.z - params.aabb.m_min.z);
        float diagonalLength = sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
        RTXMG_LOG(str::format("RTX MegaGeo: aabb (world space) min=(", params.aabb.m_min.x, ",", params.aabb.m_min.y, ",", params.aabb.m_min.z,
            ") max=(", params.aabb.m_max.x, ",", params.aabb.m_max.y, ",", params.aabb.m_max.z, ") diag=", diagonalLength));
#endif
    }

    // One-shot matrix debug dump (instance 0 only, fires once)
    {
        static bool s_matrixDumped = false;
        if (!s_matrixDumped && instanceIndex == 0) {
            s_matrixDumped = true;
            auto eye = m_tessellatorConfig.camera->GetEye();
            auto lookat = m_tessellatorConfig.camera->GetLookat();
            Logger::warn(str::format("MATRIX_DBG: camera eye=(", eye.x, ",", eye.y, ",", eye.z,
                ") lookat=(", lookat.x, ",", lookat.y, ",", lookat.z, ")"));
            Logger::warn(str::format("MATRIX_DBG: viewProj[0]=(", viewProj.data[0][0], ",", viewProj.data[0][1], ",", viewProj.data[0][2], ",", viewProj.data[0][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: viewProj[1]=(", viewProj.data[1][0], ",", viewProj.data[1][1], ",", viewProj.data[1][2], ",", viewProj.data[1][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: viewProj[2]=(", viewProj.data[2][0], ",", viewProj.data[2][1], ",", viewProj.data[2][2], ",", viewProj.data[2][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: viewProj[3]=(", viewProj.data[3][0], ",", viewProj.data[3][1], ",", viewProj.data[3][2], ",", viewProj.data[3][3], ")"));

            // Test project the camera lookat point — should be near screen center with positive clip.w
            // Column-vector: clip[i] = sum_j VP[i][j] * pos[j]  (same as mul(VP, pos) in shader)
            float px = lookat.x, py = lookat.y, pz = lookat.z;
            float clipX = viewProj.data[0][0]*px + viewProj.data[0][1]*py + viewProj.data[0][2]*pz + viewProj.data[0][3];
            float clipY = viewProj.data[1][0]*px + viewProj.data[1][1]*py + viewProj.data[1][2]*pz + viewProj.data[1][3];
            float clipZ = viewProj.data[2][0]*px + viewProj.data[2][1]*py + viewProj.data[2][2]*pz + viewProj.data[2][3];
            float clipW = viewProj.data[3][0]*px + viewProj.data[3][1]*py + viewProj.data[3][2]*pz + viewProj.data[3][3];
            Logger::warn(str::format("MATRIX_DBG: project lookat -> clip=(", clipX, ",", clipY, ",", clipZ, ",", clipW,
                ") ndc=(", clipX/clipW, ",", clipY/clipW, ") clipW>0=", clipW > 0 ? "YES" : "NO ***WRONG***"));

            // Log localToWorld raw DXVK data vs transposed sent to shader
            Logger::warn(str::format("MATRIX_DBG: localToWorld DXVK[0]=(", localToWorld.data[0][0], ",", localToWorld.data[0][1], ",", localToWorld.data[0][2], ",", localToWorld.data[0][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld DXVK[1]=(", localToWorld.data[1][0], ",", localToWorld.data[1][1], ",", localToWorld.data[1][2], ",", localToWorld.data[1][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld DXVK[2]=(", localToWorld.data[2][0], ",", localToWorld.data[2][1], ",", localToWorld.data[2][2], ",", localToWorld.data[2][3], ")"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld DXVK[3]=(", localToWorld.data[3][0], ",", localToWorld.data[3][1], ",", localToWorld.data[3][2], ",", localToWorld.data[3][3], ") <- translation"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld shader[0]=(", params.localToWorld[0].x, ",", params.localToWorld[0].y, ",", params.localToWorld[0].z, ",", params.localToWorld[0].w, ")"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld shader[1]=(", params.localToWorld[1].x, ",", params.localToWorld[1].y, ",", params.localToWorld[1].z, ",", params.localToWorld[1].w, ")"));
            Logger::warn(str::format("MATRIX_DBG: localToWorld shader[2]=(", params.localToWorld[2].x, ",", params.localToWorld[2].y, ",", params.localToWorld[2].z, ",", params.localToWorld[2].w, ")"));

            // Log AABB
            Logger::warn(str::format("MATRIX_DBG: aabb local min=(", subdivisionSurface.m_aabb.m_mins[0], ",", subdivisionSurface.m_aabb.m_mins[1], ",", subdivisionSurface.m_aabb.m_mins[2],
                ") max=(", subdivisionSurface.m_aabb.m_maxs[0], ",", subdivisionSurface.m_aabb.m_maxs[1], ",", subdivisionSurface.m_aabb.m_maxs[2], ")"));
            Logger::warn(str::format("MATRIX_DBG: aabb world min=(", params.aabb.m_min.x, ",", params.aabb.m_min.y, ",", params.aabb.m_min.z,
                ") max=(", params.aabb.m_max.x, ",", params.aabb.m_max.y, ",", params.aabb.m_max.z, ")"));
        }
    }

    params.enableBackfaceVisibility = m_tessellatorConfig.enableBackfaceVisibility;
    params.enableFrustumVisibility = m_tessellatorConfig.enableFrustumVisibility;
    params.enableHiZVisibility = m_tessellatorConfig.enableHiZVisibility && m_tessellatorConfig.zbuffer != nullptr;
    params.edgeSegments = m_tessellatorConfig.edgeSegments;
    params.globalDisplacementScale = m_tessellatorConfig.displacementScale;

    ONCE(RTXMG_LOG(str::format("RTX MegaGeo: Visibility params - frustum=", params.enableFrustumVisibility,
        " backface=", params.enableBackfaceVisibility, " HiZ=", params.enableHiZVisibility,
        " edgeSegments=(", params.edgeSegments.x, ",", params.edgeSegments.y, ",", params.edgeSegments.z, ",", params.edgeSegments.w, ")")));

    params.maxClasBlocks = uint32_t(m_maxClasBytes / size_t(cluster::kClasByteAlignment));
    params.maxClusters = m_maxClusters;
    params.maxVertices = m_maxVertices;
    params.clusterVertexPositionsBaseAddress = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
    params.clasDataBaseAddress = accels.clasBuffer.GetGpuVirtualAddress();
    params.instanceIndex = instanceIndex;

    // Safety check: if clasDataBaseAddress is 0, all CLAS addresses will be invalid
    if (params.clasDataBaseAddress == 0) {
        Logger::err("RTX MegaGeo: ComputeInstanceClusterTiling - clasDataBaseAddress is NULL! CLAS addresses will be invalid.");
    }
    if (params.clusterVertexPositionsBaseAddress == 0) {
        Logger::err("RTX MegaGeo: ComputeInstanceClusterTiling - clusterVertexPositionsBaseAddress is NULL!");
    }

    // Log addresses for debugging
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - clusterVertexPositionsBaseAddress=", std::hex, params.clusterVertexPositionsBaseAddress));
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - clasDataBaseAddress=", std::hex, params.clasDataBaseAddress));
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - maxClusters=", params.maxClusters, " maxVertices=", params.maxVertices, " maxClasBlocks=", params.maxClasBlocks));

    if (m_tessellatorConfig.zbuffer)
    {
        params.numHiZLODs = m_tessellatorConfig.zbuffer->GetNumHiZLODs();
        params.invHiZSize = m_tessellatorConfig.zbuffer->GetInvHiZSize();
    }
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - params filled, creating bindingSetDesc");

    // Create binding layouts - matching sample's 3 descriptor set structure:
    // Set 0: Main bindings (SRVs, UAVs, samplers, constant buffer)
    // Set 1: HiZ textures (space 1)
    // Set 2: Bindless textures
    if (!m_computeClusterTilingBL)
    {
        RTXMG_LOG("RTX MegaGeo: Creating main binding layout (set 0)");
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.setVisibility(nvrhi::ShaderType::Compute)
            .setRegisterSpace(0)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(6))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(7))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(9))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(10))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(11))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(12))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(13))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(14))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(15))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(16))
            // HiZ textures - Slang ignores space1 and puts them at t17 in Set 0
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(17).setSize(HIZ_MAX_LODS))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(5))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(6))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(7))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(8));

        m_computeClusterTilingBL = m_device->createBindingLayout(layoutDesc);
        if (!m_computeClusterTilingBL)
        {
            Logger::err("Failed to create main binding layout for compute_cluster_tiling.hlsl");
        }
        RTXMG_LOG("RTX MegaGeo: Main binding layout created");
    }

    // Create HiZ binding layout (set 1) - shader expects HiZ at binding 0, set 1
    if (!m_computeClusterTilingHizBL)
    {
        RTXMG_LOG("RTX MegaGeo: Creating HiZ binding layout (set 1)");
        nvrhi::BindingLayoutDesc hizLayoutDesc;
        hizLayoutDesc.setVisibility(nvrhi::ShaderType::Compute)
            .setRegisterSpace(1)
            .setRegisterSpaceIsDescriptorSet(true)
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0).setSize(HIZ_MAX_LODS));

        m_computeClusterTilingHizBL = m_device->createBindingLayout(hizLayoutDesc);
        if (!m_computeClusterTilingHizBL)
        {
            Logger::err("Failed to create HiZ binding layout for compute_cluster_tiling.hlsl");
        }
        RTXMG_LOG("RTX MegaGeo: HiZ binding layout created");

        // Create dummy HiZ binding set for when zbuffer is null
        nvrhi::BindingSetDesc dummyHizSetDesc;
        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            dummyHizSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_dummyHiZTextures[i]).setArrayElement(i));
        }
        m_dummyHizBindingSet = m_device->createBindingSet(dummyHizSetDesc, m_computeClusterTilingHizBL);
        if (!m_dummyHizBindingSet)
        {
            Logger::err("Failed to create dummy HiZ binding set");
        }
        RTXMG_LOG("RTX MegaGeo: Dummy HiZ binding set created");
    }

    // Main binding set (set 0) - no HiZ textures, they're in set 1
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_computeClusterTilingParamsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, subdivisionSurface.m_positionsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, scene.GetGeometryBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene.GetMaterialBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subdivisionSurface.m_surfaceToGeometryIndexBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subdivisionSurface.m_vertexDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subdivisionSurface.m_vertexDeviceData.controlPointIndices));
    // DEBUG: Log surface descriptor buffer info
    RTXMG_LOG(str::format("RTX MegaGeo: Binding SurfaceDescriptors SRV(4) - buffer=",
        (void*)subdivisionSurface.m_vertexDeviceData.surfaceDescriptors.Get(),
        " bytes=", subdivisionSurface.m_vertexDeviceData.surfaceDescriptors ?
            subdivisionSurface.m_vertexDeviceData.surfaceDescriptors->getDesc().byteSize : 0,
        " surfaceCount=", surfaceCount));
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subdivisionSurface.m_vertexDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subdivisionSurface.GetTopologyMap()->plansBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subdivisionSurface.GetTopologyMap()->subpatchTreesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subdivisionSurface.GetTopologyMap()->patchPointIndicesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subdivisionSurface.GetTopologyMap()->stencilMatrixArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_templateBuffers.instantiationSizesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_templateBuffers.addressesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, subdivisionSurface.m_texcoordDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subdivisionSurface.m_texcoordDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subdivisionSurface.m_texcoordDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subdivisionSurface.m_texcoordsBuffer));
    // HiZ textures at binding 17 - Slang ignores space1 and puts them here
    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
    {
        nvrhi::ITexture* hizTex = nullptr;
        if (m_tessellatorConfig.zbuffer && m_tessellatorConfig.enableHiZVisibility)
        {
            hizTex = m_tessellatorConfig.zbuffer->GetHierarchyTexture(i);
        }
        bindingSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(17, hizTex ? hizTex : m_dummyHiZTextures[i]).setArrayElement(i));
    }
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))
        .addItem(nvrhi::BindingSetItem::Sampler(1, m_commonPasses->m_LinearClampSampler)) // hiZ sampler
        // UAV bindings
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_gridSamplersBuffer,
            nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(surfaceOffset * m_gridSamplersBuffer.GetElementBytes(), surfaceCount * m_gridSamplersBuffer.GetElementBytes())))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange));
    RTXMG_LOG(str::format("RTX MegaGeo: Binding tessCounters UAV(1) - range offset=", tessCounterRange.byteOffset,
                             " size=", tessCounterRange.byteSize, " buffer=", (void*)m_tessellationCountersBuffer.Get()));
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_clustersBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterShadingDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_clasIndirectArgDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(5, accels.clasPtrsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(6, subdivisionSurface.m_vertexDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(7, subdivisionSurface.m_texcoordDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(8, m_debugBuffer));
    // One-shot diagnostic: log UAV 6/7/8 buffer handles for first instance
    {
        static bool s_loggedUavBuffers = false;
        if (!s_loggedUavBuffers) {
            s_loggedUavBuffers = true;
            Logger::warn(str::format("TILING_UAV_DBG: UAV(6) vertexPatchPoints ptr=",
                (void*)subdivisionSurface.m_vertexDeviceData.patchPoints.Get(),
                " bytes=", subdivisionSurface.m_vertexDeviceData.patchPoints ?
                    subdivisionSurface.m_vertexDeviceData.patchPoints->getDesc().byteSize : 0));
            Logger::warn(str::format("TILING_UAV_DBG: UAV(7) texcoordPatchPoints ptr=",
                (void*)subdivisionSurface.m_texcoordDeviceData.patchPoints.Get(),
                " bytes=", subdivisionSurface.m_texcoordDeviceData.patchPoints ?
                    subdivisionSurface.m_texcoordDeviceData.patchPoints->getDesc().byteSize : 0));
            Logger::warn(str::format("TILING_UAV_DBG: UAV(8) debugBuffer ptr=",
                (void*)m_debugBuffer.Get(),
                " bytes=", m_debugBuffer.GetBytes()));
        }
    }
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - main bindingSetDesc built");

    // NOTE: Binding set creation deferred to after writeBuffer to ensure volatile CB
    // descriptor offset matches the CURRENT version (not the stale previous version).
    // See createBindingSet calls in monolithic/loop paths below.

    // HiZ binding set (set 1) - use real zbuffer textures if available
    nvrhi::BindingSetHandle hizBindingSet;
    if (m_tessellatorConfig.zbuffer && m_tessellatorConfig.enableHiZVisibility)
    {
        RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating real HiZ binding set");

        // Initialize HiZ textures on first use by transitioning layout and clearing
        if (!m_hizInitialized)
        {
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - initializing HiZ textures (first use)");

            // Get underlying DxvkContext for layout transitions
            dxvk::NvrhiDxvkCommandList* dxvkCmdList = static_cast<dxvk::NvrhiDxvkCommandList*>(commandList);

            nvrhi::Color clearColor(std::numeric_limits<float>::max());

            // Clear each HiZ texture - this will transition from UNDEFINED layout
            for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
            {
                nvrhi::ITexture* hizTex = m_tessellatorConfig.zbuffer->GetHierarchyTexture(i);
                if (hizTex)
                {
                    RTXMG_LOG(str::format("RTX MegaGeo: Clearing HiZ texture level ", i));
                    commandList->clearTextureFloat(hizTex, nvrhi::AllSubresources, clearColor);
                }
            }
            m_hizInitialized = true;
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - HiZ textures initialized");
        }

        // Use cached binding set if zbuffer hasn't changed AND we're in the same frame
        // (ZBuffer textures can be recreated between frames even if the pointer is the same)
        if (m_cachedHizBuffer == m_tessellatorConfig.zbuffer &&
            m_cachedHizFrame == m_currentFrameIndex &&
            m_cachedHizBindingSet)
        {
            hizBindingSet = m_cachedHizBindingSet;
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - reusing cached HiZ binding set");
        }
        else
        {
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating and caching HiZ binding set");
            nvrhi::BindingSetDesc hizSetDesc;
            for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
            {
                nvrhi::ITexture* hizTex = m_tessellatorConfig.zbuffer->GetHierarchyTexture(i);
                hizSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, hizTex ? hizTex : m_dummyHiZTextures[i]).setArrayElement(i));
            }
            m_cachedHizBindingSet = m_device->createBindingSet(hizSetDesc, m_computeClusterTilingHizBL);
            m_cachedHizBuffer = m_tessellatorConfig.zbuffer;
            m_cachedHizFrame = m_currentFrameIndex;
            hizBindingSet = m_cachedHizBindingSet;
        }
    }
    else
    {
        RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - using dummy HiZ binding set");
        hizBindingSet = m_dummyHizBindingSet;
    }

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating shaderPermutation");
    // TEMPORARY: Force disable displacement - geometry/material buffers are not populated in RTX Remix
    ComputeClusterTilingPermutation shaderPermutation(false /*subdivisionSurface.m_hasDisplacementMaterial*/,
        m_tessellatorConfig.enableFrustumVisibility,
        m_tessellatorConfig.tessMode,
        m_tessellatorConfig.visMode,
        ShaderPermutationSurfaceType::PureBSpline);
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - shaderPermutation index=", shaderPermutation.index()));

    // Log tessellation mode once per frame (instance 0 only) for debugging screenspace approach
    if (instanceIndex == 0) {
        RTXMG_LOG(str::format("RTX MegaGeo: TessellationMode=", toString(shaderPermutation.tessellationMode()),
            " (SCREENSPACE: uses clip.w for LOD, NOT cameraPos)",
            " backface=", m_tessellatorConfig.enableBackfaceVisibility ? "YES (uses clip-space normal)" : "NO",
            " frustum=", m_tessellatorConfig.enableFrustumVisibility ? "YES" : "NO",
            " hiZ=", m_tessellatorConfig.enableHiZVisibility ? "YES" : "NO"));
    }

    auto GetComputeClusterTilingPSO = [this](const ComputeClusterTilingPermutation& shaderPermutation)
        {
            RTXMG_LOG(str::format("RTX MegaGeo: GetComputeClusterTilingPSO - index=", shaderPermutation.index()));
            if (!m_computeClusterTilingPSOs[shaderPermutation.index()])
            {
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - creating PSO");
                std::vector<donut::engine::ShaderMacro> macros;
                macros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("TESS_MODE", toString(shaderPermutation.tessellationMode())));
                macros.push_back(donut::engine::ShaderMacro("ENABLE_FRUSTUM_VISIBILITY", shaderPermutation.isFrustumVisibilityEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("VIS_MODE", toString(shaderPermutation.visibilityMode())));
                macros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - calling CreateShader");

                nvrhi::ShaderDesc tilingDesc(nvrhi::ShaderType::Compute);
                nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/compute_cluster_tiling.hlsl", "main", &macros, tilingDesc);
                RTXMG_LOG(str::format("RTX MegaGeo: GetComputeClusterTilingPSO - shader=", (void*)shader.Get()));

                // Store HiZ descriptor set layout in device for command list to use when binding set 1
                // This only needs to be done once (the layout is shared across all shader permutations)
                auto* nvrhiDevice = static_cast<NvrhiDxvkDevice*>(m_device.Get());
                if (nvrhiDevice && nvrhiDevice->getHiZDescriptorSetLayout() == VK_NULL_HANDLE) {
                    VkDescriptorSetLayout hiZLayout = m_shaderFactory.getHiZDescriptorSetLayout();
                    if (hiZLayout != VK_NULL_HANDLE) {
                        nvrhiDevice->setHiZDescriptorSetLayout(hiZLayout);
                        RTXMG_LOG("RTX MegaGeo: Stored HiZ descriptor set layout in device");
                    }
                }

                auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_computeClusterTilingBL)      // Set 0: Main bindings
                    .addBindingLayout(m_computeClusterTilingHizBL)   // Set 1: HiZ textures
                    .addBindingLayout(m_bindlessBL);                 // Set 2: Bindless textures
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - creating pipeline");

                m_computeClusterTilingPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - pipeline created");
            }
            return m_computeClusterTilingPSOs[shaderPermutation.index()];
        };

    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - enableMonolithicClusterBuild=", m_tessellatorConfig.enableMonolithicClusterBuild));

    if (m_tessellatorConfig.enableMonolithicClusterBuild)
    {
        params.surfaceStart = 0;
        params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
        uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;

        if (dispatchCount == 0) {
            uint32_t noLimitOffset = subdivisionSurface.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
            Logger::err(str::format("RTX MegaGeo DISPATCH: *** ZERO DISPATCH! surfaceEnd=", params.surfaceEnd,
                " NoLimitOffset=", noLimitOffset, " - no surfaces will be tessellated ***"));
        }

        RTXMG_LOG(str::format("RTX MegaGeo: Monolithic - surfaceStart=", params.surfaceStart,
            " surfaceEnd=", params.surfaceEnd, " dispatchCount=", dispatchCount,
            " surfaceOffsets=[", subdivisionSurface.m_surfaceOffsets[0], ",",
            subdivisionSurface.m_surfaceOffsets[1], ",", subdivisionSurface.m_surfaceOffsets[2], ",",
            subdivisionSurface.m_surfaceOffsets[3], "]"));

        // Write params FIRST so volatile CB advances to current version
        RTXMG_LOG("RTX MegaGeo: Monolithic - writeBuffer");
        commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));

        // Create binding set AFTER writeBuffer so volatile CB descriptor offset
        // snapshots the CURRENT version (not the stale previous version)
        nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_computeClusterTilingBL);
        if (!bindingSet)
        {
            Logger::err("Failed to create main binding set for compute_cluster_tiling.hlsl");
        }

        ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType::All;
        shaderPermutation.setSurfaceType(shaderSurfaceType);
        RTXMG_LOG("RTX MegaGeo: Monolithic - GetComputeClusterTilingPSO");
        auto state = nvrhi::ComputeState()
            .addBindingSet(bindingSet)           // Set 0: Main bindings
            .addBindingSet(hizBindingSet)        // Set 1: HiZ textures
            .addBindingSet(m_descriptorTable);   // Set 2: Bindless textures
        state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
        RTXMG_LOG("RTX MegaGeo: Monolithic - setComputeState");
        commandList->setComputeState(state);

        RTXMG_LOG("RTX MegaGeo: Monolithic - dispatch");
        commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);
        RTXMG_LOG("RTX MegaGeo: Monolithic - dispatch complete");

        // Save cluster offset for this instance.
        // CopyClusterOffset(AllTypes) populates:
        //   - ClusterOffsetCounts slot 3 (All): total cluster offset + count for this instance
        //   - FillClustersIndirectArgs slot 2 (Limit): vertex dispatch workgroups (used by FillInstanceClusters)
        //   - FillClustersIndirectArgs slot 3 (All): texcoord dispatch workgroups
        ClusterDispatchType dispatchType = ClusterDispatchType::AllTypes;
        RTXMG_LOG("RTX MegaGeo: Monolithic - CopyClusterOffset");
        CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
        RTXMG_LOG("RTX MegaGeo: Monolithic - CopyClusterOffset complete");
    }
    else
    {
        RTXMG_LOG("RTX MegaGeo: Loop mode - entering loop");
        // Loop
        for (uint32_t i = 0; i <= uint32_t(ClusterDispatchType::Limit); i++)
        {
            RTXMG_LOG(str::format("RTX MegaGeo: Loop iteration i=", i));
            SubdivisionSurface::SurfaceType subdSurfaceType = SubdivisionSurface::SurfaceType(i);

            // Skip no limit surfaces
            params.surfaceStart = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType)];
            params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType) + 1];

            uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;
            RTXMG_LOG(str::format("RTX MegaGeo: Loop - surfaceStart=", params.surfaceStart, " surfaceEnd=", params.surfaceEnd, " dispatchCount=", dispatchCount));
            if (dispatchCount)
            {
                // Write params FIRST so volatile CB advances to current version
                RTXMG_LOG("RTX MegaGeo: Loop - writeBuffer");
                commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));

                // Create binding set AFTER writeBuffer so volatile CB descriptor offset
                // snapshots the CURRENT version (not the stale previous version)
                nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_computeClusterTilingBL);
                if (!bindingSet)
                {
                    Logger::err("Failed to create main binding set for compute_cluster_tiling.hlsl (loop)");
                }

                ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType(i);
                shaderPermutation.setSurfaceType(shaderSurfaceType);
                RTXMG_LOG("RTX MegaGeo: Loop - GetComputeClusterTilingPSO");
                auto state = nvrhi::ComputeState()
                    .addBindingSet(bindingSet)           // Set 0: Main bindings
                    .addBindingSet(hizBindingSet)        // Set 1: HiZ textures
                    .addBindingSet(m_descriptorTable);   // Set 2: Bindless textures
                state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
                RTXMG_LOG("RTX MegaGeo: Loop - setComputeState");
                commandList->setComputeState(state);

                RTXMG_LOG("RTX MegaGeo: Loop - dispatch");
                commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);
                RTXMG_LOG("RTX MegaGeo: Loop - dispatch complete");
            }
            // Save cluster offset for this instance
            ClusterDispatchType dispatchType = ClusterDispatchType(i);
            RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset");
            CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
            RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset complete");
        }
        // Also copy to AllTypes slot so FillBlasFromClasArgs can read from it
        // (FillBlasFromClasArgs reads from ClusterDispatchType::All for all instances)
        RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset for AllTypes");
        CopyClusterOffset(instanceIndex, ClusterDispatchType::AllTypes, tessCounterRange, commandList);
        RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset for AllTypes complete");
        RTXMG_LOG("RTX MegaGeo: Loop mode - loop complete");
    }

    // Debug output download disabled for performance - enable ENABLE_SHADER_DEBUG to use
#if ENABLE_SHADER_DEBUG
    {
        RTXMG_LOG(str::format("RTX MegaGeo: Debug Instance:", instanceIndex, " Mesh:", donutMeshInfo.name));

        auto debugOutput = m_debugBuffer.Download(commandList);
        uint numElements = debugOutput.front().payloadType;
        RTXMG_LOG(str::format("RTX MegaGeo: Debug buffer numElements=", numElements));

        for (uint32_t i = 1; i <= std::min(numElements, 40u); ++i) {
            const auto& elem = debugOutput[i];
            if (elem.payloadType >= 9 && elem.payloadType <= 12) {
                RTXMG_LOG(str::format("RTX MegaGeo: Debug[", i, "] line=", elem.lineNumber,
                    " floats=[", elem.floatData.x, ",", elem.floatData.y, ",", elem.floatData.z, ",", elem.floatData.w, "]"));
            } else {
                RTXMG_LOG(str::format("RTX MegaGeo: Debug[", i, "] line=", elem.lineNumber,
                    " vals=[", elem.uintData.x, ",", elem.uintData.y, ",", elem.uintData.z, ",", elem.uintData.w, "]"));
            }
        }

        if (numElements > 0) {
            vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
        }
    }
#endif
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling complete for instance ", instanceIndex));

    // NOTE: Patchpoints logging removed - DownloadBuffer closes/reopens command list which
    // destroys bound HiZ image views and causes VK_ERROR_DEVICE_LOST in DXVK.
    // The sample code works differently because it uses a different nvrhi backend.
}

void ClusterAccelBuilder::CopyClusterOffset(uint32_t instanceIndex,
    ClusterDispatchType dispatchType, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    // Bounds check: CopyClusterOffset shader writes to m_fillClustersDispatchIndirectBuffer at indices
    // based on instanceIndex. Prevent out-of-bounds writes by checking instanceIndex.
    if (instanceIndex >= m_numInstances) {
        Logger::err(str::format("RTX MegaGeo: CopyClusterOffset - instanceIndex ", instanceIndex,
            " >= m_numInstances ", m_numInstances, ", skipping to prevent buffer overflow"));
        return;
    }

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::CopyClusterOffset");
    CopyClusterOffsetParams params;
    params.instanceIndex = instanceIndex;
    params.dispatchTypeIndex = uint32_t(dispatchType);
    commandList->writeBuffer(m_copyClusterOffsetParamsBuffer, &params, sizeof(CopyClusterOffsetParams));

    // Use binding indices from copy_cluster_offset_binding_indices.h
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(COPY_CLUSTER_OFFSET_TESS_COUNTERS_INPUT, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_CLUSTER_OFFSET_COUNTS_OUTPUT, m_clusterOffsetCountsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_FILL_INDIRECT_ARGS_OUTPUT, m_fillClustersDispatchIndirectBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(COPY_CLUSTER_OFFSET_PARAMS, m_copyClusterOffsetParamsBuffer));

    // Create layout once, then reuse for all binding sets
    if (!m_copyClusterOffsetBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(COPY_CLUSTER_OFFSET_TESS_COUNTERS_INPUT))  // SRV t0
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_CLUSTER_OFFSET_COUNTS_OUTPUT))  // UAV u0
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_FILL_INDIRECT_ARGS_OUTPUT))  // UAV u1
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(COPY_CLUSTER_OFFSET_PARAMS));  // CB b0
        m_copyClusterOffsetBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_copyClusterOffsetBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for copy_cluster_offset shader");
    }

    if (!m_copyClusterOffsetPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/copy_cluster_offset.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_copyClusterOffsetBL);

        m_copyClusterOffsetPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_copyClusterOffsetPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(1, 1, 1);
}

void ClusterAccelBuilder::BuildBlasFromClas(ClusterAccels& accels, const Instance* instances, size_t instanceCount, nvrhi::ICommandList* commandList)
{
    //// Allocate and build BLASes
    nvrhi::utils::ScopedMarker marker(commandList, "Blas Build from Clas");
    stats::clusterAccelSamplers.buildBlasTime.Start(commandList);

    uint32_t numInstances = static_cast<uint32_t>(instanceCount);

    // Debug logging for NULL address detection
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - numInstances=", numInstances));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - clasPtrsBuffer ptr=", (void*)accels.clasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasPtrsBuffer ptr=", (void*)accels.blasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasSizesBuffer ptr=", (void*)accels.blasSizesBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasBuffer ptr=", (void*)accels.blasBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - m_blasFromClasIndirectArgsBuffer ptr=", (void*)m_blasFromClasIndirectArgsBuffer.Get()));

    nvrhi::GpuVirtualAddress clasPtrsBaseAddress = accels.clasPtrsBuffer.GetGpuVirtualAddress();

    // Always log critical addresses for debugging GPU crashes
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - clasPtrsBaseAddress=0x", std::hex, clasPtrsBaseAddress,
        " numInstances=", std::dec, numInstances,
        " clasPtrsBuffer.size=", accels.clasPtrsBuffer.GetBytes(),
        " clusterOffsetCountsBuffer.size=", m_clusterOffsetCountsBuffer.GetBytes()));

    if (clasPtrsBaseAddress == 0) {
        Logger::err("RTX MegaGeo: BuildBlasFromClas - clasPtrsBaseAddress is NULL!");
    }

    FillBlasFromClasArgs(m_blasFromClasIndirectArgsBuffer, m_clusterOffsetCountsBuffer, clasPtrsBaseAddress, numInstances, commandList);

    // CRITICAL BARRIER: FillBlasFromClasArgs dispatches a compute shader that writes to
    // m_blasFromClasIndirectArgsBuffer as UAV. The subsequent BLAS build (via
    // executeMultiIndirectClusterOperation) reads this buffer as ShaderResource. However,
    // the NVRHI adapter's m_BufferStates has stale state (CopyDest from the clear earlier),
    // because the compute dispatch went through DXVK which doesn't update m_BufferStates.
    // This causes commitBarriers to emit a TRANSFER→COMPUTE barrier instead of the required
    // COMPUTE→COMPUTE barrier, missing the compute shader's write entirely.
    // This global memory barrier ensures compute shader writes are visible to the BLAS build.
    commandList->bufferBarrier(m_blasFromClasIndirectArgsBuffer,
        nvrhi::ResourceStates::UnorderedAccess,
        nvrhi::ResourceStates::ShaderResource);

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    m_blasFromClasIndirectArgsBuffer.Log(commandList, [](std::ostream& ss, const cluster::IndirectArgs& e)
        {
            ss << "{c: " << std::dec << e.clusterCount <<
                " | addr: " << std::hex << e.clusterAddresses << "}";
            return true;
        });
#endif

    // Check all addresses before the operation
    nvrhi::GpuVirtualAddress blasPtrsAddr = accels.blasPtrsBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress blasBufferAddr = accels.blasBuffer.GetBuffer() ? accels.blasBuffer.GetBuffer()->getGpuVirtualAddress() : 0;

    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasPtrsAddr=", std::hex, blasPtrsAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasBufferAddr=", std::hex, blasBufferAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - scratchSizeInBytes=", m_createBlasSizeInfo.scratchSizeInBytes));

    if (blasPtrsAddr == 0) Logger::err("RTX MegaGeo: BuildBlasFromClas - blasPtrsAddr is NULL!");
    if (blasBufferAddr == 0) Logger::err("RTX MegaGeo: BuildBlasFromClas - blasBufferAddr is NULL!");

    // DIAGNOSTIC: Download and inspect BLAS indirect args + CLAS pointers to find why most BLAS addresses are zero
    {
        static uint32_t s_diagCount = 0;
        if (s_diagCount < 3) {
            s_diagCount++;

            // 1. Download BLAS indirect args
            auto blasIndirectArgs = m_blasFromClasIndirectArgsBuffer.Download(commandList);
            uint32_t zeroCountArgs = 0, zeroAddrArgs = 0, validArgs = 0;
            uint32_t totalClusterCount = 0;
            for (uint32_t i = 0; i < std::min(uint32_t(blasIndirectArgs.size()), numInstances); ++i) {
                const auto& arg = blasIndirectArgs[i];
                if (arg.clusterCount == 0) zeroCountArgs++;
                if (arg.clusterAddresses == 0) zeroAddrArgs++;
                if (arg.clusterCount > 0 && arg.clusterAddresses != 0) validArgs++;
                totalClusterCount += arg.clusterCount;
            }
            Logger::warn(str::format("DIAG BLAS-ARGS: numInstances=", numInstances,
                " validArgs=", validArgs, " zeroCount=", zeroCountArgs, " zeroAddr=", zeroAddrArgs,
                " totalClusters=", totalClusterCount));

            // Log first 10 and any problematic entries
            for (uint32_t i = 0; i < std::min(uint32_t(blasIndirectArgs.size()), numInstances); ++i) {
                const auto& arg = blasIndirectArgs[i];
                if (i < 10 || arg.clusterCount == 0 || arg.clusterAddresses == 0) {
                    Logger::warn(str::format("DIAG BLAS-ARGS[", i, "]: clusterCount=", arg.clusterCount,
                        " addr=0x", std::hex, arg.clusterAddresses, std::dec,
                        " stride=", arg.clusterReferencesStride));
                    if (i == 10) Logger::warn("  ... (showing only problematic entries after this)");
                }
            }

            // 2. Download clusterOffsetCounts to see per-instance offsets
            auto offsetCounts = m_clusterOffsetCountsBuffer.Download(commandList);
            Logger::warn(str::format("DIAG OFFSET-COUNTS: buffer elements=", offsetCounts.size()));
            uint32_t zeroOffsetCount = 0;
            for (uint32_t i = 0; i < std::min(uint32_t(offsetCounts.size()), numInstances); ++i) {
                // Index: instanceIndex * NumTypes + All = i * 4 + 3
                uint32_t idx = i * 4 + 3; // ClusterDispatchType::All = 3, NumTypes = 4
                if (idx < offsetCounts.size()) {
                    uint32_t offset = offsetCounts[idx].x;
                    uint32_t count = offsetCounts[idx].y;
                    if (count == 0) zeroOffsetCount++;
                    if (i < 10 || count == 0) {
                        Logger::warn(str::format("DIAG OFFSET-COUNTS[inst", i, " idx", idx, "]: offset=", offset, " count=", count));
                    }
                }
            }
            Logger::warn(str::format("DIAG OFFSET-COUNTS: zeroCountInstances=", zeroOffsetCount, " of ", numInstances));

            // 3. Download CLAS pointers to check if they're valid
            auto clasPtrs = accels.clasPtrsBuffer.Download(commandList);
            uint32_t zeroClas = 0, nonZeroClas = 0;
            for (size_t i = 0; i < clasPtrs.size(); ++i) {
                if (clasPtrs[i] == 0) zeroClas++;
                else nonZeroClas++;
            }
            Logger::warn(str::format("DIAG CLAS-PTRS: total=", clasPtrs.size(), " nonZero=", nonZeroClas, " zero=", zeroClas));

            // Sample CLAS ptrs for first few instances based on their offsets
            for (uint32_t i = 0; i < std::min(5u, numInstances); ++i) {
                uint32_t idx = i * 4 + 3;
                if (idx < offsetCounts.size()) {
                    uint32_t offset = offsetCounts[idx].x;
                    uint32_t count = offsetCounts[idx].y;
                    uint32_t instNonZero = 0;
                    for (uint32_t c = 0; c < std::min(count, uint32_t(clasPtrs.size()) - offset); ++c) {
                        if (clasPtrs[offset + c] != 0) instNonZero++;
                    }
                    Logger::warn(str::format("DIAG CLAS-PTRS inst[", i, "]: offset=", offset,
                        " count=", count, " nonZeroClas=", instNonZero,
                        " first=0x", std::hex, (offset < clasPtrs.size() ? clasPtrs[offset] : 0), std::dec));
                }
            }
        }
    }

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    // Download and log the first few BLAS indirect args to diagnose misaligned address errors (GPU readback - only when logging enabled)
    {
        auto blasIndirectArgs = m_blasFromClasIndirectArgsBuffer.Download(commandList);
        uint32_t numToLog = std::min(uint32_t(blasIndirectArgs.size()), numInstances);
        numToLog = std::min(numToLog, 10u); // Limit to first 10
        RTXMG_LOG(str::format("RTX MegaGeo: BLAS indirect args (first ", numToLog, " of ", numInstances, "):"));
        for (uint32_t i = 0; i < numToLog; ++i) {
            const auto& arg = blasIndirectArgs[i];
            bool aligned128 = (arg.clusterAddresses % 128 == 0);
            RTXMG_LOG(str::format("  [", i, "] clusterCount=", arg.clusterCount,
                " clusterAddresses=0x", std::hex, arg.clusterAddresses,
                std::dec, " stride=", arg.clusterReferencesStride,
                " aligned(128)=", aligned128 ? "YES" : "NO"));
        }
    }
#endif

    //// Build Operation
    // =================================================================================
    // Compute fresh params with numInstances (not m_instanceCapacity) for each build.
    // This matches the sample's behavior where getClusterOperationSizeInfo() and
    // the build always use the same maxArgCount = numInstances.
    //
    // The blasBuffer is sized for m_instanceCapacity (always >= numInstances), so
    // the output buffer is always big enough. The driver only processes the first
    // numInstances entries of the indirect args buffer.
    // =================================================================================
    cluster::OperationParams buildParams = m_createBlasParams;
    buildParams.maxArgCount = numInstances;

    // Use CAPACITY scratch (m_createBlasSizeInfo), not per-build scratch.
    // The Vulkan spec guarantees scratch from getClusterOperationSizeInfo(capacity)
    // is sufficient for any maxArgCount <= capacity. The output buffer (blasBuffer)
    // is capacity-sized, and the driver's internal scratch needs may depend on the
    // output buffer layout, not just maxArgCount.
    Logger::warn(str::format("RTX MegaGeo: BuildBlasFromClas - numInstances=", numInstances,
        " maxArgCount=", buildParams.maxArgCount,
        " capacity=", m_createBlasParams.maxArgCount,
        " capacityScratch=", m_createBlasSizeInfo.scratchSizeInBytes));

    cluster::OperationDesc createBlasDesc =
    {
        .params = buildParams,
        .scratchSizeInBytes = m_createBlasSizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = nullptr,
        .inIndirectArgCountOffsetInBytes = 0,
        .inIndirectArgsBuffer = m_blasFromClasIndirectArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.blasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = accels.blasSizesBuffer,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = accels.blasBuffer,
        .outAccelerationStructuresOffsetInBytes = 0,
    };

    RTXMG_LOG("RTX MegaGeo: BuildBlasFromClas - calling executeMultiIndirectClusterOperation");
    Logger::warn(str::format("RTX MegaGeo: BuildBlasFromClas - numInstances=", numInstances,
        " buildMaxArgCount=", buildParams.maxArgCount,
        " capacityMaxArgCount=", m_createBlasParams.maxArgCount,
        " blasPtrsAddr=0x", std::hex, blasPtrsAddr,
        " blasBufferAddr=0x", blasBufferAddr, std::dec,
        " blasPtrsElements=", accels.blasPtrsBuffer.GetNumElements(),
        " blasBufferBytes=", accels.blasBuffer.GetBytes(),
        " capacityScratchBytes=", m_createBlasSizeInfo.scratchSizeInBytes));
    commandList->executeMultiIndirectClusterOperation(createBlasDesc);
    RTXMG_LOG("RTX MegaGeo: BuildBlasFromClas - executeMultiIndirectClusterOperation complete");

    // DIAG: Download and log BLAS addresses after build
    {
        auto blasPtrs = accels.blasPtrsBuffer.Download(commandList);
        uint32_t numToLog = std::min(uint32_t(blasPtrs.size()), numInstances);
        uint32_t zeroCount = 0, nonZeroCount = 0;
        for (uint32_t i = 0; i < numToLog; ++i) {
            if (blasPtrs[i] == 0) zeroCount++;
            else nonZeroCount++;
        }
        Logger::warn(str::format("DIAG POST-BLAS: numInstances=", numInstances,
            " nonZero=", nonZeroCount, " zero=", zeroCount));
        for (uint32_t i = 0; i < std::min(numToLog, 10u); ++i) {
            Logger::warn(str::format("  blasPtr[", i, "] = 0x", std::hex, blasPtrs[i], std::dec));
        }
    }

    stats::clusterAccelSamplers.buildBlasTime.Stop();
}
void ClusterAccelBuilder::UpdateMemoryAllocations(ClusterAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches)
{
    uint32_t maxClusters = std::min(kMaxApiClusterCount, m_tessellatorConfig.memorySettings.maxClusters);
    maxClusters = std::max(1u, maxClusters);

    // Reallocate memory if settings changed
    size_t maxClusterBlocks = (m_tessellatorConfig.memorySettings.clasBufferBytes + (size_t(cluster::kClasByteAlignment) - 1ull)) / size_t(cluster::kClasByteAlignment);
    maxClusterBlocks = std::max(1ull, maxClusterBlocks);
    size_t maxClasBytes = size_t(cluster::kClasByteAlignment) * maxClusterBlocks;

    // Calculate max vertices based on vertex buffer bytes
    // NOTE: GPU StructuredBuffer<float3> uses stride=12 (3 floats), NOT C++ sizeof(float3)=16 (SIMD padded)
    static constexpr uint32_t kGpuFloat3Stride = 3 * sizeof(float); // 12 bytes - matches Slang sizeof(float3)
    uint32_t maxVertices = uint32_t(m_tessellatorConfig.memorySettings.vertexBufferBytes / kGpuFloat3Stride);
    maxVertices = std::max(kClusterMaxVertices, maxVertices);

    // Capture old values for logging
    uint32_t oldNumInstances = m_numInstances;
    uint32_t oldCapacity = m_instanceCapacity;
    uint32_t oldSceneSubdPatches = m_sceneSubdPatches;
    uint32_t oldMaxClusters = m_maxClusters;
    size_t oldMaxClasBytes = m_maxClasBytes;
    uint32_t oldMaxVertices = m_maxVertices;

    // Always update actual instance count (used for build operations)
    m_numInstances = numInstances;

    // ==================================================================================
    // SMART INSTANCE BUFFER SCALING (hysteresis + sustained check + cooldown)
    // Grow immediately when needed, only shrink after sustained underuse, with cooldown
    // to prevent oscillation. Modeled after dynamic worker pool scaling.
    // ==================================================================================
    bool instanceBuffersNeedResize = false;

    if (numInstances > m_instanceCapacity)
    {
        // GROW: immediate, with headroom to prevent micro-reallocations
        uint32_t newCapacity = numInstances + kInstanceGrowHeadroom;
        Logger::warn(str::format("RTX MegaGeo: Instance GROW - instances=", numInstances,
            " oldCapacity=", m_instanceCapacity, " newCapacity=", newCapacity,
            " frame=", m_currentFrameIndex));
        m_instanceCapacity = newCapacity;
        instanceBuffersNeedResize = true;
        m_instanceShrinkCounter = 0;
        m_instanceResizeCooldown = kInstanceResizeCooldownFrames; // Enter cooldown
    }
    else if (m_instanceCapacity > 0 && m_instanceResizeCooldown == 0)
    {
        // Check if we should SHRINK (only outside cooldown)
        float usage = float(numInstances) / float(m_instanceCapacity);
        if (usage < kInstanceShrinkThreshold)
        {
            // Below threshold - increment sustained counter
            m_instanceShrinkCounter++;
            if (m_instanceShrinkCounter >= kInstanceShrinkSustainedFrames)
            {
                // Sustained low usage - shrink with headroom
                uint32_t newCapacity = numInstances + kInstanceGrowHeadroom;
                Logger::warn(str::format("RTX MegaGeo: Instance SHRINK (sustained ", m_instanceShrinkCounter,
                    " frames) - instances=", numInstances,
                    " oldCapacity=", m_instanceCapacity, " newCapacity=", newCapacity,
                    " frame=", m_currentFrameIndex));
                m_instanceCapacity = newCapacity;
                instanceBuffersNeedResize = true;
                m_instanceShrinkCounter = 0;
                m_instanceResizeCooldown = kInstanceResizeCooldownFrames; // Enter cooldown
            }
        }
        else
        {
            // Usage in dead zone or above - reset shrink counter (hysteresis)
            m_instanceShrinkCounter = 0;
        }
    }

    // Tick cooldown
    if (m_instanceResizeCooldown > 0)
        m_instanceResizeCooldown--;

    // GridSamplers: grow immediately, shrink only after sustained low usage (same pattern as instances).
    // Prevents per-frame buffer churn while reclaiming memory in long sessions.
    bool gridSamplersNeedResize = false;
    if (sceneSubdPatches > m_gridSamplersCapacity) {
      gridSamplersNeedResize = true;
      m_gridSamplersShrinkCounter = 0;
      m_gridSamplersResizeCooldown = kInstanceResizeCooldownFrames;
    } else if (m_gridSamplersCapacity > 0 && m_gridSamplersResizeCooldown == 0) {
      float usage = (m_gridSamplersCapacity > 0) ? float(sceneSubdPatches) / float(m_gridSamplersCapacity) : 1.0f;
      if (usage < kInstanceShrinkThreshold) {
        m_gridSamplersShrinkCounter++;
        if (m_gridSamplersShrinkCounter >= kInstanceShrinkSustainedFrames) {
          gridSamplersNeedResize = true;
          m_gridSamplersShrinkCounter = 0;
          m_gridSamplersResizeCooldown = kInstanceResizeCooldownFrames;
        }
      } else {
        m_gridSamplersShrinkCounter = 0;
      }
    }
    if (m_gridSamplersResizeCooldown > 0)
      m_gridSamplersResizeCooldown--;
    bool sceneSubdPatchesChanged = m_sceneSubdPatches != sceneSubdPatches;
    bool numClustersChanged = m_maxClusters != maxClusters;
    bool clasBytesChanged = m_maxClasBytes != maxClasBytes;
    bool maxVerticesChanged = m_maxVertices != maxVertices;

    // Check if vertex normals setting changed by comparing current setting to buffer state
    bool prevVertexNormalsEnabled = accels.clusterVertexNormalsBuffer.GetBuffer() != nullptr && accels.clusterVertexNormalsBuffer.GetNumElements() == m_maxVertices;
    bool enableVertexNormalsChanged = (m_tessellatorConfig.enableVertexNormals != prevVertexNormalsEnabled);

    m_sceneSubdPatches = sceneSubdPatches;
    m_maxClusters = maxClusters;
    m_maxClasBytes = maxClasBytes;
    m_maxVertices = maxVertices;

    // No allocations needed
    if (!instanceBuffersNeedResize && !sceneSubdPatchesChanged && !numClustersChanged && !clasBytesChanged && !maxVerticesChanged && !enableVertexNormalsChanged)
    {
        return;
    }

    // Log which conditions triggered reallocation - ALWAYS visible (Logger::warn)
    Logger::warn(str::format("RTX MegaGeo: UpdateMemoryAllocations REALLOC - "
        "instResize=", instanceBuffersNeedResize, "(cap:", oldCapacity, "->", m_instanceCapacity, " actual:", numInstances, ") "
        "subdPatches=", sceneSubdPatchesChanged, "(", oldSceneSubdPatches, "->", sceneSubdPatches, ") "
        "clusters=", numClustersChanged, "(", oldMaxClusters, "->", maxClusters, ") "
        "clasBytes=", clasBytesChanged, " vertices=", maxVerticesChanged, "(", oldMaxVertices, "->", maxVertices, ") "
        "frame=", m_currentFrameIndex));
    if (sceneSubdPatchesChanged)
        RTXMG_LOG(str::format("  sceneSubdPatches: ", oldSceneSubdPatches, " -> ", sceneSubdPatches));
    if (numClustersChanged)
        RTXMG_LOG(str::format("  maxClusters: ", oldMaxClusters, " -> ", maxClusters));
    if (clasBytesChanged)
        RTXMG_LOG(str::format("  maxClasBytes: ", oldMaxClasBytes, " -> ", maxClasBytes));
    if (maxVerticesChanged)
        RTXMG_LOG(str::format("  maxVertices: ", oldMaxVertices, " -> ", maxVertices));

    // ==========================================================================
    // DEFERRED RELEASE: Save old buffer handles before creating replacements.
    // Old buffers are kept alive for kDeferredReleaseFrames frames so in-flight
    // GPU work (which may reference them via raw device addresses) can finish
    // before the memory is freed. This avoids the expensive flushCommandList +
    // waitForIdle that would stall the CPU and split the command buffer mid-frame.
    // ==========================================================================
    DeferredBufferRelease deferred;
    deferred.frameIndex = m_currentFrameIndex;

    auto deferBuffer = [&deferred](auto& rtxmgBuffer) {
        if (auto buf = rtxmgBuffer.GetBuffer()) {
            deferred.buffers.push_back(buf);
        }
        rtxmgBuffer.Release();
    };

    if (instanceBuffersNeedResize)
    {
        deferBuffer(m_clusterOffsetCountsBuffer);
        deferBuffer(m_fillClustersDispatchIndirectBuffer);
        deferBuffer(m_blasFromClasIndirectArgsBuffer);
        deferBuffer(accels.blasPtrsBuffer);
        deferBuffer(accels.blasSizesBuffer);
    }

    if (gridSamplersNeedResize)
    {
        deferBuffer(m_gridSamplersBuffer);
    }

    if (numClustersChanged)
    {
        deferBuffer(m_clustersBuffer);
        deferBuffer(m_clasIndirectArgDataBuffer);
        deferBuffer(accels.clusterShadingDataBuffer);
        deferBuffer(accels.clasPtrsBuffer);
    }

    if (numClustersChanged || instanceBuffersNeedResize)
    {
        deferBuffer(accels.blasBuffer);
    }

    if (clasBytesChanged)
    {
        deferBuffer(accels.clasBuffer);
    }

    if (maxVerticesChanged)
    {
        deferBuffer(accels.clusterVertexPositionsBuffer);
    }

    if (maxVerticesChanged || enableVertexNormalsChanged)
    {
        deferBuffer(accels.clusterVertexNormalsBuffer);
    }

    if (!deferred.buffers.empty()) {
        Logger::warn(str::format("RTX MegaGeo: UpdateMemoryAllocations - deferred ",
            deferred.buffers.size(), " old buffers for release in ",
            kDeferredReleaseFrames, " frames"));
        m_deferredReleases.push_back(std::move(deferred));
    }

    // ==========================================================================
    // CREATE NEW BUFFERS
    // ==========================================================================

    if (instanceBuffersNeedResize)
    {
        // Volatile CB: each writeBuffer call gets a new ring buffer version, preventing data races
        // when CopyClusterOffset is dispatched once per instance per frame (matching sample exactly).
        m_copyClusterOffsetParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(CopyClusterOffsetParams), "CopyClusterOffsetParams", m_numInstances * ClusterDispatchType::NumTypes * kFrameCount));

        // Use m_instanceCapacity (not m_numInstances) for buffer sizing - capacity >= actual count

        m_clusterOffsetCountsBuffer.Create(m_instanceCapacity * ClusterDispatchType::NumTypes, "ClusterOffsets", m_device.Get());

        nvrhi::BufferDesc dispatchIndirectDesc =
        {
            .byteSize = m_instanceCapacity * ClusterDispatchType::NumTypes * sizeof(uint3),
            .debugName = "FillClustersIndirectArgs",
            .structStride = uint32_t(sizeof(uint3)),
            .canHaveUAVs = true,
            .isDrawIndirectArgs = true,
            .initialState = nvrhi::ResourceStates::IndirectArgument,
            .keepInitialState = true,
        };
        m_fillClustersDispatchIndirectBuffer.Create(dispatchIndirectDesc, m_device.Get());

        // Align structStride to 16 bytes for Vulkan minStorageBufferOffsetAlignment
        uint32_t indirectArgElementSize = sizeof(cluster::IndirectArgs);
        uint32_t indirectArgAlignedStride = (indirectArgElementSize + 15) & ~15;
        nvrhi::BufferDesc clusterIndirectArgsDesc = {
            .byteSize = indirectArgAlignedStride * m_instanceCapacity,
            .debugName = "cluster::IndirectArgs",
            .structStride = indirectArgAlignedStride,
            .canHaveUAVs = true,
            .isAccelStructBuildInput = true,
            .initialState = nvrhi::ResourceStates::ShaderResource,
            .keepInitialState = true,
        };
        m_blasFromClasIndirectArgsBuffer.Create(clusterIndirectArgsDesc, m_device.Get());
        accels.blasPtrsBuffer.Create(m_instanceCapacity, "BlasPtrs", m_device.Get());
        accels.blasSizesBuffer.Create(m_instanceCapacity, "BlasSizes", m_device.Get());
    }

    if (gridSamplersNeedResize)
    {
        uint32_t newCapacity = sceneSubdPatches + (sceneSubdPatches / 4) + 256; // 25% headroom + minimum
        m_gridSamplersCapacity = newCapacity;
        m_gridSamplersBuffer.Create(newCapacity, "GridSamplers", m_device.Get());
    }

    if (numClustersChanged)
    {
        m_clustersBuffer.Create(m_maxClusters, "clusters", m_device.Get());
        m_clasIndirectArgDataBuffer.Create(m_maxClusters, "indirect arg data", m_device.Get());
        accels.clusterShadingDataBuffer.Create(m_maxClusters, "cluster shading data", m_device.Get());
        accels.clasPtrsBuffer.Create(m_maxClusters, "ClasAddresses", m_device.Get());
    }

    RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - DEBUG: numClustersChanged=", numClustersChanged, " instanceBuffersNeedResize=", instanceBuffersNeedResize));

    if (numClustersChanged || instanceBuffersNeedResize)
    {
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - creating BLAS buffers, m_instanceCapacity=", m_instanceCapacity, " m_maxClusters=", m_maxClusters));

        // Use m_instanceCapacity for sizing (not m_numInstances) - capacity is the buffer size upper bound
        m_createBlasParams =
        {
            .maxArgCount = m_instanceCapacity,
            .type = cluster::OperationType::BlasBuild,
            .mode = cluster::OperationMode::ImplicitDestinations,
            .flags = cluster::OperationFlags::None,
            .blas =
            {
                .maxClasPerBlasCount = m_maxClusters,
                .maxTotalClasCount = m_maxClusters
            }
        };
        m_createBlasSizeInfo = m_device->getClusterOperationSizeInfo(m_createBlasParams);
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - BLAS sizeInfo: resultMaxSizeInBytes=", m_createBlasSizeInfo.resultMaxSizeInBytes,
            " scratchSizeInBytes=", m_createBlasSizeInfo.scratchSizeInBytes));

        if (m_createBlasSizeInfo.resultMaxSizeInBytes == 0) {
            Logger::err("RTX MegaGeo: UpdateMemoryAllocations - resultMaxSizeInBytes is 0!");
        }
        if (m_createBlasSizeInfo.scratchSizeInBytes == 0) {
            Logger::warn("RTX MegaGeo: UpdateMemoryAllocations - scratchSizeInBytes is 0 (may be OK if no scratch needed)");
        }

        nvrhi::BufferDesc blasBufferDesc = {
            .byteSize = m_createBlasSizeInfo.resultMaxSizeInBytes,
            .debugName = "Blas Data",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        if (m_createBlasSizeInfo.resultMaxSizeInBytes > 0)
        {
            accels.blasBuffer.Create(blasBufferDesc, m_device.Get());
            RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - blasBuffer created, ptr=", (void*)accels.blasBuffer.GetBuffer().Get()));
            if (!accels.blasBuffer.GetBuffer())
            {
                Logger::err("RTX MegaGeo: UpdateMemoryAllocations - blasBuffer creation FAILED!");
            }
        }
        else
        {
            Logger::err("RTX MegaGeo: UpdateMemoryAllocations - cannot create blasBuffer with size 0!");
        }
    }
    else
    {
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - skipping BLAS buffer creation (no change)"));
    }

    if (clasBytesChanged)
    {
        nvrhi::BufferDesc clasDataDesc =
        {
            .byteSize = m_maxClasBytes,
            .debugName = "ClasData",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        accels.clasBuffer.Create(clasDataDesc, m_device.Get());
    }

    if (maxVerticesChanged)
    {
        // CRITICAL: clusterVertexPositionsBuffer is read by CLAS instantiation via device address,
        // which requires VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        // NOTE: GPU stride is 12 (3 floats), NOT C++ sizeof(float3)=16 (SIMD padded)
        static constexpr uint32_t kGpuFloat3Stride = 3 * sizeof(float); // 12 bytes
        size_t byteSize = m_maxVertices * kGpuFloat3Stride;
        size_t alignedByteSize = (byteSize + 3) & ~3;  // Round up to multiple of 4
        nvrhi::BufferDesc vertexPosDesc = {
            .byteSize = alignedByteSize,
            .debugName = "cluster vertex positions",
            .structStride = kGpuFloat3Stride,
            .canHaveUAVs = true,
            .canHaveTypedViews = true,
            .canHaveRawViews = true,
            .isAccelStructBuildInput = true,  // Required for CLAS to read via device address
            .initialState = nvrhi::ResourceStates::UnorderedAccess,
            .keepInitialState = true
        };
        accels.clusterVertexPositionsBuffer.Create(vertexPosDesc, m_device.Get());
    }

    if (maxVerticesChanged || enableVertexNormalsChanged)
    {
        // WARNING: This uses RTXMGBuffer<float3>::Create(nElements, ...) which sizes with
        // sizeof(MathLib::float3)=16. GPU shader writes at 12-byte stride. Buffer is oversized
        // but normals aren't used for CLAS geometry, only for shading, so this is OK for now.
        // TODO: Use explicit 12-byte stride allocation like vertexPositionsBuffer above.
        accels.clusterVertexNormalsBuffer.Create(m_tessellatorConfig.enableVertexNormals ? m_maxVertices : 1, "cluster vertex normals", m_device.Get());
    }
}

void ClusterAccelBuilder::EnsureTemplatesInitialized(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList)
{
    // Initialize cluster templates early, before any image views are bound
    // The sync Downloads in InitStructuredClusterTemplates close/reopen the command list
    // which destroys any bound resources (like HiZ textures)
    InitStructuredClusterTemplates(maxGeometryCountPerMesh, commandList);
}

void ClusterAccelBuilder::BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config,
    ClusterAccels& accels, ClusterStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList)
{
    m_currentFrameIndex = frameIndex;

    // Release deferred buffers that are old enough (GPU has finished using them)
    while (!m_deferredReleases.empty() &&
           m_currentFrameIndex >= m_deferredReleases.front().frameIndex + kDeferredReleaseFrames) {
        m_deferredReleases.pop_front();
    }

#if RTXMG_CHRONO_TIMING
    auto chronoStart = std::chrono::high_resolution_clock::now();
    auto chronoSectionStart = chronoStart;
#endif
    m_tessellatorConfig = config;

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    if (subdMeshes.empty() || instances.empty())
        return;

    uint32_t totalSubdPatches = scene.TotalSubdPatchCount();
    RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instances=", instances.size(),
        " subdMeshes=", subdMeshes.size(), " totalSubdPatches=", totalSubdPatches));
#if RTXMG_CHRONO_TIMING
    auto setupStart = std::chrono::high_resolution_clock::now();
#endif
    UpdateMemoryAllocations(accels, uint32_t(instances.size()), totalSubdPatches);
#if RTXMG_CHRONO_TIMING
    auto afterMemAlloc = std::chrono::high_resolution_clock::now();
    float memAllocMs = std::chrono::duration_cast<std::chrono::microseconds>(afterMemAlloc - setupStart).count() * 0.001f;
    if (memAllocMs > 1.0f) {
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: UpdateMemoryAllocations=", memAllocMs, "ms (SLOW - likely waitForIdle)"));
    }
#endif

    const uint32_t maxGeometryCountPerMesh = uint32_t(scene.GetSceneGraph()->GetMaxGeometryCountPerMesh());
    InitStructuredClusterTemplates(maxGeometryCountPerMesh, commandList);
#if RTXMG_CHRONO_TIMING
    auto afterTemplates = std::chrono::high_resolution_clock::now();
    float templatesMs = std::chrono::duration_cast<std::chrono::microseconds>(afterTemplates - afterMemAlloc).count() * 0.001f;
    if (templatesMs > 1.0f) {
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: InitStructuredClusterTemplates=", templatesMs, "ms (SLOW)"));
    }
#endif
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after InitStructuredClusterTemplates");

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildAccel");
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after ScopedMarker");

    uint32_t tessCounterIndex = (m_buildAccelFrameIndex % kFrameCount);
    nvrhi::BufferRange tessCounterRange = { m_tessellationCountersBuffer.GetElementBytes() * tessCounterIndex, m_tessellationCountersBuffer.GetElementBytes() };
    RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - tessCounterIndex=", tessCounterIndex));
    RTXMG_LOG(str::format("RTX MegaGeo: tessCounterRange offset=", tessCounterRange.byteOffset,
                             " size=", tessCounterRange.byteSize,
                             " bufferSize=", m_tessellationCountersBuffer.GetBytes(),
                             " elementSize=", m_tessellationCountersBuffer.GetElementBytes()));

    // Clear tessellation counters for this frame
    // On first build, zero ALL slots to prevent garbage readback from uninitialized slots
    TessellationCounters tessCounters = {};
    if (m_buildAccelFrameIndex < kFrameCount) {
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            m_tessellationCountersBuffer.UploadElement(tessCounters, i, commandList);
        }
    } else {
        m_tessellationCountersBuffer.UploadElement(tessCounters, tessCounterIndex, commandList);
    }

#if RTXMG_CHRONO_TIMING
    auto beforeClears = std::chrono::high_resolution_clock::now();
#endif
    // =====================================================================================
    // Per-frame buffer clears
    // =====================================================================================
    // All these clears go through our NVRHI adapter's clearBufferUInt(), which:
    //   1. Calls requireBufferState(buffer, CopyDest) → transitions to VK_PIPELINE_STAGE_TRANSFER_BIT
    //   2. Calls commitBarriers() → emits VkBufferMemoryBarrier
    //   3. Calls DXVK's m_context->clearBuffer() → vkCmdFillBuffer (TRANSFER operation)
    //
    // IMPORTANT: DXVK's clearBuffer() also does its OWN internal barrier tracking via
    // m_execBarriers.accessBuffer() at TRANSFER stage. This creates DUAL barrier tracking
    // (our NVRHI m_BufferStates + DXVK's m_execBarriers). For most buffers this is harmless
    // because the extra DXVK barrier is just redundant. But for blasBuffer specifically,
    // the dual tracking matters - see the blasBuffer clear comment below.
    // =====================================================================================
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 1");
    commandList->clearBufferUInt(m_clusterOffsetCountsBuffer.Get(), 0);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 2");
    commandList->clearBufferUInt(m_fillClustersDispatchIndirectBuffer.Get(), 0);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 3");
    commandList->clearBufferUInt(m_blasFromClasIndirectArgsBuffer.Get(), 0);

    // blasPtrsBuffer: zero prevents garbage/stale BLAS addresses from being patched into TLAS.
    // After reallocation, new entries are uninitialized. Instances without BLAS (overflow,
    // culled) would keep stale addresses. Zero = no BLAS → patch shader writes
    // accelerationStructureReference=0 safely.
    commandList->clearBufferUInt(accels.blasPtrsBuffer.Get(), 0);
    commandList->clearBufferUInt(accels.blasSizesBuffer.Get(), 0);

    // =====================================================================================
    // blasBuffer clear - DO NOT REMOVE
    // =====================================================================================
    // This clear is REQUIRED for correctness in ImplicitDestinations mode.
    //
    // WHY IT'S NEEDED:
    //   When the blasBuffer is resized (new VkBuffer allocated), its contents are undefined.
    //   The NVIDIA driver in ImplicitDestinations mode reads from the output buffer during
    //   BLAS builds (presumably to manage internal allocation state). Undefined/garbage data
    //   causes the driver to access invalid memory → GPU page fault → VK_ERROR_DEVICE_LOST
    //   (TDR after ~2-4 seconds).
    //
    // WHY THE SAMPLE DOESN'T NEED IT:
    //   The sample (donut NVRHI) uses a different Vulkan backend where buffer creation may
    //   zero-initialize, or the sample's usage patterns avoid the stale data issue.
    //   Our DXVK backend does NOT zero-initialize buffers.
    //
    // TESTED: Removing this clear causes GPU hangs on the frame after instance buffer resize.
    //   With the clear present + heavy barrier, resizes survive reliably.
    //
    // BARRIER IMPLICATIONS:
    //   This clear puts the blasBuffer in CopyDest state (TRANSFER stage) in our NVRHI
    //   tracking. The pre-BLAS-build barrier in executeMultiIndirectClusterOperation then
    //   transitions it from CopyDest → UnorderedAccess|AccelStructWrite. The heavy barrier
    //   (ALL_COMMANDS→ALL_COMMANDS) inserted before the BLAS build provides additional
    //   synchronization that covers any gaps in the dual barrier tracking.
    // =====================================================================================
    commandList->clearBufferUInt(accels.blasBuffer.Get(), 0);
#if RTXMG_CHRONO_TIMING
    auto afterClears = std::chrono::high_resolution_clock::now();
    float clearsMs = std::chrono::duration_cast<std::chrono::microseconds>(afterClears - beforeClears).count() * 0.001f;
    if (clearsMs > 1.0f) {
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: BufferClears=", clearsMs, "ms (SLOW)"));
    }
#endif

    // Transition dummy HiZ textures to ShaderResource on first use only
    // Use RtxContext's initImage to initialize the textures from UNDEFINED to their stable layout
    if (!m_tessellatorConfig.zbuffer && !m_dummyHiZTexturesInitialized)
    {
        RTXMG_LOG("RTX MegaGeo: Initializing dummy HiZ textures via initImage");

        VkImageSubresourceRange subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, VK_REMAINING_MIP_LEVELS,
            0, VK_REMAINING_ARRAY_LAYERS
        };

        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            if (m_dummyHiZTextures[i])
            {
                // Get the underlying DxvkImage from our NVRHI texture
                NvrhiDxvkTexture* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(m_dummyHiZTextures[i].Get());
                const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();

                // Use initImage to transition from UNDEFINED to the image's stable layout
                // This is DXVK's standard way to initialize newly created images
                m_rtxContext->initImage(
                    dxvkImage,
                    subresourceRange,
                    VK_IMAGE_LAYOUT_UNDEFINED);

                RTXMG_LOG(str::format("RTX MegaGeo: Initialized DummyHiZ_Level_", i, " via initImage"));
            }
        }

        // Force the command list to be flushed so the init barriers are executed
        // This ensures the images are in the correct layout before any subsequent use
        m_rtxContext->flushCommandList();
        RTXMG_LOG("RTX MegaGeo: Flushed command list after dummy HiZ initialization");

        m_dummyHiZTexturesInitialized = true;
        RTXMG_LOG("RTX MegaGeo: Dummy HiZ texture initialization complete");
    }

    // Clear debug buffer before tiling loop so compute_cluster_tiling can write to slots 0-7
    // and fill_clusters can write to slots 8-15. Only do it until we get data.
    if (!g_megageoDbgGotData) {
        Logger::warn(str::format("TILING_DBG: Clearing debug buffer (size=", m_debugBuffer.GetBytes(), " bytes) + barrier"));
        commandList->clearBufferUInt(m_debugBuffer.Get(), 0);
        // Explicit barrier: clearBufferUInt leaves dual-tracked state (NVRHI CopyDest + DXVK TRANSFER).
        // Without this barrier, the tiling shader's UAV writes to the debug buffer may be lost.
        commandList->bufferBarrier(m_debugBuffer, nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RTXMG_LOG("RTX MegaGeo: BuildAccel - entering ComputeClusterTiling");
        nvrhi::utils::ScopedMarker marker(commandList, "ComputeClusterTiling");
        stats::clusterAccelSamplers.clusterTilingTime.Start(commandList);
        uint32_t surfaceOffset = 0;
        // Limit to m_numInstances to avoid buffer overflows
        uint32_t maxInstances = std::min(uint32_t(instances.size()), m_numInstances);
#if RTXMG_CHRONO_TIMING
        float totalInstanceMs = 0.0f;
        uint32_t totalSurfaces = 0;
#endif
        for (uint32_t i = 0; i < maxInstances; ++i)
        {
#if RTXMG_CHRONO_TIMING
            auto instanceStart = std::chrono::high_resolution_clock::now();
#endif
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - loop iteration start i=", i));
            const auto& inst = instances[i];
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - got inst, meshID=", inst.meshID, " subdMeshes.size()=", subdMeshes.size()));

            // Bounds check to prevent crash
            if (inst.meshID >= subdMeshes.size()) {
                Logger::err(str::format("RTX MegaGeo: BuildAccel - meshID ", inst.meshID, " out of bounds (subdMeshes.size()=", subdMeshes.size(), "), skipping instance ", i));
                continue;
            }

            const auto& subd = *subdMeshes[inst.meshID];
            RTXMG_LOG("RTX MegaGeo: BuildAccel - got subd");

            uint32_t surfaceCount{ subd.SurfaceCount() };
            // Log surface type distribution for each mesh (only first few frames)
            {
                uint32_t pureBSpline = subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::RegularBSpline)] - subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::PureBSpline)];
                uint32_t regularBSpline = subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::Limit)] - subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::RegularBSpline)];
                uint32_t limit = subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)] - subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::Limit)];
                uint32_t noLimit = subd.m_surfaceCount - subd.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
                Logger::err(str::format("RTX MegaGeo SUBD: inst[", i, "] meshID=", inst.meshID,
                    " surfaces=", surfaceCount,
                    " PureBSpline=", pureBSpline,
                    " RegularBSpline=", regularBSpline,
                    " Limit=", limit,
                    " NoLimit=", noLimit,
                    " offsets=[", subd.m_surfaceOffsets[0], ",", subd.m_surfaceOffsets[1], ",", subd.m_surfaceOffsets[2], ",", subd.m_surfaceOffsets[3], "]",
                    " isolationLevel=", m_tessellatorConfig.isolationLevel));
            }
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instance ", i, " surfaceCount=", surfaceCount));

            ComputeInstanceClusterTiling(accels, scene, i, surfaceOffset, surfaceCount, tessCounterRange, commandList);
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instance ", i, " ComputeInstanceClusterTiling complete"));

            surfaceOffset += surfaceCount;
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - loop iteration end i=", i, " surfaceOffset=", surfaceOffset));
#if RTXMG_CHRONO_TIMING
            auto instanceEnd = std::chrono::high_resolution_clock::now();
            float instanceMs = std::chrono::duration_cast<std::chrono::microseconds>(instanceEnd - instanceStart).count() * 0.001f;
            totalInstanceMs += instanceMs;
            totalSurfaces += surfaceCount;
            // Log every 10th instance or if it took >5ms
            if (i % 10 == 0 || instanceMs > 5.0f) {
                RTXMG_LOG(str::format(">>> RTXMG CHRONO: Instance[", i, "] surfaces=", surfaceCount, " time=", instanceMs, "ms"));
            }
#endif
        }
        stats::clusterAccelSamplers.clusterTilingTime.Stop();
#if RTXMG_CHRONO_TIMING
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float tilingMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        float avgPerInstance = maxInstances > 0 ? totalInstanceMs / maxInstances : 0.0f;
        float avgPerSurface = totalSurfaces > 0 ? totalInstanceMs / totalSurfaces : 0.0f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: ComputeClusterTiling TOTAL=", tilingMs, "ms instances=", maxInstances,
            " surfaces=", totalSurfaces, " avgPerInst=", avgPerInstance, "ms avgPerSurf=", avgPerSurface, "ms"));
        chronoSectionStart = chronoNow;
#endif
    }

    // One-shot readback of debug buffer after tiling loop (before fill_clusters)
    // This tests whether compute_cluster_tiling's UAV(8) writes actually persist.
    // NOTE: Download() closes/reopens command list - safe here since no HiZ image views
    // are bound between tiling and fill (binding sets are created fresh per dispatch).
    {
        // Delay debug readback to frame 3 so we skip UI-only frames (frame 1 has 0 clusters)
        // NOTE: removed !g_megageoDbgGotData guard — DumpDiagnosticData sets it true on frame 1,
        // which prevented this from ever firing on frame 3.
        static uint32_t s_tilingDbgFrameCount = 0;
        s_tilingDbgFrameCount++;
        Logger::warn(str::format("TILING_DBG_CANARY: frame=", s_tilingDbgFrameCount, " instances=", instances.size()));
        if (s_tilingDbgFrameCount == 3) {
            Logger::warn(str::format("TILING_DBG: ENTERING readback block (frame=3, instances=", instances.size(), ")"));
            Logger::warn(str::format("TILING_DBG: Readback AFTER tiling loop (frame=3, instances=", instances.size(), ")"));

            // FIRST: read back debug buffer as-is (should have shader writes)
            Logger::warn("TILING_DBG: === Phase 1: Read back shader writes ===");
            Logger::warn("TILING_DBG: about to call m_debugBuffer.Download()...");
            auto tilingDbg = m_debugBuffer.Download(commandList);
            Logger::warn(str::format("TILING_DBG: Download returned, size=", tilingDbg.size(), " elements, instances=", instances.size()));
            uint32_t numSlots = std::min((uint32_t)instances.size() + 2, (uint32_t)tilingDbg.size());
            numSlots = std::min(numSlots, 48u);  // cap at 48
            uint32_t slotsWithData = 0;
            uint32_t slotsZero = 0;
            for (uint32_t i = 0; i < numSlots; ++i) {
                const auto& e = tilingDbg[i];
                bool hasData = (e.uintData.x || e.uintData.y || e.payloadType);
                if (hasData) slotsWithData++; else slotsZero++;
                if (i < 8 || i >= numSlots - 4 || hasData) {
                    Logger::warn(str::format("TILING_DBG[", i, "] uint=(0x", std::hex, e.uintData.x, std::dec,
                        ",", e.uintData.y, ",", e.uintData.z, ",", e.uintData.w,
                        ") pt=", e.payloadType, " ln=", e.lineNumber,
                        hasData ? "" : " ZERO"));
                }
            }
            Logger::warn(str::format("TILING_DBG: SUMMARY: ", slotsWithData, " slots with data, ", slotsZero, " zero slots (of ", numSlots, " checked)"));

            // SECOND: Read back u_Clusters[0] to check if UAV2 (binding 202) got the marker
            Logger::warn("TILING_DBG: === Phase 2: Check u_Clusters[0] for marker (UAV2/binding202) ===");
            auto tilingClusters = m_clustersBuffer.Download(commandList);
            if (!tilingClusters.empty()) {
                bool clusterMarkerFound = (tilingClusters[0].iSurface == 0xDEAD0202);
                Logger::warn(str::format("TILING_DBG: u_Clusters[0].iSurface = 0x", std::hex, tilingClusters[0].iSurface, std::dec,
                    " - marker ", clusterMarkerFound ? "FOUND (UAV2 WORKS!)" : "NOT FOUND"));
                Logger::warn(str::format("TILING_DBG: u_Clusters[0].nVertexOffset = ", tilingClusters[0].nVertexOffset));
                for (uint32_t i = 0; i < 4 && i < tilingClusters.size(); ++i) {
                    Logger::warn(str::format("TILING_DBG: cluster[", i, "] iSurface=0x", std::hex, tilingClusters[i].iSurface,
                        std::dec, " vtxOff=", tilingClusters[i].nVertexOffset));
                }
            }

            // THIRD: vkCmdFillBuffer test on debug buffer
            Logger::warn("TILING_DBG: === Phase 3: vkCmdFillBuffer test pattern ===");
            commandList->clearBufferUInt(m_debugBuffer.Get(), 0xDEADBEEF);
            commandList->bufferBarrier(m_debugBuffer, nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::CopySource);
            auto fillTestDbg = m_debugBuffer.Download(commandList);
            if (!fillTestDbg.empty()) {
                const auto& e = fillTestDbg[0];
                bool fillWorked = (e.uintData.x == 0xDEADBEEF);
                Logger::warn(str::format("TILING_DBG: vkCmdFillBuffer test: [0].uint.x=0x", std::hex, e.uintData.x, std::dec,
                    " - readback path ", fillWorked ? "WORKS" : "BROKEN"));
            }
        }
    }

    // UAV barriers after compute_cluster_tiling / CopyClusterOffset.
    // CopyClusterOffset writes these as UAVs; FillInstanceClusters reads them as SRVs / indirect args.
    // Without explicit barriers, fill_clusters reads stale data from previous frames.
    RTXMG_LOG("RTX MegaGeo: BuildAccel - adding UAV barriers after ComputeClusterTiling");
    // UAV→UAV barriers ensure compute writes are visible to subsequent compute reads.
    // We use UAV→UAV (not UAV→ShaderResource) because the NVRHI adapter's m_BufferStates
    // tracking gets confused by state changes — subsequent clearBufferUInt expects UAV state.
    // UAV→UAV still inserts a full VkMemoryBarrier (compute write → compute read).
    commandList->bufferBarrier(m_clusterOffsetCountsBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
    commandList->bufferBarrier(m_fillClustersDispatchIndirectBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
    commandList->bufferBarrier(m_clustersBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
    // clusterShadingDataBuffer: written here, read later by ray tracing
    commandList->bufferBarrier(accels.clusterShadingDataBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
    // patchPoints buffers: written as UAV by compute_cluster_tiling's WaveEvaluatePatchPoints,
    // read as SRV by fill_clusters' EvaluateLimitSurface. Need barrier for Limit surface data.
    for (uint32_t i = 0; i < instances.size(); ++i) {
        const auto& inst = instances[i];
        if (inst.meshID < subdMeshes.size()) {
            const auto& subd = *subdMeshes[inst.meshID];
            if (subd.m_vertexDeviceData.patchPoints)
                commandList->bufferBarrier(subd.m_vertexDeviceData.patchPoints.Get(), nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
            if (subd.m_texcoordDeviceData.patchPoints)
                commandList->bufferBarrier(subd.m_texcoordDeviceData.patchPoints.Get(), nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);
        }
    }

    FillInstanceClusters(scene, accels, commandList);

    // Full diagnostic readback (vertex positions, clusters, CLAS args, etc.)
    DumpDiagnosticData(accels, commandList);

#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float fillMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: FillInstanceClusters=", fillMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // CRITICAL: Add UAV barrier after FillInstanceClusters to ensure vertex positions
    // are written before CLAS instantiation reads them via device addresses
    RTXMG_LOG("RTX MegaGeo: BuildAccel - adding UAV barrier for vertex positions (UAV -> AccelStructBuildInput)");
    commandList->bufferBarrier(accels.clusterVertexPositionsBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::AccelStructBuildInput);

    // Build CLASes for all instances at once
    stats::clusterAccelSamplers.buildClasTime.Start(commandList);
    BuildStructuredCLASes(accels, maxGeometryCountPerMesh, tessCounterRange, commandList);
    stats::clusterAccelSamplers.buildClasTime.Stop();
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float clasMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: BuildStructuredCLASes=", clasMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // CRITICAL BARRIER: The CLAS build (BuildStructuredCLASes) writes to clasPtrsBuffer and
    // clasBuffer via vkCmdBuildClusterAccelerationStructureIndirectNV, which bypasses DXVK's
    // barrier tracking entirely. The subsequent FillBlasFromClasArgs compute shader reads
    // clasPtrsBuffer via raw device address, and the BLAS build reads clasBuffer data via
    // device addresses in the indirect args. Without this barrier, the GPU may read
    // incomplete/stale CLAS data, causing TDR on resize frames when all commands execute
    // in a single submission after flush+waitForIdle.
    // Combined ShaderResource|AccelStructBuildInput covers both SHADER_READ and
    // ACCELERATION_STRUCTURE_READ access, ensuring visibility for both the compute shader
    // (FillBlasFromClasArgs) and the BLAS build's AS-level reads of CLAS data.
    commandList->bufferBarrier(accels.clasPtrsBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::AccelStructBuildInput);

    // Build BLAS unconditionally (matching sample behavior)
    // NOTE: Removed sync Download check for clusters > 0 because Download closes/reopens
    // command list which destroys bound image views in DXVK
    uint32_t blasInstanceCount = std::min(uint32_t(instances.size()), m_numInstances);
    BuildBlasFromClas(accels, instances.data(), blasInstanceCount, commandList);
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float blasMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: BuildBlasFromClas=", blasMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Async read of counters (reading previous frame's results for double-buffering)
    // On early frames, previous frame data may not exist yet, so fall back to current frame's data
    uint32_t readIndex = (tessCounterIndex + 1) % kFrameCount;
    RTXMG_LOG(str::format("RTX MegaGeo: About to download counters - writeIndex=", tessCounterIndex,
                             " readIndex=", readIndex, " frame=", m_buildAccelFrameIndex));
    auto counterBufferData = m_tessellationCountersBuffer.Download(commandList, true);
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float downloadMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: DownloadCounters=", downloadMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Log ALL counter indices to see which ones have data - always log for debugging
    for (uint32_t i = 0; i < kFrameCount; ++i) {
        ONCE(RTXMG_LOG(str::format("RTX MegaGeo: Counter[", i, "] clusters=", counterBufferData[i].clusters,
                                 " desiredClusters=", counterBufferData[i].desiredClusters,
                                 " desiredTriangles=", counterBufferData[i].desiredTriangles,
                                 " desiredVertices=", counterBufferData[i].desiredVertices,
                                 " desiredClasBlocks=", counterBufferData[i].desiredClasBlocks)));
    }

    // If readIndex has no valid data (0 clusters or garbage values exceeding max),
    // find the first index with valid data.
    // This handles early frames where previous frame data doesn't exist yet.
    auto isValidCounter = [this](const TessellationCounters& c) {
        return c.clusters > 0 && c.clusters <= m_maxClusters;
    };
    TessellationCounters counters = counterBufferData[readIndex];
    if (!isValidCounter(counters)) {
        // Search for any index with valid cluster data, prefer current frame's index
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            uint32_t checkIndex = (tessCounterIndex + kFrameCount - i) % kFrameCount;
            if (isValidCounter(counterBufferData[checkIndex])) {
                readIndex = checkIndex;
                counters = counterBufferData[readIndex];
                RTXMG_LOG(str::format("RTX MegaGeo: Fallback to counter index ", readIndex, " with ", counters.clusters, " clusters"));
                break;
            }
        }
    }
    RTXMG_LOG(str::format("RTX MegaGeo: Using counters from index ", readIndex,
                             ": clusters=", counters.clusters, " desired=", counters.desiredClusters));

    RTXMG_LOG(str::format("RTX MegaGeo COUNTERS[", readIndex, "]: clusters=", counters.clusters,
        " desiredClusters=", counters.desiredClusters,
        " desiredTriangles=", counters.desiredTriangles,
        " desiredVertices=", counters.desiredVertices,
        " desiredClasBlocks=", counters.desiredClasBlocks));
    for (uint32_t ci = 0; ci < kFrameCount; ++ci) {
      RTXMG_LOG(str::format("RTX MegaGeo COUNTERS slot[", ci, "]: clusters=", counterBufferData[ci].clusters,
          " desiredClusters=", counterBufferData[ci].desiredClusters,
          " desiredTris=", counterBufferData[ci].desiredTriangles,
          " desiredVerts=", counterBufferData[ci].desiredVertices));
    }

    // Record the desired required memory instead of the max
    stats.desired.m_numTriangles = counters.desiredTriangles;
    stats.desired.m_numClusters = counters.desiredClusters;
    stats.desired.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetElementBytes() * counters.desiredVertices;
    stats.desired.m_vertexNormalsBufferSize = m_tessellatorConfig.enableVertexNormals ? 
        (accels.clusterVertexNormalsBuffer.GetElementBytes() * counters.desiredVertices) : 0;
    stats.desired.m_clasSize = counters.DesiredClasBytes();
    stats.desired.m_clusterDataSize = (m_clustersBuffer.GetElementBytes() + 
        accels.clusterShadingDataBuffer.GetElementBytes() +
        accels.clasPtrsBuffer.GetElementBytes()) * counters.desiredClusters;
    stats.desired.m_blasSize = m_createBlasSizeInfo.resultMaxSizeInBytes;
    stats.desired.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    // Atomics are expensive so we don't track the number of allocated triangles
    stats.allocated.m_numTriangles = counters.desiredTriangles;
    stats.allocated.m_numClusters = m_maxClusters;
    stats.allocated.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetBytes();
    stats.allocated.m_vertexNormalsBufferSize = accels.clusterVertexNormalsBuffer.GetBytes();
    stats.allocated.m_clasSize = accels.clasBuffer.GetBytes();
    stats.allocated.m_clusterDataSize = m_clustersBuffer.GetBytes() + accels.clusterShadingDataBuffer.GetBytes() + accels.clasPtrsBuffer.GetBytes();
    stats.allocated.m_blasSize = accels.blasBuffer.GetBytes();
    stats.allocated.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    m_buildAccelFrameIndex++;

    // VRAM usage summary - log every 60 frames
    if ((m_buildAccelFrameIndex % 60) == 1) {
        size_t vtxPosBuf = accels.clusterVertexPositionsBuffer.GetBytes();
        size_t vtxNrmBuf = accels.clusterVertexNormalsBuffer.GetBytes();
        size_t clasBuf = accels.clasBuffer.GetBytes();
        size_t clustersBuf = m_clustersBuffer.GetBytes();
        size_t shadingBuf = accels.clusterShadingDataBuffer.GetBytes();
        size_t clasPtrsBuf = accels.clasPtrsBuffer.GetBytes();
        size_t blasBuf = accels.blasBuffer.GetBytes();
        size_t blasPtrsBuf = accels.blasPtrsBuffer.GetBytes();
        size_t blasSizesBuf = accels.blasSizesBuffer.GetBytes();
        size_t templateSmallBuf = m_templateBuffers.sizesBuffer.GetBytes() + m_templateBuffers.addressesBuffer.GetBytes() + m_templateBuffers.instantiationSizesBuffer.GetBytes();
        size_t templateDataBuf = m_templateBuffers.dataBuffer ? m_templateBuffers.dataBuffer->getDesc().byteSize : 0;
        size_t templateIdxBuf = m_templateBuffers.indexBuffer ? m_templateBuffers.indexBuffer->getDesc().byteSize : 0;
        size_t templateVtxBuf = m_templateBuffers.vertexBuffer ? m_templateBuffers.vertexBuffer->getDesc().byteSize : 0;
        size_t templateBuf = templateSmallBuf + templateDataBuf + templateIdxBuf + templateVtxBuf;
        size_t dispatchBuf = m_fillClustersDispatchIndirectBuffer.GetBytes();
        size_t offsetCountsBuf = m_clusterOffsetCountsBuffer.GetBytes();
        size_t blasFromClasArgsBuf = m_blasFromClasIndirectArgsBuffer.GetBytes();
        size_t tessCountersBuf = m_tessellationCountersBuffer.GetBytes();
        size_t totalVRAM = vtxPosBuf + vtxNrmBuf + clasBuf + clustersBuf + shadingBuf + clasPtrsBuf + blasBuf + blasPtrsBuf + blasSizesBuf + templateBuf + dispatchBuf + offsetCountsBuf + blasFromClasArgsBuf + tessCountersBuf;

        RTXMG_LOG(str::format("RTX MegaGeo VRAM: TOTAL=", totalVRAM / (1024*1024), "MB (",  totalVRAM, " bytes)"));
        RTXMG_LOG(str::format("  vtxPositions=", vtxPosBuf / 1024, "KB vtxNormals=", vtxNrmBuf / 1024, "KB"));
        RTXMG_LOG(str::format("  CLAS=", clasBuf / (1024*1024), "MB clusters=", clustersBuf / 1024, "KB shading=", shadingBuf / 1024, "KB clasPtrs=", clasPtrsBuf / 1024, "KB"));
        RTXMG_LOG(str::format("  BLAS=", blasBuf / (1024*1024), "MB blasPtrs=", blasPtrsBuf / 1024, "KB blasSizes=", blasSizesBuf / 1024, "KB"));
        RTXMG_LOG(str::format("  templates=", templateBuf / 1024, "KB (data=", templateDataBuf / 1024, "KB idx=", templateIdxBuf / 1024, "KB vtx=", templateVtxBuf / 1024, "KB small=", templateSmallBuf / 1024, "KB)"));
        RTXMG_LOG(str::format("  dispatch=", dispatchBuf / 1024, "KB offsetCounts=", offsetCountsBuf / 1024, "KB blasFromClasArgs=", blasFromClasArgsBuf / 1024, "KB tessCounters=", tessCountersBuf / 1024, "KB"));
        RTXMG_LOG(str::format("  maxClusters=", m_maxClusters, " maxVertices=", m_maxVertices, " numInstances=", m_numInstances, " maxClasBytes=", m_maxClasBytes));
    }

    // Log final statistics - always visible
    Logger::warn(str::format("RTX MegaGeo: BuildAccel[", m_buildAccelFrameIndex, "] clusters=", counters.clusters,
        "/", m_maxClusters, " desired=", counters.desiredClusters,
        " instances=", m_numInstances, "/", m_instanceCapacity,
        " blasPtrs=", accels.blasPtrsBuffer.GetNumElements(),
        " blasBuf=", accels.blasBuffer.GetBytes() / 1024, "KB"));

#if RTXMG_CHRONO_TIMING
    {
        auto chronoEnd = std::chrono::high_resolution_clock::now();
        float totalMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoEnd - chronoStart).count() * 0.001f;
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: BuildAccel TOTAL=", totalMs, "ms clusters=", counters.clusters));
    }
#endif
}

void ClusterAccelBuilder::DumpDiagnosticData(ClusterAccels& accels, nvrhi::ICommandList* commandList)
{
    using dxvk::Logger;

    // Only dump when we actually have clusters (skip UI-only frames)
    if (m_maxClusters == 0 || m_numInstances == 0) {
        return;
    }

    static uint32_t s_dumpCount = 0;
    // Dump first 3 frames with actual geometry, then every 120th
    if (s_dumpCount >= 3 && (s_dumpCount % 120) != 0) {
        s_dumpCount++;
        return;
    }
    s_dumpCount++;

    Logger::warn("=== RTXMG DIAGNOSTIC DUMP START ===");
    Logger::warn(str::format("  numInstances=", m_numInstances, " instanceCapacity=", m_instanceCapacity,
        " maxClusters=", m_maxClusters, " maxVertices=", m_maxVertices));

    // 1. Download cluster offset counts (per-instance offset + count)
    Logger::warn("--- ClusterOffsetCounts (per-instance) ---");
    {
        auto offsetCounts = m_clusterOffsetCountsBuffer.Download(commandList);
        uint32_t numEntries = std::min<uint32_t>((uint32_t)offsetCounts.size(), m_numInstances * uint32_t(ClusterDispatchType::NumTypes));
        for (uint32_t inst = 0; inst < m_numInstances && inst < 10; ++inst) {
            for (uint32_t dt = 0; dt < uint32_t(ClusterDispatchType::NumTypes); ++dt) {
                uint32_t idx = inst * uint32_t(ClusterDispatchType::NumTypes) + dt;
                if (idx < numEntries) {
                    const char* dtName = (dt == 0) ? "PureBSpline" : (dt == 1) ? "RegularBSpline" : (dt == 2) ? "Limit" : (dt == 3) ? "All" : "???";
                    Logger::warn(str::format("  inst[", inst, "] ", dtName, ": offset=", offsetCounts[idx].x, " count=", offsetCounts[idx].y));
                }
            }
        }
        // Summary: check for overlapping ranges
        Logger::warn("  --- Overlap check (All dispatch type) ---");
        uint32_t prevEnd = 0;
        for (uint32_t inst = 0; inst < m_numInstances; ++inst) {
            uint32_t idx = inst * uint32_t(ClusterDispatchType::NumTypes) + uint32_t(ClusterDispatchType::AllTypes);
            if (idx < numEntries) {
                uint32_t offset = offsetCounts[idx].x;
                uint32_t count = offsetCounts[idx].y;
                if (offset != prevEnd && inst > 0) {
                    Logger::warn(str::format("  *** GAP/OVERLAP at inst[", inst, "]: expected offset=", prevEnd, " got=", offset, " (diff=", (int)offset - (int)prevEnd, ")"));
                }
                prevEnd = offset + count;
                if (inst < 10 || offset != (inst > 0 ? offsetCounts[(inst - 1) * uint32_t(ClusterDispatchType::NumTypes) + uint32_t(ClusterDispatchType::AllTypes)].x + offsetCounts[(inst - 1) * uint32_t(ClusterDispatchType::NumTypes) + uint32_t(ClusterDispatchType::AllTypes)].y : 0u)) {
                    Logger::warn(str::format("  inst[", inst, "] All: range=[", offset, "..", offset + count, ") count=", count));
                }
            }
        }
        Logger::warn(str::format("  Total clusters from offsets: ", prevEnd));
    }

    // 2. Download cluster metadata
    Logger::warn("--- Cluster Metadata (first 20 + last 5) ---");
    {
        auto clusters = m_clustersBuffer.Download(commandList);
        uint32_t numClusters = (uint32_t)clusters.size();
        Logger::warn(str::format("  Total cluster entries in buffer: ", numClusters));
        uint32_t logCount = std::min<uint32_t>(numClusters, 20);
        for (uint32_t i = 0; i < logCount; ++i) {
            const auto& c = clusters[i];
            Logger::warn(str::format("  cluster[", i, "] iSurface=", c.iSurface,
                " vtxOff=", c.nVertexOffset, " offset=(", c.offset.x, ",", c.offset.y, ")",
                " size=", c.sizeX, "x", c.sizeY, " vertsPerCluster=", (c.sizeX + 1) * (c.sizeY + 1)));
        }
        // Also log last 5 clusters (to see if there's garbage at the end)
        if (numClusters > 25) {
            Logger::warn("  ... (skipping middle) ...");
            for (uint32_t i = numClusters - 5; i < numClusters; ++i) {
                const auto& c = clusters[i];
                Logger::warn(str::format("  cluster[", i, "] iSurface=", c.iSurface,
                    " vtxOff=", c.nVertexOffset, " offset=(", c.offset.x, ",", c.offset.y, ")",
                    " size=", c.sizeX, "x", c.sizeY));
            }
        }
        // Check for discontinuities in vertex offsets
        Logger::warn("  --- Vertex offset continuity check ---");
        uint32_t checkCount = std::min<uint32_t>(numClusters, 200);
        uint32_t discontinuities = 0;
        for (uint32_t i = 1; i < checkCount; ++i) {
            uint32_t expectedVtxOff = clusters[i - 1].nVertexOffset + (clusters[i - 1].sizeX + 1) * (clusters[i - 1].sizeY + 1);
            if (clusters[i].nVertexOffset != expectedVtxOff) {
                if (discontinuities < 10) {
                    Logger::warn(str::format("  vtxOff discontinuity at cluster[", i, "]: expected=", expectedVtxOff,
                        " got=", clusters[i].nVertexOffset, " prev_surface=", clusters[i - 1].iSurface, " cur_surface=", clusters[i].iSurface));
                }
                discontinuities++;
            }
        }
        if (discontinuities > 0) {
            Logger::warn(str::format("  Total vtxOff discontinuities in first ", checkCount, " clusters: ", discontinuities));
        } else {
            Logger::warn(str::format("  Vertex offsets are CONTIGUOUS for first ", checkCount, " clusters"));
        }
    }

    // 3. Download cluster shading data and compare with cluster metadata
    Logger::warn("--- ClusterShadingData (first 20) ---");
    {
        auto shadingData = accels.clusterShadingDataBuffer.Download(commandList);
        auto clusters = m_clustersBuffer.Download(commandList);
        uint32_t numShading = (uint32_t)shadingData.size();
        uint32_t numClusters = (uint32_t)clusters.size();
        Logger::warn(str::format("  Shading data entries: ", numShading, " cluster entries: ", numClusters));
        uint32_t logCount = std::min<uint32_t>({numShading, numClusters, 20u});
        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < logCount; ++i) {
            const auto& sd = shadingData[i];
            const auto& c = clusters[i];
            bool vtxMismatch = (sd.m_vertexOffset != c.nVertexOffset);
            bool surfMismatch = (sd.m_surfaceId != c.iSurface);
            if (vtxMismatch || surfMismatch) mismatches++;
            Logger::warn(str::format("  shading[", i, "] surfId=", sd.m_surfaceId, " vtxOff=", sd.m_vertexOffset,
                " size=", sd.m_clusterSizeX, "x", sd.m_clusterSizeY,
                " edges=(", sd.m_edgeSegments.x, ",", sd.m_edgeSegments.y, ",", sd.m_edgeSegments.z, ",", sd.m_edgeSegments.w, ")",
                vtxMismatch ? " *** VTX MISMATCH vs cluster ***" : "",
                surfMismatch ? " *** SURF MISMATCH vs cluster ***" : ""));
        }
        // Full mismatch check
        uint32_t fullCheckCount = std::min<uint32_t>(numShading, numClusters);
        uint32_t fullMismatches = 0;
        for (uint32_t i = 0; i < fullCheckCount; ++i) {
            if (shadingData[i].m_vertexOffset != clusters[i].nVertexOffset || shadingData[i].m_surfaceId != clusters[i].iSurface) {
                fullMismatches++;
            }
        }
        Logger::warn(str::format("  Shading vs Cluster mismatches: ", fullMismatches, "/", fullCheckCount));
    }

    // 4. Download vertex positions - check for zeros and NaNs
    // NOTE: GPU StructuredBuffer<float3> uses stride=12 (Slang sizeof(float3)=12),
    // but C++ sizeof(float3)=16 (SIMD padded). Must read raw bytes at stride 12.
    Logger::warn("--- Vertex Positions (sampling) ---");
    {
        struct PackedFloat3 { float x, y, z; }; // 12 bytes, matches GPU stride
        static_assert(sizeof(PackedFloat3) == 12, "PackedFloat3 must be 12 bytes");

        // Download raw bytes from the buffer
        size_t bufferBytes = accels.clusterVertexPositionsBuffer.GetBytes();
        uint32_t numVerts = (uint32_t)(bufferBytes / sizeof(PackedFloat3));

        // Create a staging buffer and download raw bytes
        nvrhi::BufferDesc readbackDesc = accels.clusterVertexPositionsBuffer.GetBuffer()->getDesc();
        readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        readbackDesc.debugName = "vtx pos readback";
        readbackDesc.initialState = nvrhi::ResourceStates::CopyDest;
        readbackDesc.keepInitialState = true;
        readbackDesc.canHaveUAVs = false;
        readbackDesc.canHaveTypedViews = false;
        readbackDesc.canHaveRawViews = false;
        readbackDesc.isAccelStructBuildInput = false;
        auto readbackBuf = commandList->getDevice()->createBuffer(readbackDesc);
        commandList->copyBuffer(readbackBuf.Get(), 0, accels.clusterVertexPositionsBuffer.Get(), 0, bufferBytes);

        // Submit and wait for the copy to complete before mapping
        commandList->close();
        commandList->getDevice()->executeCommandList(commandList);
        commandList->getDevice()->waitForIdle();

        // Map and read as packed float3 (stride 12)
        const PackedFloat3* rawVerts = static_cast<const PackedFloat3*>(commandList->getDevice()->mapBuffer(readbackBuf.Get(), nvrhi::CpuAccessMode::Read));

        Logger::warn(str::format("  Total vertex entries: ", numVerts, " (", bufferBytes, " bytes, stride=12)"));

        // Log first 20 vertices
        uint32_t logCount = std::min<uint32_t>(numVerts, 20);
        for (uint32_t i = 0; i < logCount; ++i) {
            Logger::warn(str::format("  vtx[", i, "] = (", rawVerts[i].x, ", ", rawVerts[i].y, ", ", rawVerts[i].z, ")"));
        }

        // Count zeros, NaNs, and check cluster boundaries
        auto clusters = m_clustersBuffer.Download(commandList);
        uint32_t numClusters = (uint32_t)clusters.size();
        uint32_t checkClusters = std::min<uint32_t>(numClusters, 50);
        uint32_t zeroVertClusters = 0;
        uint32_t nanVertClusters = 0;
        for (uint32_t ci = 0; ci < checkClusters; ++ci) {
            const auto& c = clusters[ci];
            uint32_t numClusterVerts = (c.sizeX + 1) * (c.sizeY + 1);
            uint32_t baseIdx = c.nVertexOffset;
            bool allZero = true;
            bool hasNan = false;
            PackedFloat3 firstVert = {0.f, 0.f, 0.f};
            for (uint32_t vi = 0; vi < numClusterVerts && (baseIdx + vi) < numVerts; ++vi) {
                const auto& v = rawVerts[baseIdx + vi];
                if (v.x != 0.f || v.y != 0.f || v.z != 0.f) allZero = false;
                if (std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z)) hasNan = true;
                if (vi == 0) firstVert = v;
            }
            if (allZero) zeroVertClusters++;
            if (hasNan) nanVertClusters++;
            if (ci < 10 || allZero || hasNan) {
                Logger::warn(str::format("  cluster[", ci, "] vtxOff=", baseIdx, " nVerts=", numClusterVerts,
                    " firstVtx=(", firstVert.x, ",", firstVert.y, ",", firstVert.z, ")",
                    allZero ? " *** ALL ZEROS ***" : "",
                    hasNan ? " *** HAS NaN ***" : ""));
            }
        }
        Logger::warn(str::format("  Clusters with ALL-ZERO vertices: ", zeroVertClusters, "/", checkClusters));
        Logger::warn(str::format("  Clusters with NaN vertices: ", nanVertClusters, "/", checkClusters));

        // Count total zero vertices in used range
        uint32_t totalZeros = 0;
        uint32_t totalUsed = 0;
        if (!clusters.empty()) {
            uint32_t lastCluster = std::min<uint32_t>((uint32_t)clusters.size(), checkClusters) - 1;
            uint32_t lastVtx = clusters[lastCluster].nVertexOffset + (clusters[lastCluster].sizeX + 1) * (clusters[lastCluster].sizeY + 1);
            totalUsed = std::min<uint32_t>(lastVtx, numVerts);
            for (uint32_t i = 0; i < totalUsed; ++i) {
                if (rawVerts[i].x == 0.f && rawVerts[i].y == 0.f && rawVerts[i].z == 0.f) totalZeros++;
            }
            Logger::warn(str::format("  Zero vertices in used range [0..", totalUsed, "): ", totalZeros, " (",
                totalUsed > 0 ? (totalZeros * 100 / totalUsed) : 0, "%)"));
        }

        commandList->getDevice()->unmapBuffer(readbackBuf.Get());

        // Reopen command list for subsequent commands
        commandList->open();
    }

    // 5. Download CLAS indirect args
    Logger::warn("--- CLAS Indirect Args (first 10) ---");
    {
        auto args = m_clasIndirectArgDataBuffer.Download(commandList);
        uint32_t numArgs = (uint32_t)args.size();
        Logger::warn(str::format("  Total CLAS indirect arg entries: ", numArgs));
        nvrhi::GpuVirtualAddress vtxBaseAddr = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
        Logger::warn(str::format("  Vertex positions base addr: 0x", std::hex, vtxBaseAddr, std::dec));
        uint32_t logCount = std::min<uint32_t>(numArgs, 10);
        for (uint32_t i = 0; i < logCount; ++i) {
            const auto& a = args[i];
            nvrhi::GpuVirtualAddress vtxAddr = a.vertexBuffer.startAddress;
            uint32_t stride = a.vertexBuffer.strideInBytes;
            uint32_t geomIdx = a.geometryIndexOffsetPacked & 0xFFFFFF;
            int64_t vtxOffset = (vtxAddr >= vtxBaseAddr) ? (int64_t)(vtxAddr - vtxBaseAddr) / 12 : -(int64_t)(vtxBaseAddr - vtxAddr) / 12;
            Logger::warn(str::format("  clasArg[", i, "] clusterIdOff=", a.clusterIdOffset,
                " geomIdx=", geomIdx, " template=0x", std::hex, a.clusterTemplate, std::dec,
                " vtxAddr=0x", std::hex, vtxAddr, std::dec,
                " stride=", stride,
                " vtxOffset=", vtxOffset,
                (stride != 12) ? " *** WRONG STRIDE ***" : ""));
        }
    }

    // 6. Download dispatch indirect args
    Logger::warn("--- Fill Clusters Dispatch Indirect Args ---");
    {
        auto dispatchArgs = m_fillClustersDispatchIndirectBuffer.Download(commandList);
        uint32_t numArgs = (uint32_t)dispatchArgs.size();
        uint32_t logCount = std::min<uint32_t>(numArgs, m_numInstances * uint32_t(ClusterDispatchType::NumTypes));
        for (uint32_t inst = 0; inst < m_numInstances && inst < 10; ++inst) {
            for (uint32_t dt = 0; dt < uint32_t(ClusterDispatchType::NumTypes); ++dt) {
                uint32_t idx = inst * uint32_t(ClusterDispatchType::NumTypes) + dt;
                if (idx < logCount) {
                    const char* dtName = (dt == 0) ? "PureBSpline" : (dt == 1) ? "RegularBSpline" : (dt == 2) ? "Limit" : (dt == 3) ? "All" : "???";
                    Logger::warn(str::format("  inst[", inst, "] ", dtName, ": groups=(", dispatchArgs[idx].x, ",", dispatchArgs[idx].y, ",", dispatchArgs[idx].z, ")"));
                }
            }
        }
    }

    Logger::warn("=== RTXMG DIAGNOSTIC DUMP END ===");
}


