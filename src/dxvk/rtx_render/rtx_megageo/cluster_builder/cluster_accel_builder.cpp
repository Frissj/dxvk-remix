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

// Global debug flag shared between BuildAccel (clear) and FillInstanceClusters (readback)
bool g_megageoDbgGotData = false;

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

    m_computeClusterTilingParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ComputeClusterTilingParams), "ComputeClusterTilingParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillClustersParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillClustersParams), "FillClustersParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillBlasFromClasArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillBlasFromClasArgsParams), "FillBlasFromClasArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    // CopyClusterOffsetParams is always 256 bytes regardless of numInstances.
    // Create it once here and reuse forever - no need to destroy/recreate on reallocation.
    {
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = 256; // CopyClusterOffsetParams is 16 bytes, align to 256 for constant buffer
        cbDesc.debugName = "CopyClusterOffsetParams";
        cbDesc.isConstantBuffer = true;
        cbDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        cbDesc.keepInitialState = true;
        m_copyClusterOffsetParamsBuffer = m_device->createBuffer(cbDesc);
    }

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
    // BuildStructuredCLASes uses m_maxClusters-1 because we store clusterIndex in
    // geometryIndexOffsetPacked for hit shader lookup of per-cluster shading data.
    uint32_t maxGeometryIndex = m_maxClusters > 0 ? m_maxClusters - 1 : 0;

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

    // Calculate total size with 128-byte alignment padding for CLAS requirements
    // Each template must be 128-byte aligned, and we need padding for base address alignment too
    size_t totalTemplateSize = 0;
    size_t totalAlignedTemplateSize = 0;
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        totalTemplateSize += templateSizes[i];
        // Round up each template to 128-byte alignment
        totalAlignedTemplateSize += (templateSizes[i] + cluster::kClasByteAlignment - 1) & ~(cluster::kClasByteAlignment - 1);
    }
    // Add extra padding for potential base address alignment (up to 127 bytes)
    size_t bufferSizeWithPadding = totalAlignedTemplateSize + cluster::kClasByteAlignment;

    RTXMG_LOG(str::format("RTX MegaGeo: Template sizes - raw total=", totalTemplateSize,
        " aligned total=", totalAlignedTemplateSize, " buffer size with padding=", bufferSizeWithPadding));

    // Create template data buffer based off of totalSize of all templates with alignment padding
    nvrhi::BufferDesc destDataDesc = {
        .byteSize = bufferSizeWithPadding,
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

    std::vector<nvrhi::GpuVirtualAddress> addresses(kNumTemplates);
    totalTemplateSize = 0;

    // CRITICAL FIX: Align base address up to 128-byte boundary for CLAS requirements
    // The buffer itself may not be 128-aligned, so we need to account for this
    nvrhi::GpuVirtualAddress alignedBaseAddress = (baseAddress + (cluster::kClasByteAlignment - 1)) & ~(nvrhi::GpuVirtualAddress(cluster::kClasByteAlignment - 1));
    size_t baseAlignmentPadding = alignedBaseAddress - baseAddress;

    RTXMG_LOG(str::format("RTX MegaGeo: Template alignment: baseAddr=", std::hex, baseAddress,
        " alignedBase=", alignedBaseAddress, " padding=", std::dec, baseAlignmentPadding, " bytes"));

    for (size_t i = 0; i < addresses.size(); i++)
    {
        // Each template address must be 128-byte aligned
        addresses[i] = alignedBaseAddress + totalTemplateSize;
        // Round up template size to 128-byte alignment for next template
        totalTemplateSize += (templateSizes[i] + cluster::kClasByteAlignment - 1) & ~(cluster::kClasByteAlignment - 1);
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

    cluster::OperationParams instantiateClasParams =
    {
        .maxArgCount = m_maxClusters,
        .type = cluster::OperationType::ClasInstantiateTemplates,
        .mode = cluster::OperationMode::ExplicitDestinations,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            // The compute_cluster_tiling shader writes clusterIndex into
            // geometryIndexOffsetPacked (not localGeometryIndex like the sample),
            // so maxGeometryIndex must cover the full cluster index range.
            .maxGeometryIndex = m_maxClusters > 0 ? m_maxClusters - 1 : 0,
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

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    // Download and log CLAS indirect args before building (GPU readback - only when logging enabled)
    {
        auto clasIndirectArgs = m_clasIndirectArgDataBuffer.Download(commandList);
        uint32_t numToLog = std::min(uint32_t(clasIndirectArgs.size()), 10u);
        RTXMG_LOG(str::format("RTX MegaGeo: CLAS indirect args before build (first ", numToLog, " of ", clasIndirectArgs.size(), "):"));
        for (uint32_t i = 0; i < numToLog; ++i) {
            const auto& arg = clasIndirectArgs[i];
            bool templateAligned = (arg.clusterTemplate % 128 == 0);
            bool vertexAligned = (arg.vertexBuffer.startAddress % 16 == 0);
            RTXMG_LOG(str::format("  [", i, "] clusterIdOff=", arg.clusterIdOffset,
                " geomIdxPacked=", arg.geometryIndexOffsetPacked,
                " template=0x", std::hex, arg.clusterTemplate, " (128-aligned=", templateAligned ? "YES" : "NO", ")",
                " vertexAddr=0x", arg.vertexBuffer.startAddress, " (16-aligned=", vertexAligned ? "YES" : "NO", ")",
                std::dec, " stride=", arg.vertexBuffer.strideInBytes));
        }
    }
#endif

    RTXMG_LOG("RTX MegaGeo: BuildStructuredCLASes - calling executeMultiIndirectClusterOperation");
    commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);

#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
    // Download and log the first few CLAS pointers to diagnose misaligned address errors (GPU readback - only when logging enabled)
    {
        auto clasPtrs = accels.clasPtrsBuffer.Download(commandList);
        uint32_t numToLog = std::min(uint32_t(clasPtrs.size()), 10u);
        RTXMG_LOG(str::format("RTX MegaGeo: CLAS pointers after build (first ", numToLog, " of ", clasPtrs.size(), "):"));
        for (uint32_t i = 0; i < numToLog; ++i) {
            bool aligned128 = (clasPtrs[i] % 128 == 0);
            RTXMG_LOG(str::format("  [", i, "] 0x", std::hex, clasPtrs[i], std::dec,
                " aligned(128)=", aligned128 ? "YES" : "NO"));
        }
    }
#endif

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
                Logger::warn(str::format("CHECKPOINT: Instance 0 mesh: meshID=", instance.meshID,
                    " posBufBytes=", posBufBytes, " surfaceCount=", surfaceCount,
                    " AABB min=(", aabb.m_mins[0], ",", aabb.m_mins[1], ",", aabb.m_mins[2],
                    ") max=(", aabb.m_maxs[0], ",", aabb.m_maxs[1], ",", aabb.m_maxs[2], ")"));
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
        params.disableSubdivision = m_tessellatorConfig.disableSubdivision ? 1 : 0;
        params.firstGeometryIndex = firstGeometryIndex;
        params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
        params.debugClusterIndex = uint32_t(m_tessellatorConfig.debugClusterIndex);
        params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
        commandList->writeBuffer(m_fillClustersParamsBuffer, &params, sizeof(FillClustersParams));

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
                    Logger::warn(str::format("CHECKPOINT: debug readback (attempt=", s_fillDbgAttempts, ")"));

                    // Slots 0-7: compute_cluster_tiling debug (payloadType=103)
                    // Slot 0: entry point marker (lineNumber=1, uintData.x=999)
                    // Slots 1-7: per-cluster data (GPU addresses for CLAS)
                    Logger::warn("  === compute_cluster_tiling debug ===");
                    for (uint32_t i = 0; i < 8 && i < debugOutput.size(); ++i) {
                        const auto& e = debugOutput[i];
                        if (e.payloadType == 103) {
                            if (e.lineNumber == 1) {
                                // Entry point marker
                                Logger::warn(str::format("  tiling[", i, "] ENTRY: surfRange=[", e.uintData.y,
                                    ",", e.uintData.z, ") instanceIndex=", e.uintData.w));
                            } else {
                                // Per-cluster: iSurface, vtxOff, vtxAddr(lo,hi), clusterIdx, templateAddr(lo,hi), sizeof(float3)
                                uint64_t vtxAddr = uint64_t(e.uintData.z) | (uint64_t(e.uintData.w) << 32);
                                uint32_t clusterIdx, templateLo, templateHi, sizeofF3;
                                memcpy(&clusterIdx, &e.floatData.x, 4);
                                memcpy(&templateLo, &e.floatData.y, 4);
                                memcpy(&templateHi, &e.floatData.z, 4);
                                memcpy(&sizeofF3, &e.floatData.w, 4);
                                uint64_t templateAddr = uint64_t(templateLo) | (uint64_t(templateHi) << 32);
                                Logger::warn(str::format("  tiling[", i, "] iSurface=", e.uintData.x,
                                    " vtxOff=", e.uintData.y, " vtxAddr=0x", std::hex, vtxAddr, std::dec,
                                    " clusterIdx=", clusterIdx, " templateAddr=0x", std::hex, templateAddr, std::dec,
                                    " sizeof(float3)=", sizeofF3));
                            }
                        } else if (e.payloadType != 0) {
                            Logger::warn(str::format("  tiling[", i, "] payloadType=", e.payloadType));
                        } else {
                            Logger::warn(str::format("  tiling[", i, "] EMPTY"));
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
                } else if (s_fillDbgAttempts % 50 == 0) {
                    Logger::warn(str::format("CHECKPOINT: debug readback attempt ", s_fillDbgAttempts, " - still empty"));
                }
            }
        }

#if RTXMG_CHRONO_TIMING
        auto instEnd = std::chrono::high_resolution_clock::now();
        dispatchTimeMs += std::chrono::duration_cast<std::chrono::microseconds>(instEnd - afterBinding).count() * 0.001f;
#endif
    }

    stats::clusterAccelSamplers.fillClustersTime.Stop();
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

    // Convert dxvk matrices to float4x4
    RTXMG_LOG("RTX MegaGeo: params - getting projection matrix");
    auto projMatrix = m_tessellatorConfig.camera->GetProjectionMatrix();
    RTXMG_LOG("RTX MegaGeo: params - getting view matrix");
    auto viewMatrix = m_tessellatorConfig.camera->GetViewMatrix();
    RTXMG_LOG("RTX MegaGeo: params - multiplying matrices");
    auto viewProj = projMatrix * viewMatrix;
    RTXMG_LOG("RTX MegaGeo: params - copying matWorldToClip");
    memcpy(&params.matWorldToClip, &viewProj.data[0][0], sizeof(float) * 16);

    // Log viewProj matrix values
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row0=(", viewProj.data[0][0], ",", viewProj.data[0][1], ",", viewProj.data[0][2], ",", viewProj.data[0][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row1=(", viewProj.data[1][0], ",", viewProj.data[1][1], ",", viewProj.data[1][2], ",", viewProj.data[1][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row2=(", viewProj.data[2][0], ",", viewProj.data[2][1], ",", viewProj.data[2][2], ",", viewProj.data[2][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row3=(", viewProj.data[3][0], ",", viewProj.data[3][1], ",", viewProj.data[3][2], ",", viewProj.data[3][3], ")"));

    // DEBUG: Test project a sample point at (5, 5, 10) to see screen coordinates
#if RTXMG_VERBOSE_LOGGING
    {
        float testX = 5.0f, testY = 5.0f, testZ = 10.0f;
        float clipX = testX * viewProj.data[0][0] + testY * viewProj.data[1][0] + testZ * viewProj.data[2][0] + viewProj.data[3][0];
        float clipY = testX * viewProj.data[0][1] + testY * viewProj.data[1][1] + testZ * viewProj.data[2][1] + viewProj.data[3][1];
        float clipZ = testX * viewProj.data[0][2] + testY * viewProj.data[1][2] + testZ * viewProj.data[2][2] + viewProj.data[3][2];
        float clipW = testX * viewProj.data[0][3] + testY * viewProj.data[1][3] + testZ * viewProj.data[2][3] + viewProj.data[3][3];
        float ndcX = clipX / clipW;
        float ndcY = clipY / clipW;
        float screenX = (ndcX * 0.5f + 0.5f) * m_tessellatorConfig.viewportSize.x;
        float screenY = (ndcY * 0.5f + 0.5f) * m_tessellatorConfig.viewportSize.y;
        RTXMG_LOG(str::format("RTX MegaGeo: TEST POINT (5,5,10) -> clip=(", clipX, ",", clipY, ",", clipZ, ",", clipW,
            ") ndc=(", ndcX, ",", ndcY, ") screen=(", screenX, ",", screenY, ")"));
    }
#endif

    RTXMG_LOG("RTX MegaGeo: params - copying localToWorld");
    // Store in row-major format for use with dot products in shader
    // Each row is (linear_row + translation_component): dot(row, (x,y,z,1)) = linear*pos + translation
    // Matrix4::data[i] is row i, data[i][j] is row i, column j
    // Translation is in column 3 (data[i][3]), NOT row 3 (data[3][i])
    params.localToWorld[0] = float4(localToWorld.data[0][0], localToWorld.data[0][1], localToWorld.data[0][2], localToWorld.data[0][3]);
    params.localToWorld[1] = float4(localToWorld.data[1][0], localToWorld.data[1][1], localToWorld.data[1][2], localToWorld.data[1][3]);
    params.localToWorld[2] = float4(localToWorld.data[2][0], localToWorld.data[2][1], localToWorld.data[2][2], localToWorld.data[2][3]);

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
    // Using the fast AABB transform algorithm from donut's box3::operator*
    {
        auto& aabb = subdivisionSurface.m_aabb;

        // Start with translation (DXVK Matrix4 column-vector convention: translation in column 3)
        float4 translation = float4(localToWorld.data[0][3], localToWorld.data[1][3], localToWorld.data[2][3], 0.0f);
        params.aabb.m_min = translation;
        params.aabb.m_max = translation;

        // Apply the linear transform to bounds using the standard AABB transform algorithm
        // For each output axis i, accumulate contributions from all input axes j
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float m = localToWorld.data[i][j];  // M[i][j] = row i, col j
                float minVal = aabb.m_mins[j];
                float maxVal = aabb.m_maxs[j];

                float e = m * minVal;
                float f = m * maxVal;

                if (i == 0) {
                    params.aabb.m_min.x += std::min(e, f);
                    params.aabb.m_max.x += std::max(e, f);
                } else if (i == 1) {
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

    params.enableBackfaceVisibility = m_tessellatorConfig.enableBackfaceVisibility && !m_tessellatorConfig.disableSubdivision;
    params.enableFrustumVisibility = m_tessellatorConfig.enableFrustumVisibility && !m_tessellatorConfig.disableSubdivision;
    // Disable HiZ when subdivision is disabled to avoid descriptor binding issues with NoLimit surfaces
    params.enableHiZVisibility = m_tessellatorConfig.enableHiZVisibility && m_tessellatorConfig.zbuffer != nullptr && !m_tessellatorConfig.disableSubdivision;
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
    params.disableSubdivision = m_tessellatorConfig.disableSubdivision ? 1 : 0;
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
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - main bindingSetDesc built");

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_computeClusterTilingBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create main binding set for compute_cluster_tiling.hlsl");
    }
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - main binding set created");

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

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating compute state");
    auto state = nvrhi::ComputeState()
        .addBindingSet(bindingSet)           // Set 0: Main bindings
        .addBindingSet(hizBindingSet)        // Set 1: HiZ textures
        .addBindingSet(m_descriptorTable);   // Set 2: Bindless textures
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - enableMonolithicClusterBuild=", m_tessellatorConfig.enableMonolithicClusterBuild));

    if (m_tessellatorConfig.enableMonolithicClusterBuild)
    {
        // When subdivision is disabled, process ALL surfaces using bilinear interpolation
        // When enabled, skip NoLimit surfaces as they don't have valid limit data
        params.surfaceStart = 0;
        params.surfaceEnd = m_tessellatorConfig.disableSubdivision
            ? subdivisionSurface.m_surfaceCount
            : subdivisionSurface.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
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

        RTXMG_LOG("RTX MegaGeo: Monolithic - writeBuffer");
        commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));
        ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType::All;
        shaderPermutation.setSurfaceType(shaderSurfaceType);
        RTXMG_LOG("RTX MegaGeo: Monolithic - GetComputeClusterTilingPSO");
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
                RTXMG_LOG("RTX MegaGeo: Loop - writeBuffer");
                commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));

                ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType(i);
                shaderPermutation.setSurfaceType(shaderSurfaceType);
                RTXMG_LOG("RTX MegaGeo: Loop - GetComputeClusterTilingPSO");
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
    Logger::warn("CHECKPOINT: BuildBlasFromClas pre-indirectArgs-barrier");
    commandList->bufferBarrier(m_blasFromClasIndirectArgsBuffer,
        nvrhi::ResourceStates::UnorderedAccess,
        nvrhi::ResourceStates::ShaderResource);
    Logger::warn("CHECKPOINT: BuildBlasFromClas post-indirectArgs-barrier");

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
        // m_copyClusterOffsetParamsBuffer is created once at init (always 256 bytes, no resizing needed)
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
    Logger::warn(str::format("CHECKPOINT: BuildAccel pre-UpdateMemoryAllocations inst=", instances.size(), " patches=", totalSubdPatches));
    UpdateMemoryAllocations(accels, uint32_t(instances.size()), totalSubdPatches);
#if RTXMG_CHRONO_TIMING
    auto afterMemAlloc = std::chrono::high_resolution_clock::now();
    float memAllocMs = std::chrono::duration_cast<std::chrono::microseconds>(afterMemAlloc - setupStart).count() * 0.001f;
    if (memAllocMs > 1.0f) {
        RTXMG_LOG(str::format(">>> RTXMG CHRONO: UpdateMemoryAllocations=", memAllocMs, "ms (SLOW - likely waitForIdle)"));
    }
#endif
    Logger::warn("CHECKPOINT: BuildAccel post-UpdateMemoryAllocations");

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
    Logger::warn("CHECKPOINT: BuildAccel post-clears");

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
        Logger::warn(str::format("CHECKPOINT: BuildAccel pre-tiling-loop maxInstances=", maxInstances));
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
        Logger::warn("CHECKPOINT: BuildAccel post-tiling-loop");
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

    // NOTE: enableLogging block removed - Log()/Download() calls close/reopen command list
    // which destroys bound image views and causes VK_ERROR_DEVICE_LOST in DXVK.

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

    Logger::warn("CHECKPOINT: BuildAccel pre-FillInstanceClusters");
    FillInstanceClusters(scene, accels, commandList);
    Logger::warn("CHECKPOINT: BuildAccel post-FillInstanceClusters");

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
    // Using AccelStructBuildInput state which maps to:
    // - VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    // - VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
    RTXMG_LOG("RTX MegaGeo: BuildAccel - adding UAV barrier for vertex positions (UAV -> AccelStructBuildInput)");
    commandList->bufferBarrier(accels.clusterVertexPositionsBuffer, nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::AccelStructBuildInput);

    // Build CLASes for all instances at once
    Logger::warn("CHECKPOINT: BuildAccel pre-BuildStructuredCLASes");
    stats::clusterAccelSamplers.buildClasTime.Start(commandList);
    BuildStructuredCLASes(accels, maxGeometryCountPerMesh, tessCounterRange, commandList);
    stats::clusterAccelSamplers.buildClasTime.Stop();
    Logger::warn("CHECKPOINT: BuildAccel post-BuildStructuredCLASes");
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
    Logger::warn("CHECKPOINT: BuildAccel pre-CLAS-barrier");
    commandList->bufferBarrier(accels.clasPtrsBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::AccelStructBuildInput);
    Logger::warn("CHECKPOINT: BuildAccel post-CLAS-barrier");

    // Build BLAS unconditionally (matching sample behavior)
    // NOTE: Removed sync Download check for clusters > 0 because Download closes/reopens
    // command list which destroys bound image views in DXVK
    Logger::warn(str::format("CHECKPOINT: BuildAccel pre-BuildBlasFromClas count=", std::min(uint32_t(instances.size()), m_numInstances)));
    uint32_t blasInstanceCount = std::min(uint32_t(instances.size()), m_numInstances);
    BuildBlasFromClas(accels, instances.data(), blasInstanceCount, commandList);
    Logger::warn("CHECKPOINT: BuildAccel post-BuildBlasFromClas");
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
    Logger::warn("CHECKPOINT: BuildAccel pre-Download-counters");
    auto counterBufferData = m_tessellationCountersBuffer.Download(commandList, true);
    Logger::warn("CHECKPOINT: BuildAccel post-Download-counters");
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


