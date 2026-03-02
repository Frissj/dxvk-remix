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

// STL includes
#include <algorithm>
#include <limits>
#include <chrono>
#include <cmath>

// RTX Remix includes
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"
#include "../nvrhi_adapter/nvrhi_types.h"
#include "../nvrhi_adapter/nvrhi_dxvk_device.h"
#include "../nvrhi_adapter/nvrhi_dxvk_command_list.h"
#include "../nvrhi_adapter/nvrhi_dxvk_buffer.h"
#include "../../rtx_context.h"

// RTX MG includes
#include "cluster_accels.h"
#include "cluster_accel_builder.h"
#include "copy_cluster_offset_params.h"
#include "tessellation_counters.h"
#include "fill_meshlet_clusters_params.h"

using namespace dxvk;

#include "../utils/buffer.h"
#include "../profiler/profiler_stub.h"

#include "../rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_CLUSTER_ACCEL_BUILDER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

using namespace donut;
using namespace nvrhi::rt;

// Verify ClusterShadingData matches expected GPU layout (56 bytes with scalar layout)
// MathLib float2 = 8 bytes (__m64), so struct should not have extra padding
static_assert(sizeof(ClusterShadingData) == 56, "ClusterShadingData size mismatch! Expected 56 bytes for GPU compatibility");

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
}

void ClusterAccelBuilder::UpdateMemoryAllocationsMeshlet(
    ClusterAccels& accels, uint32_t numInstances, uint32_t maxClusters, uint32_t maxVertices, size_t maxClasBytes)
{
    // Simplified memory allocation for the meshlet path.
    // Similar to UpdateMemoryAllocations but without grid-specific buffers.

    m_numInstances = numInstances;
    bool instanceBuffersNeedResize = (numInstances > m_instanceCapacity);
    if (instanceBuffersNeedResize) {
        m_instanceCapacity = numInstances + kInstanceGrowHeadroom;
    }

    bool clustersChanged = (m_maxClusters != maxClusters);
    bool verticesChanged = (m_maxVertices != maxVertices);
    bool clasBytesChanged = (m_maxClasBytes != maxClasBytes);

    m_maxClusters = maxClusters;
    m_maxVertices = maxVertices;
    m_maxClasBytes = maxClasBytes;

    if (!instanceBuffersNeedResize && !clustersChanged && !verticesChanged && !clasBytesChanged)
        return;

    Logger::warn(str::format("RTX MegaGeo MESHLET: UpdateMemoryAllocationsMeshlet - "
        "instances=", numInstances, " cap=", m_instanceCapacity,
        " clusters=", maxClusters, " vertices=", maxVertices,
        " clasBytes=", maxClasBytes));

    // Deferred release of old buffers
    DeferredBufferRelease deferred;
    deferred.frameIndex = m_currentFrameIndex;
    auto deferBuffer = [&deferred](auto& rtxmgBuffer) {
        if (auto buf = rtxmgBuffer.GetBuffer()) {
            deferred.buffers.push_back(buf);
        }
        rtxmgBuffer.Release();
    };

    if (instanceBuffersNeedResize) {
        deferBuffer(m_clusterOffsetCountsBuffer);
        deferBuffer(m_fillClustersDispatchIndirectBuffer);
        deferBuffer(m_blasFromClasIndirectArgsBuffer);
        deferBuffer(accels.blasPtrsBuffer);
        deferBuffer(accels.blasSizesBuffer);
    }

    if (clustersChanged) {
        deferBuffer(m_clustersBuffer);
        deferBuffer(m_clasIndirectArgDataBuffer);
        deferBuffer(accels.clusterShadingDataBuffer);
        deferBuffer(accels.clasPtrsBuffer);
    }

    if (clustersChanged || instanceBuffersNeedResize) {
        deferBuffer(accels.blasBuffer);
    }

    if (clasBytesChanged) {
        deferBuffer(accels.clasBuffer);
    }

    if (verticesChanged) {
        deferBuffer(accels.clusterVertexPositionsBuffer);
        deferBuffer(accels.clusterVertexNormalsBuffer);
    }

    if (!deferred.buffers.empty()) {
        m_deferredReleases.push_back(std::move(deferred));
    }

    // Create new buffers
    if (instanceBuffersNeedResize) {
        m_clusterOffsetCountsBuffer.Create(m_instanceCapacity * ClusterDispatchType::NumTypes, "ClusterOffsets", m_device.Get());

        nvrhi::BufferDesc dispatchIndirectDesc = {
            .byteSize = m_instanceCapacity * ClusterDispatchType::NumTypes * sizeof(uint3),
            .debugName = "FillClustersIndirectArgs",
            .structStride = uint32_t(sizeof(uint3)),
            .canHaveUAVs = true,
            .isDrawIndirectArgs = true,
            .initialState = nvrhi::ResourceStates::IndirectArgument,
            .keepInitialState = true,
        };
        m_fillClustersDispatchIndirectBuffer.Create(dispatchIndirectDesc, m_device.Get());

        uint32_t indirectArgAlignedStride = (sizeof(cluster::IndirectArgs) + 15) & ~15;
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

    if (clustersChanged) {
        m_clustersBuffer.Create(m_maxClusters, "clusters", m_device.Get());
        m_clasIndirectArgDataBuffer.Create(m_maxClusters, "indirect arg data", m_device.Get());
        accels.clusterShadingDataBuffer.Create(m_maxClusters, "cluster shading data", m_device.Get());
        accels.clasPtrsBuffer.Create(m_maxClusters, "ClasAddresses", m_device.Get());
    }

    if (clustersChanged || instanceBuffersNeedResize) {
        m_createBlasParams = {
            .maxArgCount = m_instanceCapacity,
            .type = cluster::OperationType::BlasBuild,
            .mode = cluster::OperationMode::ImplicitDestinations,
            .flags = cluster::OperationFlags::None,
            .blas = {
                .maxClasPerBlasCount = m_maxClusters,
                .maxTotalClasCount = m_maxClusters
            }
        };
        m_createBlasSizeInfo = m_device->getClusterOperationSizeInfo(m_createBlasParams);

        if (m_createBlasSizeInfo.resultMaxSizeInBytes > 0) {
            nvrhi::BufferDesc blasBufferDesc = {
                .byteSize = m_createBlasSizeInfo.resultMaxSizeInBytes,
                .debugName = "Blas Data",
                .canHaveUAVs = true,
                .isAccelStructStorage = true,
                .initialState = nvrhi::ResourceStates::AccelStructWrite,
                .keepInitialState = true,
            };
            accels.blasBuffer.Create(blasBufferDesc, m_device.Get());
        }
    }

    if (clasBytesChanged) {
        nvrhi::BufferDesc clasDataDesc = {
            .byteSize = m_maxClasBytes,
            .debugName = "ClasData",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        accels.clasBuffer.Create(clasDataDesc, m_device.Get());
    }

    if (verticesChanged) {
        static constexpr uint32_t kGpuFloat3Stride = 3 * sizeof(float);
        size_t byteSize = m_maxVertices * kGpuFloat3Stride;
        size_t alignedByteSize = (byteSize + 3) & ~3;
        nvrhi::BufferDesc vertexPosDesc = {
            .byteSize = alignedByteSize,
            .debugName = "cluster vertex positions",
            .structStride = kGpuFloat3Stride,
            .canHaveUAVs = true,
            .canHaveTypedViews = true,
            .canHaveRawViews = true,
            .isAccelStructBuildInput = true,
            .initialState = nvrhi::ResourceStates::UnorderedAccess,
            .keepInitialState = true
        };
        accels.clusterVertexPositionsBuffer.Create(vertexPosDesc, m_device.Get());

        // Normals buffer must use 12-byte stride (GPU float3) not sizeof(MathLib float3) = 16
        nvrhi::BufferDesc vertexNormDesc = {
            .byteSize = alignedByteSize,
            .debugName = "cluster vertex normals",
            .structStride = kGpuFloat3Stride,
            .canHaveUAVs = true,
            .canHaveTypedViews = true,
            .canHaveRawViews = true,
            .initialState = nvrhi::ResourceStates::UnorderedAccess,
            .keepInitialState = true
        };
        accels.clusterVertexNormalsBuffer.Create(vertexNormDesc, m_device.Get());
    }
}

void ClusterAccelBuilder::BuildAccelMeshlet(
    const std::vector<MeshletInstance>& instances,
    const float cameraPos[3],
    float cameraProj,
    float cameraNear,
    float errorThreshold,
    ClusterAccels& accels,
    ClusterStatistics& stats,
    uint32_t frameIndex,
    nvrhi::ICommandList* commandList)
{
    m_currentFrameIndex = frameIndex;

    // Release deferred buffers
    while (!m_deferredReleases.empty() &&
           m_currentFrameIndex >= m_deferredReleases.front().frameIndex + kDeferredReleaseFrames) {
        m_deferredReleases.pop_front();
    }

    if (instances.empty())
        return;

    uint32_t numInstances = static_cast<uint32_t>(instances.size());

    // =====================================================================
    // CPU LOD selection: select clusters for each instance
    // =====================================================================
    struct SelectedCluster {
        uint32_t instanceIdx;
        uint32_t meshletIdx;
    };
    std::vector<SelectedCluster> allSelectedClusters;
    std::vector<uint32_t> perInstanceClusterCount(numInstances, 0);
    std::vector<uint32_t> perInstanceClusterOffset(numInstances, 0);

    for (uint32_t inst = 0; inst < numInstances; ++inst) {
        const MeshletInstance& mi = instances[inst];
        if (!mi.lodData || !mi.templates || !mi.templates->isBuilt)
            continue;

        std::vector<uint32_t> selected = ClusterLODBuilder::selectClusters(
            *mi.lodData, cameraPos, cameraProj, cameraNear, errorThreshold);

        perInstanceClusterOffset[inst] = static_cast<uint32_t>(allSelectedClusters.size());
        perInstanceClusterCount[inst] = static_cast<uint32_t>(selected.size());

        for (uint32_t meshletIdx : selected) {
            allSelectedClusters.push_back({inst, meshletIdx});
        }
    }

    uint32_t totalClusters = static_cast<uint32_t>(allSelectedClusters.size());

    if (totalClusters == 0)
        return;

    // DIAG: Log LOD selection results
    Logger::info(str::format("RTX MegaGeo MESHLET: === BuildAccelMeshlet frame=", frameIndex,
        " instances=", numInstances, " totalClusters=", totalClusters, " ==="));
    for (uint32_t inst = 0; inst < numInstances; ++inst) {
        Logger::info(str::format("RTX MegaGeo MESHLET:   inst[", inst, "] clusters=", perInstanceClusterCount[inst],
            " offset=", perInstanceClusterOffset[inst],
            " surfaceId=", instances[inst].surfaceId));
    }

    // =====================================================================
    // CPU prefix sums: compute per-frame offsets for each selected cluster
    // This is trivial O(N) work, stays on CPU since it depends on LOD selection
    // =====================================================================
    static constexpr uint32_t kGpuFloat3Stride = 12;

    uint32_t totalExpandedVtx = 0;
    uint32_t totalUniqueVtx = 0;
    uint32_t totalTriangles = 0;

    // Build compact GPU info array + compute prefix sums
    std::vector<MeshletClusterGPUInfo> clusterInfos(totalClusters);
    nvrhi::GpuVirtualAddress clasBaseAddr = 0; // set after allocation
    uint64_t clasOffset = 0;

    // First pass: compute totals for memory allocation
    size_t totalClasBytes = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        totalTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
        totalUniqueVtx += mi.templates->meshletVertexCounts[sc.meshletIdx];
        totalExpandedVtx += mi.templates->meshletTriangleCounts[sc.meshletIdx] * 3;
        uint32_t instSize = mi.templates->instantiationSizes[sc.meshletIdx];
        totalClasBytes += ((instSize + cluster::kClasByteAlignment - 1) / cluster::kClasByteAlignment) * cluster::kClasByteAlignment;
    }
    totalClasBytes = std::max(totalClasBytes, (size_t)cluster::kClasByteAlignment);

    uint32_t maxClusters = std::max(totalClusters, 1u);
    uint32_t maxVertices = std::max(totalExpandedVtx + totalUniqueVtx, 1u);

    uint32_t maxTriPerMeshlet = 0;
    uint32_t maxVtxPerMeshlet = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        maxTriPerMeshlet = std::max(maxTriPerMeshlet, mi.templates->maxTrianglesPerMeshlet);
        maxVtxPerMeshlet = std::max(maxVtxPerMeshlet, mi.templates->maxVerticesPerMeshlet);
    }
    if (maxTriPerMeshlet == 0) maxTriPerMeshlet = 128;
    if (maxVtxPerMeshlet == 0) maxVtxPerMeshlet = 128;

    // DIAG: Log allocation sizes
    Logger::info(str::format("RTX MegaGeo MESHLET: totalExpandedVtx=", totalExpandedVtx,
        " totalUniqueVtx=", totalUniqueVtx, " totalTriangles=", totalTriangles,
        " totalClasBytes=", totalClasBytes, " maxTriPerMeshlet=", maxTriPerMeshlet,
        " maxVtxPerMeshlet=", maxVtxPerMeshlet));

    UpdateMemoryAllocationsMeshlet(accels, numInstances, maxClusters, maxVertices, totalClasBytes);

    // Now that buffers are allocated, compute GPU addresses
    clasBaseAddr = accels.clasBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress vertexBaseAddr = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();

    // Second pass: fill per-cluster GPU info with prefix sums
    uint32_t expandedVtxOffset = 0;
    uint32_t uniqueVtxOffset = 0;
    clasOffset = 0;
    for (uint32_t ci = 0; ci < totalClusters; ++ci) {
        const SelectedCluster& sc = allSelectedClusters[ci];
        const MeshletInstance& mi = instances[sc.instanceIdx];
        uint32_t triCount = mi.templates->meshletTriangleCounts[sc.meshletIdx];
        uint32_t vtxCount = mi.templates->meshletVertexCounts[sc.meshletIdx];

        clusterInfos[ci].persistentExpandedVtxOffset = mi.templates->meshletExpandedVtxOffsets[sc.meshletIdx];
        clusterInfos[ci].persistentUniqueVtxOffset = mi.templates->meshletVertexOffsets[sc.meshletIdx];
        clusterInfos[ci].perFrameExpandedVtxOffset = expandedVtxOffset;
        clusterInfos[ci].perFrameUniqueVtxOffset = uniqueVtxOffset;
        clusterInfos[ci].triCount = triCount;
        clusterInfos[ci].uniqueVtxCount = vtxCount;
        clusterInfos[ci].surfaceId = mi.surfaceId;
        clusterInfos[ci].clusterIndex = ci;
        clusterInfos[ci].templateAddr = mi.templates->templateAddresses[sc.meshletIdx];
        clusterInfos[ci].clasDestAddr = clasBaseAddr + clasOffset;

        uint32_t instSize = mi.templates->instantiationSizes[sc.meshletIdx];
        clasOffset += ((instSize + cluster::kClasByteAlignment - 1) / cluster::kClasByteAlignment) * cluster::kClasByteAlignment;
        expandedVtxOffset += triCount * 3;
        uniqueVtxOffset += vtxCount;
    }

    // DIAG: Log GPU addresses and first few cluster infos
    Logger::info(str::format("RTX MegaGeo MESHLET: clasBaseAddr=0x", std::hex, clasBaseAddr,
        " vertexBaseAddr=0x", vertexBaseAddr, std::dec));
    for (uint32_t ci = 0; ci < std::min(totalClusters, 3u); ++ci) {
        const auto& info = clusterInfos[ci];
        Logger::info(str::format("RTX MegaGeo MESHLET:   cluster[", ci, "] triCount=", info.triCount,
            " uniqueVtxCount=", info.uniqueVtxCount,
            " persistExpVtxOff=", info.persistentExpandedVtxOffset,
            " persistUniqVtxOff=", info.persistentUniqueVtxOffset,
            " perFrameExpVtxOff=", info.perFrameExpandedVtxOffset,
            " perFrameUniqVtxOff=", info.perFrameUniqueVtxOffset));
        Logger::info(str::format("RTX MegaGeo MESHLET:     templateAddr=0x", std::hex, info.templateAddr,
            " clasDestAddr=0x", info.clasDestAddr, std::dec,
            " surfaceId=", info.surfaceId, " clusterIndex=", info.clusterIndex));
    }

    // =====================================================================
    // Per-frame buffer clears
    // =====================================================================
    commandList->clearBufferUInt(m_clusterOffsetCountsBuffer.Get(), 0);
    commandList->clearBufferUInt(m_fillClustersDispatchIndirectBuffer.Get(), 0);
    commandList->clearBufferUInt(m_blasFromClasIndirectArgsBuffer.Get(), 0);
    commandList->clearBufferUInt(accels.blasPtrsBuffer.Get(), 0);
    commandList->clearBufferUInt(accels.blasSizesBuffer.Get(), 0);
    commandList->clearBufferUInt(accels.blasBuffer.Get(), 0);

    // =====================================================================
    // Upload cluster selection info to GPU (small: 48 bytes × numClusters)
    // =====================================================================
    // Create temporary buffer for cluster infos (per-frame upload)
    size_t clusterInfoBytes = clusterInfos.size() * sizeof(MeshletClusterGPUInfo);
    nvrhi::BufferDesc clusterInfoDesc = {
        .byteSize = clusterInfoBytes,
        .debugName = "MeshletClusterInfos",
        .structStride = sizeof(MeshletClusterGPUInfo),
        .initialState = nvrhi::ResourceStates::ShaderResource,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle clusterInfoBuffer = m_device->createBuffer(clusterInfoDesc);
    commandList->writeBuffer(clusterInfoBuffer, clusterInfos.data(), clusterInfoBytes);
    commandList->bufferBarrier(clusterInfoBuffer.Get(),
        nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::ShaderResource);

    // =====================================================================
    // GPU compute dispatch: fill per-frame buffers from persistent data
    // Replaces ALL CPU vertex/normal/CLAS-args loops
    // =====================================================================
    {
        nvrhi::utils::ScopedMarker marker(commandList, "FillMeshletClusters_GPU");

        // Get the persistent buffers from the first valid instance's template set
        // (all instances with the same mesh share the same template set)
        const MeshletTemplateSet* templateSet = nullptr;
        for (const auto& sc : allSelectedClusters) {
            templateSet = instances[sc.instanceIdx].templates;
            if (templateSet && templateSet->isBuilt)
                break;
        }

        if (!templateSet || !templateSet->expandedVerticesBuffer || !templateSet->expandedNormalsBuffer) {
            Logger::err("RTX MegaGeo MESHLET: Missing persistent expanded buffers!");
            return;
        }

        // Create push constant params buffer (non-volatile, uses vkCmdUpdateBuffer)
        FillMeshletClustersParams params = {};
        params.numClusters = totalClusters;
        params.totalExpandedVertices = totalExpandedVtx;
        params.perFrameVertexAddr = vertexBaseAddr;

        nvrhi::BufferDesc paramsDesc;
        paramsDesc.byteSize = 256; // Aligned for constant buffer requirements
        paramsDesc.debugName = "FillMeshletClustersParams";
        paramsDesc.isConstantBuffer = true;
        paramsDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        paramsDesc.keepInitialState = true;
        nvrhi::BufferHandle paramsBuffer = m_device->createBuffer(paramsDesc);
        commandList->writeBuffer(paramsBuffer, &params, sizeof(params));

        // Create binding set
        auto bindingSetDesc = nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, paramsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, clusterInfoBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, templateSet->expandedVerticesBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, templateSet->expandedNormalsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, templateSet->vertexBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, accels.clusterVertexPositionsBuffer.Get()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, accels.clusterVertexNormalsBuffer.Get()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_clasIndirectArgDataBuffer.Get()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clasPtrsBuffer.Get()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(4, accels.clusterShadingDataBuffer.Get()));

        nvrhi::BindingSetHandle bindingSet;
        if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillMeshletClustersBL, bindingSet)) {
            Logger::err("RTX MegaGeo: Failed to create binding set for fill_meshlet_clusters");
            return;
        }

        // Create pipeline on first use
        if (!m_fillMeshletClustersPSO) {
            nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader(
                "cluster_builder/fill_meshlet_clusters.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
            if (!shader) {
                Logger::err("RTX MegaGeo: Failed to create fill_meshlet_clusters shader");
                return;
            }

            auto pipelineDesc = nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_fillMeshletClustersBL);

            m_fillMeshletClustersPSO = m_device->createComputePipeline(pipelineDesc);
            if (!m_fillMeshletClustersPSO) {
                Logger::err("RTX MegaGeo: Failed to create fill_meshlet_clusters pipeline");
                return;
            }
        }

        auto state = nvrhi::ComputeState()
            .setPipeline(m_fillMeshletClustersPSO)
            .addBindingSet(bindingSet);

        commandList->setComputeState(state);

        uint32_t numGroups = (totalClusters + kFillMeshletClustersThreads - 1) / kFillMeshletClustersThreads;
        commandList->dispatch(numGroups, 1, 1);

        Logger::info(str::format("RTX MegaGeo MESHLET: GPU compute dispatched ", numGroups, " groups for ", totalClusters, " clusters"));
        Logger::info(str::format("RTX MegaGeo MESHLET: FillParams numClusters=", params.numClusters,
            " totalExpandedVertices=", params.totalExpandedVertices,
            " perFrameVertexAddr=0x", std::hex, params.perFrameVertexAddr, std::dec));
    }

    // DIAG: Readback CLAS indirect args after GPU compute
    {
        static uint32_t s_argsDiagCount = 0;
        if (s_argsDiagCount < 3) {
            s_argsDiagCount++;
            std::vector<cluster::IndirectInstantiateTemplateArgs> clasArgs =
                m_clasIndirectArgDataBuffer.Download(commandList);
            Logger::info(str::format("RTX MegaGeo MESHLET DIAG: CLAS args readback: total=", clasArgs.size()));
            for (size_t i = 0; i < std::min(clasArgs.size(), (size_t)3); ++i) {
                Logger::info(str::format("RTX MegaGeo MESHLET DIAG:   clasArgs[", i, "] clusterIdOffset=", clasArgs[i].clusterIdOffset,
                    " geometryIndexOffsetPacked=", clasArgs[i].geometryIndexOffsetPacked,
                    " clusterTemplate=0x", std::hex, clasArgs[i].clusterTemplate,
                    " vtxAddr=0x", clasArgs[i].vertexBuffer.startAddress,
                    " vtxStride=", std::dec, clasArgs[i].vertexBuffer.strideInBytes));
            }

            // Also readback first few per-frame vertex positions
            // Use raw byte readback since structStride is 12 (float3)
            size_t vtxBufBytes = accels.clusterVertexPositionsBuffer.GetBytes();
            size_t numFloat3 = vtxBufBytes / 12;
            size_t readCount = std::min(numFloat3, (size_t)10);
            if (readCount > 0) {
                // Create readback buffer
                nvrhi::BufferDesc rbDesc = {
                    .byteSize = readCount * 12,
                    .debugName = "DiagVtxReadback",
                    .cpuAccess = nvrhi::CpuAccessMode::Read,
                    .initialState = nvrhi::ResourceStates::CopyDest,
                    .keepInitialState = true
                };
                nvrhi::BufferHandle rbBuf = m_device->createBuffer(rbDesc);
                commandList->copyBuffer(rbBuf.Get(), 0, accels.clusterVertexPositionsBuffer.Get(), 0, readCount * 12);
                commandList->close();
                m_device->executeCommandList(commandList);
                m_device->waitForIdle();
                float* mapped = (float*)m_device->mapBuffer(rbBuf.Get(), nvrhi::CpuAccessMode::Read);
                if (mapped) {
                    for (size_t i = 0; i < readCount; ++i) {
                        Logger::info(str::format("RTX MegaGeo MESHLET DIAG:   vtxPos[", i, "] = (",
                            mapped[i*3+0], ", ", mapped[i*3+1], ", ", mapped[i*3+2], ")"));
                    }
                }
                m_device->unmapBuffer(rbBuf.Get());
                commandList->open();
            }

            // Also readback ClusterShadingData to verify hit shader lookup data
            {
                std::vector<ClusterShadingData> shadingData = accels.clusterShadingDataBuffer.Download(commandList);
                Logger::warn(str::format("RTX MegaGeo MESHLET DIAG: ClusterShadingData readback: total=", shadingData.size()));
                for (size_t i = 0; i < std::min(shadingData.size(), (size_t)5); ++i) {
                    const auto& sd = shadingData[i];
                    Logger::warn(str::format("RTX MegaGeo MESHLET DIAG:   shadingData[", i, "] surfaceId=", sd.m_surfaceId,
                        " vertexOffset=", sd.m_vertexOffset,
                        " clusterSizeX=", sd.m_clusterSizeX, " clusterSizeY=", sd.m_clusterSizeY,
                        " edgeSegments=(", sd.m_edgeSegments.x, ",", sd.m_edgeSegments.y, ",", sd.m_edgeSegments.z, ",", sd.m_edgeSegments.w, ")",
                        " clusterOffset=(", sd.m_clusterOffset.x, ",", sd.m_clusterOffset.y, ")"));
                    // For meshlet path, vertexOffset should point to expanded vertices (primId*3 + vertexOffset)
                    // Verify: expanded vertices at vertexOffset should match what we expect
                    if (sd.m_clusterSizeX == 0) {
                        Logger::warn(str::format("RTX MegaGeo MESHLET DIAG:     -> meshlet path (sizeX==0), hit shader will read vtxPos[vertexOffset + primId*3 + {0,1,2}]"));
                        Logger::warn(str::format("RTX MegaGeo MESHLET DIAG:     -> for primId=0: vtxPos[", sd.m_vertexOffset, "], vtxPos[", sd.m_vertexOffset+1, "], vtxPos[", sd.m_vertexOffset+2, "]"));
                    } else {
                        Logger::warn(str::format("RTX MegaGeo MESHLET DIAG:     -> subdivision path (sizeX=", sd.m_clusterSizeX, ")"));
                    }
                }
            }
        }
    }

    // =====================================================================
    // Barriers: GPU compute output → CLAS instantiation
    // =====================================================================
    commandList->bufferBarrier(accels.clusterVertexPositionsBuffer,
        nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(m_clasIndirectArgDataBuffer,
        nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::ShaderResource);
    commandList->bufferBarrier(accels.clasPtrsBuffer,
        nvrhi::ResourceStates::UnorderedAccess, nvrhi::ResourceStates::UnorderedAccess);

    // =====================================================================
    // CLAS Instantiation
    // =====================================================================
    {
        nvrhi::utils::ScopedMarker marker(commandList, "BuildStructuredCLASes_Meshlet");

        constexpr uint32_t kMaxMeshletGeometryIndex = 16383;

        cluster::OperationParams instantiateClasParams = {
            .maxArgCount = totalClusters,
            .type = cluster::OperationType::ClasInstantiateTemplates,
            .mode = cluster::OperationMode::ExplicitDestinations,
            .flags = cluster::OperationFlags::None,
            .clas = {
                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                .maxGeometryIndex = kMaxMeshletGeometryIndex,
                .maxUniqueGeometryCount = 1,
                .maxTriangleCount = maxTriPerMeshlet,
                .maxVertexCount = maxVtxPerMeshlet,
                .maxTotalTriangleCount = totalClusters * maxTriPerMeshlet,
                .maxTotalVertexCount = totalUniqueVtx,
                .minPositionTruncateBitCount = 0,
            }
        };

        cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(instantiateClasParams);

        cluster::OperationDesc instantiateClasDesc = {
            .params = instantiateClasParams,
            .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
            .inIndirectArgCountBuffer = nullptr,
            .inIndirectArgCountOffsetInBytes = 0,
            .inIndirectArgsBuffer = m_clasIndirectArgDataBuffer,
            .inIndirectArgsOffsetInBytes = 0,
            .inOutAddressesBuffer = accels.clasPtrsBuffer,
            .inOutAddressesOffsetInBytes = 0,
            .outSizesBuffer = nullptr,
            .outSizesOffsetInBytes = 0,
            .outAccelerationStructuresBuffer = nullptr,
            .outAccelerationStructuresOffsetInBytes = 0
        };

        Logger::info(str::format("RTX MegaGeo MESHLET: CLAS Instantiate: totalClusters=", totalClusters,
            " maxTriPerMeshlet=", maxTriPerMeshlet, " maxVtxPerMeshlet=", maxVtxPerMeshlet,
            " maxTotalTriangleCount=", totalClusters * maxTriPerMeshlet,
            " maxTotalVertexCount=", totalUniqueVtx));
        Logger::info(str::format("RTX MegaGeo MESHLET: CLAS Instantiate: argsBuffer=0x", std::hex,
            m_clasIndirectArgDataBuffer.GetGpuVirtualAddress(),
            " argsStride=", std::dec, m_clasIndirectArgDataBuffer.GetElementBytes(),
            " argsByteSize=", m_clasIndirectArgDataBuffer.GetBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: CLAS Instantiate: clasPtrsBuffer=0x", std::hex,
            accels.clasPtrsBuffer.GetGpuVirtualAddress(),
            " ptrsByteSize=", std::dec, accels.clasPtrsBuffer.GetBytes(),
            " ptrsStride=", accels.clasPtrsBuffer.GetElementBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: CLAS Instantiate: scratchSize=", sizeInfo.scratchSizeInBytes,
            " mode=ExplicitDestinations outAccelBuf=nullptr outSizesBuf=nullptr"));

        commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);
        Logger::info("RTX MegaGeo MESHLET: CLAS Instantiate completed");
    }

    // CLAS → BLAS barriers
    commandList->bufferBarrier(accels.clasPtrsBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(accels.clasBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::AccelStructBuildInput);

    // =====================================================================
    // Fill BLAS from CLAS args (per-instance) — stays on CPU (tiny data)
    // =====================================================================
    {
        nvrhi::GpuVirtualAddress clasPtrsBaseAddress = accels.clasPtrsBuffer.GetGpuVirtualAddress();
        uint32_t indirectArgAlignedStride = (sizeof(cluster::IndirectArgs) + 15) & ~15;
        std::vector<uint8_t> blasArgsData(indirectArgAlignedStride * numInstances, 0);

        for (uint32_t inst = 0; inst < numInstances; ++inst) {
            auto* arg = reinterpret_cast<cluster::IndirectArgs*>(
                blasArgsData.data() + indirectArgAlignedStride * inst);
            arg->clusterCount = perInstanceClusterCount[inst];
            arg->clusterReferencesStride = 8;
            arg->clusterAddresses = clasPtrsBaseAddress + (uint64_t)perInstanceClusterOffset[inst] * sizeof(nvrhi::GpuVirtualAddress);
        }

        // DIAG: Log BLAS from CLAS args
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS args: clasPtrsBaseAddress=0x", std::hex, clasPtrsBaseAddress, std::dec,
            " indirectArgAlignedStride=", indirectArgAlignedStride, " numInstances=", numInstances));
        for (uint32_t inst = 0; inst < numInstances; ++inst) {
            auto* arg = reinterpret_cast<cluster::IndirectArgs*>(
                blasArgsData.data() + indirectArgAlignedStride * inst);
            Logger::info(str::format("RTX MegaGeo MESHLET:   blasArg[", inst, "] clusterCount=", arg->clusterCount,
                " referencesStride=", arg->clusterReferencesStride,
                " clusterAddresses=0x", std::hex, arg->clusterAddresses, std::dec));
        }

        commandList->writeBuffer(m_blasFromClasIndirectArgsBuffer.Get(), blasArgsData.data(), blasArgsData.size());
        commandList->bufferBarrier(m_blasFromClasIndirectArgsBuffer,
            nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::ShaderResource);
    }

    // =====================================================================
    // Build BLAS from CLAS
    // =====================================================================
    {
        nvrhi::utils::ScopedMarker marker(commandList, "Blas Build from Clas (Meshlet)");

        cluster::OperationParams buildParams = m_createBlasParams;
        buildParams.maxArgCount = numInstances;

        cluster::OperationDesc createBlasDesc = {
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

        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: maxArgCount=", buildParams.maxArgCount,
            " maxClasPerBlasCount=", buildParams.blas.maxClasPerBlasCount,
            " maxTotalClasCount=", buildParams.blas.maxTotalClasCount));
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: argsBuffer=0x", std::hex,
            m_blasFromClasIndirectArgsBuffer.GetGpuVirtualAddress(), std::dec,
            " argsStride=", m_blasFromClasIndirectArgsBuffer.GetElementBytes(),
            " argsByteSize=", m_blasFromClasIndirectArgsBuffer.GetBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: blasPtrsBuffer=0x", std::hex,
            accels.blasPtrsBuffer.GetGpuVirtualAddress(), std::dec,
            " ptrsStride=", accels.blasPtrsBuffer.GetElementBytes(),
            " ptrsByteSize=", accels.blasPtrsBuffer.GetBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: blasBuffer=0x", std::hex,
            accels.blasBuffer.GetGpuVirtualAddress(), std::dec,
            " blasByteSize=", accels.blasBuffer.GetBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: blasSizesBuffer=0x", std::hex,
            accels.blasSizesBuffer.GetGpuVirtualAddress(), std::dec,
            " sizesStride=", accels.blasSizesBuffer.GetElementBytes(),
            " sizesByteSize=", accels.blasSizesBuffer.GetBytes()));
        Logger::info(str::format("RTX MegaGeo MESHLET: BLAS Build: scratchSize=", m_createBlasSizeInfo.scratchSizeInBytes,
            " mode=ImplicitDestinations"));

        commandList->executeMultiIndirectClusterOperation(createBlasDesc);
        Logger::info("RTX MegaGeo MESHLET: BLAS Build completed");
    }

    // DIAG: Readback blasPtrsBuffer after BLAS build to check if addresses are populated
    {
        static uint32_t s_blasDiagCount = 0;
        if (s_blasDiagCount < 3) {
            s_blasDiagCount++;
            std::vector<nvrhi::GpuVirtualAddress> blasAddrs = accels.blasPtrsBuffer.Download(commandList);
            uint32_t nullCount = 0, nonNullCount = 0;
            for (size_t i = 0; i < blasAddrs.size(); ++i) {
                if (blasAddrs[i] == 0) nullCount++;
                else nonNullCount++;
            }
            Logger::info(str::format("RTX MegaGeo MESHLET DIAG: blasPtrsBuffer after BLAS build: total=", blasAddrs.size(),
                " null=", nullCount, " nonNull=", nonNullCount));
            for (size_t i = 0; i < std::min(blasAddrs.size(), (size_t)5); ++i) {
                Logger::info(str::format("RTX MegaGeo MESHLET DIAG:   blasPtrs[", i, "] = 0x", std::hex, blasAddrs[i]));
            }

            // Also readback clasPtrsBuffer to check CLAS addresses
            std::vector<nvrhi::GpuVirtualAddress> clasAddrs = accels.clasPtrsBuffer.Download(commandList);
            uint32_t clasNull = 0, clasNonNull = 0;
            for (size_t i = 0; i < clasAddrs.size(); ++i) {
                if (clasAddrs[i] == 0) clasNull++;
                else clasNonNull++;
            }
            Logger::info(str::format("RTX MegaGeo MESHLET DIAG: clasPtrsBuffer after CLAS instantiate: total=", clasAddrs.size(),
                " null=", clasNull, " nonNull=", clasNonNull));
            for (size_t i = 0; i < std::min(clasAddrs.size(), (size_t)5); ++i) {
                Logger::info(str::format("RTX MegaGeo MESHLET DIAG:   clasPtrs[", i, "] = 0x", std::hex, clasAddrs[i]));
            }
        }
    }

    // Update stats
    stats.allocated.m_numClusters = totalClusters;
    stats.allocated.m_numTriangles = totalTriangles;
    stats.desired = stats.allocated;

    m_buildAccelFrameIndex++;
}

void ClusterAccelBuilder::EnsureMeshletTemplatesInitialized(
    const std::vector<MeshletInstance>& instances,
    uint32_t maxGeometryIndex,
    nvrhi::ICommandList* commandList)
{
    // No-op for now — templates are built per-mesh in RtxMegaGeoBuilder
    // This is a placeholder for future use if we want to pre-initialize
    (void)instances;
    (void)maxGeometryIndex;
    (void)commandList;
}


