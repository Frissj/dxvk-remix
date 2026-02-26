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
        uint32_t meshletIdx;     // Index into ClusterLODData::clusters
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

    if (totalClusters == 0) {
        Logger::warn("RTX MegaGeo MESHLET: No clusters selected for any instance");
        return;
    }

    Logger::info(str::format("RTX MegaGeo MESHLET: Selected ", totalClusters,
        " clusters across ", numInstances, " instances"));

    // =====================================================================
    // Calculate total vertices needed
    // Per-triangle expanded (3 per tri) for hit shader vertex/normal buffers
    // Unique vertices for CLAS instantiation (uses template's own vertex buffer)
    // =====================================================================
    uint32_t totalVertices = 0;     // per-tri expanded for hit shader
    uint32_t totalUniqueVtx = 0;    // unique for CLAS params
    uint32_t totalTriangles = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        totalTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
        totalUniqueVtx += mi.templates->meshletVertexCounts[sc.meshletIdx];
        totalVertices += mi.templates->meshletTriangleCounts[sc.meshletIdx] * 3;
    }

    // =====================================================================
    // Memory allocation
    // =====================================================================
    uint32_t maxClusters = std::max(totalClusters, 1u);
    uint32_t maxVertices = std::max(totalVertices, 1u);

    // Determine actual max triangles/vertices per meshlet across all selected clusters.
    // These MUST match the template build params for CLAS instantiation to succeed.
    uint32_t maxTriPerMeshlet = 0;
    uint32_t maxVtxPerMeshlet = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        maxTriPerMeshlet = std::max(maxTriPerMeshlet, mi.templates->maxTrianglesPerMeshlet);
        maxVtxPerMeshlet = std::max(maxVtxPerMeshlet, mi.templates->maxVerticesPerMeshlet);
    }
    if (maxTriPerMeshlet == 0) maxTriPerMeshlet = 128;
    if (maxVtxPerMeshlet == 0) maxVtxPerMeshlet = 128;

    // Estimate CLAS size: sum of instantiation sizes for selected clusters
    size_t totalClasBytes = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        uint32_t instSize = mi.templates->instantiationSizes[sc.meshletIdx];
        totalClasBytes += ((instSize + cluster::kClasByteAlignment - 1) / cluster::kClasByteAlignment) * cluster::kClasByteAlignment;
    }
    totalClasBytes = std::max(totalClasBytes, (size_t)cluster::kClasByteAlignment);

    UpdateMemoryAllocationsMeshlet(accels, numInstances, maxClusters, maxVertices, totalClasBytes);

    // STRIDE/SIZE DIAGNOSTICS — log buffer strides and struct sizes for debugging
    {
        Logger::warn(str::format("RTX MegaGeo STRIDE-DIAG: sizeof(ClusterShadingData)=", sizeof(ClusterShadingData),
            " sizeof(float2)=", sizeof(float2), " sizeof(float3)=", sizeof(float3),
            " sizeof(IndirectInstantiateTemplateArgs)=", sizeof(cluster::IndirectInstantiateTemplateArgs),
            " sizeof(IndirectArgs)=", sizeof(cluster::IndirectArgs)));
        Logger::warn(str::format("RTX MegaGeo STRIDE-DIAG: vtxPosBuffer structStride=",
            accels.clusterVertexPositionsBuffer.GetBuffer() ? accels.clusterVertexPositionsBuffer.GetBuffer()->getDesc().structStride : 0,
            " byteSize=", accels.clusterVertexPositionsBuffer.GetBytes(),
            " normBuffer structStride=",
            accels.clusterVertexNormalsBuffer.GetBuffer() ? accels.clusterVertexNormalsBuffer.GetBuffer()->getDesc().structStride : 0,
            " byteSize=", accels.clusterVertexNormalsBuffer.GetBytes(),
            " shadingBuffer structStride=",
            accels.clusterShadingDataBuffer.GetBuffer() ? accels.clusterShadingDataBuffer.GetBuffer()->getDesc().structStride : 0,
            " byteSize=", accels.clusterShadingDataBuffer.GetBytes()));
        Logger::warn(str::format("RTX MegaGeo STRIDE-DIAG: clasArgsBuffer structStride=",
            m_clasIndirectArgDataBuffer.GetBuffer() ? m_clasIndirectArgDataBuffer.GetBuffer()->getDesc().structStride : 0,
            " clasPtrsBuffer structStride=",
            accels.clasPtrsBuffer.GetBuffer() ? accels.clasPtrsBuffer.GetBuffer()->getDesc().structStride : 0,
            " blasArgsBuffer structStride=",
            m_blasFromClasIndirectArgsBuffer.GetBuffer() ? m_blasFromClasIndirectArgsBuffer.GetBuffer()->getDesc().structStride : 0));
        Logger::warn(str::format("RTX MegaGeo STRIDE-DIAG: kGpuFloat3Stride=", 12,
            " totalVertices(perTriExpanded)=", totalVertices, " totalUniqueVtx=", totalUniqueVtx,
            " totalTriangles=", totalTriangles, " totalClusters=", totalClusters));
    }

    // DEBUG ISOLATION: Set to 0-4 to progressively enable GPU stages
    // 0 = skip all GPU work (just stats)
    // 1 = buffer clears only
    // 2 = clears + uploads + barriers
    // 3 = clears + uploads + barriers + CLAS instantiation (no BLAS)
    // 4 = full pipeline (normal operation)
    constexpr int kMeshletDebugStage = 4;

    if (kMeshletDebugStage < 1) {
        Logger::warn("RTX MegaGeo MESHLET DEBUG: Stage 0 — skipping all GPU work");
        stats.allocated.m_numClusters = totalClusters;
        stats.allocated.m_numTriangles = 0;
        for (const auto& sc : allSelectedClusters) {
            const MeshletInstance& mi = instances[sc.instanceIdx];
            stats.allocated.m_numTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
        }
        stats.desired = stats.allocated;
        m_buildAccelFrameIndex++;
        return;
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
    // CPU fill: vertex positions, CLAS indirect args, CLAS dest addrs, shading data
    // =====================================================================
    static constexpr uint32_t kGpuFloat3Stride = 12;

    // Build CPU-side data for all selected clusters
    std::vector<float> vertexData(totalVertices * 3);
    std::vector<cluster::IndirectInstantiateTemplateArgs> clasArgs(totalClusters);
    std::vector<nvrhi::GpuVirtualAddress> clasDests(totalClusters);
    std::vector<ClusterShadingData> shadingData(totalClusters);

    nvrhi::GpuVirtualAddress vertexBaseAddr = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress clasBaseAddr = accels.clasBuffer.GetGpuVirtualAddress();

    uint32_t vertexOffset = 0;
    uint64_t clasOffset = 0;

    for (uint32_t ci = 0; ci < totalClusters; ++ci) {
        const SelectedCluster& sc = allSelectedClusters[ci];
        const MeshletInstance& mi = instances[sc.instanceIdx];
        const MeshletCluster& meshlet = mi.lodData->clusters[sc.meshletIdx];

        // Copy vertex positions per-triangle expanded (3 vertices per triangle, sequential)
        // This is for the HIT SHADER to index with primId * 3 + {0,1,2}
        float* dst = vertexData.data() + vertexOffset * 3;
        for (uint32_t t = 0; t < meshlet.triangleCount; ++t) {
            for (uint32_t vi = 0; vi < 3; ++vi) {
                uint32_t localIdx = meshlet.localTriangles[t * 3 + vi];
                uint32_t globalVtx = meshlet.localVertices[localIdx];
                uint32_t dstIdx = (t * 3 + vi) * 3;
                dst[dstIdx + 0] = mi.lodData->vertexPositions[globalVtx * 3 + 0];
                dst[dstIdx + 1] = mi.lodData->vertexPositions[globalVtx * 3 + 1];
                dst[dstIdx + 2] = mi.lodData->vertexPositions[globalVtx * 3 + 2];
            }
        }

        // Fill CLAS indirect args
        // IMPORTANT: CLAS instantiation uses the TEMPLATE's own vertex buffer (unique-vertex layout)
        // since meshlet geometry is static. The template was built with these same positions.
        nvrhi::GpuVirtualAddress templateVtxAddr = mi.templates->vertexBuffer->getGpuVirtualAddress()
            + (uint64_t)mi.templates->meshletVertexOffsets[sc.meshletIdx] * kGpuFloat3Stride;
        clasArgs[ci].clusterIdOffset = ci;
        clasArgs[ci].geometryIndexOffsetPacked = ci & 0xFFFFFF; // lower 24 bits = geometryIndex
        clasArgs[ci].clusterTemplate = mi.templates->templateAddresses[sc.meshletIdx];
        clasArgs[ci].vertexBuffer.startAddress = templateVtxAddr;
        clasArgs[ci].vertexBuffer.strideInBytes = kGpuFloat3Stride;

        // Fill CLAS destination address (128-byte aligned)
        clasDests[ci] = clasBaseAddr + clasOffset;
        uint32_t instSize = mi.templates->instantiationSizes[sc.meshletIdx];
        clasOffset += ((instSize + cluster::kClasByteAlignment - 1) / cluster::kClasByteAlignment) * cluster::kClasByteAlignment;

        // Fill shading data — m_clusterSizeX stays 0 to signal meshlet path to hit shader
        shadingData[ci] = {};
        shadingData[ci].m_surfaceId = mi.surfaceId;
        shadingData[ci].m_vertexOffset = vertexOffset;

        vertexOffset += meshlet.triangleCount * 3;  // per-triangle expanded for hit shader
    }

    // Log key addresses for debugging
    Logger::warn(str::format("RTX MegaGeo MESHLET BUFFERS: vtxBase=0x", std::hex, vertexBaseAddr,
        " clasBase=0x", clasBaseAddr, std::dec,
        " totalClusters=", totalClusters, " totalVtx=", totalVertices,
        " totalClasBytes=", totalClasBytes));
    if (!clasDests.empty()) {
        Logger::warn(str::format("RTX MegaGeo MESHLET CLAS-DEST[0]: addr=0x", std::hex, clasDests[0], std::dec,
            " instSize=", instances[allSelectedClusters[0].instanceIdx].templates->instantiationSizes[allSelectedClusters[0].meshletIdx]));
    }

    // Log first few shading data entries for debugging
    for (uint32_t i = 0; i < std::min(totalClusters, 5u); ++i) {
        Logger::warn(str::format("RTX MegaGeo SHADING[", i, "]: surfaceId=", shadingData[i].m_surfaceId,
            " vertexOffset=", shadingData[i].m_vertexOffset,
            " clusterSizeX=", (uint32_t)shadingData[i].m_clusterSizeX,
            " clusterSizeY=", (uint32_t)shadingData[i].m_clusterSizeY));
    }

    // Log upload sizes for debugging stride/overrun issues
    Logger::warn(str::format("RTX MegaGeo UPLOAD-SIZES: vtxData=", vertexData.size() * sizeof(float),
        " bytes, clasArgs=", clasArgs.size() * sizeof(clasArgs[0]),
        " bytes, clasDests=", clasDests.size() * sizeof(clasDests[0]),
        " bytes, shadingData=", shadingData.size() * sizeof(shadingData[0]),
        " bytes (sizeof(ClusterShadingData)=", sizeof(ClusterShadingData), ")"));

    // =====================================================================
    // Upload CPU data to GPU buffers
    // =====================================================================

    // Vertex positions
    if (!vertexData.empty()) {
        commandList->writeBuffer(accels.clusterVertexPositionsBuffer.Get(), vertexData.data(),
            vertexData.size() * sizeof(float));
    }

    // CLAS indirect args
    if (!clasArgs.empty()) {
        commandList->writeBuffer(m_clasIndirectArgDataBuffer.Get(), clasArgs.data(),
            clasArgs.size() * sizeof(clasArgs[0]));
    }

    // CLAS destination addresses
    if (!clasDests.empty()) {
        commandList->writeBuffer(accels.clasPtrsBuffer.Get(), clasDests.data(),
            clasDests.size() * sizeof(clasDests[0]));
    }

    // Shading data
    if (!shadingData.empty()) {
        commandList->writeBuffer(accels.clusterShadingDataBuffer.Get(), shadingData.data(),
            shadingData.size() * sizeof(shadingData[0]));
    }

    // Compute face normals per-triangle expanded and upload
    {
        std::vector<float> normalData(totalVertices * 3, 0.0f);
        for (uint32_t ci = 0; ci < totalClusters; ++ci) {
            const SelectedCluster& sc = allSelectedClusters[ci];
            const MeshletInstance& mi = instances[sc.instanceIdx];
            const MeshletCluster& meshlet = mi.lodData->clusters[sc.meshletIdx];
            uint32_t baseVtx = shadingData[ci].m_vertexOffset;

            // Per-triangle expanded: each triangle gets its own 3 normal slots
            for (uint32_t t = 0; t < meshlet.triangleCount; ++t) {
                uint32_t i0 = meshlet.localTriangles[t * 3 + 0];
                uint32_t i1 = meshlet.localTriangles[t * 3 + 1];
                uint32_t i2 = meshlet.localTriangles[t * 3 + 2];
                uint32_t g0 = meshlet.localVertices[i0];
                uint32_t g1 = meshlet.localVertices[i1];
                uint32_t g2 = meshlet.localVertices[i2];

                float ax = mi.lodData->vertexPositions[g1 * 3 + 0] - mi.lodData->vertexPositions[g0 * 3 + 0];
                float ay = mi.lodData->vertexPositions[g1 * 3 + 1] - mi.lodData->vertexPositions[g0 * 3 + 1];
                float az = mi.lodData->vertexPositions[g1 * 3 + 2] - mi.lodData->vertexPositions[g0 * 3 + 2];
                float bx = mi.lodData->vertexPositions[g2 * 3 + 0] - mi.lodData->vertexPositions[g0 * 3 + 0];
                float by = mi.lodData->vertexPositions[g2 * 3 + 1] - mi.lodData->vertexPositions[g0 * 3 + 1];
                float bz = mi.lodData->vertexPositions[g2 * 3 + 2] - mi.lodData->vertexPositions[g0 * 3 + 2];

                float nx = ay * bz - az * by;
                float ny = az * bx - ax * bz;
                float nz = ax * by - ay * bx;
                float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }

                // Write face normal to all 3 expanded vertex slots for this triangle
                uint32_t vtxBase = baseVtx + t * 3;
                normalData[vtxBase * 3 + 0] = nx; normalData[vtxBase * 3 + 1] = ny; normalData[vtxBase * 3 + 2] = nz;
                normalData[(vtxBase + 1) * 3 + 0] = nx; normalData[(vtxBase + 1) * 3 + 1] = ny; normalData[(vtxBase + 1) * 3 + 2] = nz;
                normalData[(vtxBase + 2) * 3 + 0] = nx; normalData[(vtxBase + 2) * 3 + 1] = ny; normalData[(vtxBase + 2) * 3 + 2] = nz;
            }
        }

        if (accels.clusterVertexNormalsBuffer.GetBuffer()) {
            commandList->writeBuffer(accels.clusterVertexNormalsBuffer.Get(), normalData.data(),
                normalData.size() * sizeof(float));
        }
    }

    if (kMeshletDebugStage < 3) {
        Logger::warn(str::format("RTX MegaGeo MESHLET DEBUG: Stage ", kMeshletDebugStage, " — skipping CLAS+BLAS"));
        stats.allocated.m_numClusters = totalClusters;
        stats.allocated.m_numTriangles = 0;
        for (const auto& sc : allSelectedClusters) {
            const MeshletInstance& mi = instances[sc.instanceIdx];
            stats.allocated.m_numTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
        }
        stats.desired = stats.allocated;
        m_buildAccelFrameIndex++;
        return;
    }

    // =====================================================================
    // Barriers: CPU uploads → CLAS instantiation
    // Use ShaderResource as destination state since that's what executeMultiIndirectClusterOperation
    // expects for inIndirectArgsBuffer (matching sample's NVRHI adapter)
    // =====================================================================
    commandList->bufferBarrier(accels.clusterVertexPositionsBuffer,
        nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(m_clasIndirectArgDataBuffer,
        nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::ShaderResource);
    commandList->bufferBarrier(accels.clasPtrsBuffer,
        nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::UnorderedAccess);

    // =====================================================================
    // CLAS Instantiation
    // Use srcInfosCount=0 (no indirect count buffer) — driver uses maxArgCount directly.
    // This avoids tessellation counter buffer barrier issues.
    // =====================================================================
    {
        nvrhi::utils::ScopedMarker marker(commandList, "BuildStructuredCLASes_Meshlet");

        // Must match the maxGeometryIndex used during template build (MeshletTemplateBuilder).
        // The Vulkan spec requires the resulting geometryIndex (baseGeometryIndex + geometryIndexOffset)
        // not exceed maxGeometryIndex from BOTH the template build AND the instantiation call.
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
                .maxTotalVertexCount = totalUniqueVtx,  // unique vertices (matches template)
                .minPositionTruncateBitCount = 0,
            }
        };

        cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(instantiateClasParams);

        Logger::warn(str::format("RTX MegaGeo MESHLET CLAS-INST: maxArg=", totalClusters,
            " maxTri=", maxTriPerMeshlet, " maxVtx=", maxVtxPerMeshlet,
            " totalTri=", totalClusters * maxTriPerMeshlet, " totalVtx=", totalUniqueVtx,
            " maxGeomIdx=", kMaxMeshletGeometryIndex,
            " scratchSize=", sizeInfo.scratchSizeInBytes,
            " clasArgs={addr=0x", std::hex, m_clasIndirectArgDataBuffer.GetGpuVirtualAddress(),
            " stride=", std::dec, m_clasIndirectArgDataBuffer.GetElementBytes(),
            " size=", m_clasIndirectArgDataBuffer.GetBytes(), "}",
            " clasDests={addr=0x", std::hex, accels.clasPtrsBuffer.GetGpuVirtualAddress(),
            " stride=", std::dec, accels.clasPtrsBuffer.GetElementBytes(),
            " size=", accels.clasPtrsBuffer.GetBytes(), "}"));

        // Log first CLAS arg for verification
        if (!clasArgs.empty()) {
            Logger::warn(str::format("RTX MegaGeo MESHLET CLAS-ARG[0]: template=0x", std::hex, clasArgs[0].clusterTemplate,
                " vtxAddr=0x", clasArgs[0].vertexBuffer.startAddress,
                " vtxStride=", std::dec, clasArgs[0].vertexBuffer.strideInBytes,
                " clusterIdOff=", clasArgs[0].clusterIdOffset,
                " geomIdxPacked=", clasArgs[0].geometryIndexOffsetPacked));
        }

        // No indirect count buffer: driver uses maxArgCount (= totalClusters) directly.
        // This avoids barrier issues with tessellation counter buffer + writeBuffer/updateBuffer.
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

        commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);
    }

    if (kMeshletDebugStage < 4) {
        Logger::warn("RTX MegaGeo MESHLET DEBUG: Stage 3 — CLAS done, skipping BLAS");
        stats.allocated.m_numClusters = totalClusters;
        stats.allocated.m_numTriangles = 0;
        for (const auto& sc : allSelectedClusters) {
            const MeshletInstance& mi = instances[sc.instanceIdx];
            stats.allocated.m_numTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
        }
        stats.desired = stats.allocated;
        m_buildAccelFrameIndex++;
        return;
    }

    // CLAS → BLAS barriers:
    // 1. clasPtrsBuffer: CLAS instantiation wrote CLAS addresses here, BLAS build reads them
    // 2. clasBuffer: CLAS instantiation wrote CLAS data here (via explicit destination addresses),
    //    BLAS build reads the CLAS data through those addresses. Without this barrier, the BLAS
    //    build could read stale/partial CLAS data.
    commandList->bufferBarrier(accels.clasPtrsBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(accels.clasBuffer, nvrhi::ResourceStates::AccelStructWrite,
        nvrhi::ResourceStates::AccelStructBuildInput);

    // =====================================================================
    // Fill BLAS from CLAS args (per-instance)
    // =====================================================================
    {
        nvrhi::GpuVirtualAddress clasPtrsBaseAddress = accels.clasPtrsBuffer.GetGpuVirtualAddress();

        // Fill per-instance BLAS args on CPU
        uint32_t indirectArgAlignedStride = (sizeof(cluster::IndirectArgs) + 15) & ~15;
        std::vector<uint8_t> blasArgsData(indirectArgAlignedStride * numInstances, 0);

        for (uint32_t inst = 0; inst < numInstances; ++inst) {
            auto* arg = reinterpret_cast<cluster::IndirectArgs*>(
                blasArgsData.data() + indirectArgAlignedStride * inst);
            arg->clusterCount = perInstanceClusterCount[inst];
            arg->clusterReferencesStride = 8; // sizeof(VkDeviceAddress)
            arg->clusterAddresses = clasPtrsBaseAddress + (uint64_t)perInstanceClusterOffset[inst] * sizeof(nvrhi::GpuVirtualAddress);
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

        Logger::warn(str::format("RTX MegaGeo MESHLET BLAS-BUILD: numInstances=", numInstances,
            " maxClasPerBlas=", buildParams.blas.maxClasPerBlasCount,
            " maxTotalClas=", buildParams.blas.maxTotalClasCount,
            " scratchSize=", m_createBlasSizeInfo.scratchSizeInBytes,
            " resultMaxSize=", m_createBlasSizeInfo.resultMaxSizeInBytes,
            " blasBuf={addr=0x", std::hex, accels.blasBuffer.GetGpuVirtualAddress(),
            " size=", std::dec, accels.blasBuffer.GetBytes(), "}",
            " blasPtrs={addr=0x", std::hex, accels.blasPtrsBuffer.GetGpuVirtualAddress(),
            " size=", std::dec, accels.blasPtrsBuffer.GetBytes(), "}"));

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

        commandList->executeMultiIndirectClusterOperation(createBlasDesc);
    }

    // Diagnostic: readback BLAS pointers and vertex data when real geometry loads
    {
        static uint32_t s_diagCount = 0;
        static uint32_t s_lastClusterCount = 0;
        // Reset counter when cluster count changes significantly (>50% difference = level transition)
        uint32_t diff = (totalClusters > s_lastClusterCount) ? (totalClusters - s_lastClusterCount) : (s_lastClusterCount - totalClusters);
        bool significantChange = (s_lastClusterCount == 0) || (diff > s_lastClusterCount / 2);
        if (significantChange && totalClusters != s_lastClusterCount) {
            Logger::warn(str::format("RTX MegaGeo DIAG: cluster count changed ", s_lastClusterCount, " -> ", totalClusters));
            s_diagCount = 0;
            s_lastClusterCount = totalClusters;
        }
        // Only readback first 2 frames after each significant level transition, and only for real geometry (>5 clusters)
        if (s_diagCount < 2 && totalClusters > 5) {
            s_diagCount++;

            Logger::warn(str::format("RTX MegaGeo DIAG (frame ", s_diagCount, "): totalClusters=", totalClusters,
                " numInstances=", numInstances, " totalVertices=", totalVertices));

            // Readback BLAS pointers
            std::vector<nvrhi::GpuVirtualAddress> blasAddrs = accels.blasPtrsBuffer.Download(commandList);
            Logger::warn(str::format("  BLAS-PTRS: ", blasAddrs.size(), " entries (buffer elements=", accels.blasPtrsBuffer.GetNumElements(), ")"));
            for (uint32_t i = 0; i < std::min((uint32_t)blasAddrs.size(), 10u); ++i) {
                Logger::warn(str::format("  blasPtr[", i, "] = 0x", std::hex, blasAddrs[i], std::dec));
            }
            // Count zero BLAS pointers
            uint32_t zeroCount = 0;
            for (auto addr : blasAddrs) { if (addr == 0) zeroCount++; }
            Logger::warn(str::format("  BLAS zero-ptr count: ", zeroCount, " / ", blasAddrs.size()));

            // Readback BLAS sizes
            std::vector<uint32_t> blasSizes = accels.blasSizesBuffer.Download(commandList);
            Logger::warn(str::format("  BLAS-SIZES: ", blasSizes.size(), " entries"));
            for (uint32_t i = 0; i < std::min((uint32_t)blasSizes.size(), 10u); ++i) {
                Logger::warn(str::format("  blasSize[", i, "] = ", blasSizes[i]));
            }

            // Readback CLAS pointers (destination addresses)
            std::vector<nvrhi::GpuVirtualAddress> clasAddrs = accels.clasPtrsBuffer.Download(commandList);
            Logger::warn(str::format("  CLAS-PTRS: ", clasAddrs.size(), " entries"));
            for (uint32_t i = 0; i < std::min((uint32_t)clasAddrs.size(), 10u); ++i) {
                Logger::warn(str::format("  clasPtr[", i, "] = 0x", std::hex, clasAddrs[i], std::dec));
            }

            // Readback first few vertex positions
            Logger::warn(str::format("  CPU-VERTICES (first 10 of ", totalVertices, "):"));
            for (uint32_t i = 0; i < std::min(totalVertices, 10u); ++i) {
                Logger::warn(str::format("  vtx[", i, "] = (", vertexData[i*3+0], ", ", vertexData[i*3+1], ", ", vertexData[i*3+2], ")"));
            }

            // Log CLAS args details for first few clusters
            Logger::warn("  CLAS instantiate args (first 5):");
            for (uint32_t i = 0; i < std::min(totalClusters, 5u); ++i) {
                const auto& sel = allSelectedClusters[i];
                const MeshletInstance& mi = instances[sel.instanceIdx];
                Logger::warn(str::format("  cluster[", i, "] instIdx=", sel.instanceIdx,
                    " meshletIdx=", sel.meshletIdx,
                    " triCount=", mi.templates->meshletTriangleCounts[sel.meshletIdx],
                    " vtxCount=", mi.templates->meshletVertexCounts[sel.meshletIdx],
                    " templateAddr=0x", std::hex, mi.templates->templateAddresses[sel.meshletIdx], std::dec));
            }
        }
    }

    // Update stats
    stats.allocated.m_numClusters = totalClusters;
    stats.allocated.m_numTriangles = 0;
    for (const auto& sc : allSelectedClusters) {
        const MeshletInstance& mi = instances[sc.instanceIdx];
        stats.allocated.m_numTriangles += mi.templates->meshletTriangleCounts[sc.meshletIdx];
    }
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


