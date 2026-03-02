/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// GPU-driven meshlet cluster fill shader.
// Replaces CPU loops that computed per-tri expanded vertices, normals,
// CLAS instantiation args, CLAS destination addresses, and shading data.
//
// Each thread processes one selected cluster:
// 1. Copies per-tri expanded vertices from persistent buffer to per-frame buffer
// 2. Copies per-tri expanded normals from persistent buffer to per-frame buffer
// 3. Copies unique vertices from persistent template buffer to per-frame buffer
// 4. Writes CLAS instantiation args (IndirectInstantiateTemplateArgs)
// 5. Writes CLAS destination address
// 6. Writes ClusterShadingData

//!variant fill_meshlet_clusters.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_meshlet_clusters_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/cluster.h"

// SRVs (t0-t2 → bindings 0-2)
StructuredBuffer<MeshletClusterGPUInfo> t_ClusterInfos : register(t0);
StructuredBuffer<float3> t_PersistentExpandedVertices : register(t1);
StructuredBuffer<float3> t_PersistentExpandedNormals : register(t2);
StructuredBuffer<float3> t_PersistentUniqueVertices : register(t3);

// UAVs (u0-u4 → bindings 200-204)
RWStructuredBuffer<float3> u_ClusterVertexPositions : register(u0);
RWStructuredBuffer<float3> u_ClusterVertexNormals : register(u1);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> u_IndirectArgData : register(u2);
RWStructuredBuffer<nvrhi::GpuVirtualAddress> u_ClasAddresses : register(u3);
RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u4);

layout(push_constant)
ConstantBuffer<FillMeshletClustersParams> g_Params;

[shader("compute")]
[numthreads(kFillMeshletClustersThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t ci = threadIdx.x;
    if (ci >= g_Params.numClusters)
        return;

    MeshletClusterGPUInfo info = t_ClusterInfos[ci];

    // 1. Copy per-tri expanded vertices from persistent to per-frame buffer
    uint32_t expandedVtxCount = info.triCount * 3;
    for (uint32_t v = 0; v < expandedVtxCount; ++v) {
        u_ClusterVertexPositions[info.perFrameExpandedVtxOffset + v] =
            t_PersistentExpandedVertices[info.persistentExpandedVtxOffset + v];
    }

    // 2. Copy per-tri expanded normals from persistent to per-frame buffer
    for (uint32_t v = 0; v < expandedVtxCount; ++v) {
        u_ClusterVertexNormals[info.perFrameExpandedVtxOffset + v] =
            t_PersistentExpandedNormals[info.persistentExpandedVtxOffset + v];
    }

    // 3. Copy unique vertices from persistent template buffer to per-frame unique section
    uint32_t uniqueVtxBase = g_Params.totalExpandedVertices;
    for (uint32_t v = 0; v < info.uniqueVtxCount; ++v) {
        u_ClusterVertexPositions[uniqueVtxBase + info.perFrameUniqueVtxOffset + v] =
            t_PersistentUniqueVertices[info.persistentUniqueVtxOffset + v];
    }

    // 4. Write CLAS instantiation args
    // vertexBuffer.startAddress points to per-frame unique section (matching sample pattern)
    nvrhi::GpuVirtualAddress clasVtxAddr = g_Params.perFrameVertexAddr
        + nvrhi::GpuVirtualAddress((uniqueVtxBase + info.perFrameUniqueVtxOffset) * 12);

    nvrhi::rt::cluster::IndirectInstantiateTemplateArgs clasArgs =
        (nvrhi::rt::cluster::IndirectInstantiateTemplateArgs)0;
    clasArgs.clusterIdOffset = info.clusterIndex;
    clasArgs.geometryIndexOffsetPacked = info.clusterIndex & 0xFFFFFF;
    clasArgs.clusterTemplate = info.templateAddr;
    clasArgs.vertexBuffer.startAddress = clasVtxAddr;
    clasArgs.vertexBuffer.strideInBytes = 12; // sizeof(float3) with scalar layout
    u_IndirectArgData[ci] = clasArgs;

    // 5. Write CLAS destination address
    u_ClasAddresses[ci] = info.clasDestAddr;

    // 6. Write ClusterShadingData
    // m_clusterSizeX/Y = 0 signals meshlet path to hit shader
    ClusterShadingData shadingData = (ClusterShadingData)0;
    shadingData.m_surfaceId = info.surfaceId;
    shadingData.m_vertexOffset = info.perFrameExpandedVtxOffset;
    u_ClusterShadingData[ci] = shadingData;
}
