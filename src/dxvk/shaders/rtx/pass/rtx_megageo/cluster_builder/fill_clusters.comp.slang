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

// NOTE: This shader has 16 permutations (displacement × normals × surface type) that are compiled
// at runtime via Donut ShaderFactory. The single variant below is a placeholder for the build
// system; actual runtime compilation uses macro defines.
//!variant fill_clusters.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/utils/shader_debug.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_clusters_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/copy_cluster_offset_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_clusters_binding_indices.h"
#include "rtx/pass/rtx_megageo/cluster_builder/material_data.h"
#include "rtxmg/subdivision/vertex.h"

// Default compile-time defines for static build (can be overridden at runtime)
// Note: subdivision_eval.hlsli (included below) defines SURFACE_TYPE_* constants
#ifndef DISPLACEMENT_MAPS
#define DISPLACEMENT_MAPS 0
#endif

#ifndef VERTEX_NORMALS
#define VERTEX_NORMALS 0
#endif

#include "rtxmg/subdivision/subdivision_eval.hlsli"

#define ENABLE_GROUP_PUREBSPLINE 1

// No other way to pass groupshared array to SubdivisionSurfaceEvaluator
#if SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE && ENABLE_GROUP_PUREBSPLINE
groupshared float3 s_patchControlPoints[kPatchSize * kFillClustersVerticesWaves];
#define GROUP_SHARED_CONTROL_POINTS(waveIndex, controlPointIndex) s_patchControlPoints[waveIndex * kPatchSize + controlPointIndex]
#endif
#include "rtx/pass/rtx_megageo/cluster_builder/tessellator_constants.h"
#include "rtxmg/subdivision/subdivision_plan_hlsl.h"
#include "rtx/pass/rtx_megageo/cluster_builder/cluster.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_instantiate_template_args_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/displacement.hlsli"
#include "rtxmg/subdivision/osd_ports/tmr/surfaceDescriptor.h"

// Slang maps HLSL registers to Vulkan bindings: t→0-99, s→100-199, u→200-299, b→300-399
// SRVs (t0-t19 → bindings 0-19)
StructuredBuffer<GridSampler> t_GridSamplers : register(t0);
StructuredBuffer<uint2> t_ClusterOffsetCounts : register(t1);
StructuredBuffer<Cluster> t_Clusters : register(t2);
StructuredBuffer<float3> t_VertexControlPoints : register(t3);
StructuredBuffer<SurfaceDescriptor> t_VertexSurfaceDescriptors : register(t4);
StructuredBuffer<Index> t_VertexControlPointIndices : register(t5);
StructuredBuffer<uint32_t> t_VertexPatchPointsOffsets : register(t6);
StructuredBuffer<SubdivisionPlanHLSL> t_Plans : register(t7);
StructuredBuffer<uint32_t> t_SubpatchTrees : register(t8);
StructuredBuffer<Index> t_PatchPointIndices : register(t9);
StructuredBuffer<float> t_StencilMatrix : register(t10);
StructuredBuffer<float3> t_VertexPatchPoints : register(t11);
StructuredBuffer<GeometryData> t_GeometryData : register(t12);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t13);
StructuredBuffer<uint16_t> t_SurfaceToGeometryIndex : register(t14);
StructuredBuffer<LinearSurfaceDescriptor> t_TexCoordSurfaceDescriptors : register(t15);
StructuredBuffer<Index> t_TexCoordControlPointIndices : register(t16);
StructuredBuffer<uint32_t> t_TexCoordPatchPointsOffsets : register(t17);
StructuredBuffer<float2> t_TexCoordPatchPoints : register(t18);
StructuredBuffer<float2> t_TexCoords : register(t19);

// Bindless texture array for displacement mapping (t20 → binding 20)
Texture2D bindlessTextures[] : register(t20);

// Sampler (s0 → binding 100)
SamplerState s_DisplacementSampler : register(s0);

// UAVs (u0-u3 → bindings 200-203)
RWStructuredBuffer<float3> u_ClusterVertexPositions : register(u0);
RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u1);
RWStructuredBuffer<ShaderDebugElement> u_Debug : register(u2);
#if VERTEX_NORMALS
RWStructuredBuffer<float3> u_ClusterVertexNormals : register(u3);
#endif

// Constant buffer (b0 → binding 300)
ConstantBuffer<FillClustersParams> g_TessParams : register(b0);

void GathererWriteLimit(LimitFrame vertexLimit, Cluster cluster, uint32_t vertexIndex)
{
    u_ClusterVertexPositions[cluster.nVertexOffset + vertexIndex] = quantize(vertexLimit.p, g_TessParams.quantNBits);
}

#if VERTEX_NORMALS
void GathererWriteNormal(LimitFrame vertexLimit, Cluster cluster, uint32_t vertexIndex, SubdivisionEvaluatorHLSL subd)
{
    // Use the subdivision evaluator's robust normal calculation method
    float3 normal = subd.CalculateLimitFrameNormal(vertexLimit);
    u_ClusterVertexNormals[cluster.nVertexOffset + vertexIndex] = normal;
}
#endif

void GathererWriteTexcoord(TexCoordLimitFrame texcoord, uint32_t clusterIndex, uint32_t cornerIndex)
{
    u_ClusterShadingData[clusterIndex].m_texcoords[cornerIndex] = texcoord.uv;
}

// Entry point function - called by main()
void FillClustersMain(uint3 threadIdx, uint3 groupIdx)
{
    uint32_t iLane = threadIdx.x % kFillClustersVerticesLanes;
    uint32_t iWave = threadIdx.x / kFillClustersVerticesLanes;
    const uint32_t groupClusterIndex = groupIdx.x * kFillClustersVerticesWaves + iWave;

#if SURFACE_TYPE == SURFACE_TYPE_ALL
    uint32_t kDispatchTypeIndex = (uint32_t)ClusterDispatchType::All;
#elif SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE
    uint32_t kDispatchTypeIndex = (uint32_t)ClusterDispatchType::PureBSpline;
#elif SURFACE_TYPE == SURFACE_TYPE_REGULARBSPLINE
    uint32_t kDispatchTypeIndex = (uint32_t)ClusterDispatchType::RegularBSpline;
#elif SURFACE_TYPE == SURFACE_TYPE_LIMIT
    uint32_t kDispatchTypeIndex = (uint32_t)ClusterDispatchType::Limit;
#endif

    uint2 offsetCount = t_ClusterOffsetCounts[g_TessParams.instanceIndex * (uint32_t)ClusterDispatchType::NumTypes + kDispatchTypeIndex];
    if (groupClusterIndex >= offsetCount.y)
        return; // early out waves beyond cluster array end

    uint32_t clusterIndex = groupClusterIndex + offsetCount.x;
    const Cluster rCluster = t_Clusters[clusterIndex];

    // Shader debug only
    uint surfaceIndex = rCluster.iSurface;
    uint linearClusterOffset = (rCluster.offset.y * rCluster.sizeX) + rCluster.offset.x;

    SHADER_DEBUG_INIT(u_Debug, uint3(g_TessParams.debugSurfaceIndex, g_TessParams.debugClusterIndex, g_TessParams.debugLaneIndex), uint3(surfaceIndex, linearClusterOffset, iLane));

    const uint32_t iSurface = rCluster.iSurface;
    const GridSampler rSampler = t_GridSamplers[iSurface];

    SubdivisionEvaluatorHLSL subd;
    subd.m_surfaceIndex = iSurface;

    subd.m_isolationLevel = uint16_t(g_TessParams.isolationLevel);
    subd.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
    subd.m_plans = t_Plans;
    subd.m_subpatchTrees = t_SubpatchTrees;
    subd.m_vertexPatchPointIndices = t_PatchPointIndices;
    subd.m_stencilMatrix = t_StencilMatrix;

    subd.m_vertexControlPointIndices = t_VertexControlPointIndices;
    subd.m_vertexControlPoints = t_VertexControlPoints;
    subd.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
    subd.m_vertexPatchPoints = t_VertexPatchPoints;

#if DISPLACEMENT_MAPS
    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = t_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;

    float displacementScale;
    int displacementTexIndex;

    uint32_t geometryIndex = t_SurfaceToGeometryIndex[iSurface] + g_TessParams.firstGeometryIndex;
    GeometryData geometry = t_GeometryData[geometryIndex];
    MaterialConstants material = t_MaterialConstants[geometry.materialIndex];

    GetDisplacement(material, g_TessParams.globalDisplacementScale, displacementTexIndex, displacementScale);
#endif

#if SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE && ENABLE_GROUP_PUREBSPLINE
    subd.PrefetchPatchControlPoints(iWave, iLane);
#endif

    {
        // wave wide loop
        for (uint16_t pointIndex = (uint16_t)iLane; pointIndex < rCluster.VerticesPerCluster(); pointIndex += kFillClustersVerticesLanes)
        {
            float2 uv = rSampler.UV(rCluster.Linear2Idx2D(pointIndex) + rCluster.offset, (ClusterPattern)g_TessParams.clusterPattern);

            // always do the non-displaced evaluation first.  Displacement maps will perturb this calculation below
            LimitFrame limit;
            if (g_TessParams.disableSubdivision != 0)
            {
                // Bypass subdivision: use bilinear interpolation of control points
                limit = subd.EvaluateBilinear(uv);
            }
            else
            {
#if SURFACE_TYPE == SURFACE_TYPE_ALL
                limit = subd.Evaluate(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE
            #if ENABLE_GROUP_PUREBSPLINE
                limit = subd.EvaluatePureBsplinePatchGroupShared(iWave, uv);
            #else
                limit = subd.EvaluatePureBsplinePatch(uv);
            #endif
#elif SURFACE_TYPE == SURFACE_TYPE_REGULARBSPLINE
                limit = subd.EvaluateBsplinePatch(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_LIMIT
                limit = subd.EvaluateLimitSurface(uv);
#endif
            }

#if DISPLACEMENT_MAPS
            if (displacementTexIndex >= 0)
            {
                Texture2D displacementTexture = bindlessTextures[displacementTexIndex];

                float du = rSampler.DU(uv);
                float dv = rSampler.DV(uv);

                limit = DoDisplacement(texcoordEval,
                        limit, iSurface, uv, du, dv,
                        displacementTexture,
                        s_DisplacementSampler, displacementScale);
            }
#endif

            GathererWriteLimit(limit, rCluster, pointIndex);

            // DEBUG: Log cluster metadata + vertex positions to debug buffer (instance 0 only)
            // Slots 0-7: reserved for compute_cluster_tiling debug (payloadType=3)
            // Slots 8: offsetCount for this fill_clusters dispatch
            // Slots 9-15: cluster metadata (iSurface, nVertexOffset, etc.) for first 7 clusters
            if (g_TessParams.instanceIndex == 0 && groupClusterIndex == 0 && pointIndex == 0) {
                // Slot 8: store offsetCount for this instance
                u_Debug[8].uintData = uint4(offsetCount.x, offsetCount.y, g_TessParams.instanceIndex, kDispatchTypeIndex);
                u_Debug[8].floatData = float4(limit.p.x, limit.p.y, limit.p.z, 0);
                u_Debug[8].payloadType = 102; // Mark as offset data (102 to avoid conflict with ShaderDebug PayloadType_Uint2=2)
                u_Debug[8].lineNumber = 0;
            }
            if (g_TessParams.instanceIndex == 0 && pointIndex == 0 && groupClusterIndex < 7) {
                uint32_t debugIdx = 9 + groupClusterIndex;
                // Log cluster metadata: iSurface, nVertexOffset, sizeX, sizeY, offset, clusterIndex
                u_Debug[debugIdx].uintData = uint4(iSurface, rCluster.nVertexOffset, rCluster.sizeX | (rCluster.sizeY << 16), clusterIndex);
                u_Debug[debugIdx].floatData = float4(limit.p.x, limit.p.y, limit.p.z, asfloat(rCluster.offset.x | (rCluster.offset.y << 16)));
                u_Debug[debugIdx].payloadType = 104; // Mark as fill_clusters cluster metadata (104 to avoid conflict with ShaderDebug PayloadType_Uint4=4)
                u_Debug[debugIdx].lineNumber = 0;
            }

#if VERTEX_NORMALS
            GathererWriteNormal(limit, rCluster, pointIndex, subd);
#endif
        }
    }
}

// Texcoord entry point function - needs separate shader variant for runtime
void FillClustersTexcoordsMain(uint3 threadIdx, uint3 groupIdx)
{
    const uint32_t clusterThreadIdx = threadIdx.x % kFillClustersTexcoordsThreadsX;
    const uint32_t cornerIdx = threadIdx.x / kFillClustersTexcoordsThreadsX;
    const uint32_t groupClusterIndex = groupIdx.x * kFillClustersTexcoordsThreadsX + clusterThreadIdx;
    uint2 offsetCount = t_ClusterOffsetCounts[g_TessParams.instanceIndex * (uint32_t)ClusterDispatchType::NumTypes + (uint32_t)ClusterDispatchType::All];
    if (groupClusterIndex >= offsetCount.y)
        return; // early out waves beyond cluster array end

    uint32_t clusterIndex = groupClusterIndex + offsetCount.x;
    const Cluster rCluster = t_Clusters[clusterIndex];
    const uint32_t iSurface = rCluster.iSurface;
    const GridSampler rSampler = t_GridSamplers[iSurface];

    // Extra shading data: texcoords on corners of surfaces (patches). This might not exactly match texcoords used in displacement
    // above if the subd evaluator is cubic.

    // NOTE: For higher-order texture coord eval, these per-cluster corner UVs could be moved to a
    // dedicated array in the coarse rasterizer. Current linear evaluation works correctly as-is.
    const float2 kSurfaceUVs[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = t_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;

    TexCoordLimitFrame texcoord = texcoordEval.EvaluateLinearSubd(kSurfaceUVs[cornerIdx], iSurface);

    GathererWriteTexcoord(texcoord, clusterIndex, cornerIdx);
}

// Placeholder main() for build system - actual runtime compilation uses FillClustersMain
[shader("compute")]
[numthreads(kFillClustersVerticesLanes * kFillClustersVerticesWaves, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    FillClustersMain(threadIdx, groupIdx);
}
