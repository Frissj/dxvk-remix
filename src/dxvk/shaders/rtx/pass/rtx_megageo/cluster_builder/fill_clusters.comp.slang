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
#if SURFACE_TYPE == SURFACE_TYPE_ALL
            LimitFrame limit = subd.Evaluate(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE
        #if ENABLE_GROUP_PUREBSPLINE
            LimitFrame limit = subd.EvaluatePureBsplinePatchGroupShared(iWave, uv);
        #else
            LimitFrame limit = subd.EvaluatePureBsplinePatch(uv);
        #endif
#elif SURFACE_TYPE == SURFACE_TYPE_REGULARBSPLINE
            LimitFrame limit = subd.EvaluateBsplinePatch(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_LIMIT
            LimitFrame limit = subd.EvaluateLimitSurface(uv);
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
                u_Debug[debugIdx].uintData = uint4(iSurface, rCluster.nVertexOffset, rCluster.sizeX | (uint(rCluster.sizeY) << 16), clusterIndex);
                u_Debug[debugIdx].floatData = float4(limit.p.x, limit.p.y, limit.p.z, asfloat(rCluster.offset.x | (uint(rCluster.offset.y) << 16)));
                u_Debug[debugIdx].payloadType = 104; // Mark as fill_clusters cluster metadata (104 to avoid conflict with ShaderDebug PayloadType_Uint4=4)
                u_Debug[debugIdx].lineNumber = 0;
            }

            // DIAG: Dump control point data chain for first cluster, first vertex of instance 0
            // Slots 16-23: control point diagnostic data (payloadType=105)
            if (g_TessParams.instanceIndex == 0 && groupClusterIndex == 0 && pointIndex == 0) {
                SurfaceDescriptor desc = t_VertexSurfaceDescriptors[iSurface];

                // Slot 16: surface descriptor raw data
                u_Debug[16].uintData = uint4(iSurface, desc.field0, desc.firstControlPoint, 0);
                u_Debug[16].floatData = float4(0, 0, 0, 0);
                u_Debug[16].payloadType = 105;
                u_Debug[16].lineNumber = 1;

                // Slot 17: control point indices [0..3]
                uint cpi0 = t_VertexControlPointIndices[desc.firstControlPoint + 0];
                uint cpi1 = t_VertexControlPointIndices[desc.firstControlPoint + 1];
                uint cpi2 = t_VertexControlPointIndices[desc.firstControlPoint + 2];
                uint cpi3 = t_VertexControlPointIndices[desc.firstControlPoint + 3];
                u_Debug[17].uintData = uint4(cpi0, cpi1, cpi2, cpi3);
                u_Debug[17].floatData = float4(asfloat(desc.firstControlPoint), 0, 0, 0);
                u_Debug[17].payloadType = 105;
                u_Debug[17].lineNumber = 2;

                // Slot 18: control point position p00 = positions[cpi0]
                float3 p00 = t_VertexControlPoints[cpi0];
                u_Debug[18].floatData = float4(p00.x, p00.y, p00.z, 0);
                u_Debug[18].uintData = uint4(cpi0, 0, 0, 0);
                u_Debug[18].payloadType = 105;
                u_Debug[18].lineNumber = 3;

                // Slot 19: control point position p10 = positions[cpi1]
                float3 p10 = t_VertexControlPoints[cpi1];
                u_Debug[19].floatData = float4(p10.x, p10.y, p10.z, 0);
                u_Debug[19].uintData = uint4(cpi1, 0, 0, 0);
                u_Debug[19].payloadType = 105;
                u_Debug[19].lineNumber = 4;

                // Slot 20: control point position p11 = positions[cpi2]
                float3 p11 = t_VertexControlPoints[cpi2];
                u_Debug[20].floatData = float4(p11.x, p11.y, p11.z, 0);
                u_Debug[20].uintData = uint4(cpi2, 0, 0, 0);
                u_Debug[20].payloadType = 105;
                u_Debug[20].lineNumber = 5;

                // Slot 21: control point position p01 = positions[cpi3]
                float3 p01 = t_VertexControlPoints[cpi3];
                u_Debug[21].floatData = float4(p01.x, p01.y, p01.z, 0);
                u_Debug[21].uintData = uint4(cpi3, 0, 0, 0);
                u_Debug[21].payloadType = 105;
                u_Debug[21].lineNumber = 6;

                // Slot 22: buffer dimension info
                uint numDescriptors, descStride;
                t_VertexSurfaceDescriptors.GetDimensions(numDescriptors, descStride);
                uint numCPIndices, cpiStride;
                t_VertexControlPointIndices.GetDimensions(numCPIndices, cpiStride);
                uint numPositions, posStride;
                t_VertexControlPoints.GetDimensions(numPositions, posStride);
                u_Debug[22].uintData = uint4(numDescriptors, numCPIndices, numPositions, descStride);
                u_Debug[22].floatData = float4(asfloat(cpiStride), asfloat(posStride), 0, 0);
                u_Debug[22].payloadType = 105;
                u_Debug[22].lineNumber = 7;

                // Slot 23: evaluated limit position at uv=(0,0) for comparison
                u_Debug[23].floatData = float4(limit.p.x, limit.p.y, limit.p.z, 0);
                u_Debug[23].uintData = uint4(asuint(uv.x), asuint(uv.y), 0, 0);
                u_Debug[23].payloadType = 105;
                u_Debug[23].lineNumber = 8;

                // Slots 24-30: Compare bilinear vs Evaluate() at multiple UV values
                {
                    SubdivisionEvaluatorHLSL cmpEval;
                    cmpEval.m_surfaceIndex = iSurface;
                    cmpEval.m_isolationLevel = g_TessParams.isolationLevel;
                    cmpEval.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
                    cmpEval.m_plans = t_Plans;
                    cmpEval.m_subpatchTrees = t_SubpatchTrees;
                    cmpEval.m_vertexPatchPointIndices = t_PatchPointIndices;
                    cmpEval.m_stencilMatrix = t_StencilMatrix;
                    cmpEval.m_vertexControlPointIndices = t_VertexControlPointIndices;
                    cmpEval.m_vertexControlPoints = t_VertexControlPoints;
                    cmpEval.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
                    cmpEval.m_vertexPatchPoints = t_VertexPatchPoints;

                    SurfaceDescriptor cmpDesc = t_VertexSurfaceDescriptors[iSurface];
                    uint planIdx = cmpDesc.GetSubdivisionPlanIndex();
                    bool isPure = (planIdx == 0);
                    bool isBSp = !isPure && cmpEval.IsBSplinePatch();
                    uint evalPath = isPure ? 0 : (isBSp ? 1 : 2);

                    // Slot 24: path info + bilinear vs Evaluate at UV=(0.5, 0.5)
                    LimitFrame bilin05 = cmpEval.EvaluateBilinear(float2(0.5, 0.5));
                    LimitFrame subd05 = cmpEval.Evaluate(float2(0.5, 0.5));
                    u_Debug[24].floatData = float4(bilin05.p.x, bilin05.p.y, bilin05.p.z, 0);
                    u_Debug[24].uintData = uint4(evalPath, planIdx, asuint(subd05.p.x - bilin05.p.x), asuint(subd05.p.y - bilin05.p.y));
                    u_Debug[24].payloadType = 106;
                    u_Debug[24].lineNumber = 1;

                    // Slot 25: Evaluate result at UV=(0.5, 0.5)
                    u_Debug[25].floatData = float4(subd05.p.x, subd05.p.y, subd05.p.z, 0);
                    u_Debug[25].uintData = uint4(asuint(subd05.p.z - bilin05.p.z), 0, 0, 0);
                    u_Debug[25].payloadType = 106;
                    u_Debug[25].lineNumber = 2;

                    // Slot 26: bilinear vs Evaluate at UV=(0.25, 0.75)
                    LimitFrame bilin2575 = cmpEval.EvaluateBilinear(float2(0.25, 0.75));
                    LimitFrame subd2575 = cmpEval.Evaluate(float2(0.25, 0.75));
                    u_Debug[26].floatData = float4(bilin2575.p.x, bilin2575.p.y, bilin2575.p.z, 0);
                    u_Debug[26].uintData = uint4(asuint(subd2575.p.x - bilin2575.p.x), asuint(subd2575.p.y - bilin2575.p.y), asuint(subd2575.p.z - bilin2575.p.z), 0);
                    u_Debug[26].payloadType = 106;
                    u_Debug[26].lineNumber = 3;

                    // Slot 27: Evaluate at UV=(0.25, 0.75)
                    u_Debug[27].floatData = float4(subd2575.p.x, subd2575.p.y, subd2575.p.z, 0);
                    u_Debug[27].uintData = uint4(0, 0, 0, 0);
                    u_Debug[27].payloadType = 106;
                    u_Debug[27].lineNumber = 4;

                    // Slot 28: Check a DIFFERENT surface (surface 0) - bilinear vs Evaluate
                    {
                        SubdivisionEvaluatorHLSL cmpEval0;
                        cmpEval0.m_surfaceIndex = 0;
                        cmpEval0.m_isolationLevel = g_TessParams.isolationLevel;
                        cmpEval0.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
                        cmpEval0.m_plans = t_Plans;
                        cmpEval0.m_subpatchTrees = t_SubpatchTrees;
                        cmpEval0.m_vertexPatchPointIndices = t_PatchPointIndices;
                        cmpEval0.m_stencilMatrix = t_StencilMatrix;
                        cmpEval0.m_vertexControlPointIndices = t_VertexControlPointIndices;
                        cmpEval0.m_vertexControlPoints = t_VertexControlPoints;
                        cmpEval0.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
                        cmpEval0.m_vertexPatchPoints = t_VertexPatchPoints;

                        SurfaceDescriptor desc0 = t_VertexSurfaceDescriptors[0];
                        uint planIdx0 = desc0.GetSubdivisionPlanIndex();
                        bool isPure0 = (planIdx0 == 0);
                        bool isBSp0 = !isPure0 && cmpEval0.IsBSplinePatch();
                        uint path0 = isPure0 ? 0 : (isBSp0 ? 1 : 2);

                        LimitFrame bilin0 = cmpEval0.EvaluateBilinear(float2(0.5, 0.5));
                        LimitFrame subd0 = cmpEval0.Evaluate(float2(0.5, 0.5));
                        u_Debug[28].floatData = float4(bilin0.p.x - subd0.p.x, bilin0.p.y - subd0.p.y, bilin0.p.z - subd0.p.z, 0);
                        u_Debug[28].uintData = uint4(path0, planIdx0, desc0.firstControlPoint, 0);
                        u_Debug[28].payloadType = 106;
                        u_Debug[28].lineNumber = 5;
                    }

                    // Slot 29: numCP for current plan, HasSharpness, actual sharpness from node
                    {
                        SubdivisionPlanContext dbgPlan = cmpEval.GetPlan();
                        SubdivisionNode rootNode = dbgPlan.GetRootNode();
                        NodeDescriptor nDesc = rootNode.GetDesc();
                        bool hasSharp = nDesc.HasSharpness();
                        float sharp = hasSharp ? rootNode.GetSharpness() : 0.0f;
                        float computedSharp = computeSingleCreaseSharpness(rootNode, nDesc, (uint16_t)nDesc.GetDepth(), g_TessParams.isolationLevel);
                        u_Debug[29].floatData = float4(sharp, computedSharp, 0, 0);
                        u_Debug[29].uintData = uint4(hasSharp ? 1 : 0, nDesc.GetBoundaryMask(), dbgPlan.m_data.numControlPoints, evalPath);
                        u_Debug[29].payloadType = 106;
                        u_Debug[29].lineNumber = 6;
                    }

                    // Slot 30: BSpline weights at UV=(0.5,0.5) - show positions 5,6,9,10 weights
                    {
                        SubdivisionPlanContext dbgPlan = cmpEval.GetPlan();
                        float wP[16], wDs[16], wDt[16];
                        uint16_t quadrant = 0;
                        SubdivisionNode node = dbgPlan.EvaluateBasis(float2(0.5, 0.5), wP, wDs, wDt, quadrant, g_TessParams.isolationLevel);
                        // Show weights at the 4 inner positions (5,6,9,10) and sum of outer ring
                        float outerSum = 0;
                        for (int i = 0; i < 16; i++) {
                            if (i != 5 && i != 6 && i != 9 && i != 10) outerSum += wP[i];
                        }
                        u_Debug[30].floatData = float4(wP[5], wP[6], wP[9], wP[10]);
                        u_Debug[30].uintData = uint4(asuint(outerSum), asuint(wP[0]), asuint(wP[3]), asuint(wP[15]));
                        u_Debug[30].payloadType = 106;
                        u_Debug[30].lineNumber = 7;
                    }
                }
            }

            // Slots 40-55: LIMIT SURFACE diagnostics (first Limit surface cluster, any instance)
            // Uses atomic CAS on pad0.x to ensure exactly one thread writes all slots
            if (pointIndex == 0 && !subd.IsPureBSplinePatch() && !subd.IsBSplinePatch()) {
                uint origFlag;
                InterlockedCompareExchange(u_Debug[40].pad0.x, 0, 1, origFlag);
                if (origFlag == 0) {

                    SubdivisionPlanContext limitPlan = subd.GetPlan();
                    SurfaceDescriptor limitDesc = subd.GetSurfaceDesc();
                    uint32_t numPP = limitPlan.GetTreeDescriptor().GetNumPatchPoints(subd.m_isolationLevel);
                    uint32_t ppOffset = t_VertexPatchPointsOffsets[iSurface];
                    uint32_t ppOffsetNext = t_VertexPatchPointsOffsets[iSurface + 1]; // next surface's offset for size calc

                    // Slot 40: Limit surface identity + plan info
                    u_Debug[40].uintData = uint4(iSurface, limitDesc.GetSubdivisionPlanIndex(),
                        limitPlan.m_data.numControlPoints, numPP);
                    u_Debug[40].floatData = float4(limit.p.x, limit.p.y, limit.p.z, 0);
                    u_Debug[40].payloadType = 110;
                    u_Debug[40].lineNumber = 0;

                    // Slot 41: surface descriptor + patch points offset
                    u_Debug[41].uintData = uint4(limitDesc.field0, limitDesc.firstControlPoint, ppOffset, ppOffsetNext);
                    u_Debug[41].floatData = float4(asfloat(limitPlan.m_data.stencilMatrixOffset),
                        asfloat(limitPlan.m_data.stencilMatrixSize),
                        asfloat(limitPlan.m_data.treeOffset), 0);
                    u_Debug[41].payloadType = 110;
                    u_Debug[41].lineNumber = 1;

                    // Slot 42: cluster metadata for this Limit cluster
                    u_Debug[42].uintData = uint4(clusterIndex, rCluster.nVertexOffset,
                        rCluster.sizeX | (uint(rCluster.sizeY) << 16),
                        groupClusterIndex);
                    u_Debug[42].floatData = float4(asfloat(rCluster.offset.x | (uint(rCluster.offset.y) << 16)),
                        asfloat(offsetCount.x), asfloat(offsetCount.y), 0);
                    u_Debug[42].payloadType = 110;
                    u_Debug[42].lineNumber = 2;

                    // Slots 43-46: First 4 patch points from the buffer (what compute_cluster_tiling wrote)
                    for (uint pp = 0; pp < 4 && pp < numPP; pp++) {
                        float3 patchPt = t_VertexPatchPoints[ppOffset + pp];
                        u_Debug[43 + pp].floatData = float4(patchPt.x, patchPt.y, patchPt.z, 0);
                        u_Debug[43 + pp].uintData = uint4(ppOffset + pp, pp, numPP, 0);
                        u_Debug[43 + pp].payloadType = 110;
                        u_Debug[43 + pp].lineNumber = 3 + pp;
                    }

                    // Slots 47-50: First 4 control points for this Limit surface
                    uint numCP = min(limitPlan.m_data.numControlPoints, 4u);
                    for (uint cp = 0; cp < numCP; cp++) {
                        uint cpi = t_VertexControlPointIndices[limitDesc.firstControlPoint + cp];
                        float3 cpPos = t_VertexControlPoints[cpi];
                        u_Debug[47 + cp].floatData = float4(cpPos.x, cpPos.y, cpPos.z, 0);
                        u_Debug[47 + cp].uintData = uint4(cpi, cp, limitDesc.firstControlPoint + cp, limitPlan.m_data.numControlPoints);
                        u_Debug[47 + cp].payloadType = 110;
                        u_Debug[47 + cp].lineNumber = 7 + cp;
                    }

                    // Slots 51-54: First 4 stencil matrix entries for patch point 0
                    if (numPP > 0) {
                        uint stencilBase = limitPlan.m_data.stencilMatrixOffset;
                        uint numCPForStencil = limitPlan.m_data.numControlPoints;
                        for (uint s = 0; s < 4 && s < numCPForStencil; s++) {
                            float stencilVal = t_StencilMatrix[stencilBase + s];
                            u_Debug[51 + s].floatData = float4(stencilVal, 0, 0, 0);
                            u_Debug[51 + s].uintData = uint4(stencilBase + s, s, numCPForStencil, stencilBase);
                            u_Debug[51 + s].payloadType = 110;
                            u_Debug[51 + s].lineNumber = 11 + s;
                        }
                    }

                    // Slot 55: Evaluate at (0.5,0.5) for this Limit surface specifically
                    LimitFrame limitEval = subd.EvaluateLimitSurface(float2(0.5, 0.5));
                    u_Debug[55].floatData = float4(limitEval.p.x, limitEval.p.y, limitEval.p.z, 0);
                    u_Debug[55].uintData = uint4(iSurface, numPP, ppOffset, g_TessParams.instanceIndex);
                    u_Debug[55].payloadType = 110;
                    u_Debug[55].lineNumber = 15;
                }
            }

            // Slots 31-38: UV and position for first 8 vertices of first cluster (instance 0)
            // This helps verify the UV computation and bilinear interpolation across the grid
            if (g_TessParams.instanceIndex == 0 && groupClusterIndex == 0 && pointIndex < 8) {
                uint32_t slot = 31 + pointIndex;
                u_Debug[slot].floatData = float4(limit.p.x, limit.p.y, limit.p.z, 0);
                u_Debug[slot].uintData = uint4(asuint(uv.x), asuint(uv.y), pointIndex,
                    rCluster.sizeX | (uint32_t(rCluster.sizeY) << 16));
                u_Debug[slot].payloadType = 107;
                u_Debug[slot].lineNumber = pointIndex;
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
