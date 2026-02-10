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

// NOTE: This shader has 96 permutations (displacement × frustum × tess mode × vis mode × surface type)
// that are compiled at runtime via Donut ShaderFactory. The single variant below is a placeholder
// for the build system; actual runtime compilation uses macro defines.
//!variant compute_cluster_tiling.comp
//!end-variants

// Use row-major matrix layout to match CPU struct packing (float3x4 = 48 bytes)
// Note: Slang doesn't support #pragma pack_matrix, it uses -matrix-layout-column-major compiler flag
#ifndef __SLANG__
#pragma pack_matrix(row_major)
#endif

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/compute_cluster_tiling_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/material_data.h"

// TESS_MODE
#define TESS_MODE_SPHERICAL_PROJECTION 0
#define TESS_MODE_WORLD_SPACE_EDGE_LENGTH 1
#define TESS_MODE_UNIFORM 2

// VIS_MODE
#define VIS_MODE_LIMIT_EDGES 0
#define VIS_MODE_SURFACE 1

// Default compile-time defines for static build (can be overridden at runtime)
#ifndef VIS_MODE
#define VIS_MODE VIS_MODE_LIMIT_EDGES
#endif

#ifndef TESS_MODE
#define TESS_MODE TESS_MODE_SPHERICAL_PROJECTION
#endif

#ifndef SURFACE_TYPE
#define SURFACE_TYPE SURFACE_TYPE_ALL
#endif

#ifndef DISPLACEMENT_MAPS
#define DISPLACEMENT_MAPS 0
#endif

#define PATCH_POINTS_WRITEABLE

// Group atomics is significantly faster by reducing pressure on the global atomic
// This reduces the global atomic by a factor of 4 (kComputeClusterTilingWavesPerSurface)
// DISABLED: GroupMemoryBarrierWithGroupSync() inside divergent if(iLane==0) is UB
// in Vulkan SPIR-V (requires ALL workgroup threads to reach barrier). The SDK uses
// D3D12 where NVIDIA handles this pattern, but Slang->SPIR-V on Vulkan deadlocks/fails.
#define ENABLE_GROUP_ATOMICS 0

// Wave intrinsics do not appear to make any difference in performance
// This is most likely since we use the same number of registers (otherwise we pay for local data stores/reads)
#define ENABLE_WAVE_INTRINSICS 0

#include "rtx/pass/rtx_megageo/utils/shader_debug.h"
#include "rtx/pass/rtx_megageo/cluster_builder/cluster.h"
#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include "rtx/pass/rtx_megageo/cluster_builder/tessellation_counters.h"
#include "rtx/pass/rtx_megageo/cluster_builder/tessellator_constants.h"
#include "rtx/pass/rtx_megageo/cluster_builder/tilings.h"
#include "rtxmg/subdivision/osd_ports/tmr/surfaceDescriptor.h"
#include "rtxmg/subdivision/osd_ports/tmr/subdivisionNode.h"
#include "rtxmg/subdivision/osd_ports/tmr/nodeDescriptor.h"
#include "rtx/pass/rtx_megageo/utils/box3.h"
#include "rtx/pass/rtx_megageo/hiz/hiz_buffer_constants.h"
#include "rtx/pass/rtx_megageo/cluster_builder/displacement.hlsli"

static const uint32_t kComputeClusterTilingLanes = 32;
static const float kEpsilon = 1e-6f;

// Slang maps HLSL registers to Vulkan bindings: t→0-99, s→100-199, u→200-299, b→300-399
// SRVs (t0-t16 → bindings 0-16)
StructuredBuffer<float3> t_VertexControlPoints : register(t0);
StructuredBuffer<GeometryData> t_GeometryData : register(t1);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t2);
StructuredBuffer<uint16_t> t_SurfaceToGeometryIndex : register(t3);
StructuredBuffer<SurfaceDescriptor> t_VertexSurfaceDescriptors : register(t4);
StructuredBuffer<Index> t_VertexControlPointIndices : register(t5);
StructuredBuffer<uint32_t> t_VertexPatchPointsOffsets : register(t6);
StructuredBuffer<SubdivisionPlanHLSL> t_Plans : register(t7);
StructuredBuffer<uint32_t> t_SubpatchTrees : register(t8);
StructuredBuffer<Index> t_PatchPointIndices : register(t9);
StructuredBuffer<float> t_StencilMatrix : register(t10);
StructuredBuffer<uint32_t> t_ClasInstantiationBytes : register(t11);
StructuredBuffer<nvrhi::GpuVirtualAddress> t_TemplateAddresses : register(t12);
StructuredBuffer<LinearSurfaceDescriptor> t_TexCoordSurfaceDescriptors : register(t13);
StructuredBuffer<Index> t_TexCoordControlPointIndices : register(t14);
StructuredBuffer<uint32_t> t_TexCoordPatchPointsOffsets : register(t15);
StructuredBuffer<float2> t_TexCoords : register(t16);

// HiZ buffer array (t17-t25 → bindings 17-25)
Texture2D t_HiZBuffer[HIZ_MAX_LODS] : register(t17);

// Bindless texture array for displacement mapping (t26 → binding 26)
Texture2D bindlessTextures[] : register(t26);

// Samplers (s0-s1 → bindings 100-101)
SamplerState s_DisplacementSampler : register(s0);
SamplerState s_HizSampler : register(s1);

// UAVs (u0-u8 → bindings 200-208)
RWStructuredBuffer<GridSampler> u_GridSamplers : register(u0);
RWStructuredBuffer<TessellationCounters> u_TessellationCounters : register(u1);
RWStructuredBuffer<Cluster> u_Clusters : register(u2);
RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u3);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> u_IndirectArgData : register(u4);
RWStructuredBuffer<nvrhi::GpuVirtualAddress> u_ClasAddresses : register(u5);
RWStructuredBuffer<float3> u_VertexPatchPoints : register(u6);
RWStructuredBuffer<float2> u_TexCoordPatchPoints : register(u7);
RWStructuredBuffer<ShaderDebugElement> u_Debug : register(u8);

// Constant buffer (b0 → binding 300)
ConstantBuffer<ComputeClusterTilingParams> g_Params : register(b0);

// Helper to transform float4 by localToWorld (stored as 3 float4 rows)
float3 transformByLocalToWorld(float4 v)
{
    return float3(
        dot(g_Params.localToWorld[0], v),
        dot(g_Params.localToWorld[1], v),
        dot(g_Params.localToWorld[2], v)
    );
}

const static uint32_t nSamples = kComputeClusterTilingWaves * kNumWaveSurfaceUVSamples;
groupshared LimitFrame samples[nSamples];

#if ENABLE_GROUP_ATOMICS
groupshared uint32_t s_clusters;
groupshared uint32_t s_vertices;
groupshared uint32_t s_clasBlocks;
groupshared uint32_t s_triangles;

groupshared uint32_t s_groupClasBlocksOffset;
groupshared uint32_t s_groupClusterOffset;
groupshared uint32_t s_groupVertexOffset;

groupshared bool s_allocationSucceeded;
#endif

bool HasLimit(uint32_t iSurface)
{
    return t_VertexSurfaceDescriptors[iSurface].HasLimit();
}

bool HiZIsVisible(Box3 aabb)
{
    // tests visibility of a screen-space aligned aabb against hi-z buffer
    if (aabb.m_min.z > 0.f)
    {
        float oldminy = aabb.m_min.y;
        float oldmaxy = aabb.m_max.y;

        aabb.m_max.y = g_Params.viewportSize.y - oldminy;
        aabb.m_min.y = g_Params.viewportSize.y - oldmaxy;

        float invTileSize = 1.f / float(HIZ_LOD0_TILE_SIZE);

        float sizeInTiles = max(aabb.m_max.x - aabb.m_min.x, aabb.m_max.y - aabb.m_min.y) * invTileSize;
        uint32_t level = (uint32_t)ceil(log2(sizeInTiles));

        if (level < g_Params.numHiZLODs)
        {
            float2 uv = float2(aabb.m_min.x + aabb.m_max.x, aabb.m_min.y + aabb.m_max.y) * .5f;
            uv *= invTileSize * g_Params.invHiZSize;

            int lw, lh;
            t_HiZBuffer[level].GetDimensions(lw, lh);

            // gather & reduce 4 texels in case the aabb straddles pixel boundaries
            float4 z4 = t_HiZBuffer[level].Gather(s_HizSampler, uv, 0);
            float zfar = max(max(z4.x, z4.y), max(z4.z, z4.w));

            if (aabb.m_min.z > zfar)
            {
                return false;
            }
        }
    }
    return true;
}

float FrustumVisibility(SubdivisionEvaluatorHLSL subd, uint iLane)
{
    SurfaceDescriptor desc = subd.GetSurfaceDesc();
    uint32_t numControlPoints = subd.GetPlan().m_data.numControlPoints;

    Box3 aabb;
    aabb.Init();

    // if bit 0 is set, then X value is > +1
    // if bit 1 is set, then X value is < -1
    // bits 2, 3: same for Y
    // bit 4: behind eye
    uint signBits = 0xFF;

    for (uint32_t i = iLane; i < numControlPoints; i += kComputeClusterTilingLanes)
    {
        Index index = subd.m_vertexControlPointIndices[desc.firstControlPoint + i];
        float3 cp = subd.m_vertexControlPoints[index];

        float4 p = { cp.x, cp.y, cp.z, 1.0f };
        float3 pWorld = transformByLocalToWorld(p);
        float4 pClip = mul(g_Params.matWorldToClip, float4(pWorld, 1.0f));

        uint16_t bits = 0u;
        bits |= (uint16_t(pClip.x > pClip.w) << 0);
        bits |= (uint16_t(pClip.x < -pClip.w) << 1);
        bits |= (uint16_t(pClip.y > pClip.w) << 2);
        bits |= (uint16_t(pClip.y < -pClip.w) << 3);
        bits |= (uint16_t(pClip.w < 0) << 4); // behind eye

        signBits &= bits;

        // accumulate screen-space AABB

        float3 pScreen = float3(
            (.5f + pClip.x / pClip.w * .5f) * g_Params.viewportSize.x,
            (.5f + pClip.y / pClip.w * .5f) * g_Params.viewportSize.y,
            pClip.w);
        aabb.Include(pScreen);
    }

    // reduce sign bits across lanes
    uint surfaceSignBits = WaveActiveBitAnd(signBits);

    bool visible = (surfaceSignBits == 0);
    bool surfaceVisible = WaveReadLaneAt(visible, 0);

    if (!surfaceVisible || !g_Params.enableHiZVisibility)
    {
        return surfaceVisible;
    }

    // butterfly reduction of AABB across lanes
    for (int i = (kComputeClusterTilingLanes/2); i >= 1; i /= 2)
    {
        uint targetLane = WaveGetLaneIndex() ^ i;
        aabb.m_min.x = min(aabb.m_min.x, WaveReadLaneAt(aabb.m_min.x, targetLane));
        aabb.m_min.y = min(aabb.m_min.y, WaveReadLaneAt(aabb.m_min.y, targetLane));
        aabb.m_min.z = min(aabb.m_min.z, WaveReadLaneAt(aabb.m_min.z, targetLane));

        aabb.m_max.x = max(aabb.m_max.x, WaveReadLaneAt(aabb.m_max.x, targetLane));
        aabb.m_max.y = max(aabb.m_max.y, WaveReadLaneAt(aabb.m_max.y, targetLane));
        aabb.m_max.z = max(aabb.m_max.z, WaveReadLaneAt(aabb.m_max.z, targetLane));
    }


    if (iLane == 0 && aabb.Valid())
    {
        aabb.m_min.x = clamp(aabb.m_min.x, 0.f, g_Params.viewportSize.x);
        aabb.m_max.x = clamp(aabb.m_max.x, 0.f, g_Params.viewportSize.x);
        aabb.m_min.y = clamp(aabb.m_min.y, 0.f, g_Params.viewportSize.y);
        aabb.m_max.y = clamp(aabb.m_max.y, 0.f, g_Params.viewportSize.y);

        surfaceVisible = HiZIsVisible(aabb);
    }

    surfaceVisible = WaveReadLaneAt(surfaceVisible, 0);

    return (float)surfaceVisible;
}

float CalculateVisibility(SubdivisionEvaluatorHLSL subd, uint iLane)
{
    float visibility = 1.0; // fully visible
#if VIS_MODE == VIS_MODE_SURFACE
    if (g_Params.enableFrustumVisibility)
    {
        visibility = FrustumVisibility(subd, iLane);
    }
#elif VIS_MODE == VIS_MODE_LIMIT_EDGES
    // for edge visibility, all the work IsBSplinePatch done below int `calculateEdgeVisibility`
#else
#error UNKNOWN VIS MODE VIS_MODE
#endif
    return visibility;
}

float CalculateEdgeVisibility(uint32_t iLane, uint32_t waveSampleOffset, float visibility)
{
#if VIS_MODE == VIS_MODE_SURFACE
    return visibility;
#elif VIS_MODE == VIS_MODE_LIMIT_EDGES
    //
    // expected limitFrames layout:
    //
    //          e2
    //
    //     p6---p5---p4
    //     |          |
    // e3  p7        p3  e1
    //     |          |
    //     p0---p1---p2
    //
    //          e0
    //

    // each lane is assigned to one of the 4 surface edges
    if (iLane > 3)
        return 1.f;

    visibility = 1.f;

    // compute a normalized visibility factor for the 3 limit samples locations
    // along the edge against the frustum, hi-z and back-facing criteria.
    float3 pworld[3];
    float4 pclip[3];

    float frustumFactor = 0.f;
    for (int i = 0; i < 3; ++i)
    {
        pworld[i] = samples[waveSampleOffset + (iLane * 2 + i) % kNumWaveSurfaceUVSamples].p;
        pclip[i] = mul(g_Params.matWorldToClip, float4(pworld[i], 1.0));

        float dist = (1.f / pclip[i].w) * sqrt(pclip[i].x * pclip[i].x + pclip[i].y * pclip[i].y);
        frustumFactor = max(frustumFactor, pclip[i].w < 0.f ? 1.f : smoothstep(1.3f, 2.5f, dist));
    }

    if (g_Params.enableFrustumVisibility)
    {
        visibility *= (1.f - frustumFactor);
    }

    if (visibility == 0.f)
        return visibility;

    if (g_Params.enableHiZVisibility)
    {
        float3 pscreen[3];
        [unroll]
        for (uint16_t i = 0; i < 3; ++i)
        {
            pscreen[i].x = (.5f + pclip[i].x / pclip[i].w * .5f) * g_Params.viewportSize.x;
            pscreen[i].y = (.5f + pclip[i].y / pclip[i].w * .5f) * g_Params.viewportSize.y;
            pscreen[i].z = pclip[i].w;
        }

        Box3 aabb;
        aabb.Init(pscreen[0], pscreen[1], pscreen[2]);
        aabb.m_min.x = clamp(aabb.m_min.x, 0.f, g_Params.viewportSize.x);
        aabb.m_max.x = clamp(aabb.m_max.x, 0.f, g_Params.viewportSize.x);
        aabb.m_min.y = clamp(aabb.m_min.y, 0.f, g_Params.viewportSize.y);
        aabb.m_max.y = clamp(aabb.m_max.y, 0.f, g_Params.viewportSize.y);

        if (!HiZIsVisible(aabb))
            return 0.f;
    }

    if (g_Params.enableBackfaceVisibility)
    {
        for (uint16_t i = 0; i < 3; ++i)
        {
            uint32_t sampleIndex = (iLane * 2 + i) % kNumWaveSurfaceUVSamples;
            float3 t0 = samples[waveSampleOffset + sampleIndex].deriv1;
            float3 t1 = samples[waveSampleOffset + sampleIndex].deriv2;

            if (!IsParallel(t0, t1))
            {
                float3 nobj = cross(t0, t1);
                float3 nworld = normalize(transformByLocalToWorld(float4(nobj, 0.f)));
                // Use view direction from world-to-clip matrix (camera at origin in translated world)
                // The view direction to a point is just -normalize(point) when camera is at origin
                float3 viewDir = -normalize(pworld[i]);
                float cosTheta = dot(viewDir, nworld);
                float backfaceFactor = smoothstep(.6f, 1.f, cosTheta);

                visibility *= (1.f - backfaceFactor);
            }
        }
    }
    return visibility;
#else
#error UNKNOWN VIS MODE VIS_MODE
#endif
}

void WaveEvaluateBSplinePatch8(uint32_t iWave,
    uint32_t iLane,
    SubdivisionEvaluatorHLSL subd,
    TexcoordEvaluatorHLSL texcoordEval)
{
    uint32_t waveSampleOffset = kNumWaveSurfaceUVSamples * iWave;

    // always do the non-displaced evaluation first.  Displacement maps will perturb this calculation below
    // Check for subdivision bypass - use bilinear interpolation instead
    if (g_Params.disableSubdivision != 0)
    {
        if (iLane < kNumWaveSurfaceUVSamples)
        {
            LimitFrame limit = subd.EvaluateBilinear(kWaveSurfaceUVSamples[iLane]);
            samples[waveSampleOffset + iLane] = limit;
        }
    }
    else
    {
#if SURFACE_TYPE == SURFACE_TYPE_ALL
        if (subd.IsPureBSplinePatch())
        {
            LimitFrame limit = subd.WaveEvaluatePureBsplinePatch8(iLane);
            if (iLane < kNumWaveSurfaceUVSamples)
            {
                samples[waveSampleOffset + iLane] = limit;
            }
        }
        else if (subd.IsBSplinePatch())
        {
            LimitFrame limit = subd.WaveEvaluateBsplinePatch(iLane);
            if (iLane < kNumWaveSurfaceUVSamples)
            {
                samples[waveSampleOffset + iLane] = limit;
            }
        }
        else
        {
            // there is no wave parallel implementation for non-bspline patches falling back to single thread
            subd.WaveEvaluatePatchPoints(iLane, kComputeClusterTilingLanes);
            if (iLane < kNumWaveSurfaceUVSamples)
            {
                LimitFrame limit = subd.EvaluateLimitSurface(kWaveSurfaceUVSamples[iLane]);
                samples[waveSampleOffset + iLane] = limit;
            }
        }
#elif SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE
        LimitFrame limit = subd.WaveEvaluatePureBsplinePatch8(iLane);
        if (iLane < kNumWaveSurfaceUVSamples)
            samples[waveSampleOffset + iLane] = limit;
#elif SURFACE_TYPE == SURFACE_TYPE_REGULARBSPLINE
        LimitFrame limit = subd.WaveEvaluateBsplinePatch(iLane);
        if (iLane < kNumWaveSurfaceUVSamples)
        {
            samples[waveSampleOffset + iLane] = limit;
        }
#elif SURFACE_TYPE == SURFACE_TYPE_LIMIT
        // there is no wave parallel implementation for non-bspline patches falling back to single thread
        subd.WaveEvaluatePatchPoints(iLane, kComputeClusterTilingLanes);
        if (iLane < kNumWaveSurfaceUVSamples)
        {
            LimitFrame limit = subd.EvaluateLimitSurface(kWaveSurfaceUVSamples[iLane]);
            samples[waveSampleOffset + iLane] = limit;
        }
#endif
    }

#if DISPLACEMENT_MAPS
    uint32_t geometryIndex = t_SurfaceToGeometryIndex[subd.m_surfaceIndex] + g_Params.firstGeometryIndex;
    GeometryData geometry = t_GeometryData[geometryIndex];
    MaterialConstants material = t_MaterialConstants[geometry.materialIndex];

    float displacementScale;
    int displacementTexIndex;
    GetDisplacement(material, g_Params.globalDisplacementScale, displacementTexIndex, displacementScale);
    if (displacementTexIndex >= 0 && iLane < kNumWaveSurfaceUVSamples)
    {
        Texture2D displacementTexture = bindlessTextures[displacementTexIndex];
        LimitFrame displaced = DoDisplacement(texcoordEval,
            samples[waveSampleOffset + iLane], subd.m_surfaceIndex, kWaveSurfaceUVSamples[iLane], 0, 0,
            displacementTexture,
            s_DisplacementSampler, displacementScale);
        samples[waveSampleOffset + iLane] = displaced;
    }
#endif
}

// Global debug counter to limit log spam
static uint32_t g_edgeRateLogCount = 0;

// Project world position to screen pixels
// Returns screen position, sets valid=false if behind camera
float2 worldToScreen(float3 worldPos, out bool valid)
{
    float4 clipPos = mul(g_Params.matWorldToClip, float4(worldPos, 1.0f));

    // Check if point is behind camera (w <= 0) or too close
    if (clipPos.w <= 0.001f)
    {
        valid = false;
        return float2(0, 0);
    }

    valid = true;
    // Perspective divide to NDC (-1 to 1)
    float2 ndc = clipPos.xy / clipPos.w;
    // Convert to screen pixels
    float2 screen = (ndc * 0.5f + 0.5f) * g_Params.viewportSize;
    return screen;
}

float CalculateEdgeRates(LimitFrame limitFrame, uint32_t iSurface, uint32_t iLane)
{
#if TESS_MODE == TESS_MODE_SPHERICAL_PROJECTION
    // Not used for true screen-space - kept for compatibility
    return g_Params.fineTessellationRate;
#elif TESS_MODE == TESS_MODE_WORLD_SPACE_EDGE_LENGTH
    float diagonalLength = length(g_Params.aabb.Extent());
    return g_Params.fineTessellationRate * 1000.f / diagonalLength;
#else // TESS_MODE_UNIFORM or other
    return g_Params.fineTessellationRate;
#endif
}

uint16_t EvaluateEdgeSegments(uint32_t iWave, uint32_t iLane, uint32_t iSurface, float visibility, float visibilityRateMultiplier)
{
    // BYPASS TESSELLATION: Force 8 segments per edge for testing
    return iLane < 4 ? (uint16_t)8 : (uint16_t)0;

#if 0 // Original tessellation code disabled for bypass testing
    uint32_t waveSampleOffset = kNumWaveSurfaceUVSamples * iWave;
#if TESS_MODE == TESS_MODE_UNIFORM
    if (iLane < 4)
    {
        uint32_t segments = g_Params.edgeSegments[iLane];
        float segmentVisibility = CalculateEdgeVisibility(iLane, waveSampleOffset, visibility);

        segments = float(segments) * (visibilityRateMultiplier + segmentVisibility * (1.f - visibilityRateMultiplier));
        return (uint16_t)clamp(segments, 1u, 1024u);
    }
    return 0;
#else
    // TRUE SCREEN-SPACE TESSELLATION
    // Transform points to world space first
    if (iLane < kNumWaveSurfaceUVSamples)
    {
        samples[waveSampleOffset + iLane].p = transformByLocalToWorld(float4(samples[waveSampleOffset + iLane].p, 1.0));
    }

    // Project to screen space and compute edge length in PIXELS
    bool valid0, valid1;
    float2 screenPos0 = worldToScreen(samples[waveSampleOffset + iLane % kNumWaveSurfaceUVSamples].p, valid0);
    float2 screenPos1 = worldToScreen(samples[waveSampleOffset + (iLane + 1) % kNumWaveSurfaceUVSamples].p, valid1);

    // If either point is behind camera, use fallback tessellation
    float edgeLengthPixels = 0.0f;
    if (valid0 && valid1)
    {
        edgeLengthPixels = length(screenPos1 - screenPos0);
        // Sanity check - clamp huge values that might cause issues
        edgeLengthPixels = min(edgeLengthPixels, 10000.0f);
    }
    else
    {
        // Fallback: use minimum tessellation for offscreen/behind-camera edges
        edgeLengthPixels = g_Params.fineTessellationRate; // Will result in 1 segment
    }

    // Accumulate edge segment lengths (each edge spans 2 sample segments)
    float finalEdgeLengthPixels = edgeLengthPixels;
    float edgeLength2 = edgeLengthPixels;
    if (iLane < 8)
    {
        edgeLength2 = edgeLengthPixels + WaveReadLaneAt(edgeLengthPixels, iLane + 1);
        // Get the edge lengths starting at the corner vertices (in 2*iLane threads)
        finalEdgeLengthPixels = WaveReadLaneAt(edgeLength2, 2 * iLane);
    }

    // Apply visibility
    float edgeVisibility = CalculateEdgeVisibility(iLane, waveSampleOffset, visibility);
    float visibilityFactor = (visibilityRateMultiplier + edgeVisibility * (1.f - visibilityRateMultiplier));

    // TRUE SCREEN-SPACE: tessellate based on pixel coverage
    // Use a fixed 16 pixels per segment for reasonable tessellation
    // (fineTessellationRate=0.05 is meant for distance-based, not pixel-based)
    const float pixelsPerSegment = 16.0f;
    float edgeSegments = (finalEdgeLengthPixels / pixelsPerSegment) * visibilityFactor;

    // Clamp to reasonable range
    edgeSegments = clamp(edgeSegments, 1.0f, 64.0f);

    // Debug logging for first surface, edge 0
    if (iSurface == 0 && iLane == 0)
    {
        float3 worldP0 = samples[waveSampleOffset + 0].p;
        float3 worldP1 = samples[waveSampleOffset + 1].p;
        SHADER_DEBUG_FORCE(float4(worldP0.x, worldP0.y, worldP0.z, 1111.0f));
        SHADER_DEBUG_FORCE(float4(screenPos0.x, screenPos0.y, float(valid0), 2222.0f));
        SHADER_DEBUG_FORCE(float4(screenPos1.x, screenPos1.y, float(valid1), 3333.0f));
        SHADER_DEBUG_FORCE(float4(edgeLengthPixels, finalEdgeLengthPixels, pixelsPerSegment, 4444.0f));
        SHADER_DEBUG_FORCE(float4(edgeSegments, g_Params.viewportSize.x, g_Params.viewportSize.y, 5555.0f));
    }

    return iLane < 4 ? ((uint16_t)round(edgeSegments)) : 0;

#endif
#endif // End of #if 0 bypass block
}

void GathererWriteCluster(Cluster cluster, uint32_t clusterIndex, GridSampler gridSampler, uint32_t localGeometryIndex, nvrhi::GpuVirtualAddress templateAddress)
{
    const nvrhi::GpuVirtualAddress vertexBufferAddress = g_Params.clusterVertexPositionsBaseAddress + nvrhi::GpuVirtualAddress(cluster.nVertexOffset * sizeof(float3));

    nvrhi::rt::cluster::IndirectInstantiateTemplateArgs indirectArgs = (nvrhi::rt::cluster::IndirectInstantiateTemplateArgs)0;
    indirectArgs.clusterIdOffset = clusterIndex;
    // Use clusterIndex as geometry index so we can retrieve it via GeometryIndex() in hit shaders
    // The original surface ID is available via ClusterShadingData.m_surfaceId
    indirectArgs.geometryIndexOffsetPacked = clusterIndex; // Lower 24 bits = clusterIndex for shader lookup
    indirectArgs.clusterTemplate = templateAddress;
    indirectArgs.vertexBuffer.startAddress = vertexBufferAddress;
    indirectArgs.vertexBuffer.strideInBytes = sizeof(float3);
    u_IndirectArgData[clusterIndex] = indirectArgs;

    ClusterShadingData shadingData = (ClusterShadingData)0;
    shadingData.m_edgeSegments = gridSampler.edgeSegments;
    shadingData.m_surfaceId = cluster.iSurface;
    shadingData.m_vertexOffset = cluster.nVertexOffset;
    shadingData.m_clusterOffset = cluster.offset;
    shadingData.m_clusterSizeX = cluster.sizeX;
    shadingData.m_clusterSizeY = cluster.sizeY;

    u_ClusterShadingData[clusterIndex] = shadingData;
}

void WriteSurfaceWave(uint32_t iWave, uint32_t iLane, uint32_t iSurface, GridSampler rSampler)
{
    uint32_t waveSampleOffset = kNumWaveSurfaceUVSamples * iWave;
    if (iLane == 0)
    {
        u_GridSamplers[iSurface] = rSampler;
    }

    uint16_t2 surfaceSize = rSampler.GridSize();

    // Skip degenerate surfaces with zero size
    if (surfaceSize.x == 0 || surfaceSize.y == 0)
    {
        if (iLane == 0)
        {
            SHADER_DEBUG_FORCE(uint4(0xDEAD0000, iSurface, surfaceSize.x, surfaceSize.y));
        }
        return;
    }

    SurfaceTiling surfaceTiling = MakeSurfaceTiling(surfaceSize);

    uint32_t clusterCount = 0;
    uint32_t vertexCount = 0;
    uint32_t clasBlocks = 0;
    uint32_t clusterTris = 0;

#if ENABLE_WAVE_INTRINSICS
    _Static_assert(SurfaceTiling::N_SUB_TILINGS <= kComputeClusterTilingLanes, "Must have enough lanes to use wave ops");
    if (iLane < SurfaceTiling::N_SUB_TILINGS)
    {
        ClusterTiling clusterTiling = surfaceTiling.subTilings[iLane];
        uint32_t templateIndex = GetTemplateIndex(clusterTiling.clusterSize);
        uint32_t tilingClusterCount = clusterTiling.ClusterCount();
        uint32_t tilingVertexCount = clusterTiling.VertexCount();

        clasBlocks = (t_ClasInstantiationBytes[templateIndex] / nvrhi::rt::cluster::kClasByteAlignment) * tilingClusterCount;

        clasBlocks += WaveReadLaneAt(clasBlocks, WaveGetLaneIndex() + 2); // Lanes: [0+2], [1+3]
        clasBlocks += WaveReadLaneAt(clasBlocks, WaveGetLaneIndex() + 1); // Lanes: [0+1]

        clusterCount = tilingClusterCount;
        clusterCount += WaveReadLaneAt(clusterCount, WaveGetLaneIndex() + 2);
        clusterCount += WaveReadLaneAt(clusterCount, WaveGetLaneIndex() + 1);

        vertexCount = tilingVertexCount;
        vertexCount += WaveReadLaneAt(vertexCount, WaveGetLaneIndex() + 2);
        vertexCount += WaveReadLaneAt(vertexCount, WaveGetLaneIndex() + 1);
    }
#endif

    bool allocationSucceeded = false;
    // compute cluster and vertex offsets into linear storage using global counters
    uint32_t surfaceClusterOffset, surfaceVertexOffset, clasBlocksOffset;
    if (iLane == 0)
    {
#if !ENABLE_WAVE_INTRINSICS
        [unroll]
        for (uint16_t iTiling = 0; iTiling < SurfaceTiling_N_SUB_TILINGS; ++iTiling)
        {
            ClusterTiling clusterTiling = surfaceTiling.subTilings[iTiling];
            uint32_t templateIndex = GetTemplateIndex(clusterTiling.clusterSize);
            uint32_t tilingClusterCount = ClusterTiling_ClusterCount(clusterTiling);
            uint32_t tilingVertexCount = ClusterTiling_VertexCount(clusterTiling);

            clusterCount += tilingClusterCount;
            vertexCount += tilingVertexCount;
            clasBlocks += (t_ClasInstantiationBytes[templateIndex] / nvrhi::rt::cluster::kClasByteAlignment) * tilingClusterCount;
        }
#endif
        clusterTris = 2 * (uint32_t)surfaceSize.x * (uint32_t)surfaceSize.y;
    }

#if ENABLE_GROUP_ATOMICS
    uint32_t waveClasBlocksOffset, waveSurfaceClusterOffset, waveSurfaceVertexOffset;
    if (iLane == 0)
    {
        // Coalesce waves into group shared memory
        uint32_t dummy;
        InterlockedAdd(s_clasBlocks, clasBlocks, waveClasBlocksOffset);
        InterlockedAdd(s_clusters, clusterCount, waveSurfaceClusterOffset);
        InterlockedAdd(s_vertices, vertexCount, waveSurfaceVertexOffset);
        InterlockedAdd(s_triangles, clusterTris, dummy);

        GroupMemoryBarrierWithGroupSync();

        // 1 global atomic per thread group
        if (iWave == 0)
        {
            uint32_t dummy;
            InterlockedAdd(u_TessellationCounters[0].desiredClasBlocks, s_clasBlocks, s_groupClasBlocksOffset);
            InterlockedAdd(u_TessellationCounters[0].desiredClusters, s_clusters, s_groupClusterOffset);
            InterlockedAdd(u_TessellationCounters[0].desiredVertices, s_vertices, s_groupVertexOffset);
            InterlockedAdd(u_TessellationCounters[0].desiredTriangles, s_triangles, dummy);

            allocationSucceeded = ((s_groupClasBlocksOffset + s_clasBlocks) <= g_Params.maxClasBlocks) &&
                ((s_groupClusterOffset + s_clusters) <= g_Params.maxClusters) &&
                ((s_groupVertexOffset + s_vertices) <= g_Params.maxVertices);

            // If we passed all allocations increment the real cluster counter.
            // This code used to track allocated clasBlocks, vertices, triangles
            // but atomics are expensive and it doubled the number of stalls on atomics
            if (allocationSucceeded)
            {
                InterlockedAdd(u_TessellationCounters[0].clusters, s_clusters, s_groupClusterOffset);
            }

            // write back global offset
            s_allocationSucceeded = allocationSucceeded;
        }

        GroupMemoryBarrierWithGroupSync();

        // Read back to each wave
        clasBlocksOffset = s_groupClasBlocksOffset + waveClasBlocksOffset;
        surfaceClusterOffset = s_groupClusterOffset + waveSurfaceClusterOffset;
        surfaceVertexOffset = s_groupVertexOffset + waveSurfaceVertexOffset;
        allocationSucceeded = s_allocationSucceeded;
    }
#else
    if (iLane == 0)
    {
        uint32_t dummy;
        InterlockedAdd(u_TessellationCounters[0].desiredClasBlocks, clasBlocks, clasBlocksOffset);
        InterlockedAdd(u_TessellationCounters[0].desiredClusters, clusterCount, surfaceClusterOffset);
        InterlockedAdd(u_TessellationCounters[0].desiredVertices, vertexCount, surfaceVertexOffset);
        InterlockedAdd(u_TessellationCounters[0].desiredTriangles, clusterTris, dummy);

        allocationSucceeded = ((clasBlocksOffset + clasBlocks) <= g_Params.maxClasBlocks) &&
            ((surfaceClusterOffset + clusterCount) <= g_Params.maxClusters) &&
            ((surfaceVertexOffset + vertexCount) <= g_Params.maxVertices);

        // If we passed all allocations increment the real cluster counter.
        // This code used to track allocated clasBlocks, vertices, triangles
        // but atomics are expensive and it doubled the number of stalls on atomics
        if (allocationSucceeded)
        {
            InterlockedAdd(u_TessellationCounters[0].clusters, clusterCount, surfaceClusterOffset);
        }
    }
#endif

    allocationSucceeded = WaveReadLaneAt(allocationSucceeded, 0);
    if (!allocationSucceeded)
    {
        return;
    }

    surfaceClusterOffset = WaveReadLaneAt(surfaceClusterOffset, 0);
    surfaceVertexOffset = WaveReadLaneAt(surfaceVertexOffset, 0);
    clasBlocksOffset = WaveReadLaneAt(clasBlocksOffset, 0);

    uint32_t tilingClusterOffset = surfaceClusterOffset;
    uint32_t tilingVertexOffset = surfaceVertexOffset;

    nvrhi::GpuVirtualAddress tilingClusterBaseAddress = g_Params.clasDataBaseAddress + clasBlocksOffset * nvrhi::rt::cluster::kClasByteAlignment;
    uint32_t localGeometryIndex = t_SurfaceToGeometryIndex[iSurface];

    // Unroll so that we don't have local data loads from the surface tiling array
    [unroll]
    for (uint16_t iTiling = 0; iTiling < SurfaceTiling_N_SUB_TILINGS; ++iTiling)
    {
        const ClusterTiling clusterTiling = surfaceTiling.subTilings[iTiling];
        uint32_t tilingClusterCount = ClusterTiling_ClusterCount(clusterTiling);
        uint32_t tilingClusterVertexCount = ClusterTiling_ClusterVertexCount(clusterTiling);
        uint16_t2 tilingClusterSize = clusterTiling.clusterSize;
        uint32_t tilingVertexCount =  tilingClusterCount * tilingClusterVertexCount;

        const uint32_t templateIndex = GetTemplateIndex(tilingClusterSize);
        nvrhi::GpuVirtualAddress templateAddress = t_TemplateAddresses[templateIndex];
        uint32_t clasBytes = t_ClasInstantiationBytes[templateIndex];

        // make clusters with tilingSize
        for (uint32_t iCluster = iLane;
            iCluster < tilingClusterCount;
            iCluster += kComputeClusterTilingLanes)
        {
            Cluster cluster = MakeCluster(iSurface, tilingVertexOffset + tilingClusterVertexCount * iCluster,
                    SurfaceTiling_ClusterOffset(surfaceTiling, iTiling, iCluster), tilingClusterSize.x, tilingClusterSize.y);
            uint32_t clusterIndex = tilingClusterOffset + iCluster;
            u_Clusters[clusterIndex] = cluster;

            // DEBUG: Log what compute_cluster_tiling writes to clusters buffer for instance 0
            if (g_Params.instanceIndex == 0 && clusterIndex < 8) {
                u_Debug[clusterIndex].uintData = uint4(iSurface, clusterIndex, surfaceClusterOffset, clusterCount);
                u_Debug[clusterIndex].floatData = float4(asfloat(tilingClusterOffset), asfloat(g_Params.surfaceStart), asfloat(g_Params.surfaceEnd), 0);
                u_Debug[clusterIndex].payloadType = 3; // Mark as tiling debug
                u_Debug[clusterIndex].lineNumber = 0;
            }

            GathererWriteCluster(cluster, clusterIndex, rSampler, localGeometryIndex, templateAddress);

            u_ClasAddresses[clusterIndex] = tilingClusterBaseAddress + clasBytes * iCluster;
        }
        tilingClusterOffset += tilingClusterCount;
        tilingVertexOffset += tilingVertexCount;
        tilingClusterBaseAddress += clasBytes * tilingClusterCount;
    }
}

[shader("compute")]
[numthreads(kComputeClusterTilingLanes * kComputeClusterTilingWaves, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    const uint32_t iLane = threadIdx.x % kComputeClusterTilingLanes;
    const uint32_t iWave = threadIdx.x / kComputeClusterTilingLanes;
    const uint32_t iSurface = kComputeClusterTilingWaves * groupIdx.x + iWave + g_Params.surfaceStart;

    SHADER_DEBUG_INIT(u_Debug, uint2(g_Params.debugSurfaceIndex, g_Params.debugLaneIndex), uint2(iSurface, iLane));


#if ENABLE_GROUP_ATOMICS
    if (iLane == 0 && iWave == 0)
    {
        s_clusters = 0;
        s_vertices = 0;
        s_clasBlocks = 0;
        s_triangles = 0;
        s_allocationSucceeded = false;
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    uint numSurfaceDescriptors, surfaceDescriptorStride;
    t_VertexSurfaceDescriptors.GetDimensions(numSurfaceDescriptors, surfaceDescriptorStride);

    if (iSurface >= g_Params.surfaceEnd)
    {
        return; // early out waves beyond cluster array end
    }

    if (!HasLimit(iSurface))
    {
        return; // don't process surfaces that have no limit
    }

    SubdivisionEvaluatorHLSL subd;
    subd.m_surfaceIndex = iSurface;

    subd.m_isolationLevel = uint16_t(g_Params.isolationLevel);
    subd.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
    subd.m_plans = t_Plans;
    subd.m_subpatchTrees = t_SubpatchTrees;
    subd.m_vertexPatchPointIndices = t_PatchPointIndices;
    subd.m_stencilMatrix = t_StencilMatrix;

    subd.m_vertexControlPointIndices = t_VertexControlPointIndices;
    subd.m_vertexControlPoints = t_VertexControlPoints;
    subd.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
    subd.m_vertexPatchPoints = u_VertexPatchPoints;

    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = u_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;
    texcoordEval.WaveEvaluateTexCoordPatchPoints(iLane, iSurface);

    // Frustum "culling"
    float visibility = CalculateVisibility(subd, iLane);

    // -------------------------------------------------------------------------
    // Evaluate corner and mid points for surface quad
    // -------------------------------------------------------------------------
    //
    // sample locations:
    //
    //          e2
    //
    //     p6---p5---p4
    //     |          |
    // e3  p7        p3  e1
    //     |          |
    //     p0---p1---p2
    //
    //          e0
    //
    WaveEvaluateBSplinePatch8(iWave, iLane, subd, texcoordEval);

    const float tessFactor = g_Params.coarseTessellationRate / g_Params.fineTessellationRate;

    uint16_t edgeSegments = EvaluateEdgeSegments(iWave, iLane, iSurface, visibility, tessFactor);

    GridSampler rSampler;
    rSampler.edgeSegments[0] = WaveReadLaneAt(edgeSegments, 0);
    rSampler.edgeSegments[1] = WaveReadLaneAt(edgeSegments, 1);
    rSampler.edgeSegments[2] = WaveReadLaneAt(edgeSegments, 2);
    rSampler.edgeSegments[3] = WaveReadLaneAt(edgeSegments, 3);

    // Debug: Log visibility and edge segments for first lane of first wave
    if (iLane == 0 && iWave == 0)
    {
        uint16_t2 gridSize = rSampler.GridSize();
        uint32_t triangles = 2u * (uint32_t)gridSize.x * (uint32_t)gridSize.y;
        SHADER_DEBUG_FORCE(uint4(iSurface, (uint32_t)rSampler.edgeSegments[0] | ((uint32_t)rSampler.edgeSegments[1] << 16),
                                 (uint32_t)rSampler.edgeSegments[2] | ((uint32_t)rSampler.edgeSegments[3] << 16), triangles));
        SHADER_DEBUG_FORCE(float4(g_Params.coarseTessellationRate, g_Params.fineTessellationRate, tessFactor, visibility));
        SHADER_DEBUG_FORCE(float4((float)gridSize.x, (float)gridSize.y, (float)g_Params.viewportSize.x, (float)g_Params.viewportSize.y));
    }

    WriteSurfaceWave(iWave, iLane, iSurface, rSampler);
}
