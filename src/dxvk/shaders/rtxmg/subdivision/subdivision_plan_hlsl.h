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

#include "rtxmg/subdivision/osd_ports/tmr/types.h"

#ifndef __cplusplus
#include "rtxmg/subdivision/osd_ports/tmr/treeDescriptor.h"
#include "rtxmg/subdivision/osd_ports/tmr/nodeDescriptor.h"
#include "rtxmg/subdivision/osd_ports/tmr/subdivisionNode.h"
#include "patch_param.h"

enum class SingleCreaseDynamicIsolation : uint16_t { SHARP = 0, SMOOTH = 1, };
static const SingleCreaseDynamicIsolation kSingleCreaseDynamicIsolation = SingleCreaseDynamicIsolation::SHARP;
static const float kMinFloat = 1.17549435e-38f;
// translators for Tmr => Far types
inline PatchDescriptorType
RegularBasisType(SchemeType scheme)
{
    switch (scheme)
    {
    case SchemeType::SCHEME_CATMARK: return PatchDescriptorType::REGULAR;
    case SchemeType::SCHEME_LOOP: return PatchDescriptorType::LOOP;
    default:
        break;
    }
    return PatchDescriptorType::NON_PATCH;
}

inline PatchDescriptorType
IrregularBasisType(SchemeType scheme, EndCapType endcap)
{

    if (scheme == SchemeType::SCHEME_CATMARK)
    {
        switch (endcap)
        {
        case EndCapType::ENDCAP_BILINEAR_BASIS: return PatchDescriptorType::QUADS;
        case EndCapType::ENDCAP_BSPLINE_BASIS: return PatchDescriptorType::REGULAR;
        case EndCapType::ENDCAP_GREGORY_BASIS: return PatchDescriptorType::GREGORY_BASIS;
        default:
            break;
        }
    }
    else if (scheme == SchemeType::SCHEME_LOOP)
    {
        switch (endcap)
        {
        case EndCapType::ENDCAP_BILINEAR_BASIS: return PatchDescriptorType::TRIANGLES;
        case EndCapType::ENDCAP_BSPLINE_BASIS: return PatchDescriptorType::LOOP;
        case EndCapType::ENDCAP_GREGORY_BASIS: return PatchDescriptorType::GREGORY_TRIANGLE;
        default:
            break;
        }
    }
    return PatchDescriptorType::NON_PATCH;
}

// both Loop & Catmark quadrant traversals expect Z-curve winding
// (see subdivisionPlanBuilder for details)
inline void TraverseCatmark(inout float u, inout float v, inout uint16_t quadrant)
{
    if (u >= 0.5f)
    {
        quadrant ^= 1;
        u = 1.0f - u;
    }
    if (v >= 0.5f)
    {
        quadrant ^= 2;
        v = 1.0f - v;
    }
    u *= 2.0f;
    v *= 2.0f;
}


// note: Z-winding of triangle faces rotates sub-domains every subdivision level,
// but the center face is always at index (2)
//
//                0,1                                    0,1
//                 *                                      *
//               /   \                                  /   \
//              /     \                                /  3  \
//             /       \                              /       \
//            /         \           ==>        0,0.5 . ------- . 0.5,0.5
//           /           \                          /   \ 2 /   \
//          /             \                        /  0  \ /  1  \
//         * ------------- *                      * ----- . ----- *
//      0,0                 1,0                0,0      0.5,0      1,0

inline uint16_t TraverseLoop(float median, inout float u, inout float v, inout bool rotated)
{
    if (!rotated)
    {
        if (u >= median)
        {
            u -= median;
            return 1;
        }
        if (v >= median)
        {
            v -= median;
            return 3;
        }
        if ((u + v) >= median)
        {
            rotated = true;
            return 2;
        }
    }
    else
    {
        if (u < median)
        {
            v -= median;
            return 1;
        }
        if (v < median)
        {
            u -= median;
            return 3;
        }
        u -= median;
        v -= median;
        if ((u + v) < median)
        {
            rotated = true;
            return 2;
        }
    }
    return 0;
}

//
//  Cubic BSpline curve basis evaluation:
//
inline void EvalBSplineCurve(float t, out float wP[4], out float wDP[4], out float wDP2[4], bool calcD1, bool calcD2)
{
    const float one6th = (float)(1.0 / 6.0);

    float t2 = t * t;
    float t3 = t * t2;

    wP[0] = one6th * (1.0f - 3.0f * (t - t2) - t3);
    wP[1] = one6th * (4.0f - 6.0f * t2 + 3.0f * t3);
    wP[2] = one6th * (1.0f + 3.0f * (t + t2 - t3));
    wP[3] = one6th * (t3);

    if (calcD1)
    {
        wDP[0] = -0.5f * t2 + t - 0.5f;
        wDP[1] = 1.5f * t2 - 2.0f * t;
        wDP[2] = -1.5f * t2 + t + 0.5f;
        wDP[3] = 0.5f * t2;
    }
    if (calcD2)
    {
        wDP2[0] = -t + 1.0f;
        wDP2[1] = 3.0f * t - 2.0f;
        wDP2[2] = -3.0f * t + 1.0f;
        wDP2[3] = t;
    }
}


inline float mix(float s1, float s2, float t)
{
    return ((float)1.0 - t) * s1 + t * s2;
}


inline void ComputeMixedCreaseMatrix(float sharp1, float sharp2, float t, float tInf, out float m[16])
{
    float s1 = (float)exp2(sharp1), s2 = (float)exp2(sharp2);

    float sOver3 = mix(s1, s2, t) / float(3), oneOverS1 = (float)1 / s1, oneOverS2 = (float)1 / s2,
        oneOver6S = mix(oneOverS1, oneOverS2, t) / (float)6, sSqr = mix(s1 * s1, s2 * s2, t);

    float A = -sSqr + sOver3 * (float)5.5 + oneOver6S - (float)1.0, B = sOver3 + oneOver6S + (float)0.5,
        C = sOver3 - oneOver6S * (float)2.0 + (float)1.0, E = sOver3 + oneOver6S - (float)0.5,
        F = -sOver3 * (float)0.5 + oneOver6S;

    m[0] = (float)1.0;
    m[1] = A * tInf;
    m[2] = (float)-2.0 * A * tInf;
    m[3] = A * tInf;
    m[4] = (float)0.0;
    m[5] = mix((float)1.0, B, tInf);
    m[6] = (float)-2.0 * E * tInf;
    m[7] = E * tInf;
    m[8] = (float)0.0;
    m[9] = F * tInf;
    m[10] = mix((float)1.0, C, tInf);
    m[11] = F * tInf;
    m[12] = (float)0.0;
    m[13] = mix((float)-1.0, E, tInf);
    m[14] = mix((float)2.0, -(float)2.0 * E, tInf);
    m[15] = B * tInf;
}

// compute the "crease matrix" for modifying basis weights at parametric
// location 't', given a sharpness value (see Matthias Niessner derivation
// for 'single crease' regular patches)
inline void ComputeCreaseMatrix(float sharpness, float t, out float m[16])
{

    float sharpFloor = (float)floor(sharpness), sharpCeil = sharpFloor + 1, sharpFrac = sharpness - sharpFloor;

    float creaseWidthFloor = (float)1.0 - exp2(-sharpFloor), creaseWidthCeil = (float)1.0 - exp2(-sharpCeil);

    // we compute the matrix for both the floor and ceiling of
    // the sharpness value, and then interpolate between them
    // as needed.
    float tA = (t > creaseWidthCeil) ? sharpFrac : (float)0.0, tB = (float)0.0;
    if (t > creaseWidthFloor)
        tB = (float)1.0 - sharpFrac;
    if (t > creaseWidthCeil)
        tB = (float)1.0;

    ComputeMixedCreaseMatrix(sharpFloor, sharpCeil, tA, tB, m);
}

inline void swap(inout float a, inout float b)
{
    float t = a;
    a = b;
    b = t;
}

inline void FlipMatrix(inout float m[16])
{
    swap(m[0], m[15]);
    swap(m[1], m[14]);
    swap(m[2], m[13]);
    swap(m[3], m[12]);
    swap(m[4], m[11]);
    swap(m[5], m[10]);
    swap(m[6], m[9]);
    swap(m[7], m[8]);
}

inline void FlipMatrix(float a[16], out float m[16])
{
    m[0] = a[15];
    m[1] = a[14];
    m[2] = a[13];
    m[3] = a[12];
    m[4] = a[11];
    m[5] = a[10];
    m[6] = a[9];
    m[7] = a[8];
    m[8] = a[7];
    m[9] = a[6];
    m[10] = a[5];
    m[11] = a[4];
    m[12] = a[3];
    m[13] = a[2];
    m[14] = a[1];
    m[15] = a[0];
}

// v x m (column major)
inline void ApplyMatrix(inout float v[4], float m[16])
{
    float r[4];
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
    r[3] = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
    v[0] = r[0];
    v[1] = r[1];
    v[2] = r[2];
    v[3] = r[3];
}

void EvalBasisBSpline(float2 st, out float wP[16], out float wDs[16], out float wDt[16], int boundaryMask, float sharpness, bool PURE_BSPLINE)
{
    float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4], unused[4];

    EvalBSplineCurve(st.x, sWeights, dsWeights, unused, true, false);
    EvalBSplineCurve(st.y, tWeights, dtWeights, unused, true, false);

    if ((boundaryMask != 0 && sharpness > (float)0.0) && !PURE_BSPLINE)
    {
        float m[16], mflip[16];
        if (boundaryMask & 1)
        {
            ComputeCreaseMatrix(sharpness, (float)1.0 - st.y, m);
            FlipMatrix(m, mflip);
            ApplyMatrix(tWeights, mflip);
            ApplyMatrix(dtWeights, mflip);
        }
        if (boundaryMask & 2)
        {
            ComputeCreaseMatrix(sharpness, st.x, m);
            ApplyMatrix(sWeights, m);
            ApplyMatrix(dsWeights, m);
        }
        if (boundaryMask & 4)
        {
            ComputeCreaseMatrix(sharpness, st.y, m);
            ApplyMatrix(tWeights, m);
            ApplyMatrix(dtWeights, m);
        }
        if (boundaryMask & 8)
        {
            ComputeCreaseMatrix(sharpness, (float)1.0 - st.x, m);
            FlipMatrix(m, mflip);
            ApplyMatrix(sWeights, mflip);
            ApplyMatrix(dsWeights, mflip);
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            wP[4 * i + j] = sWeights[j] * tWeights[i];
            wDs[4 * i + j] = dsWeights[j] * tWeights[i];
            wDt[4 * i + j] = sWeights[j] * dtWeights[i];
        }
    }
}

//
//  Weight adjustments to account for phantom end points:
//
inline void AdjustBSplineBoundaryWeights(int boundary, inout float w[16])
{

    if ((boundary & 1) != 0)
    {
        for (int i = 0; i < 4; ++i)
        {
            w[i + 8] -= w[i + 0];
            w[i + 4] += w[i + 0] * 2.0f;
            w[i + 0] = 0.0f;
        }
    }
    if ((boundary & 2) != 0)
    {
        for (int i = 0; i < 16; i += 4)
        {
            w[i + 1] -= w[i + 3];
            w[i + 2] += w[i + 3] * 2.0f;
            w[i + 3] = 0.0f;
        }
    }
    if ((boundary & 4) != 0)
    {
        for (int i = 0; i < 4; ++i)
        {
            w[i + 4] -= w[i + 12];
            w[i + 8] += w[i + 12] * 2.0f;
            w[i + 12] = 0.0f;
        }
    }
    if ((boundary & 8) != 0)
    {
        for (int i = 0; i < 16; i += 4)
        {
            w[i + 2] -= w[i + 0];
            w[i + 1] += w[i + 0] * 2.0f;
            w[i + 0] = 0.0f;
        }
    }
}

inline void AdjustBSplineBoundaryTop(inout float w[4])
{
    w[1] -= w[3];
    w[2] += w[3] * 2.0f;
    w[3] = 0.0f;
}

inline void AdjustBSplineBoundaryBottom(inout float w[4])
{
    w[2] -= w[0];
    w[1] += w[0] * 2.0f;
    w[0] = 0.0f;
}

inline void AdjustBoundaries(float2 st, inout float sWeights[4], inout float tWeights[4], inout float dsWeights[4], inout float dtWeights[4], int boundaryMask)
{
    if ((boundaryMask & 1) != 0)
    {
        AdjustBSplineBoundaryBottom(tWeights);
        AdjustBSplineBoundaryBottom(dtWeights);
    }
    if ((boundaryMask & 4) != 0)
    {
        AdjustBSplineBoundaryTop(tWeights);
        AdjustBSplineBoundaryTop(dtWeights);
    }
    if ((boundaryMask & 2) != 0)
    {
        AdjustBSplineBoundaryTop(sWeights);
        AdjustBSplineBoundaryTop(dsWeights);
    }
    if ((boundaryMask & 8) != 0)
    {
        AdjustBSplineBoundaryBottom(sWeights);
        AdjustBSplineBoundaryBottom(dsWeights);
    }
}

// compute the "crease matrix" for modifying basis weights at parametric
// location 't', given a sharpness value (see Matthias Niessner derivation
// for 'single crease' regular patches)
inline void ComputeCreaseMatrixTop(float sharpness, float t, out float m[16])
{

    float sharpFloor = (float)floor(sharpness), sharpCeil = sharpFloor + 1, sharpFrac = sharpness - sharpFloor;
    float creaseWidthFloor = (float)1.0 - exp2(-sharpFloor), creaseWidthCeil = (float)1.0 - exp2(-sharpCeil);

    // we compute the matrix for both the floor and ceiling of
    // the sharpness value, and then interpolate between them
    // as needed.
    float tA = (t > creaseWidthCeil) ? sharpFrac : 0.0f;
    float tB = 0.0f;
    if (t > creaseWidthFloor)
        tB = 1.0f - sharpFrac;
    if (t > creaseWidthCeil)
        tB = 1.0f;

    ComputeMixedCreaseMatrix(sharpFloor, sharpCeil, tA, tB, m);
}

inline void ComputeCreaseMatrixBottom(float sharpness, float t, out float m[16])
{
    ComputeCreaseMatrixTop(sharpness, 1.0f - t, m);
    FlipMatrix(m);
}

inline void AdjustCreases(float2 st, inout float sWeights[4], inout float tWeights[4], inout float dsWeights[4], inout float dtWeights[4], int boundaryMask, float sharpness)
{
    if (boundaryMask & 1)
    {
        float m[16];
        ComputeCreaseMatrixBottom(sharpness, st.y, m);
        ApplyMatrix(tWeights, m);
        ApplyMatrix(dtWeights, m);
    }
    if (boundaryMask & 4)
    {
        float m[16];
        ComputeCreaseMatrixTop(sharpness, st.y, m);
        ApplyMatrix(tWeights, m);
        ApplyMatrix(dtWeights, m);
    }
    if (boundaryMask & 2)
    {
        float m[16];
        ComputeCreaseMatrixTop(sharpness, st.x, m);
        ApplyMatrix(sWeights, m);
        ApplyMatrix(dsWeights, m);
    }
    if (boundaryMask & 8)
    {
        float m[16];
        ComputeCreaseMatrixBottom(sharpness, st.x, m);
        ApplyMatrix(sWeights, m);
        ApplyMatrix(dsWeights, m);
    }
}



inline  void BoundBasisBSpline(int boundary, inout float wP[16], inout float wDs[16], inout float wDt[16])
{
    AdjustBSplineBoundaryWeights(boundary, wP);
    AdjustBSplineBoundaryWeights(boundary, wDs);
    AdjustBSplineBoundaryWeights(boundary, wDt);
}


//
//  Higher level basis evaluation functions that deal with parameterization and
//  boundary issues (reflected in PatchParam) for all patch types:
//
inline bool EvaluatePatchBasisNormalized(PatchDescriptorType  patchType,
    PatchParam param,
    float2                             st,
    out float                          wP[16],
    out float                          wDs[16],
    out float                          wDt[16],
    float                              sharpness)
{
    int boundaryMask = PatchParam_GetBoundary(param);

    bool hasPoints = false;
    if (patchType == PatchDescriptorType::REGULAR)
    {
        hasPoints = true;
        EvalBasisBSpline(st, wP, wDs, wDt, boundaryMask, sharpness, false);
        if (boundaryMask && (sharpness == float(0)))
        {
            BoundBasisBSpline(boundaryMask, wP, wDs, wDt);
        }
    }
    return hasPoints;
}


inline void EvaluatePatchBasis(PatchDescriptorType patchType,
    PatchParam param,
    float2                             st,
    out float                          wP[16],
    out float                          wDs[16],
    out float                          wDt[16],
    float                              sharpness = 0)
{
    float derivSign = 1.0f;

    if ((patchType == PatchDescriptorType::LOOP) || (patchType == PatchDescriptorType::GREGORY_TRIANGLE)
        || (patchType == PatchDescriptorType::TRIANGLES))
    {
        PatchParam_NormalizeTriangle(param, st.x, st.y);
        if (PatchParam_IsTriangleRotated(param))
        {
            derivSign = -1.0f;
        }
    }
    else
    {
        PatchParam_Normalize(param, st.x, st.y);
    }

    bool hasPoints = EvaluatePatchBasisNormalized(patchType, param, st, wP, wDs, wDt, sharpness);
    float d1Scale = derivSign * (float)(1U << PatchParam_GetDepth(param));

    if (hasPoints)
    {
        for (int i = 0; i < 16; ++i)
        {
            wDs[i] *= d1Scale;
            wDt[i] *= d1Scale;
        }
    }
    
}

#endif

struct SubdivisionPlanHLSL
{
    static const SchemeType kScheme = SchemeType::SCHEME_CATMARK;
    static const EndCapType kEndCap = EndCapType::ENDCAP_BSPLINE_BASIS;

    // scheme, endcap used to be dynamically defined
    SchemeType GetScheme() { return kScheme; }
    EndCapType GetEndCap() { return kEndCap; }

    uint16_t numControlPoints;
    uint16_t coarseFaceSize;
    int16_t  coarseFaceQuadrant;  // locates a surface within a non-quad parent face

    uint32_t treeOffset; // offset into m_subpatchTrees;
    uint32_t treeSize; // size of elements in m_subpatchTrees

    uint32_t patchPointsOffset; // index into m_patchPoints
    uint32_t patchPointsSize; // size of elements in m_patchPoints

    // Stencil matrix for computing patch points from control points:
    // - columns contain 1 scalar weight per control point of the 1-ring
    // - rows contain a stencil of weights for each patch point
    uint32_t stencilMatrixOffset; // index into m_stencilMatrix
    uint32_t stencilMatrixSize; // size of elements in m_stencilMatrix

    // member variables are not referenced (see GetScheme(), GetEndCap()) but are left in here
    // in case the caller wants to implement a dynamic path
    SchemeType scheme;
    EndCapType endCap;
};

#ifndef __cplusplus

float
computeSingleCreaseSharpness(SubdivisionNode n, NodeDescriptor desc, uint16_t depth, uint16_t level)
{
    if (kSingleCreaseDynamicIsolation == SingleCreaseDynamicIsolation::SHARP)
    {
        return desc.HasSharpness() ? n.GetSharpness() : 0.0f;
    }
    else if (kSingleCreaseDynamicIsolation == SingleCreaseDynamicIsolation::SMOOTH)
    {
        if (desc.HasSharpness())
        {
            float sharpness = n.GetSharpness();
            
            // single-crease patches require a non-null boundary mask and sharpness > 0.f
            // std::numeric_limits::min() ensures EvalBasisBSpline() evaluates the crease
            // matrix
            return max(kMinFloat, min(sharpness, float(level) - float(depth)));
        }
        return 0.f;
    }

    // Impossible path
    return 0.f;
}

struct SubdivisionPlanContext
{
    SubdivisionPlanHLSL m_data;

    StructuredBuffer<uint32_t> m_subpatchTrees;
    StructuredBuffer<Index> m_patchPoints; // indices into the stencil matrix weights
    StructuredBuffer<float> m_stencilMatrix; 

    TreeDescriptorHLSL GetTreeDescriptor()
    {
        TreeDescriptorHLSL treeDescriptor;
        treeDescriptor.m_subpatchTrees = m_subpatchTrees;
        treeDescriptor.m_treeOffset = m_data.treeOffset;
        return treeDescriptor;
    }

    bool IsBSplinePatch(uint16_t level)
    {
        return GetTreeDescriptor().GetNumPatchPoints(level) == 0;
    }

    SubdivisionNode GetRootNode()
    {
        SubdivisionNode node;
        node.m_subpatchTrees = m_subpatchTrees;
        node.m_patchPoints = m_patchPoints;
        node.m_nodeOffset = SubdivisionNode::rootNodeOffset();
        node.m_treeOffset = m_data.treeOffset;
        node.m_patchPointsOffset = m_data.patchPointsOffset;
        return node;
    }

    SubdivisionNode GetNode(float2 uv, inout uint16_t quadrant, uint16_t level)
    {
        // start at root node
        SubdivisionNode node = GetRootNode();
        NodeDescriptor desc = node.GetDesc();
        NodeType type = desc.GetType();
        bool isIrregular = !(GetTreeDescriptor().IsRegularFace());
        bool rotated = false;
        float median = 0.5f;

        while (type == NodeType::NODE_RECURSIVE)
        {
            if (desc.GetDepth() == level)
            {
                break;
            }
            switch (m_data.GetScheme())
            {
            case SchemeType::SCHEME_CATMARK:
                TraverseCatmark(uv.x, uv.y, quadrant);
                break;
            case SchemeType::SCHEME_LOOP:
                quadrant = TraverseLoop(median, uv.x, uv.y, rotated);
                break;
            default:
                // SCHEME_BILINEAR is not supported — only CATMARK and LOOP are used
                break;
            }

            node = node.GetChild(quadrant);
            desc = node.GetDesc();
            type = desc.GetType();

            median *= 0.5f;
        }
        return node;
    }

    SubdivisionNode EvaluateBasis(float2 st, out float wP[16], out float wDs[16], out float wDt[16], out uint16_t subpatch, uint16_t level)
    {
        PatchDescriptorType regularBasis = RegularBasisType(m_data.GetScheme());
        PatchDescriptorType irregularBasis = IrregularBasisType(m_data.GetScheme(), m_data.GetEndCap());

        bool isIrregular = !(GetTreeDescriptor().IsRegularFace());

        uint16_t quadrant = 0;
        SubdivisionNode node = GetNode(st, quadrant, level);

        NodeDescriptor desc = node.GetDesc();

        NodeType nodeType = desc.GetType();
        uint16_t depth = (uint16_t)desc.GetDepth();

        bool dynamicIsolation = (nodeType == NodeType::NODE_RECURSIVE) && (depth >= level) && desc.HasEndcap();

        uint16_t u = (uint16_t)desc.GetU();
        uint16_t v = (uint16_t)desc.GetV();

        PatchParam param;

        if (dynamicIsolation)
        {
            PatchParam_Set(param, INDEX_INVALID, u, v, depth, isIrregular, 0, 0, true);
            EvaluatePatchBasis(irregularBasis, param, st, wP, wDs, wDt);
        }
        else
        {
            PatchParam_Set(param, INDEX_INVALID, u, v, depth, isIrregular, (uint16_t)desc.GetBoundaryMask(), 0, true);
            switch (nodeType)
            {
            case NodeType::NODE_REGULAR:
            {
                float sharpness = computeSingleCreaseSharpness(node, desc, depth, level);
                EvaluatePatchBasis(regularBasis, param, st, wP, wDs, wDt, sharpness);
                break;
            }

            case NodeType::NODE_END:
                EvaluatePatchBasis(irregularBasis, param, st, wP, wDs, wDt);
                break;

            case NodeType::NODE_TERMINAL:
            case NodeType::NODE_RECURSIVE:
                // not handled
                break;
            }
        }
        subpatch = quadrant;
        return node;
    }
};
#endif


#ifndef __cplusplus
// clang-format off
static const float kBSplineWeights[] = {
    // cubic b-spline weight matrix
    1 / 6.0f, -3 / 6.0f,  3 / 6.0f, -1 / 6.0f,
    4 / 6.0f,  0 / 6.0f, -6 / 6.0f,  3 / 6.0f,
    1 / 6.0f,  3 / 6.0f,  3 / 6.0f, -3 / 6.0f,
    0 / 6.0f,  0 / 6.0f,  0 / 6.0f,  1 / 6.0f,
};
// clang-format on

inline float CubicBSplineWeight(float t, uint32_t index)
{
    float result = 0.0f;
    float factor = 1.0f;

    for (uint32_t i = 0; i < 4; ++i)
    {
        result += kBSplineWeights[4 * index + i] * factor;
        factor *= t;
    }
    return result;
}

// clang-format off
static const float kDerivativeWeights[] = {
    // cubic b-spline derivative weight matrix
    -1 / 2.0f,  2 / 2.0f, -1 / 2.0f,
     0 / 2.0f, -4 / 2.0f,  3 / 2.0f,
     1 / 2.0f,  2 / 2.0f, -3 / 2.0f,
     0 / 2.0f,  0 / 2.0f,  1 / 2.0f
};
// clang-format on

inline float CubicBSplineDerivativeWeight(float t, uint32_t index)
{
    float result = 0.0f;
    float factor = 1.0f;

    for (uint32_t i = 0; i < 3; ++i)
    {
        result += kDerivativeWeights[3 * index + i] * factor;
        factor *= t;
    }
    return result;
}

#endif