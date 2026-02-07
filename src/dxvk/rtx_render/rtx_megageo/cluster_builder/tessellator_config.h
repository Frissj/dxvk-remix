/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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


#pragma once

#include "cluster.h"
#include "../scene/camera.h"

class ZBuffer;

struct TessellatorConfig
{
    // Default rate for SPHERICAL_PROJECTION mode (screen-space tessellation)
    // Formula: edgeRate = viewportHeight * rate / viewDepth
    // With viewport 720 and rate 0.05: at depth 10 -> edgeRate = 720*0.05/10 = 3.6 segments per unit
    // Lower values = less tessellation, higher = more tessellation
    static constexpr float kDefaultFineTessellationRate = 0.05f;
    static constexpr float kDefaultCoarseTessellationRate = 0.05f / 15.0f;

    // 2M clusters
    static constexpr uint32_t kDefaultMaxClusters = (1u << 21);

    // 1024MB vertices at 1440p render res
    static constexpr size_t kDefaultVertexBufferBytes = (1024ull << 20);

    // 3GB CLAS memory at 1440p render res
    static constexpr size_t kDefaultClasBufferBytes = (3076ull << 20);

    static constexpr uint32_t kMinIsolationLevel = 1u;
    static constexpr uint32_t kMaxIsolationLevel = 6u;

    enum class VisibilityMode
    {
        VIS_LIMIT_EDGES = 0,
        VIS_SURFACE = 1,
        COUNT
    };

    enum class AdaptiveTessellationMode
    {
        UNIFORM = 0,
        WORLD_SPACE_EDGE_LENGTH,
        SPHERICAL_PROJECTION,
        COUNT
    };
    
    struct MemorySettings
    {
        uint32_t maxClusters = kDefaultMaxClusters;
        size_t clasBufferBytes = kDefaultClasBufferBytes;
        size_t vertexBufferBytes = kDefaultVertexBufferBytes;

        bool operator==(const MemorySettings& o) const
        {
            return vertexBufferBytes == o.vertexBufferBytes &&
                maxClusters == o.maxClusters &&
                clasBufferBytes == o.clasBufferBytes;
        }
    };
    
    MemorySettings memorySettings;
    VisibilityMode visMode = VisibilityMode::VIS_LIMIT_EDGES;
    // SPHERICAL_PROJECTION using screenspace (clip.w depth) - no camera position needed
    AdaptiveTessellationMode tessMode = AdaptiveTessellationMode::SPHERICAL_PROJECTION;

    float fineTessellationRate = kDefaultFineTessellationRate;
    float coarseTessellationRate = kDefaultCoarseTessellationRate;
    bool  enableFrustumVisibility = true;
    bool  enableHiZVisibility = true;
    // Backface visibility using screenspace (clip-space normal) - no camera position needed
    bool  enableBackfaceVisibility = true;
    bool  enableLogging = false; // enable debug logging for tessellator build
    bool  enableMonolithicClusterBuild = true;  // Must use monolithic - only one shader permutation is pre-compiled
    bool  enableVertexNormals = false; // enable vertex normal computation
    bool  disableSubdivision = false; // bypass stencil evaluation, use bilinear interpolation of control points

    uint2            viewportSize = { 0u, 0u };
    uint4            edgeSegments = { 8, 8, 8, 8 };
    uint32_t         isolationLevel = kMaxIsolationLevel; // Match sample: use max isolation level (6) for correct patch point evaluation
    ClusterPattern   clusterPattern = ClusterPattern::SLANTED;
    unsigned char    quantNBits = 0;

    float            displacementScale = 1.0f;

    const dxvk::Camera* camera = nullptr;
    const ZBuffer* zbuffer = nullptr;

    int debugSurfaceIndex = -1;  // -1 to disable debug
    int debugClusterIndex = -1;
    int debugLaneIndex = -1;
};

#if __cplusplus
#include <array>
constexpr std::array<const char*, 3> kAdaptiveTessellationModeNames = {
    "Uniform",
    "WS Edge Length",
    "Spherical Projection"
};
static_assert(kAdaptiveTessellationModeNames.size() == size_t(TessellatorConfig::AdaptiveTessellationMode::COUNT));

constexpr std::array<const char*, 2> kVisibilityModeNames = {
    "Limit Edge",
    "Surface 1-Ring"
};
static_assert(kVisibilityModeNames.size() == size_t(TessellatorConfig::VisibilityMode::COUNT));
#endif