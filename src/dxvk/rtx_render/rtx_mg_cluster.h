/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include "rtx_types.h"

namespace dxvk {

  // Cluster tessellation constants (from RTXMG)
  constexpr uint32_t kMaxClusterEdgeSegments = 11;
  constexpr uint32_t kMaxClusterVertices = (kMaxClusterEdgeSegments + 1) * (kMaxClusterEdgeSegments + 1); // 144
  constexpr uint32_t kMaxClusterQuads = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments; // 121
  constexpr uint32_t kMaxClusterTriangles = kMaxClusterQuads * 2; // 242
  constexpr uint32_t kMaxApiClusterCount = 1 << 22; // 4M clusters max per NVAPI
  constexpr uint32_t kClusterTemplateCount = 121; // 11x11 combinations

  /**
   * \brief Cluster pattern type
   */
  enum class ClusterPattern : uint32_t {
    Regular = 0,  // Standard grid
    Slanted = 1   // Slanted grid for better distribution
  };

  /**
   * \brief Adaptive tessellation mode
   */
  enum class AdaptiveTessellationMode : uint32_t {
    Uniform = 0,                // Uniform tessellation
    WorldSpaceEdgeLength = 1,   // Based on world-space edge length
    SphericalProjection = 2     // Based on spherical projection (camera distance)
  };

  /**
   * \brief Visibility mode for cluster culling
   */
  enum class VisibilityMode : uint32_t {
    VisSurface = 0,      // Per-surface visibility
    VisLimitEdges = 1    // Limit based on edge visibility
  };

  /**
   * \brief Cluster vertex data
   *
   * Stores tessellated vertex position and normals.
   * Matches RTXMG Cluster structure.
   */
  struct ClusterVertex {
    Vector3 position;      // Vertex position in local space
    Vector3 normal;        // Vertex normal (optional, can be computed)
    Vector2 texCoord;      // Texture coordinates (subdivision UV)
    uint32_t clusterId;    // Cluster ID for debugging
  };

  /**
   * \brief Cluster shading data
   *
   * Per-cluster metadata for shading and visibility.
   */
  struct ClusterShadingData {
    uint32_t surfaceIndex;      // Source subdivision surface index
    uint32_t patchIndex;        // Source patch index
    uint32_t materialId;        // Material ID
    uint32_t instanceId;        // Instance ID
    Vector2 uvMin;              // Parametric UV bounds min
    Vector2 uvMax;              // Parametric UV bounds max
    float lodLevel;             // LOD level
    uint32_t vertexOffset;      // Offset into vertex buffer
    uint32_t indexOffset;       // Offset into index buffer
    uint32_t vertexCount;       // Number of vertices in cluster
    uint32_t triangleCount;     // Number of triangles in cluster
    uint32_t gridSizeX;         // Grid resolution X (1-11)
    uint32_t gridSizeY;         // Grid resolution Y (1-11)
    uint32_t visible;           // Visibility flag (1 = visible, 0 = culled)
  };

  /**
   * \brief Cluster statistics
   *
   * GPU-side statistics buffer.
   */
  struct ClusterStatistics {
    uint32_t totalClusters;
    uint32_t visibleClusters;
    uint32_t culledByHiZ;
    uint32_t culledByFrustum;
    uint32_t culledByBackface;
    uint32_t totalVertices;
    uint32_t totalTriangles;
    uint32_t memoryUsedBytes;
    float avgTessellationRate;
    float minEdgeLength;
    float maxEdgeLength;
    uint32_t pad[5]; // Pad to 64 bytes
  };

  /**
   * \brief Cluster tiling info
   *
   * Describes how a subdivision surface is tiled into clusters.
   */
  struct ClusterTilingInfo {
    uint32_t surfaceIndex;
    uint32_t patchIndex;
    uint32_t clusterOffsetX;
    uint32_t clusterOffsetY;
    uint32_t clusterCountX;
    uint32_t clusterCountY;
    uint32_t totalClusters;
    uint32_t baseClusterId;
    Matrix4 localToWorld;
    Vector4 boundingBox[2]; // Min, max
  };

  /**
   * \brief Tessellator configuration
   *
   * Configuration for adaptive tessellation and visibility culling.
   * Adapted from RTXMG TessellatorConfig.
   */
  struct TessellatorConfig {
    // Tessellation mode
    AdaptiveTessellationMode tessMode = AdaptiveTessellationMode::WorldSpaceEdgeLength;
    VisibilityMode visMode = VisibilityMode::VisLimitEdges;
    ClusterPattern clusterPattern = ClusterPattern::Regular;

    // Tessellation rates
    float fineTessellationRate = 1.0f;    // Fine tessellation multiplier
    float coarseTessellationRate = 0.5f;  // Coarse tessellation multiplier
    Vector4 edgeSegments = Vector4(11, 11, 11, 11); // Target edge segments [left, right, top, bottom]

    // Culling
    bool enableFrustumVisibility = true;
    bool enableHiZVisibility = true;
    bool enableBackfaceVisibility = true;

    // Vertex processing
    bool enableVertexNormals = true;
    uint32_t isolationLevel = 0;         // Subdivision isolation level
    uint32_t quantNBits = 16;            // Vertex quantization bits

    // Memory limits
    uint32_t maxClusters = 2097152;      // 2M default
    uint64_t clasBufferBytes = 3ULL * 1024 * 1024 * 1024;  // 3GB for CLAS
    uint64_t vertexBufferBytes = 1024 * 1024 * 1024;       // 1GB for vertices

    // View parameters (updated per frame)
    Vector2 viewportSize = Vector2(1920, 1080);
    Matrix4 viewMatrix;
    Matrix4 projMatrix;
    Matrix4 viewProjMatrix;
    Vector3 cameraPosition;
    Vector4 frustumPlanes[6];

    // HiZ buffer
    const void* hizBuffer = nullptr;
    uint32_t hizWidth = 0;
    uint32_t hizHeight = 0;
    uint32_t hizMipLevels = 0;
  };

  /**
   * \brief Cluster builder constants
   *
   * Shader constants for cluster building compute shaders.
   */
  struct ClusterBuilderConstants {
    TessellatorConfig config;
    uint32_t surfaceCount;
    uint32_t instanceCount;
    uint32_t frameIndex;
    uint32_t enableDebug;
  };

} // namespace dxvk
