/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Adapted from NVIDIA RTX Mega Geometry SDK for RTX Remix integration
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*/

#pragma once

#include <cstdint>
#include "rtxmg_math_types.h"
#include "rtxmg_cluster.h"

namespace dxvk {

// RTXMG Tessellator configuration
// Simplified version adapted from NVIDIA RTXMG SDK for RTX Remix
struct RtxmgConfig {
  // Default tessellation rates
  static constexpr float kDefaultFineTessellationRate = 1.0f;
  static constexpr float kDefaultCoarseTessellationRate = 1.0f / 15.0f;

  // Default memory limits
  static constexpr uint32_t kDefaultMaxClusters = (1u << 21);        // 2M clusters
  static constexpr size_t kDefaultVertexBufferBytes = (1024ull << 20);  // 1GB vertices
  static constexpr size_t kDefaultClasBufferBytes = (3076ull << 20);    // 3GB CLAS

  static constexpr uint32_t kMinIsolationLevel = 1u;
  static constexpr uint32_t kMaxIsolationLevel = 6u;

  // Visibility culling mode
  enum class VisibilityMode {
    VIS_LIMIT_EDGES = 0,   // Cull based on cluster edge visibility
    VIS_SURFACE = 1,       // Cull based on surface 1-ring visibility
    COUNT
  };

  // Adaptive tessellation mode
  enum class AdaptiveTessellationMode {
    UNIFORM = 0,                 // Uniform tessellation (no adaptation)
    WORLD_SPACE_EDGE_LENGTH,     // Adapt based on world-space edge length
    SPHERICAL_PROJECTION,        // Adapt based on spherical projection
    COUNT
  };

  // Memory allocation settings
  struct MemorySettings {
    uint32_t maxClusters = kDefaultMaxClusters;
    uint32_t maxVertices = (1u << 24);  // 16M vertices
    uint32_t maxClasBlocks = (1u << 20); // 1M CLAS blocks
    size_t clasBufferBytes = kDefaultClasBufferBytes;
    size_t vertexBufferBytes = kDefaultVertexBufferBytes;

    bool operator==(const MemorySettings& o) const {
      return vertexBufferBytes == o.vertexBufferBytes &&
             maxClusters == o.maxClusters &&
             maxVertices == o.maxVertices &&
             maxClasBlocks == o.maxClasBlocks &&
             clasBufferBytes == o.clasBufferBytes;
    }
  };

  // Configuration members
  MemorySettings memorySettings;
  VisibilityMode visMode = VisibilityMode::VIS_LIMIT_EDGES;
  AdaptiveTessellationMode tessMode = AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH;

  float fineTessellationRate = kDefaultFineTessellationRate;
  float coarseTessellationRate = kDefaultCoarseTessellationRate;

  bool enableFrustumVisibility = true;
  bool enableHiZVisibility = true;
  bool enableBackfaceVisibility = true;
  bool enableHiZCulling = true;           // Enable HiZ occlusion culling (Phase 4)
  bool useGPUCompute = true;              // Use GPU compute shaders (Phase 5, falls back to CPU if false)
  bool enableLogging = false;
  bool enableMonolithicClusterBuild = false;
  bool enableVertexNormals = false;

  uint2 viewportSize = {0u, 0u};
  uint4 edgeSegments = {8, 8, 8, 8};
  uint32_t isolationLevel = 0;  // 0 is dynamic, >0 is fixed
  ClusterPattern clusterPattern = ClusterPattern::Slanted;
  unsigned char quantNBits = 0;

  // Per-frame rendering parameters (updated before each dispatch)
  Matrix4 matWorldToClip;  // World-to-clip space transformation matrix
  Vector3 cameraPos;       // Camera position in world space

  // Note: displacementScale excluded per user request (no displacement mapping)
  // RTX Remix uses POM (parallax occlusion mapping) instead

  int debugSurfaceIndex = 0;
  int debugClusterIndex = 0;
  int debugLaneIndex = 0;
};

// Visibility mode names for UI
static constexpr const char* kVisibilityModeNames[] = {
  "Limit Edge",
  "Surface 1-Ring"
};
static_assert(sizeof(kVisibilityModeNames) / sizeof(kVisibilityModeNames[0]) ==
              static_cast<size_t>(RtxmgConfig::VisibilityMode::COUNT));

// Adaptive tessellation mode names for UI
static constexpr const char* kAdaptiveTessellationModeNames[] = {
  "Uniform",
  "WS Edge Length",
  "Spherical Projection"
};
static_assert(sizeof(kAdaptiveTessellationModeNames) / sizeof(kAdaptiveTessellationModeNames[0]) ==
              static_cast<size_t>(RtxmgConfig::AdaptiveTessellationMode::COUNT));

} // namespace dxvk
