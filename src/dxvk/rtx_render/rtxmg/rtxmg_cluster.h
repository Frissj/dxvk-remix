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
#include <cmath>
#include <algorithm>
#include "../../util/util_vector.h"
#include "rtxmg_constants.h"
#include "../rtx_mg_cluster.h"  // Use existing types

namespace dxvk {

// Use ClusterPattern and ClusterShadingData from rtx_mg_cluster.h

// RTXMG cluster shape types
enum class ClusterShape {
  RECTANGULAR,
  SQUARE,
};

// Grid sampler for computing UV sample locations on rectangular grid
struct GridSampler {
  uint16_t edgeSegments[4];  // [x, y, z, w] edge segments

  uint16_t GridSizeX() const {
    return std::max(edgeSegments[0], edgeSegments[2]);
  }

  uint16_t GridSizeY() const {
    return std::max(edgeSegments[1], edgeSegments[3]);
  }

  bool IsEmpty() const {
    return GridSizeX() == 0 && GridSizeY() == 0;
  }
};

// RTXMG Cluster structure
// Represents a tessellated cluster primitive with grid-based sampling
struct RtxmgCluster {
  uint32_t iSurface;        // Index of the surface (patch) generating this cluster
  uint32_t nVertexOffset;   // Vertex array index of this cluster's [0,0] corner
  uint16_t offsetX;         // Cluster's X offset inside sample grid
  uint16_t offsetY;         // Cluster's Y offset inside sample grid
  uint8_t sizeX;            // Cluster's X size (number of quads)
  uint8_t sizeY;            // Cluster's Y size (number of quads)
  uint8_t pad0;
  uint8_t pad1;

  uint32_t VerticesPerCluster() const {
    return (sizeX + 1) * (sizeY + 1);
  }

  uint32_t QuadsPerCluster() const {
    return sizeX * sizeY;
  }

  uint32_t TrianglesPerCluster() const {
    return 2 * QuadsPerCluster();
  }
};

// RTXMG Cluster Shading Data (matches shader structure in rtxmg_bindings.slangh)
// This is GPU-side data written by compute_cluster_tiling.comp.slang
struct RtxmgClusterShadingData {
  uint16_t edgeSegments[4];    // Edge segment counts
  Vector2 texcoords[4];        // Corner texture coordinates
  uint32_t surfaceId;          // Surface identifier
  uint32_t vertexOffset;       // Offset into vertex buffer
  uint16_t clusterOffsetX;     // Cluster grid offset X
  uint16_t clusterOffsetY;     // Cluster grid offset Y
  uint8_t clusterSizeX;        // Cluster size X
  uint8_t clusterSizeY;        // Cluster size Y
  uint16_t pad0;
  uint32_t stableClusterId;    // NV-DXVK: Stable cluster ID (UV-based hash, camera-stable)
};
static_assert(sizeof(RtxmgClusterShadingData) == 60, "RtxmgClusterShadingData size mismatch with shader");

// Note: ClusterShadingData is defined in rtx_mg_cluster.h (legacy subdivision surfaces)

// Cluster instance data for NVAPI
struct ClusterInstanceData {
  uint32_t clusterIndex;     // Index into cluster template array
  uint32_t vertexOffset;     // Offset into vertex buffer
  uint32_t surfaceId;        // Surface/material identifier
  Matrix4 transform;         // Instance transform
};

} // namespace dxvk
