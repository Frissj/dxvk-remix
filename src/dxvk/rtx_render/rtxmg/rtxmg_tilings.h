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

#include "rtxmg_math_types.h"
#include "rtxmg_constants.h"

namespace dxvk {

// Cluster tiling within a surface patch
// Represents a grid of identical clusters
struct ClusterTiling {
  uint16_t2 tilingSize;    // Number of tiles in x and y direction
  uint16_t2 clusterSize;   // Number of quads in x and y direction inside each tile

  // Total number of clusters in this tiling
  uint32_t ClusterCount() const {
    return uint32_t(tilingSize.x) * uint32_t(tilingSize.y);
  }

  // Number of vertices per cluster
  uint32_t ClusterVertexCount() const {
    return (clusterSize.x + 1) * (clusterSize.y + 1);
  }

  // Total vertices in this tiling
  uint32_t VertexCount() const {
    return ClusterVertexCount() * ClusterCount();
  }

  // Convert linear cluster index to 2D grid position
  uint16_t2 ClusterIndex2D(uint32_t rowMajorIndex) const {
    return uint16_t2(
      static_cast<uint16_t>(rowMajorIndex % tilingSize.x),
      static_cast<uint16_t>(rowMajorIndex / tilingSize.x)
    );
  }

  // Get quad offset for a cluster in the surface grid
  uint16_t2 QuadOffset2D(uint32_t rowMajorIndex) const {
    uint16_t2 clusterIdx = ClusterIndex2D(rowMajorIndex);
    return uint16_t2(
      static_cast<uint16_t>(clusterIdx.x * clusterSize.x),
      static_cast<uint16_t>(clusterIdx.y * clusterSize.y)
    );
  }

  // Convert linear vertex index to 2D position within cluster
  uint2 VertexIndex2D(uint32_t rowMajorIndex) const {
    uint32_t verticesU = clusterSize.x + 1;
    return uint2(
      rowMajorIndex % verticesU,
      rowMajorIndex / verticesU
    );
  }
};

// Surface tiling subdivides a surface into multiple cluster grids
// Handles surfaces that don't divide evenly into clusters
struct SurfaceTiling {
  enum SubTiling {
    REGULAR = 0,  // Main regular grid of clusters
    RIGHT = 1,    // Right edge with different cluster size
    TOP = 2,      // Top edge with different cluster size
    CORNER = 3,   // Top-right corner with different cluster size
    N_SUB_TILINGS = 4
  };

  ClusterTiling subTilings[N_SUB_TILINGS];
  uint16_t2 quadOffsets[N_SUB_TILINGS];  // Quad offset of each sub-tiling

  // Total clusters across all sub-tilings
  uint32_t ClusterCount() const {
    uint32_t sum = 0;
    for (int i = 0; i < N_SUB_TILINGS; ++i) {
      sum += subTilings[i].ClusterCount();
    }
    return sum;
  }

  // Total vertices across all sub-tilings
  uint32_t VertexCount() const {
    uint32_t sum = 0;
    for (int i = 0; i < N_SUB_TILINGS; ++i) {
      sum += subTilings[i].VertexCount();
    }
    return sum;
  }

  // Get cluster offset in surface quad space
  uint16_t2 ClusterOffset(uint16_t iTiling, uint32_t iCluster) const {
    uint16_t2 offset = subTilings[iTiling].QuadOffset2D(iCluster);
    return uint16_t2(
      static_cast<uint16_t>(quadOffsets[iTiling].x + offset.x),
      static_cast<uint16_t>(quadOffsets[iTiling].y + offset.y)
    );
  }
};

// Create a surface tiling for a given surface size
// Divides surface into regular 8x8 clusters with edge/corner adjustment
inline SurfaceTiling MakeSurfaceTiling(uint16_t2 surfaceSize) {
  SurfaceTiling ret = {};
  const uint16_t targetEdgeSegments = 8;

  // Calculate how surface divides into target-sized clusters
  uint16_t2 regularGridSize;
  uint16_t2 modCluster;

  {
    uint16_t2 divClusters = uint16_t2(
      static_cast<uint16_t>(surfaceSize.x / targetEdgeSegments),
      static_cast<uint16_t>(surfaceSize.y / targetEdgeSegments)
    );

    modCluster = uint16_t2(
      static_cast<uint16_t>(surfaceSize.x % targetEdgeSegments),
      static_cast<uint16_t>(surfaceSize.y % targetEdgeSegments)
    );

    const uint32_t maxEdgeSegments = kMaxClusterEdgeSegments;

    // Try to merge remainder into edge clusters if it fits
    if (divClusters.x > 0 && modCluster.x + targetEdgeSegments <= maxEdgeSegments) {
      divClusters.x -= 1;
      modCluster.x += targetEdgeSegments;
    }

    if (divClusters.y > 0 && modCluster.y + targetEdgeSegments <= maxEdgeSegments) {
      divClusters.y -= 1;
      modCluster.y += targetEdgeSegments;
    }

    regularGridSize = divClusters;
  }

  // Regular grid (main body)
  ret.subTilings[SurfaceTiling::REGULAR].tilingSize = regularGridSize;
  ret.subTilings[SurfaceTiling::REGULAR].clusterSize = uint16_t2(targetEdgeSegments, targetEdgeSegments);
  ret.quadOffsets[SurfaceTiling::REGULAR] = uint16_t2(0, 0);

  // Right edge
  ret.subTilings[SurfaceTiling::RIGHT].tilingSize = uint16_t2(1, regularGridSize.y);
  ret.subTilings[SurfaceTiling::RIGHT].clusterSize = uint16_t2(modCluster.x, targetEdgeSegments);
  ret.quadOffsets[SurfaceTiling::RIGHT] = uint16_t2(
    static_cast<uint16_t>(regularGridSize.x * targetEdgeSegments),
    0
  );

  // Top edge
  ret.subTilings[SurfaceTiling::TOP].tilingSize = uint16_t2(regularGridSize.x, 1);
  ret.subTilings[SurfaceTiling::TOP].clusterSize = uint16_t2(targetEdgeSegments, modCluster.y);
  ret.quadOffsets[SurfaceTiling::TOP] = uint16_t2(
    0,
    static_cast<uint16_t>(regularGridSize.y * targetEdgeSegments)
  );

  // Top-right corner
  ret.subTilings[SurfaceTiling::CORNER].tilingSize = uint16_t2(1, 1);
  ret.subTilings[SurfaceTiling::CORNER].clusterSize = modCluster;
  ret.quadOffsets[SurfaceTiling::CORNER] = uint16_t2(
    static_cast<uint16_t>(regularGridSize.x * targetEdgeSegments),
    static_cast<uint16_t>(regularGridSize.y * targetEdgeSegments)
  );

  return ret;
}

} // namespace dxvk
