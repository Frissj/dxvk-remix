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
#include <cstddef>

namespace dxvk {

// Cluster acceleration structure byte alignment
// Per VK_NV_cluster_acceleration_structure spec
static constexpr size_t kClasByteAlignment = 256;

// Tessellation counters for cluster building
// Scratch device memory needed while filling clusters
struct TessellationCounters {
  uint32_t clusters;           // Number of clusters generated
  uint32_t desiredClusters;    // Number of clusters we want to generate
  uint32_t desiredVertices;    // Number of vertices we want to generate
  uint32_t desiredTriangles;   // Number of triangles we want to generate
  uint32_t desiredClasBlocks;  // Number of CLAS blocks we want to allocate

  // Pad for Vulkan minStorageBufferOffsetAlignment = 16
  uint32_t pad[3];

  size_t DesiredClasBytes() const {
    return size_t(desiredClasBlocks) * kClasByteAlignment;
  }
};

// Byte offset to cluster count field
static constexpr uint32_t kClusterCountByteOffset = offsetof(TessellationCounters, clusters);

// Note: ClusterStatistics is defined in rtx_mg_cluster.h
// Use RtxmgStatistics for local stats tracking
struct RtxmgStatistics {
  struct BufferStatistics {
    uint32_t m_numClusters = 0;
    uint32_t m_numTriangles = 0;
    size_t m_blasScratchSize = 0;
    size_t m_blasSize = 0;
    size_t m_vertexBufferSize = 0;
    size_t m_vertexNormalsBufferSize = 0;
    size_t m_clasSize = 0;
    size_t m_clusterDataSize = 0;
  };

  BufferStatistics desired;    // Desired allocation sizes
  BufferStatistics allocated;  // Actually allocated sizes
};

} // namespace dxvk
