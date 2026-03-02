// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// GPU-driven meshlet cluster fill shader parameters
// Shared between C++ and Slang shader code

#pragma once

#include "../nvrhi/nvrhiHLSL.h"

static const uint32_t kFillMeshletClustersThreads = 64;

// Per-cluster selection info, uploaded from CPU each frame.
// Contains everything the GPU shader needs to fill per-frame buffers.
struct MeshletClusterGPUInfo {
    uint32_t persistentExpandedVtxOffset;   // vertex offset into persistent expanded vtx/norm buffers
    uint32_t persistentUniqueVtxOffset;     // vertex offset into persistent unique (template) vtx buffer
    uint32_t perFrameExpandedVtxOffset;     // vertex offset into per-frame vtx/norm buffers (prefix sum)
    uint32_t perFrameUniqueVtxOffset;       // vertex offset into per-frame unique section (prefix sum)
    uint32_t triCount;                      // triangles in this meshlet (expanded vtx count = triCount * 3)
    uint32_t uniqueVtxCount;                // unique vertices in this meshlet
    uint32_t surfaceId;                     // surface ID from instance (for shading data)
    uint32_t clusterIndex;                  // global cluster index (for CLAS clusterIdOffset/geometryIndex)
    nvrhi::GpuVirtualAddress templateAddr;  // CLAS template GPU address
    nvrhi::GpuVirtualAddress clasDestAddr;  // pre-computed CLAS destination GPU address (128-byte aligned)
};

#if defined(__cplusplus)
static_assert(sizeof(MeshletClusterGPUInfo) == 48, "MeshletClusterGPUInfo must be 48 bytes");
#endif

// Push constant parameters for fill_meshlet_clusters shader
struct FillMeshletClustersParams {
    uint32_t numClusters;
    uint32_t totalExpandedVertices;              // total per-tri expanded vertices (for unique vtx base offset)
    nvrhi::GpuVirtualAddress perFrameVertexAddr; // per-frame vertex buffer GPU address (for CLAS vertexBuffer)
};

#if defined(__cplusplus)
static_assert(sizeof(FillMeshletClustersParams) == 16, "FillMeshletClustersParams must be 16 bytes");
static_assert(sizeof(FillMeshletClustersParams) % 16 == 0, "Must be 16 byte aligned");
#endif
