// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// meshoptimizer-based cluster LOD builder for RTX MegaGeo
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <cfloat>

// Forward declare clusterlod types
struct clodBounds;

struct MeshletCluster {
    std::vector<uint32_t> indices;     // Triangle indices into original vertex buffer
    std::vector<unsigned char> localTriangles; // Meshlet-local triangle indices (uint8)
    std::vector<uint32_t> localVertices;       // Meshlet-local vertex remap (global vertex per local slot)
    uint32_t vertexCount;              // Unique vertex count in this cluster
    uint32_t triangleCount;            // Number of triangles
    int refinedGroupId;                // Parent group in DAG, or -1 for base level

    // Bounding sphere + error metric
    float boundsCenter[3];
    float boundsRadius;
    float boundsError;
};

struct MeshletGroup {
    int depth;                         // DAG level
    // Simplified bounds for LOD selection
    float simplifiedCenter[3];
    float simplifiedRadius;
    float simplifiedError;
    std::vector<uint32_t> clusterIds;  // Clusters in this group
};

struct ClusterLODData {
    std::vector<MeshletCluster> clusters;
    std::vector<MeshletGroup> groups;

    // Original mesh data (kept for vertex positions during CLAS build)
    std::vector<float> vertexPositions;  // float3 packed (stride 12 bytes)
    size_t vertexCount;
    size_t vertexStride;  // 12 bytes (3 floats)

    // Max cluster dimensions (for template sizing)
    uint32_t maxTrianglesPerCluster;
    uint32_t maxVerticesPerCluster;
};

class ClusterLODBuilder {
public:
    // Build LOD DAG from triangle mesh
    // Called on worker thread (same as current async SubdivisionSurface creation)
    static std::unique_ptr<ClusterLODData> build(
        const uint32_t* indices, size_t indexCount,
        const float* vertexPositions, size_t vertexCount,
        size_t vertexPositionsStride);

    // Select which clusters to render for a given camera
    // Returns indices into ClusterLODData::clusters
    static std::vector<uint32_t> selectClusters(
        const ClusterLODData& data,
        const float cameraPos[3],
        float cameraProj,    // projection[1][1] = cot(fovy/2)
        float cameraZNear,
        float errorThreshold);  // screen-space error threshold (pixels / screenHeight)
};
