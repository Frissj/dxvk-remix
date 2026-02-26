// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// meshoptimizer-based cluster LOD builder for RTX MegaGeo

#include "cluster_lod_builder.h"
#include "meshoptimizer.h"
#include "clusterlod.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// Log helper - uses DXVK logger when available
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"

namespace {

// Callback context for clodBuild
struct BuildContext {
    ClusterLODData* data;
    int nextGroupId;
};

int clodOutputCallback(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count) {
    auto* ctx = static_cast<BuildContext*>(output_context);
    int groupId = ctx->nextGroupId++;

    // Store the group
    MeshletGroup mg;
    mg.depth = group.depth;
    mg.simplifiedCenter[0] = group.simplified.center[0];
    mg.simplifiedCenter[1] = group.simplified.center[1];
    mg.simplifiedCenter[2] = group.simplified.center[2];
    mg.simplifiedRadius = group.simplified.radius;
    mg.simplifiedError = group.simplified.error;

    // Store each cluster in this group
    for (size_t i = 0; i < cluster_count; ++i) {
        uint32_t clusterId = static_cast<uint32_t>(ctx->data->clusters.size());
        mg.clusterIds.push_back(clusterId);

        MeshletCluster mc;
        mc.indices.assign(clusters[i].indices, clusters[i].indices + clusters[i].index_count);
        mc.vertexCount = static_cast<uint32_t>(clusters[i].vertex_count);
        mc.triangleCount = static_cast<uint32_t>(clusters[i].index_count / 3);
        mc.refinedGroupId = clusters[i].refined;

        mc.boundsCenter[0] = clusters[i].bounds.center[0];
        mc.boundsCenter[1] = clusters[i].bounds.center[1];
        mc.boundsCenter[2] = clusters[i].bounds.center[2];
        mc.boundsRadius = clusters[i].bounds.radius;
        mc.boundsError = clusters[i].bounds.error;

        // Extract meshlet-local indices using clodLocalIndices
        mc.localVertices.resize(mc.vertexCount);
        mc.localTriangles.resize(mc.indices.size());
        clodLocalIndices(mc.localVertices.data(), mc.localTriangles.data(),
                         mc.indices.data(), mc.indices.size());

        // Track max dimensions
        ctx->data->maxTrianglesPerCluster = std::max(ctx->data->maxTrianglesPerCluster, mc.triangleCount);
        ctx->data->maxVerticesPerCluster = std::max(ctx->data->maxVerticesPerCluster, mc.vertexCount);

        ctx->data->clusters.push_back(std::move(mc));
    }

    ctx->data->groups.push_back(std::move(mg));
    return groupId;
}

} // anonymous namespace


std::unique_ptr<ClusterLODData> ClusterLODBuilder::build(
    const uint32_t* indices, size_t indexCount,
    const float* vertexPositions, size_t vertexCount,
    size_t vertexPositionsStride)
{
    auto data = std::make_unique<ClusterLODData>();
    data->vertexCount = vertexCount;
    data->vertexStride = 12; // 3 floats
    data->maxTrianglesPerCluster = 0;
    data->maxVerticesPerCluster = 0;

    // Copy vertex positions into packed float3 format
    data->vertexPositions.resize(vertexCount * 3);
    size_t srcStride = vertexPositionsStride / sizeof(float);
    for (size_t i = 0; i < vertexCount; ++i) {
        data->vertexPositions[i * 3 + 0] = vertexPositions[i * srcStride + 0];
        data->vertexPositions[i * 3 + 1] = vertexPositions[i * srcStride + 1];
        data->vertexPositions[i * 3 + 2] = vertexPositions[i * srcStride + 2];
    }

    // Configure for RT-optimized clusterization
    // max_triangles=128 fits CLAS hardware limits
    clodConfig config = clodDefaultConfigRT(128);

    // Set up input mesh
    clodMesh mesh = {};
    mesh.indices = indices;
    mesh.index_count = indexCount;
    mesh.vertex_count = vertexCount;
    mesh.vertex_positions = data->vertexPositions.data();
    mesh.vertex_positions_stride = 12; // packed float3

    BuildContext ctx;
    ctx.data = data.get();
    ctx.nextGroupId = 0;

    size_t totalClusters = clodBuild(config, mesh, &ctx, clodOutputCallback);

    dxvk::Logger::info(dxvk::str::format(
        "ClusterLODBuilder: built LOD DAG with ", data->clusters.size(), " clusters, ",
        data->groups.size(), " groups from ", indexCount / 3, " triangles (",
        vertexCount, " vertices), maxTri=", data->maxTrianglesPerCluster,
        " maxVtx=", data->maxVerticesPerCluster));

    (void)totalClusters;
    return data;
}


std::vector<uint32_t> ClusterLODBuilder::selectClusters(
    const ClusterLODData& data,
    const float cameraPos[3],
    float cameraProj,
    float cameraZNear,
    float errorThreshold)
{
    std::vector<uint32_t> selected;
    if (data.clusters.empty() || data.groups.empty())
        return selected;

    // Pre-compute projected error for each group's simplified bounds
    // projected error = bounds.error / max(distance - radius, znear) * (proj * 0.5)
    auto projectedError = [&](const float center[3], float radius, float error) -> float {
        if (error >= FLT_MAX * 0.5f)
            return FLT_MAX; // Terminal group — always over threshold
        float dx = center[0] - cameraPos[0];
        float dy = center[1] - cameraPos[1];
        float dz = center[2] - cameraPos[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        float denom = std::max(dist - radius, cameraZNear);
        return (error / denom) * (cameraProj * 0.5f);
    };

    // For each cluster, decide if it should be rendered:
    // 1. The group it belongs to has projected error OVER threshold (group wants to show detail)
    // 2. Its refined parent group is either -1 (base level) or has projected error AT/UNDER threshold
    //    (parent is satisfied with its simplified representation)
    for (uint32_t ci = 0; ci < data.clusters.size(); ++ci) {
        const MeshletCluster& cluster = data.clusters[ci];

        // Find which group this cluster belongs to
        // (linear search is fine — this runs per-frame but groups are typically small)
        int myGroupId = -1;
        for (uint32_t gi = 0; gi < data.groups.size(); ++gi) {
            for (uint32_t id : data.groups[gi].clusterIds) {
                if (id == ci) {
                    myGroupId = static_cast<int>(gi);
                    break;
                }
            }
            if (myGroupId >= 0) break;
        }

        if (myGroupId < 0) continue; // orphan cluster, skip

        const MeshletGroup& myGroup = data.groups[myGroupId];

        // Condition 1: my group's error is over threshold (this level wants to show)
        float myGroupError = projectedError(myGroup.simplifiedCenter, myGroup.simplifiedRadius, myGroup.simplifiedError);
        if (myGroupError <= errorThreshold)
            continue; // This whole group is fine at coarser level

        // Condition 2: parent group (refined) is either absent or under threshold
        if (cluster.refinedGroupId >= 0 && cluster.refinedGroupId < static_cast<int>(data.groups.size())) {
            const MeshletGroup& parentGroup = data.groups[cluster.refinedGroupId];
            float parentError = projectedError(parentGroup.simplifiedCenter, parentGroup.simplifiedRadius, parentGroup.simplifiedError);
            if (parentError > errorThreshold)
                continue; // Parent also wants to show — use parent's clusters instead
        }
        // else: refinedGroupId == -1 means base level, always eligible

        selected.push_back(ci);
    }

    return selected;
}
