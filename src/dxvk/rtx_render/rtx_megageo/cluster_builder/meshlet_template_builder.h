// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// Per-meshlet CLAS template builder for RTX MegaGeo meshoptimizer path
#pragma once

#include "../nvrhi_adapter/nvrhi_types.h"
#include "../utils/buffer.h"
#include "cluster_lod_builder.h"

#include <vector>
#include <memory>
#include <string>

// Forward declarations
namespace dxvk {
  class NvrhiDxvkDevice;
  class RtxContext;
}

struct MeshletTemplateSet {
    // GPU buffers (persistent, built once per mesh)
    nvrhi::BufferHandle indexBuffer;        // Concatenated uint8 local indices for all meshlets
    nvrhi::BufferHandle vertexBuffer;       // Concatenated float3 positions for template building
    nvrhi::BufferHandle templateDataBuffer; // Opaque CLAS template data from driver

    // Per-meshlet metadata (CPU-side)
    std::vector<nvrhi::GpuVirtualAddress> templateAddresses; // GPU address of each meshlet's template
    std::vector<uint32_t> instantiationSizes;                // Byte size to instantiate each template

    // Keep size buffers alive (GPU caches address references)
    RTXMGBuffer<uint32_t> sizesBuffer;
    RTXMGBuffer<nvrhi::GpuVirtualAddress> addressesBuffer;
    RTXMGBuffer<uint32_t> instantiationSizesBuffer;

    uint32_t numTemplates = 0;
    uint32_t maxTrianglesPerMeshlet = 0;
    uint32_t maxVerticesPerMeshlet = 0;

    // Per-meshlet index/vertex offsets into the concatenated buffers
    // Used during per-frame instantiation to set vertex buffer addresses
    std::vector<uint32_t> meshletIndexOffsets;   // byte offset into indexBuffer for each meshlet
    std::vector<uint32_t> meshletVertexOffsets;   // vertex offset into vertexBuffer for each meshlet
    std::vector<uint32_t> meshletTriangleCounts;  // triangle count per meshlet
    std::vector<uint32_t> meshletVertexCounts;    // vertex count per meshlet

    bool isBuilt = false;
};

class MeshletTemplateBuilder {
public:
    // Build per-meshlet CLAS templates from ClusterLODData.
    // This is called once per mesh after the LOD DAG is ready (needs GPU command list).
    // The maxGeometryIndex should match what's used for CLAS instantiation.
    static std::unique_ptr<MeshletTemplateSet> build(
        const ClusterLODData& lodData,
        nvrhi::IDevice* device,
        nvrhi::ICommandList* commandList,
        uint32_t maxGeometryIndex,
        uint32_t quantNBits = 0);
};
