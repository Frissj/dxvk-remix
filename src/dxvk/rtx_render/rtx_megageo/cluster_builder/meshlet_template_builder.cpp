// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// Per-meshlet CLAS template builder for RTX MegaGeo meshoptimizer path

#include "meshlet_template_builder.h"
#include "../utils/buffer.h"
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"

using namespace dxvk;
using namespace nvrhi::rt;

// Helper to align buffer sizes to 4 bytes (required for Vulkan vkCmdUpdateBuffer)
static inline size_t alignBufferSize4(size_t size) {
    return (size + 3) & ~3;
}

std::unique_ptr<MeshletTemplateSet> MeshletTemplateBuilder::build(
    const ClusterLODData& lodData,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    uint32_t maxGeometryIndex,
    uint32_t quantNBits)
{
    auto result = std::make_unique<MeshletTemplateSet>();
    const uint32_t numMeshlets = static_cast<uint32_t>(lodData.clusters.size());

    if (numMeshlets == 0) {
        Logger::warn("MeshletTemplateBuilder: No meshlets to build templates for");
        return result;
    }

    result->numTemplates = numMeshlets;
    result->maxTrianglesPerMeshlet = lodData.maxTrianglesPerCluster;
    result->maxVerticesPerMeshlet = lodData.maxVerticesPerCluster;

    Logger::info(str::format("MeshletTemplateBuilder: Building ", numMeshlets,
        " templates, maxTri=", lodData.maxTrianglesPerCluster,
        " maxVtx=", lodData.maxVerticesPerCluster,
        " maxGeomIdx=", maxGeometryIndex));

    // =====================================================================
    // Step 1: Concatenate all meshlet local indices and vertex positions
    // =====================================================================

    // Calculate total sizes and per-meshlet offsets
    result->meshletIndexOffsets.resize(numMeshlets);
    result->meshletVertexOffsets.resize(numMeshlets);
    result->meshletTriangleCounts.resize(numMeshlets);
    result->meshletVertexCounts.resize(numMeshlets);

    uint32_t totalIndices = 0;    // Total uint8 local triangle indices (3 per tri)
    uint32_t totalVertices = 0;   // Total unique vertices across all meshlets
    uint32_t totalTriangles = 0;

    for (uint32_t m = 0; m < numMeshlets; ++m) {
        const MeshletCluster& mc = lodData.clusters[m];
        result->meshletIndexOffsets[m] = totalIndices;
        result->meshletVertexOffsets[m] = totalVertices;
        result->meshletTriangleCounts[m] = mc.triangleCount;
        result->meshletVertexCounts[m] = mc.vertexCount;

        totalIndices += static_cast<uint32_t>(mc.localTriangles.size()); // 3 * triangleCount
        totalVertices += mc.vertexCount;
        totalTriangles += mc.triangleCount;
    }

    Logger::info(str::format("MeshletTemplateBuilder: Total indices=", totalIndices,
        " vertices=", totalVertices, " triangles=", totalTriangles));

    // Concatenate index data (uint8 local indices)
    std::vector<uint8_t> allIndices(totalIndices);
    for (uint32_t m = 0; m < numMeshlets; ++m) {
        const MeshletCluster& mc = lodData.clusters[m];
        memcpy(allIndices.data() + result->meshletIndexOffsets[m],
               mc.localTriangles.data(),
               mc.localTriangles.size());
    }

    // Concatenate vertex positions (float3, 12 bytes per vertex)
    // For each meshlet, extract positions from the original mesh via localVertices
    std::vector<float> allVertices(totalVertices * 3);
    for (uint32_t m = 0; m < numMeshlets; ++m) {
        const MeshletCluster& mc = lodData.clusters[m];
        float* dst = allVertices.data() + result->meshletVertexOffsets[m] * 3;
        for (uint32_t v = 0; v < mc.vertexCount; ++v) {
            uint32_t globalVtx = mc.localVertices[v];
            dst[v * 3 + 0] = lodData.vertexPositions[globalVtx * 3 + 0];
            dst[v * 3 + 1] = lodData.vertexPositions[globalVtx * 3 + 1];
            dst[v * 3 + 2] = lodData.vertexPositions[globalVtx * 3 + 2];
        }
    }

    // =====================================================================
    // Step 2: Upload index and vertex buffers to GPU
    // =====================================================================

    size_t indexDataSize = allIndices.size() * sizeof(uint8_t);
    nvrhi::BufferDesc indexBufferDesc = {
        .byteSize = alignBufferSize4(indexDataSize),
        .debugName = "MeshletTemplateIndices",
        .structStride = sizeof(uint8_t),
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };
    result->indexBuffer = device->createBuffer(indexBufferDesc);
    if (!allIndices.empty()) {
        commandList->writeBuffer(result->indexBuffer, allIndices.data(), indexDataSize);
    }

    size_t vertexDataSize = allVertices.size() * sizeof(float);
    nvrhi::BufferDesc vertexBufferDesc = {
        .byteSize = vertexDataSize,
        .debugName = "MeshletTemplateVertices",
        .format = nvrhi::Format::RGB32_FLOAT,
        .isVertexBuffer = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };
    result->vertexBuffer = device->createBuffer(vertexBufferDesc);
    if (!allVertices.empty()) {
        commandList->writeBuffer(result->vertexBuffer, allVertices.data(), vertexDataSize);
    }

    // Barrier: writeBuffer goes through DXVK transfer, but template build bypasses DXVK
    commandList->bufferBarrier(result->indexBuffer.Get(), nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->bufferBarrier(result->vertexBuffer.Get(), nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::AccelStructBuildInput);

    // =====================================================================
    // Step 3: Build IndirectTriangleTemplateArgs per meshlet
    // =====================================================================

    nvrhi::GpuVirtualAddress indexBufferAddress = result->indexBuffer->getGpuVirtualAddress();
    nvrhi::GpuVirtualAddress vertexBufferAddress = result->vertexBuffer->getGpuVirtualAddress();

    if (indexBufferAddress == 0 || vertexBufferAddress == 0) {
        Logger::err("MeshletTemplateBuilder: NULL buffer address!");
        return result;
    }

    // Use 8-bit index format for meshlet local indices
    uint32_t indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat8bit);

    std::vector<cluster::IndirectTriangleTemplateArgs> templateArgs(numMeshlets);
    for (uint32_t m = 0; m < numMeshlets; ++m) {
        templateArgs[m] = {};
        templateArgs[m] = cluster::IndirectTriangleTemplateArgs{
            .clusterId = 0,
            .clusterFlags = 0,
            .triangleCount = result->meshletTriangleCounts[m],
            .vertexCount = result->meshletVertexCounts[m],
            .positionTruncateBitCount = 0,
            .indexFormat = indexFormat,
            .opacityMicromapIndexFormat = 0,
            .baseGeometryIndexAndFlags = 0,
            .indexBufferStride = sizeof(uint8_t),
            .vertexBufferStride = 12, // float3, scalar layout
            .geometryIndexAndFlagsBufferStride = 0,
            .opacityMicromapIndexBufferStride = 0,
            .indexBuffer = indexBufferAddress + result->meshletIndexOffsets[m],
            .vertexBuffer = vertexBufferAddress + result->meshletVertexOffsets[m] * 12,
            .geometryIndexAndFlagsBuffer = 0,
            .opacityMicromapArray = 0,
            .opacityMicromapIndexBuffer = 0,
            .instantiationBoundingBoxLimit = 0
        };
    }

    nvrhi::BufferDesc templateArgsDesc = {
        .byteSize = templateArgs.size() * sizeof(templateArgs[0]),
        .debugName = "MeshletTemplateArgs",
        .structStride = sizeof(templateArgs[0]),
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle templateArgsBuffer = device->createBuffer(templateArgsDesc);
    commandList->writeBuffer(templateArgsBuffer, templateArgs.data(),
        templateArgs.size() * sizeof(templateArgs[0]));

    // =====================================================================
    // Step 4: GetSizes for template build
    // =====================================================================

    cluster::OperationParams operationParams = {
        .maxArgCount = numMeshlets,
        .type = cluster::OperationType::ClasBuildTemplates,
        .mode = cluster::OperationMode::GetSizes,
        .flags = cluster::OperationFlags::None,
        .clas = {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxGeometryIndex = maxGeometryIndex,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = lodData.maxTrianglesPerCluster,
            .maxVertexCount = lodData.maxVerticesPerCluster,
            .maxTotalTriangleCount = totalTriangles,
            .maxTotalVertexCount = totalVertices,
            .minPositionTruncateBitCount = quantNBits,
        }
    };
    cluster::OperationSizeInfo sizeInfo = device->getClusterOperationSizeInfo(operationParams);

    result->sizesBuffer.Create(numMeshlets, "MeshletTemplateSizes", device);

    cluster::OperationDesc getSizesDesc = {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = templateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = result->sizesBuffer.Get(),
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(getSizesDesc);

    // Readback template sizes (requires command list flush)
    std::vector<uint32_t> templateSizes = result->sizesBuffer.Download(commandList);

    // =====================================================================
    // Step 5: Allocate template data buffer, compute addresses
    // =====================================================================

    size_t totalTemplateSize = 0;
    for (uint32_t i = 0; i < numMeshlets; ++i) {
        totalTemplateSize += templateSizes[i];
    }

    Logger::info(str::format("MeshletTemplateBuilder: Total template data size = ",
        totalTemplateSize, " bytes"));

    nvrhi::BufferDesc dataBufferDesc = {
        .byteSize = totalTemplateSize,
        .debugName = "MeshletTemplateData",
        .canHaveUAVs = true,
        .isAccelStructStorage = true,
        .initialState = nvrhi::ResourceStates::AccelStructWrite,
        .keepInitialState = true,
    };
    result->templateDataBuffer = device->createBuffer(dataBufferDesc);

    nvrhi::GpuVirtualAddress baseAddress = result->templateDataBuffer->getGpuVirtualAddress();
    result->templateAddresses.resize(numMeshlets);
    size_t offset = 0;
    for (uint32_t i = 0; i < numMeshlets; ++i) {
        result->templateAddresses[i] = baseAddress + offset;
        offset += templateSizes[i];
    }

    result->addressesBuffer.Create(numMeshlets, "MeshletTemplateAddresses", device);
    result->addressesBuffer.Upload(result->templateAddresses, commandList);

    // =====================================================================
    // Step 6: Build templates (ExplicitDestinations mode)
    // =====================================================================

    operationParams.mode = cluster::OperationMode::ExplicitDestinations;

    cluster::OperationDesc buildDesc = {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = templateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = result->addressesBuffer.Get(),
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = 0,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(buildDesc);

    // =====================================================================
    // Step 7: Get instantiation sizes
    // =====================================================================

    // Build dummy instantiate args to query sizes
    uint32_t instantiateArgAlignedStride = (sizeof(cluster::IndirectInstantiateTemplateArgs) + 15) & ~15;
    nvrhi::BufferDesc instantiateArgsDesc = {
        .byteSize = (size_t)instantiateArgAlignedStride * numMeshlets,
        .debugName = "MeshletInstantiateArgs",
        .structStride = instantiateArgAlignedStride,
        .canHaveUAVs = true,
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle instantiateArgsBuffer = device->createBuffer(instantiateArgsDesc);

    // Fill instantiate args with template addresses
    std::vector<uint8_t> instantiateArgsData(instantiateArgAlignedStride * numMeshlets, 0);
    for (uint32_t i = 0; i < numMeshlets; ++i) {
        auto* arg = reinterpret_cast<cluster::IndirectInstantiateTemplateArgs*>(
            instantiateArgsData.data() + instantiateArgAlignedStride * i);
        arg->clusterIdOffset = 0;
        arg->geometryIndexOffsetPacked = 0;
        arg->clusterTemplate = result->templateAddresses[i];
        arg->vertexBuffer.startAddress = vertexBufferAddress + result->meshletVertexOffsets[i] * 12;
        arg->vertexBuffer.strideInBytes = 12;
    }
    commandList->writeBuffer(instantiateArgsBuffer, instantiateArgsData.data(), instantiateArgsData.size());

    result->instantiationSizesBuffer.Create(numMeshlets, "MeshletInstantiationSizes", device);

    cluster::OperationParams instParams = operationParams;
    instParams.type = cluster::OperationType::ClasInstantiateTemplates;
    instParams.mode = cluster::OperationMode::GetSizes;

    cluster::OperationDesc instGetSizesDesc = {
        .params = instParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = instantiateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = result->instantiationSizesBuffer.Get(),
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(instGetSizesDesc);

    result->instantiationSizes = result->instantiationSizesBuffer.Download(commandList);

    result->isBuilt = true;

    Logger::info(str::format("MeshletTemplateBuilder: Successfully built ", numMeshlets,
        " templates, totalData=", totalTemplateSize, " bytes"));

    // Log first few instantiation sizes
    for (uint32_t i = 0; i < std::min(numMeshlets, 5u); ++i) {
        Logger::info(str::format("  meshlet[", i, "] tri=", result->meshletTriangleCounts[i],
            " vtx=", result->meshletVertexCounts[i],
            " instSize=", result->instantiationSizes[i]));
    }

    return result;
}
