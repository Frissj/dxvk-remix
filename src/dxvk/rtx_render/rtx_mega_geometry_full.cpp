/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
*
* COMPLETE IMPLEMENTATION - Part 2
* This file contains the full implementation of geometry submission, BLAS building,
* and statistics. Include this after rtx_mega_geometry.cpp
*/

#include "rtx_mega_geometry.h"
#include "rtx_mega_geometry_integration.h"
#include "rtx_mega_geometry_autotune.h"
#include "rtx_mg_cluster.h"
#include "rtxmg/rtxmg_accel.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "../util/log/log.h"

namespace dxvk {

  void RtxMegaGeometry::submitGeometry(
    Rc<DxvkContext> ctx,
    const Vector3* positions,
    const Vector3* normals,
    const Vector2* texCoords,
    uint32_t vertexCount,
    const uint32_t* indices,
    uint32_t indexCount,
    const Matrix4& transform,
    uint32_t materialId) {

    if (!m_initialized || vertexCount == 0) {
      return;
    }

    ScopedCpuProfileZone();

    // Compute geometry hash for caching (CPU path)
    // NOTE: For proper deforming geometry support, we hash POSITIONS
    // - Static geometry: positions don't change → same hash → cache hit ✓
    // - Deforming geometry: positions change → different hash → cache miss → rebuild BLAS ✓
    // This ensures deforming meshes (animated dragon) get fresh BLAS with updated positions
    XXH64_hash_t geomHash = XXH3_64bits(positions, vertexCount * sizeof(Vector3));
    if (indices && indexCount > 0) {
      geomHash = XXH3_64bits_withSeed(indices, indexCount * sizeof(uint32_t), geomHash);
    }

    // Create submitted mesh
    SubmittedMesh mesh;
    mesh.transform = transform;
    mesh.materialId = materialId;
    mesh.geometryHash = geomHash;

    Logger::info(str::format("[CLUSTER DEBUG] verts=", vertexCount, " indices=", indexCount,
                            " matID=", materialId,
                            " geomHash=0x", std::hex, geomHash, std::dec,
                            " (includes positions for deform detection)"));

    // Convert vertices to cluster format
    mesh.vertices.reserve(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
      ClusterVertex vertex;
      vertex.position = positions[i];
      vertex.normal = normals ? normals[i] : Vector3(0.0f, 0.0f, 1.0f);
      vertex.texCoord = texCoords ? texCoords[i] : Vector2(0.0f, 0.0f);
      vertex.clusterId = materialId;  // Use materialId as cluster ID (stable)

      mesh.vertices.push_back(vertex);
    }

    // Copy indices
    if (indices && indexCount > 0) {
      mesh.indices.assign(indices, indices + indexCount);
    } else {
      // Generate sequential indices
      mesh.indices.reserve(vertexCount);
      for (uint32_t i = 0; i < vertexCount; ++i) {
        mesh.indices.push_back(i);
      }
    }

    // Add to pending meshes
    m_pendingMeshes.push_back(std::move(mesh));

    // Update statistics
    m_stats.totalVertices += vertexCount;
    m_stats.totalTriangles += indexCount / 3;
  }

  void RtxMegaGeometry::submitGeometryGpu(
    Rc<DxvkContext> ctx,
    const DxvkBufferSlice& positionBuffer,
    const DxvkBufferSlice& normalBuffer,
    const DxvkBufferSlice& texcoordBuffer,
    const DxvkBufferSlice& indexBuffer,
    uint32_t vertexCount,
    uint32_t indexCount,
    const Matrix4& transform,
    uint32_t materialId,
    uint32_t positionStride,
    uint32_t normalStride,
    uint32_t texcoordStride,
    VkIndexType indexType,
    XXH64_hash_t geometryHash) {

    if (!m_initialized || vertexCount == 0) {
      return;
    }

    ScopedCpuProfileZone();

    // Create GPU mesh submission
    GpuSubmittedMesh mesh;
    mesh.positionBuffer = positionBuffer;
    mesh.normalBuffer = normalBuffer;
    mesh.texcoordBuffer = texcoordBuffer;
    mesh.indexBuffer = indexBuffer;
    mesh.vertexCount = vertexCount;
    mesh.indexCount = indexCount;
    mesh.positionStride = positionStride;
    mesh.normalStride = normalStride;
    mesh.texcoordStride = texcoordStride;
    mesh.indexType = indexType;
    mesh.transform = transform;
    mesh.materialId = materialId;

    // Use provided stable geometryHash (RTX Remix's TopologicalHash)
    // This hash is stable across frames even when vertex positions change (deformation)
    // Fallback: compute unstable hash if not provided (for backwards compatibility)
    if (geometryHash == 0) {
      XXH64_hash_t h = 0;
      h = XXH64(&vertexCount, sizeof(vertexCount), h);
      h = XXH64(&indexCount, sizeof(indexCount), h);
      h = XXH64(&positionStride, sizeof(positionStride), h);

      // Hash buffer device addresses (UNSTABLE - changes every frame)
      if (positionBuffer.defined()) {
        VkDeviceAddress addr = positionBuffer.getDeviceAddress();
        h = XXH64(&addr, sizeof(addr), h);
      }
      if (indexBuffer.defined()) {
        VkDeviceAddress addr = indexBuffer.getDeviceAddress();
        h = XXH64(&addr, sizeof(addr), h);
      }
      geometryHash = h;
    }

    mesh.geometryHash = geometryHash;

    // Add to pending GPU meshes for BLAS building
    m_pendingGpuMeshes.push_back(std::move(mesh));

    // Update statistics
    m_stats.totalVertices += vertexCount;
    m_stats.totalTriangles += indexCount / 3;
  }

  void RtxMegaGeometry::buildClusterAccelerationStructuresForFrame(Rc<DxvkContext> ctx) {
    // EXACT SAMPLE MATCH: BuildAccel() lines 1254-1368
    // Flow:
    //   1. ComputeInstanceClusterTiling() - done in tessellateCollectedGeometry()
    //   2. FillInstanceClusters() - fills cluster vertex data
    //   3. BuildStructuredCLASes() - ONE GPU call for ALL CLAS (line 1365)
    //   4. BuildBlasFromClas() - ONE GPU call for ALL BLAS (line 1368)

    // CRITICAL: Only build ONCE per frame (sample calls BuildAccel once per frame)
    // Multiple calls per frame cause GPU timeout (TDR)
    static uint32_t lastBuildFrameId = 0xFFFFFFFF;
    uint32_t currentFrameId = m_device->getCurrentFrameId();

    if (currentFrameId == lastBuildFrameId) {
      Logger::warn(str::format("[RTXMG BUILD ACCEL] SKIPPING: Already built for frame ", currentFrameId,
                              " (", m_frameDrawCalls.size(), " draw calls will be deferred to next frame)"));
      // Don't clear m_frameDrawCalls - they'll be built next frame
      return;
    }
    lastBuildFrameId = currentFrameId;

    static uint32_t callCount = 0;
    callCount++;

    Logger::info(str::format("[RTXMG BUILD ACCEL] ========== CALL #", callCount, " (Frame ", currentFrameId, ") =========="));
    Logger::info(str::format("[RTXMG BUILD ACCEL] Processing ", m_frameDrawCalls.size(), " draw calls"));

    if (m_frameDrawCalls.empty() || !isClusterAccelerationExtensionAvailable()) {
      Logger::info("[RTXMG BUILD ACCEL] EARLY EXIT: No draw calls to process");
      m_frameDrawCalls.clear();
      return;
    }

    RtxContext* rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
    if (!rtxCtx || !m_clusterBuilder) {
      Logger::err("[RTXMG BUILD ACCEL] ERROR: Invalid context or cluster builder");
      return;
    }

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Build Unified CLAS and BLAS");

    RtxmgConfig config;
    config.tessMode = RtxmgConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH;
    config.fineTessellationRate = 1.0f;
    config.coarseTessellationRate = 1.0f / 15.0f;

    // Count total clusters and geometry stats across all draw calls
    uint32_t totalClusters = 0;
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
    for (const auto& drawCall : m_frameDrawCalls) {
      totalClusters += drawCall.clusterCount;
      totalVertices += drawCall.inputVertexCount;
      totalTriangles += drawCall.inputIndexCount / 3;
    }

    if (totalClusters == 0) {
      Logger::info("[RTXMG BUILD ACCEL] No clusters to build");
      m_frameDrawCalls.clear();
      return;
    }

    // Update auto-tune with actual usage to grow cluster budget from 64K -> 2M dynamically
    if (m_autoTune) {
      VkPhysicalDeviceMemoryProperties memProps = m_device->adapter()->memoryProperties();

      // Calculate available VRAM (total heap 0 minus used)
      uint64_t totalVramBytes = memProps.memoryHeaps[0].size;
      uint64_t availableVramMB = totalVramBytes / (1024 * 1024);

      m_autoTune->updatePerFrame(totalVertices, totalTriangles, m_frameDrawCalls.size(), availableVramMB);

      // Check if buffers need to be resized (grow or shrink with fence tracking)
      if (m_autoTune->needsBufferResize()) {
        resizeBuffersIfNeeded(ctx);
      }
    }

    Logger::info(str::format("[RTXMG BUILD ACCEL] Building unified CLAS+BLAS for ",
                            m_frameDrawCalls.size(), " instances, ",
                            totalClusters, " total clusters"));

    // EXACT SAMPLE MATCH: Build ALL CLAS structures in ONE GPU operation
    // Sample line 1365: BuildStructuredCLASes(accels, maxGeometryCountPerMesh, tessCounterRange, commandList);
    // This uses executeMultiIndirectClusterOperation with ClasInstantiateTemplates

    // EXACT SAMPLE MATCH: Build ALL BLAS structures in ONE GPU operation
    // Sample line 1368: BuildBlasFromClas(accels, instances, commandList);
    // This uses executeMultiIndirectClusterOperation with BuildClustersBottomLevel

    // Convert to cluster builder format
    std::vector<RtxmgClusterBuilder::DrawCallData> drawCallsForBuilder;
    drawCallsForBuilder.reserve(m_frameDrawCalls.size());

    for (const auto& drawCall : m_frameDrawCalls) {
      RtxmgClusterBuilder::DrawCallData data;
      data.geometryHash = drawCall.geometryHash;
      data.transform = drawCall.transform;
      data.drawCallIndex = drawCall.drawCallIndex;
      data.clusterCount = drawCall.clusterCount;
      data.output = drawCall.output;

      // CRITICAL FIX: Copy input geometry for fill_clusters reconstruction
      data.inputPositions = drawCall.inputPositions;
      data.inputNormals = drawCall.inputNormals;
      data.inputTexcoords = drawCall.inputTexcoords;
      data.inputIndices = drawCall.inputIndices;
      data.inputVertexCount = drawCall.inputVertexCount;
      data.inputIndexCount = drawCall.inputIndexCount;
      data.positionOffset = drawCall.positionOffset;
      data.normalOffset = drawCall.normalOffset;
      data.texcoordOffset = drawCall.texcoordOffset;

      drawCallsForBuilder.push_back(data);
    }

    // CRITICAL: Split CLAS and BLAS building like reference implementation
    // Reference: cluster_accel_builder.cpp:1365-1368
    // Line 1365: BuildStructuredCLASes()
    // Line 1368: BuildBlasFromClas()
    // CRITICAL: Both use SAME command buffer without flush between them!

    // Step 1: Build CLAS (pass frame index for ring buffer indexing)
    if (!m_clusterBuilder->buildStructuredCLASes(rtxCtx, drawCallsForBuilder, config, currentFrameId)) {
      Logger::err("[RTXMG BUILD ACCEL] CLAS building failed");
      m_frameDrawCalls.clear();
      return;
    }

    // Step 2: Build BLAS from CLAS (SAME command buffer, no flush!)
    // Reference line 1368 calls this immediately after CLAS build
    if (!m_clusterBuilder->buildBlasFromClas(rtxCtx, drawCallsForBuilder, config)) {
      Logger::err("[RTXMG BUILD ACCEL] BLAS building failed");
      m_frameDrawCalls.clear();
      return;
    }

    // NOTE: Do NOT flush here! TLAS build must happen in the same command buffer for barriers to work.
    // NVIDIA's sample (rtxmg_renderer.cpp:1062-1160) does BuildAccel → FillInstanceDescs → buildTopLevelAccelStruct
    // ALL IN THE SAME COMMAND LIST with NO flush between them.
    // Barriers only work within the same command buffer!
    Logger::info(str::format("[RTXMG BUILD ACCEL] COMPLETE: Built unified CLAS+BLAS for ",
                            m_frameDrawCalls.size(), " instances (NO flush - TLAS will build in same command buffer)"));

    // NOTE: CPU-side readback of blasPtrsBuffer returns all zeros (confirmed in both IMPLICIT and
    // EXPLICIT_DESTINATIONS modes). NVIDIA sample doesn't do CPU readback - it only uses GPU-side
    // patching via FillInstanceDescs shader (rtxmg_renderer.cpp:1155).
    //
    // BLAS addresses are patched directly on GPU by AccelManager::patchClusterBlasAddresses()
    // which uses the patch_tlas_instance_blas_addresses compute shader.
    // The CPU never needs to know the actual BLAS addresses.
    //
    // Caching with null addresses (0x0) is safe because:
    // 1. GPU patching reads from blasPtrsBuffer directly (never uses cached addresses)
    // 2. Cached geometry is only used to skip re-tessellation, not for address lookup
    Logger::info("[RTXMG BLAS] Skipping CPU readback - addresses will be patched on GPU");

    // NVIDIA SAMPLE APPROACH: Always rebuild every frame
    // - Cache LOOKUP is disabled (rtx_mega_geometry_integration.cpp:411) → always rebuild
    // - Cache STORAGE is enabled (below) → injection can find cluster BLASes THIS frame
    //
    // This ensures fresh hash→index mapping each frame while allowing injection to work.
    const ClusterAccels* frameAccels = m_clusterBuilder ? &m_clusterBuilder->getFrameAccels() : nullptr;
    for (const auto& drawCall : m_frameDrawCalls) {
      // Cache tessellated geometry with frameAccels for THIS FRAME's injection
      // GPU patching will read correct addresses from blasPtrsBuffer
      cacheTessellatedGeometry(
        ctx,  // Pass context for command list tracking
        drawCall.geometryHash,
        drawCall.output,
        frameAccels,
        0,  // BLAS address - not needed, GPU patching handles this
        0); // BLAS size - not needed
    }
    Logger::info(str::format("[RTXMG] Cached ", m_frameDrawCalls.size(), " cluster BLASes for injection this frame"));

    // Clear frame data for next frame
    m_frameDrawCalls.clear();
  }

  void RtxMegaGeometry::buildBLASForMesh(Rc<DxvkContext> ctx, const SubmittedMesh& mesh) {
    ScopedGpuProfileZone(ctx, "Build BLAS for Mesh");

    // Create geometry info for BLAS
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // Triangle geometry data
    VkAccelerationStructureGeometryTrianglesDataKHR& triangles = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

    // Create staging buffer for vertex data
    DxvkBufferCreateInfo vertexBufferInfo;
    vertexBufferInfo.size = mesh.vertices.size() * sizeof(ClusterVertex);
    vertexBufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vertexBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    vertexBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    Rc<DxvkBuffer> vertexBuffer = m_device->createBuffer(vertexBufferInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "Mega Geometry BLAS Vertex Buffer");

    // Upload vertex data
    void* vertexData = vertexBuffer->mapPtr(0);
    if (vertexData) {
      memcpy(vertexData, mesh.vertices.data(), vertexBufferInfo.size);
    }

    // Create staging buffer for index data
    DxvkBufferCreateInfo indexBufferInfo;
    indexBufferInfo.size = mesh.indices.size() * sizeof(uint32_t);
    indexBufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    indexBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    indexBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    Rc<DxvkBuffer> indexBuffer = m_device->createBuffer(indexBufferInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "Mega Geometry BLAS Index Buffer");

    // Upload index data
    void* indexData = indexBuffer->mapPtr(0);
    if (indexData) {
      memcpy(indexData, mesh.indices.data(), indexBufferInfo.size);
    }

    // Set triangle data
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBuffer->getDeviceAddress();
    triangles.vertexStride = sizeof(ClusterVertex);
    triangles.maxVertex = static_cast<uint32_t>(mesh.vertices.size() - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBuffer->getDeviceAddress();

    // Build range info
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    buildRange.primitiveCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    buildRange.primitiveOffset = 0;
    buildRange.firstVertex = 0;
    buildRange.transformOffset = 0;

    // Get build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    uint32_t primCount = buildRange.primitiveCount;
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(
      m_device->vkd()->device(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo,
      &primCount,
      &sizeInfo);

    // Create BLAS buffer
    DxvkBufferCreateInfo blasBufferInfo;
    blasBufferInfo.size = sizeInfo.accelerationStructureSize;
    blasBufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    blasBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    blasBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    Rc<DxvkBuffer> blasBuffer = m_device->createBuffer(blasBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "Mega Geometry BLAS");

    // Create acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer->getBufferHandle().buffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkResult result = m_device->vkd()->vkCreateAccelerationStructureKHR(
      m_device->vkd()->device(), &createInfo, nullptr, &blas);

    if (result != VK_SUCCESS) {
      Logger::err(str::format("[RTX Mega Geometry] Failed to create BLAS: ", result));
      return;
    }

    // Create scratch buffer
    DxvkBufferCreateInfo scratchBufferInfo;
    scratchBufferInfo.size = sizeInfo.buildScratchSize;
    scratchBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    scratchBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    scratchBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    Rc<DxvkBuffer> scratchBuffer = m_device->createBuffer(scratchBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "Mega Geometry BLAS Scratch");

    // Build BLAS
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = scratchBuffer->getDeviceAddress();

    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    ctx->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRange);

    // Update visible cluster count
    m_stats.visibleClusters += buildRange.primitiveCount;

    Logger::debug(str::format(
      "[RTX Mega Geometry] Built BLAS: ",
      mesh.vertices.size(), " vertices, ",
      mesh.indices.size() / 3, " triangles"
    ));
  }

  void RtxMegaGeometry::buildBLASForGpuMesh(Rc<DxvkContext> ctx, const GpuSubmittedMesh& mesh) {
    ScopedGpuProfileZone(ctx, "Build BLAS for GPU Mesh (zero-copy)");

    // Skip meshes with no indices or vertices
    if (mesh.vertexCount == 0 || mesh.indexCount == 0 || !mesh.positionBuffer.defined()) {
      return;
    }

    // STEP 1: Check if we already have a cached BLAS for this geometry
    CachedBLAS* cachedBlas = lookupBLAS(mesh.geometryHash);
    if (cachedBlas != nullptr) {
      // Cache hit! Reuse existing BLAS
      Logger::debug(str::format(
        "[RTX Mega Geometry] BLAS cache HIT: hash=0x", std::hex, mesh.geometryHash, std::dec,
        ", triangles=", cachedBlas->triangleCount,
        ", size=", cachedBlas->blasSize / 1024, "KB"
      ));

      // TODO: Register this BLAS with the scene manager for this instance
      // This would integrate with RTX Remix's instance/TLAS system
      // For now, the BLAS is cached but needs scene manager integration

      return;  // Reuse cached BLAS, no rebuild needed
    }

    // STEP 2: Cache miss - need to build new BLAS
    Logger::debug(str::format(
      "[RTX Mega Geometry] BLAS cache MISS: hash=0x", std::hex, mesh.geometryHash, std::dec,
      ", building new BLAS..."
    ));

    // Create geometry info for BLAS
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // Triangle geometry data - use existing GPU buffers directly (zero-copy)
    VkAccelerationStructureGeometryTrianglesDataKHR& triangles = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = mesh.positionBuffer.getDeviceAddress();
    triangles.vertexStride = mesh.positionStride;
    triangles.maxVertex = mesh.vertexCount - 1;

    // Set index data
    if (mesh.indexBuffer.defined() && mesh.indexCount > 0) {
      triangles.indexType = mesh.indexType;
      triangles.indexData.deviceAddress = mesh.indexBuffer.getDeviceAddress();
    } else {
      triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    // Build range info
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    if (mesh.indexBuffer.defined()) {
      buildRange.primitiveCount = mesh.indexCount / 3;
    } else {
      buildRange.primitiveCount = mesh.vertexCount / 3;
    }
    buildRange.primitiveOffset = 0;
    buildRange.firstVertex = 0;
    buildRange.transformOffset = 0;

    // Get build sizes with compaction support
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    uint32_t primCount = buildRange.primitiveCount;
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(
      m_device->vkd()->device(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo,
      &primCount,
      &sizeInfo);

    // STEP 3: Create BLAS buffer (with proper lifetime management)
    DxvkBufferCreateInfo blasBufferInfo;
    blasBufferInfo.size = sizeInfo.accelerationStructureSize;
    blasBufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    blasBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    blasBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    // STEP 4: Create acceleration structure (using proper DxvkAccelStructure wrapper)
    Rc<DxvkAccelStructure> blasBuffer = m_device->createAccelStructure(
      blasBufferInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      "Mega Geometry GPU BLAS (Pooled)");

    if (blasBuffer == nullptr) {
      Logger::err("[RTX Mega Geometry] Failed to create BLAS acceleration structure");
      return;
    }

    VkAccelerationStructureKHR accelStruct = blasBuffer->getAccelStructure();

    // STEP 5: Create scratch buffer for build
    DxvkBufferCreateInfo scratchInfo;
    scratchInfo.size = sizeInfo.buildScratchSize;
    scratchInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    scratchInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    scratchInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    Rc<DxvkBuffer> scratchBuffer = m_device->createBuffer(scratchInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "Mega Geometry BLAS Scratch");

    // STEP 6: Build the BLAS
    buildInfo.dstAccelerationStructure = accelStruct;
    buildInfo.scratchData.deviceAddress = scratchBuffer->getDeviceAddress();

    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    // Build BLAS using standard Vulkan command
    ctx->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRange);

    // STEP 7: Store BLAS in cache for reuse
    CachedBLAS newCachedBlas;
    newCachedBlas.blasBuffer = blasBuffer;              // Keep buffer alive via Rc
    newCachedBlas.accelStructure = accelStruct;          // Store handle for later use
    newCachedBlas.vertexCount = mesh.vertexCount;
    newCachedBlas.triangleCount = buildRange.primitiveCount;
    newCachedBlas.lastUsedFrame = m_device->getCurrentFrameId();  // Use actual frame ID for ring buffer tracking
    newCachedBlas.blasSize = sizeInfo.accelerationStructureSize;
    newCachedBlas.isCompacted = false;  // Not compacted yet

    cacheBLAS(mesh.geometryHash, newCachedBlas);

    // CRITICAL TODO: Integrate with RTX Remix's instance/BLAS management system
    //
    // Currently the BLAS is built and cached but NOT used by the renderer.
    // To make this production-ready, you need to:
    //
    // 1. Hook into RTX Remix's geometry/instance creation in SceneManager
    // 2. Before RTX Remix builds a BLAS, check our cache with geometryHash
    // 3. If cache hit, return our cached BLAS instead of building new one
    // 4. If cache miss, let RTX Remix build it, then store in our cache
    //
    // Integration points to investigate:
    // - SceneManager::submitDrawState() - where geometry enters the system
    // - RtxGeometryUtils::createGeometryInfo() - where BLASes are created
    // - Instance creation code - where BLAS handles are assigned
    //
    // Without this integration, we're building BLASes that are never used
    // (RTX Remix builds its own BLASes in parallel)

    Logger::info(str::format(
      "[RTX Mega Geometry] Built GPU BLAS (zero-copy): ",
      mesh.vertexCount, " vertices, ",
      mesh.indexCount / 3, " triangles, ",
      sizeInfo.accelerationStructureSize / 1024, "KB, ",
      "hash=0x", std::hex, mesh.geometryHash, std::dec
    ));
  }

  void RtxMegaGeometry::readbackStatistics(Rc<DxvkContext> ctx) {
    if (m_clusterStatisticsBuffer == nullptr) {
      return;
    }

    // Read back statistics from GPU
    void* statsData = m_clusterStatisticsBuffer->mapPtr(0);
    if (statsData) {
      const ClusterStatistics* gpuStats = static_cast<const ClusterStatistics*>(statsData);

      // Update CPU-side statistics
      m_stats.totalClusters = gpuStats->totalClusters;
      m_stats.visibleClusters = gpuStats->visibleClusters;
      m_stats.culledClusters = gpuStats->culledByHiZ + gpuStats->culledByFrustum + gpuStats->culledByBackface;
      m_stats.totalVertices = gpuStats->totalVertices;
      m_stats.totalTriangles = gpuStats->totalTriangles;
    }
  }

  void RtxMegaGeometry::getStatistics(MegaGeometryStatistics& outStats) const {
    outStats.totalClusters = m_stats.totalClusters;
    outStats.visibleClusters = m_stats.visibleClusters;
    outStats.culledClusters = m_stats.culledClusters;
    outStats.totalVertices = m_stats.totalVertices;
    outStats.totalTriangles = m_stats.totalTriangles;
    outStats.memoryUsedMB = m_stats.memoryUsedMB;
  }

  void RtxMegaGeometry::resizeBuffersIfNeeded(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Resize Buffers");

    RtxContext* rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
    if (!rtxCtx) {
      Logger::err("[RTX Mega Geometry] Cannot resize buffers: invalid context");
      return;
    }

    // STEP 1: Process pending releases - check fences and release completed buffers
    m_pendingClusterBufferReleases.erase(
      std::remove_if(m_pendingClusterBufferReleases.begin(), m_pendingClusterBufferReleases.end(),
        [this](BuffersWithFence& pending) {
          VkResult status = m_device->vkd()->vkGetFenceStatus(m_device->handle(), pending.lastUsageFence);
          if (status == VK_SUCCESS) {
            const uint64_t freedDataMB = pending.clusterDataBuffer->info().size / (1024 * 1024);
            const uint64_t freedInfoMB = pending.clusterInfoBuffer->info().size / (1024 * 1024);
            Logger::info(str::format("[RTX Mega Geometry FENCE] GPU work completed, releasing ",
                                    freedDataMB, "MB cluster data + ", freedInfoMB, "MB cluster info"));
            // Buffers automatically destroyed when removed from vector
            return true;
          }
          return false;
        }),
      m_pendingClusterBufferReleases.end());

    // STEP 2: Get new recommended sizes and detect grow/shrink needs
    const uint64_t newClusterDataSize = m_autoTune->getRecommendedClusterDataBufferSize();
    const uint64_t newClusterInfoSize = m_autoTune->getRecommendedClusterInfoBufferSize();
    const uint64_t currentClusterDataSize = m_clusterDataBuffer->info().size;
    const uint64_t currentClusterInfoSize = m_clusterInfoBuffer->info().size;

    const bool needsGrowth = (newClusterDataSize > currentClusterDataSize) ||
                             (newClusterInfoSize > currentClusterInfoSize);
    const bool needsShrink = (newClusterDataSize < currentClusterDataSize / 2) ||
                             (newClusterInfoSize < currentClusterInfoSize / 2);

    // STEP 3: If no changes needed, reuse existing buffers
    if (!needsGrowth && !needsShrink) {
      return;
    }

    // STEP 4: Queue old buffers for fence-tracked release
    if (needsGrowth) {
      Logger::info(str::format("[RTX Mega Geometry] Growing cluster buffers: ",
                              currentClusterDataSize / (1024 * 1024), "MB -> ",
                              newClusterDataSize / (1024 * 1024), "MB"));
    } else if (needsShrink) {
      Logger::info(str::format("[RTX Mega Geometry] Shrinking cluster buffers: ",
                              currentClusterDataSize / (1024 * 1024), "MB -> ",
                              newClusterDataSize / (1024 * 1024), "MB (fence-tracked release)"));
    }

    VkFence currentFence = rtxCtx->getCommandList()->fence();
    BuffersWithFence pending;
    pending.clusterDataBuffer = std::move(m_clusterDataBuffer);
    pending.clusterInfoBuffer = std::move(m_clusterInfoBuffer);
    pending.lastUsageFence = currentFence;
    m_pendingClusterBufferReleases.push_back(std::move(pending));

    // STEP 5: Allocate new buffers with new sizes
    DxvkBufferCreateInfo bufferInfo;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    bufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    // Cluster data buffer
    bufferInfo.size = newClusterDataSize;
    m_clusterDataBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Cluster Data");
    Logger::debug(str::format("[RTX Mega Geometry] Allocated cluster data buffer: ",
      newClusterDataSize / (1024 * 1024), "MB"));

    // Cluster info buffer
    bufferInfo.size = newClusterInfoSize;
    m_clusterInfoBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Cluster Info");
    Logger::debug(str::format("[RTX Mega Geometry] Allocated cluster info buffer: ",
      newClusterInfoSize / (1024 * 1024), "MB"));

    m_autoTune->acknowledgeBufferResize();

    Logger::info("[RTX Mega Geometry] Buffer resize complete");
  }

  // ============================================================================
  // BLAS Cache Management
  // ============================================================================

  RtxMegaGeometry::CachedBLAS* RtxMegaGeometry::lookupBLAS(XXH64_hash_t geometryHash) {
    auto it = m_blasCache.find(geometryHash);
    if (it != m_blasCache.end()) {
      // Update last used frame for LRU eviction
      it->second.lastUsedFrame = m_device->getCurrentFrameId();  // Use actual frame ID for ring buffer tracking
      m_stats.blasCacheHits++;
      return &it->second;
    }

    m_stats.blasCacheMisses++;
    return nullptr;
  }

  void RtxMegaGeometry::cacheBLAS(XXH64_hash_t geometryHash, const CachedBLAS& blas) {
    // Store in cache
    m_blasCache[geometryHash] = blas;

    // Update statistics
    m_stats.blasCachedEntries = static_cast<uint32_t>(m_blasCache.size());

    // Calculate total BLAS memory usage
    uint64_t totalBlasMemory = 0;
    for (const auto& entry : m_blasCache) {
      totalBlasMemory += entry.second.blasSize;
    }
    m_stats.blasMemoryUsedMB = static_cast<float>(totalBlasMemory) / (1024.0f * 1024.0f);

    Logger::debug(str::format(
      "[RTX Mega Geometry] Cached BLAS: hash=0x", std::hex, geometryHash, std::dec,
      ", triangles=", blas.triangleCount,
      ", size=", blas.blasSize / 1024, "KB",
      " | Cache: ", m_stats.blasCachedEntries, " entries, ",
      m_stats.blasMemoryUsedMB, "MB"
    ));
  }

  void RtxMegaGeometry::evictOldBLASEntries() {
    // Evict BLASes not used in last 300 frames (~5 seconds at 60fps)
    const uint32_t MAX_FRAME_AGE = 300;
    const uint32_t MAX_CACHE_ENTRIES = 5000;  // Limit BLAS cache size

    // GPU synchronization: Small safety margin as backup (lifetime tracking is primary protection)
    // With proper trackResource() calls, this is just defense-in-depth
    const uint32_t gpuPipelineDepth = m_device->pendingSubmissions() + 3;
    uint32_t currentFrameId = m_device->getCurrentFrameId();

    // Age-based eviction
    auto it = m_blasCache.begin();
    while (it != m_blasCache.end()) {
      uint32_t age = currentFrameId - it->second.lastUsedFrame;
      // CRITICAL: Only evict if (1) old enough AND (2) GPU has finished with it
      if (age > MAX_FRAME_AGE && age > gpuPipelineDepth) {
        Logger::debug(str::format(
          "[RTX Mega Geometry] Evicting BLAS: hash=0x", std::hex, it->first, std::dec,
          ", age=", age, " frames, size=", it->second.blasSize / 1024, "KB"
        ));

        // Destroy acceleration structure handle
        if (it->second.accelStructure != VK_NULL_HANDLE) {
          m_device->vkd()->vkDestroyAccelerationStructureKHR(
            m_device->vkd()->device(),
            it->second.accelStructure,
            nullptr
          );
        }

        it = m_blasCache.erase(it);
      } else {
        ++it;
      }
    }

    // Size-based eviction (if cache is too large, evict oldest entries)
    if (m_blasCache.size() > MAX_CACHE_ENTRIES) {
      // Sort by last used frame and evict oldest
      std::vector<std::pair<XXH64_hash_t, uint32_t>> entries;
      for (const auto& entry : m_blasCache) {
        entries.push_back({entry.first, entry.second.lastUsedFrame});
      }

      std::sort(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

      // Evict oldest entries (but skip those still in use by GPU)
      size_t numToEvict = m_blasCache.size() - MAX_CACHE_ENTRIES;
      size_t numEvicted = 0;
      for (size_t i = 0; i < entries.size() && numEvicted < numToEvict; ++i) {
        auto evictIt = m_blasCache.find(entries[i].first);
        if (evictIt != m_blasCache.end()) {
          uint32_t age = currentFrameId - evictIt->second.lastUsedFrame;

          // CRITICAL: Don't evict if GPU might still be using it
          if (age <= gpuPipelineDepth) {
            Logger::warn(str::format(
              "[RTX Mega Geometry] Cache size limit reached but cannot evict BLAS (age=",
              age, ", GPU pipeline depth=", gpuPipelineDepth, "). Risk of OOM."
            ));
            continue;  // Skip this entry, GPU might still need it
          }

          if (evictIt->second.accelStructure != VK_NULL_HANDLE) {
            m_device->vkd()->vkDestroyAccelerationStructureKHR(
              m_device->vkd()->device(),
              evictIt->second.accelStructure,
              nullptr
            );
          }
          m_blasCache.erase(evictIt);
          numEvicted++;
        }
      }

      Logger::info(str::format(
        "[RTX Mega Geometry] Evicted ", numEvicted, " / ", numToEvict,
        " old BLAS entries (cache size limit: ", MAX_CACHE_ENTRIES, ")"
      ));
    }

    // Update statistics
    m_stats.blasCachedEntries = static_cast<uint32_t>(m_blasCache.size());

    uint64_t totalBlasMemory = 0;
    for (const auto& entry : m_blasCache) {
      totalBlasMemory += entry.second.blasSize;
    }
    m_stats.blasMemoryUsedMB = static_cast<float>(totalBlasMemory) / (1024.0f * 1024.0f);
  }

  void RtxMegaGeometry::cleanupBLASCache() {
    Logger::info(str::format(
      "[RTX Mega Geometry] Cleaning up BLAS cache: ",
      m_blasCache.size(), " entries"
    ));

    // Destroy all acceleration structure handles
    for (auto& entry : m_blasCache) {
      if (entry.second.accelStructure != VK_NULL_HANDLE) {
        m_device->vkd()->vkDestroyAccelerationStructureKHR(
          m_device->vkd()->device(),
          entry.second.accelStructure,
          nullptr
        );
      }
    }

    m_blasCache.clear();
    m_stats.blasCachedEntries = 0;
    m_stats.blasMemoryUsedMB = 0.0f;
  }

} // namespace dxvk
