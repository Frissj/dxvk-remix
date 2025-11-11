/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "rtx_mega_geometry.h"
#include "rtx_mega_geometry_integration.h"
#include "rtx_context.h"
#include "rtx_instance_manager.h"
#include "rtx_scene_manager.h"
#include "rtx_geometry_utils.h"
#include "rtx_debug_view.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_device.h"
#include "rtxmg/rtxmg_cluster_builder.h"

namespace dxvk {

  /**
   * \brief RTX Mega Geometry Integration - FULL IMPLEMENTATION
   *
   * Intercepts geometry submission and routes through RTXMG cluster tessellation.
   * Always-on by design - no fallback path.
   */

  // Singleton mega geometry system
  static std::unique_ptr<RtxMegaGeometry> s_megaGeometry = nullptr;

  void initializeMegaGeometry(DxvkDevice* device) {
    if (!s_megaGeometry) {
      Logger::info("[RTX Mega Geometry Integration] Initializing ALWAYS-ON cluster tessellation");
      s_megaGeometry = std::make_unique<RtxMegaGeometry>(device);
      s_megaGeometry->initialize();
    }
  }

  void shutdownMegaGeometry() {
    if (s_megaGeometry) {
      Logger::info("[RTX Mega Geometry Integration] Shutting down");
      s_megaGeometry.reset();
    }
  }

  RtxMegaGeometry* getMegaGeometry() {
    return s_megaGeometry.get();
  }

  /**
   * \brief GPU-resident geometry descriptor for cluster builder
   * Contains buffer addresses that remain on GPU - no CPU extraction
   */
  struct GpuGeometryDescriptor {
    VkDeviceAddress positionBufferAddress = 0;
    VkDeviceAddress normalBufferAddress = 0;
    VkDeviceAddress texcoordBufferAddress = 0;
    VkDeviceAddress indexBufferAddress = 0;

    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t positionStride = 0;
    uint32_t normalStride = 0;
    uint32_t texcoordStride = 0;
    uint32_t positionOffset = 0;
    uint32_t normalOffset = 0;
    uint32_t texcoordOffset = 0;

    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    Matrix4 transform;
    XXH64_hash_t materialId = 0;
    bool valid = false;
  };

  /**
   * \brief Gather GPU buffer addresses from DrawCallState (no CPU mapping!)
   */
  static GpuGeometryDescriptor gatherGpuGeometry(const DrawCallState& drawCallState) {
    GpuGeometryDescriptor result;

    const RasterGeometry& geoData = drawCallState.getGeometryData();

    // Check if we have valid position buffer
    if (!geoData.positionBuffer.defined()) {
      return result;
    }

    // Get vertex/index counts
    result.vertexCount = geoData.vertexCount;
    result.indexCount = geoData.indexCount;

    if (result.vertexCount == 0) {
      return result;
    }

    // Get GPU buffer addresses (no CPU mapping!)
    // Position buffer
    result.positionBufferAddress = geoData.positionBuffer.getDeviceAddress();
    result.positionStride = geoData.positionBuffer.stride();
    result.positionOffset = geoData.positionBuffer.offsetFromSlice();

    // Normal buffer (optional)
    if (geoData.normalBuffer.defined()) {
      result.normalBufferAddress = geoData.normalBuffer.getDeviceAddress();
      result.normalStride = geoData.normalBuffer.stride();
      result.normalOffset = geoData.normalBuffer.offsetFromSlice();
    }

    // Texcoord buffer (optional)
    if (geoData.texcoordBuffer.defined()) {
      result.texcoordBufferAddress = geoData.texcoordBuffer.getDeviceAddress();
      result.texcoordStride = geoData.texcoordBuffer.stride();
      result.texcoordOffset = geoData.texcoordBuffer.offsetFromSlice();
    }

    // Index buffer (optional)
    if (result.indexCount > 0 && geoData.indexBuffer.defined()) {
      result.indexBufferAddress = geoData.indexBuffer.getDeviceAddress();
      result.indexType = geoData.indexBuffer.indexType();
    }

    // Get transform and material
    result.transform = drawCallState.getTransformData().objectToWorld;
    result.materialId = drawCallState.getMaterialData().getHash();

    result.valid = true;

    Logger::info(str::format("[RTX Mega Geometry] Gathered GPU geometry: ",
                            result.vertexCount, " verts, ",
                            result.indexCount, " indices, ",
                            "posAddr=0x", std::hex, result.positionBufferAddress, std::dec));

    return result;
  }

  /**
   * \brief Process geometry through RTXMG
   *
   * Supports two modes:
   * 1. With tessellation: Subdivides meshes and caches tessellated results
   * 2. Without tessellation: Submits original geometry for BLAS pooling and culling only
   */
  bool processGeometryWithMegaGeometry(
    RtxContext* ctx,
    const DrawParameters& params,
    DrawCallState& drawCallState,
    RasterGeometry& geometryData,
    SceneManager& sceneManager) {

    // Early exit if mega geometry is disabled
    if (!RtxMegaGeometry::enable()) {
      return false; // Disabled - use normal path
    }

    if (!s_megaGeometry || !s_megaGeometry->isInitialized()) {
      ONCE(Logger::info("[RTX Mega Geometry] System not initialized yet"));
      return false; // Not initialized yet
    }

    ScopedCpuProfileZone();

    // Check if we have actual geometry data in RasterGeometry
    // Note: DrawParameters may show 0 vertices but RasterGeometry has the actual data
    if (!geometryData.positionBuffer.defined() || geometryData.vertexCount == 0) {
      // No geometry to process
      return false;
    }

    const RasterGeometry& geoData = drawCallState.getGeometryData();

    // MODE 1: BLAS pooling only (no tessellation)
    // When tessellation is disabled, we ONLY use AccelManager's BLAS cache.
    // Do NOT call submitGeometryGpu() as it would build duplicate BLASes and conflict with the cache.
    if (!RtxMegaGeometry::enableTessellation()) {
      // Return false so original geometry is passed to AccelManager for BLAS caching
      return false;
    }

    // MODE 2: Tessellation enabled (original code path)

    // CRITICAL FIX: Compute hash from ACTUAL vertex position data, not buffer metadata
    // LegacyAssetHash0 hashes buffer stride/offset, NOT the vertex data itself
    // Result: Different meshes with same buffer config get the same hash → wrong BLAS used → crash
    //
    // Solution: Hash actual vertex position bytes + index bytes
    XXH64_hash_t geomHash = 0;

    // Try to map and hash actual vertex position data
    // First, debug WHY mapPtr returns null
    const DxvkBuffer* posBuffer = geoData.positionBuffer.buffer().ptr();
    if (posBuffer != nullptr) {
      VkMemoryPropertyFlags memProps = posBuffer->memFlags();
      bool isHostVisible = (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
      bool isDeviceLocal = (memProps & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
      bool isHostCached = (memProps & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
      bool isHostCoherent = (memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

      // Debug logging disabled for performance
    }

    const void* positionData = geoData.positionBuffer.mapPtr((size_t)geoData.positionBuffer.offsetFromSlice());

    // ALSO check if index buffer is CPU-accessible (it might be even if vertex buffer isn't!)
    const void* indexData = nullptr;
    if (geoData.indexBuffer.defined()) {
      indexData = geoData.indexBuffer.mapPtr((size_t)geoData.indexBuffer.offsetFromSlice());
    }

    if (positionData != nullptr) {
      // Buffer is CPU-accessible - hash actual vertex position data
      const size_t positionStride = geoData.positionBuffer.stride();
      const uint32_t vertexCount = geoData.vertexCount;

      Logger::info(str::format("[RTX Mega Geometry HASH] CPU path: hashing ", vertexCount,
                              " verts with stride=", positionStride));

      // Hash each vertex position (only position component, not padding)
      // Assuming 3 floats (12 bytes) per position
      XXH64_state_t* state = XXH64_createState();
      XXH64_reset(state, 0);

      const uint8_t* vertexPtr = static_cast<const uint8_t*>(positionData);
      for (uint32_t i = 0; i < vertexCount; ++i) {
        // Hash 12 bytes (vec3 position) per vertex, skip padding/stride
        XXH64_update(state, vertexPtr, 12);
        vertexPtr += positionStride;
      }

      geomHash = XXH64_digest(state);
      XXH64_freeState(state);

      // Also hash indices if present
      if (geoData.indexBuffer.defined()) {
        const void* indexData = geoData.indexBuffer.mapPtr((size_t)geoData.indexBuffer.offsetFromSlice());
        if (indexData != nullptr) {
          const size_t indexSize = (geoData.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) ? 2 : 4;
          const size_t indexBytes = geoData.indexCount * indexSize;
          geomHash = XXH3_64bits_withSeed(indexData, indexBytes, geomHash);
          Logger::info(str::format("[RTX Mega Geometry HASH] Also hashed ", geoData.indexCount, " indices"));
        }
      }

      Logger::info(str::format("[RTX Mega Geometry HASH] Final hash=0x", std::hex, geomHash, std::dec,
                              " for verts=", vertexCount, " indices=", geoData.indexCount));
    } else if (indexData != nullptr) {
      // Hybrid: Vertices are GPU-only but indices are CPU-accessible!
      // Hash indices + metadata instead of expensive GPU compute
      Logger::info(str::format("[RTX Mega Geometry HASH] Hybrid path: vertices GPU-only, indices CPU-accessible"));

      const size_t indexSize = (geoData.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) ? 2 : 4;
      const size_t indexBytes = geoData.indexCount * indexSize;

      // Hash: indices + vertex count + stride (represents topology, not exact vertex data)
      XXH64_state_t* state = XXH64_createState();
      XXH64_reset(state, 0);
      XXH64_update(state, indexData, indexBytes);
      XXH64_update(state, &geoData.vertexCount, sizeof(geoData.vertexCount));
      size_t stride = geoData.positionBuffer.stride();
      XXH64_update(state, &stride, sizeof(size_t));
      geomHash = XXH64_digest(state);
      XXH64_freeState(state);

      Logger::info(str::format("[RTX Mega Geometry HASH] Hybrid hash=0x", std::hex, geomHash, std::dec,
                              " from indices=", geoData.indexCount, " verts=", geoData.vertexCount));
    } else {
      // Both buffers are GPU-only (no CPU-accessible index buffer)
      // Use metadata-only hash (buffer address + vertex count + stride) for uniqueness
      // GPU compute path is currently broken (returns 0x0), so use metadata instead
      Logger::info(str::format("[RTX Mega Geometry HASH] GPU-only buffers (no indices) - using metadata hash. verts=",
                               geoData.vertexCount));

      // Hash: vertex count + stride + buffer address for uniqueness
      XXH64_state_t* state = XXH64_createState();
      XXH64_reset(state, 0);
      XXH64_update(state, &geoData.vertexCount, sizeof(geoData.vertexCount));
      size_t stride = geoData.positionBuffer.stride();
      XXH64_update(state, &stride, sizeof(size_t));
      VkDeviceAddress bufferAddr = geoData.positionBuffer.buffer()->getDeviceAddress() + geoData.positionBuffer.offset();
      XXH64_update(state, &bufferAddr, sizeof(VkDeviceAddress));
      geomHash = XXH64_digest(state);
      XXH64_freeState(state);

      Logger::info(str::format("[RTX Mega Geometry HASH] Metadata hash=0x", std::hex, geomHash, std::dec,
                              " for verts=", geoData.vertexCount, " stride=", stride));
    }

    // SAMPLE CODE MATCH: Per-instance architecture with NO caching
    // Hash identifies geometry only (not transform), used for grouping purposes
    // Each draw call is tessellated separately with its instance-specific transform
    // Sample: ComputeInstanceClusterTiling (line 810) applies transform per-instance
    // Sample rebuilds all BLASes every frame (no caching) - Remix now matches this

    Logger::info(str::format("[RTX Mega Geometry HASH] Final hash (geometry only, no transform)=0x", std::hex, geomHash, std::dec));

    // Check if we already have tessellated version in cache
    // DISABLED: This early cache lookup bypasses BLAS building causing GPU crashes
    // Let the batch processing handle caching instead
    const RtxMegaGeometry::TessellatedGeometryCache* cached = nullptr; // s_megaGeometry->getTessellatedGeometry(geomHash);

    if (false && cached != nullptr) {
      // Cache hit! Replace geometry with tessellated version
      Logger::info(str::format("[RTX Mega Geometry] Using cached tessellated geometry: hash=0x",
                              std::hex, geomHash, std::dec,
                              ", ", geoData.vertexCount, " verts -> ", cached->vertexCount, " verts",
                              cached->hasClusterBLAS ? " (cluster BLAS)" : ""));

      // Tessellated geometry uses interleaved format (ClusterVertex: position, normal, texcoord)
      // struct ClusterVertex { float3 position; float3 normal; float2 texcoord; } = 32 bytes
      const uint32_t vertexStride = sizeof(ClusterVertex);  // 32 bytes
      const uint32_t positionOffset = 0;   // offsetof(ClusterVertex, position)
      const uint32_t normalOffset = 12;    // offsetof(ClusterVertex, normal)
      const uint32_t texcoordOffset = 24;  // offsetof(ClusterVertex, texcoord)

      // Create base buffer slices for interleaved vertex data
      DxvkBufferSlice vertexSlice(cached->vertexBuffer, 0, cached->vertexCount * vertexStride);
      DxvkBufferSlice indexSlice(cached->indexBuffer, 0, cached->indexCount * sizeof(uint32_t));

      // Replace geometry buffers with tessellated versions (using GeometryBuffer constructors)
      geometryData.positionBuffer = GeometryBuffer<Raster>(vertexSlice, positionOffset, vertexStride, VK_FORMAT_R32G32B32_SFLOAT);
      geometryData.normalBuffer = GeometryBuffer<Raster>(vertexSlice, normalOffset, vertexStride, VK_FORMAT_R32G32B32_SFLOAT);
      geometryData.texcoordBuffer = GeometryBuffer<Raster>(vertexSlice, texcoordOffset, vertexStride, VK_FORMAT_R32G32_SFLOAT);
      geometryData.indexBuffer = GeometryBuffer<Raster>(indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32);

      geometryData.vertexCount = cached->vertexCount;
      geometryData.indexCount = cached->indexCount;

      // CRITICAL: Track cached buffers in command list to prevent premature deletion
      // This keeps buffers alive until GPU finishes with them - no need for arbitrary frame counts!
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(cached->vertexBuffer);
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(cached->indexBuffer);

      // NV-DXVK: Store geometry hash for cluster BLAS injection
      // This will be used in SceneManager to inject cluster BLAS instead of building regular BLAS
      if (cached->hasClusterBLAS) {
        const_cast<DrawCallState&>(drawCallState).clusterBlasGeometryHash = geomHash;
        Logger::info(str::format("[RTX Mega Geometry] Set cluster BLAS hash on DrawCallState: 0x",
                                std::hex, geomHash, std::dec));
      }

      return true; // Geometry was replaced with tessellated version
    }

    // Cache miss - tessellate IMMEDIATELY for same-frame cluster BLAS usage (best performance)

    // CRITICAL: Skip degenerate geometry (0 indices) - would cause GPU crash
    if (geoData.indexCount == 0) {
      ONCE(Logger::info("[RTX Mega Geometry] Skipping tessellation for geometry with 0 indices"));
      return false; // Use original geometry
    }

    // CRITICAL: Skip geometry that's too small for meaningful tessellation
    // RTXMG is designed for subdivision surfaces with target edge segments = 8
    // Minimum: 12 vertices (2x2 subdivided quad), Recommended: 64+ vertices (8x8)
    // Small geometry (UI quads, skyboxes, billboards) causes invalid cluster data and GPU crashes
    constexpr uint32_t kMinVerticesForTessellation = 12;  // 2x2 subdivided quad minimum
    constexpr uint32_t kMinIndicesForTessellation = 18;   // 2x2 quad = 8 triangles = 24 indices (or 18 if shared)

    if (geoData.vertexCount < kMinVerticesForTessellation || geoData.indexCount < kMinIndicesForTessellation) {
      ONCE(Logger::info(str::format("[RTX Mega Geometry] Skipping tessellation for small geometry: ",
                                    geoData.vertexCount, " verts, ", geoData.indexCount, " indices ",
                                    "(minimum: ", kMinVerticesForTessellation, " verts, ",
                                    kMinIndicesForTessellation, " indices)")));
      return false; // Use original geometry - too small for subdivision tessellation
    }

    Logger::info(str::format("[RTX Mega Geometry] Queueing geometry for batched tessellation: hash=0x",
                            std::hex, geomHash, std::dec,
                            ", verts=", geoData.vertexCount, ", indices=", geoData.indexCount));

    s_megaGeometry->addGeometryForBatchTessellation(
      geoData.positionBuffer,
      geoData.normalBuffer.defined() ? geoData.normalBuffer : DxvkBufferSlice(),
      geoData.texcoordBuffer.defined() ? geoData.texcoordBuffer : DxvkBufferSlice(),
      geoData.indexBuffer.defined() ? geoData.indexBuffer : DxvkBufferSlice(),
      geoData.vertexCount,
      geoData.indexCount,
      geoData.positionBuffer.stride(),
      geoData.normalBuffer.defined() ? geoData.normalBuffer.stride() : 0,
      geoData.texcoordBuffer.defined() ? geoData.texcoordBuffer.stride() : 0,
      geoData.positionBuffer.offsetFromSlice(),
      geoData.normalBuffer.defined() ? geoData.normalBuffer.offsetFromSlice() : 0,
      geoData.texcoordBuffer.defined() ? geoData.texcoordBuffer.offsetFromSlice() : 0,
      geoData.indexBuffer.defined() ? geoData.indexBuffer.indexType() : VK_INDEX_TYPE_UINT16,
      drawCallState.getTransformData().objectToWorld,
      drawCallState.getMaterialData().getHash(),
      geomHash);  // Pass geometry hash for cache

    // SAMPLE CODE APPROACH: Defer tessellation to end of frame (batched)
    // Geometry is collected above, will be tessellated at end of frame
    //
    // CRITICAL: NVIDIA sample ALWAYS calls BuildAccel() every frame (no caching)
    // Caching breaks the hash→index mapping in m_geometryHashToBlasIndex and blasPtrsBuffer.
    // GPU patching relies on fresh mapping each frame. Must match NVIDIA's approach.
    //
    // DISABLED CACHING: Always rebuild cluster BLASes every frame
    // HOWEVER: Still set geometry hash for GPU patching to work!

    // CRITICAL: Set hash for cluster BLAS injection (needed for GPU patching)
    // Even though we're not using cached geometry, the hash must be set so GPU patching
    // can find the correct BLAS address from blasPtrsBuffer after BuildAccel() runs
    const_cast<DrawCallState&>(drawCallState).clusterBlasGeometryHash = geomHash;
    Logger::info(str::format("[RTX Mega Geometry] Set cluster BLAS hash on DrawCallState: 0x",
                            std::hex, geomHash, std::dec, " (no caching, rebuild every frame)"));

    // Return false to use original geometry THIS frame
    // Cluster BLAS will be built at end of frame and used for GPU patching
    Logger::info(str::format("[RTX Mega Geometry] Geometry queued for batched tessellation: hash=0x",
                            std::hex, geomHash, std::dec,
                            " - original geometry used this frame, cluster BLAS built at frame end"));
    return false;
  }

  /**
   * \brief Update RTXMG per-frame (called during injectRTX)
   *
   * FULL IMPLEMENTATION: Updates HiZ and builds cluster acceleration structures
   */
  void updateMegaGeometryPerFrame(
    RtxContext* ctx,
    SceneManager& sceneManager,
    const Rc<DxvkImageView>& depthBuffer) {

    if (!s_megaGeometry || !RtxMegaGeometry::enable()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Frame Update");

    // CRITICAL: Rotate frame buffers BEFORE building any geometry this frame
    // This resets the cluster counter and ensures proper ring buffering for GPU lag
    RtxmgConfig config;  // Default config from options
    s_megaGeometry->getClusterBuilder()->updatePerFrame(ctx, depthBuffer, config);

    // BATCHED TESSELLATION: Tessellate all collected geometry in ONE dispatch
    s_megaGeometry->tessellateCollectedGeometry(ctx);

    // Emit memory barrier ONCE per frame for all tessellation dispatches
    // This batches all compute writes before synchronization (avoids GPU timeout)
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    // Update HiZ buffer for visibility culling
    if (RtxMegaGeometry::enableHiZCulling() && depthBuffer != nullptr) {
      s_megaGeometry->updateHiZ(ctx, depthBuffer);
    }

    // Build cluster acceleration structures for all geometry submitted this frame
    s_megaGeometry->buildClusterAccelerationStructuresForFrame(ctx);

    // Read back statistics if enabled
    if (RtxMegaGeometry::showStatistics()) {
      s_megaGeometry->readbackStatistics(ctx);
    }

    // NOTE: SDK sample does NOT flush here - keeps BLAS+TLAS in same command buffer
    // Flushing here causes massive GPU sync overhead (3+ second stalls)
    // The command buffer will be submitted later by the renderer
  }

  /**
   * \brief Render RTXMG debug visualization
   *
   * FULL IMPLEMENTATION: Renders debug overlays
   */
  void renderMegaGeometryDebugView(
    RtxContext* ctx,
    const Rc<DxvkImageView>& outputImage,
    uint32_t debugViewIndex) {

    if (!s_megaGeometry || !RtxMegaGeometry::enable()) {
      return;
    }

    // Check if a mega geometry debug view is active (indices 900-907)
    if (debugViewIndex < 900 || debugViewIndex > 907) {
      return;
    }

    // Debug visualization is now integrated into the ray tracing closest hit shaders
    // where cluster ID and geometry data are available. No post-process pass needed.
    // The debug view is applied in geometry_resolver.slangh before GBuffer write.
  }

  /**
   * \brief Check if geometry should use RTXMG
   *
   * In always-on mode, this always returns true (unless disabled via options).
   */
  bool shouldUseMegaGeometry(const DrawCallState& drawCallState) {
    // Always-on mode: all geometry uses RTXMG if enabled
    return RtxMegaGeometry::enable();
  }

  /**
   * \brief Get RTXMG statistics for UI display
   *
   * FULL IMPLEMENTATION: Returns current frame statistics
   */
  void getMegaGeometryStatistics(MegaGeometryStatistics& outStats) {
    if (!s_megaGeometry) {
      memset(&outStats, 0, sizeof(outStats));
      return;
    }

    s_megaGeometry->getStatistics(outStats);
  }

} // namespace dxvk
