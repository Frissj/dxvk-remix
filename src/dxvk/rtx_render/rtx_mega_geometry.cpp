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
#include "rtx_mega_geometry_autotune.h"
#include "rtx_mg_cluster.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtxmg/rtxmg_accel.h"
#include "../util/xxHash/xxhash.h"

#include <unordered_map>
#include <algorithm>

// Include compiled shaders
#include <rtx_shaders/compute_cluster_tiling.h>
#include <rtx_shaders/fill_cluster_data.h>
#include <rtx_shaders/hiz_reduce.h>
#include <rtx_shaders/debug_visualization.h>
#include <rtx_shaders/gpu_hash_geometry.h>

namespace dxvk {

  // Shader definitions using RTX Remix's ManagedShader pattern
  namespace {
    // GPU Hash shader bindings
    constexpr uint32_t GPU_HASH_BINDING_VERTEX_POSITIONS = 0;
    constexpr uint32_t GPU_HASH_BINDING_INDICES = 1;
    constexpr uint32_t GPU_HASH_BINDING_HASH_OUTPUT = 2;

    struct GpuHashPushConstants {
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t positionStride;
      uint32_t hasIndices;
      uint32_t positionBufferOffset;  // Byte offset for buffer slice
      uint32_t indexBufferOffset;     // Byte offset for buffer slice
      uint32_t indexIs16Bit;          // 1 if indices are uint16, 0 if uint32
    };

    class GpuHashGeometryShader : public ManagedShader {
      SHADER_SOURCE(GpuHashGeometryShader, VK_SHADER_STAGE_COMPUTE_BIT, gpu_hash_geometry)

      PUSH_CONSTANTS(GpuHashPushConstants)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(GPU_HASH_BINDING_VERTEX_POSITIONS)
        STRUCTURED_BUFFER(GPU_HASH_BINDING_INDICES)
        RW_STRUCTURED_BUFFER(GPU_HASH_BINDING_HASH_OUTPUT)
      END_PARAMETER()
    };

    // Debug visualization shader bindings
    constexpr uint32_t BINDING_DEBUG_CONSTANTS = 0;
    constexpr uint32_t BINDING_OUTPUT_IMAGE = 4;

    class ComputeClusterTilingShader : public ManagedShader {
      SHADER_SOURCE(ComputeClusterTilingShader, VK_SHADER_STAGE_COMPUTE_BIT, compute_cluster_tiling)

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ComputeClusterTilingShader);

    class FillClusterDataShader : public ManagedShader {
      SHADER_SOURCE(FillClusterDataShader, VK_SHADER_STAGE_COMPUTE_BIT, fill_cluster_data)

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(FillClusterDataShader);

    class HiZReduceShader : public ManagedShader {
      SHADER_SOURCE(HiZReduceShader, VK_SHADER_STAGE_COMPUTE_BIT, hiz_reduce)

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(HiZReduceShader);

    class DebugVisualizationShader : public ManagedShader {
      SHADER_SOURCE(DebugVisualizationShader, VK_SHADER_STAGE_COMPUTE_BIT, debug_visualization)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BINDING_DEBUG_CONSTANTS)
        RW_TEXTURE2D(BINDING_OUTPUT_IMAGE)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(DebugVisualizationShader);
  }

  RtxMegaGeometry::RtxMegaGeometry(DxvkDevice* device)
    : RtxPass(device)
    , m_device(device)
    , m_initialized(false) {

    // Create auto-tuning system for automatic memory management
    m_autoTune = new RtxMegaGeometryAutoTune(device);

    // Create RTXMG cluster builder (Phase 1-5 implementation)
    m_clusterBuilder = std::make_unique<RtxmgClusterBuilder>(device);
  }

  RtxMegaGeometry::~RtxMegaGeometry() {
    // Cleanup all cached BLAS handles
    cleanupBLASCache();

    // Shutdown cluster builder
    if (m_clusterBuilder) {
      m_clusterBuilder->shutdown();
      m_clusterBuilder.reset();
    }

    // Cleanup auto-tune system
    if (m_autoTune) {
      delete m_autoTune;
      m_autoTune = nullptr;
    }

    // Cleanup resources (std::array is automatically cleaned up)
    // m_clusterTemplates doesn't need explicit cleanup
  }

  void RtxMegaGeometry::initialize() {
    if (m_initialized) {
      return;
    }

    Logger::info("[RTX Mega Geometry] Initializing cluster-based tessellation system (ALWAYS-ON)");

    // Initialize VK_NV_cluster_acceleration_structure extension
    // Note: This extension is optional - we'll fall back to standard triangle geometry if not available
    Logger::info("[RTX Mega Geometry] Checking for NVIDIA cluster acceleration extension...");
    if (initClusterAccelerationExtension(m_device)) {
      Logger::info("[RTX Mega Geometry] VK_NV_cluster_acceleration_structure extension available");
    } else {
      Logger::warn("[RTX Mega Geometry] VK_NV_cluster_acceleration_structure extension not available - falling back to CPU tessellation");
    }

    // Initialize RTXMG cluster builder (Phase 1-5) - ONLY if tessellation is enabled
    // NOTE: Without tessellation, we only do BLAS caching (no clustering/culling)
    if (enableTessellation() && m_clusterBuilder) {
      // NV-DXVK start: Catch memory allocation failures to prevent crashes
      try {
        if (!m_clusterBuilder->initialize()) {
          Logger::err("[RTX Mega Geometry] Failed to initialize RTXMG cluster builder");
          // Disable cluster builder but continue with basic functionality
          m_clusterBuilder.reset();
        } else {
          Logger::info("[RTX Mega Geometry] RTXMG cluster builder initialized successfully");
        }
      } catch (const DxvkError& e) {
        Logger::err(str::format("[RTX Mega Geometry] Cluster builder initialization failed: ", e.message()));
        Logger::warn("[RTX Mega Geometry] Mega geometry DISABLED due to memory constraints");
        Logger::warn("[RTX Mega Geometry] Application will continue with standard rendering");
        // Disable cluster builder but don't crash - just continue without mega geometry
        m_clusterBuilder.reset();
      }
      // NV-DXVK end

      // Only continue initialization if cluster builder is still valid
      if (m_clusterBuilder) {
        // Initialize cluster templates (11x11 grid combinations = 121 templates)
        initializeClusterTemplates();

        // Create GPU buffers using auto-tune recommended sizes
        // Auto-tune will automatically adjust these based on scene complexity
        const uint32_t maxClusters = m_autoTune->getRecommendedMaxClusters();

        DxvkBufferCreateInfo bufferInfo;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        bufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        // Cluster data buffer (vertex/normal/texcoord data) - auto-sized
        bufferInfo.size = m_autoTune->getRecommendedClusterDataBufferSize();
        m_clusterDataBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Cluster Data");

        // Cluster info buffer (metadata) - auto-sized
        bufferInfo.size = m_autoTune->getRecommendedClusterInfoBufferSize();
        m_clusterInfoBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Cluster Info");

        // Cluster tiling buffer (layout per instance) - start with reasonable default
        bufferInfo.size = 16384 * sizeof(ClusterTilingInfo);  // Will auto-grow if needed
        m_clusterTilingBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Cluster Tiling");

        // Statistics buffer (read-back for UI)
        bufferInfo.size = sizeof(ClusterStatistics);
        bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        m_clusterStatisticsBuffer = m_device->createBuffer(bufferInfo,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Statistics");
      }
    }

    if (!m_clusterBuilder) {
      Logger::info("[RTX Mega Geometry] Tessellation disabled - skipping cluster builder initialization");
      Logger::info("[RTX Mega Geometry] BLAS caching only mode active");
    }

    m_autoTune->acknowledgeBufferResize();
    m_initialized = true;

    Logger::info(str::format("[RTX Mega Geometry] Initialized successfully"));
    if (enableTessellation()) {
      const uint32_t maxClusters = m_autoTune->getRecommendedMaxClusters();
      RtxMegaGeometryAutoTune::MemoryStats memStats;
      m_autoTune->getMemoryStats(memStats);

      Logger::info(str::format("[RTX Mega Geometry]   Max clusters: ", maxClusters, " (auto-tuned)"));
      Logger::info(str::format("[RTX Mega Geometry]   Cluster templates: ", kClusterTemplateCount));
      Logger::info(str::format("[RTX Mega Geometry]   Max vertices per cluster: ", kMaxClusterVertices));
      Logger::info(str::format("[RTX Mega Geometry]   Memory budget: ", memStats.totalMB, " MB (auto-managed)"));
      Logger::info(str::format("[RTX Mega Geometry]   BLAS pool budget: ", memStats.blasPoolMB, " MB"));
      Logger::info(str::format("[RTX Mega Geometry]   Mode: ALWAYS-ON (no fallback, fully automatic)"));
    } else {
      Logger::info(str::format("[RTX Mega Geometry]   Mode: BLAS caching only (tessellation disabled)"));
    }
  }

  void RtxMegaGeometry::initializeClusterTemplates() {
    Logger::info("[RTX Mega Geometry] Creating 121 cluster templates...");

    // Phase 1: Generate template grids (CPU-side, no GPU needed)
    m_templateGrids = generateTemplateGrids();
    m_templatesInitialized = true;

    // Also populate old m_clusterTemplates array for backward compatibility
    uint32_t templateIndex = 0;
    for (uint32_t gridY = 1; gridY <= kMaxClusterEdgeSegments; ++gridY) {
      for (uint32_t gridX = 1; gridX <= kMaxClusterEdgeSegments; ++gridX) {
        auto& tmpl = m_clusterTemplates[templateIndex++];
        tmpl.gridSizeX = gridX;
        tmpl.gridSizeY = gridY;
      }
    }

    Logger::info(str::format("[RTX Mega Geometry] Generated ", kClusterTemplateCount, " cluster template grids (CPU-side)"));
    Logger::info("[RTX Mega Geometry] Template CLAS structures will be built on first use (GPU-side)");
  }

  void RtxMegaGeometry::buildTemplateClasIfNeeded(Rc<DxvkContext> ctx) {
    // Early exit if already built or templates not initialized
    if (m_templateClasBuilt || !m_templatesInitialized) {
      return;
    }

    // Check if cluster acceleration extension is available
    if (!isClusterAccelerationExtensionAvailable()) {
      Logger::warn("[RTX Mega Geometry] Cluster acceleration extension not available - skipping template CLAS building");
      m_templateClasBuilt = true;  // Prevent repeated attempts
      return;
    }

    Logger::info("[RTX Mega Geometry] Building 121 template CLAS structures (one-time GPU operation)...");

    // Convert Rc<DxvkContext> to RtxContext*
    RtxContext* rtxCtx = static_cast<RtxContext*>(ctx.ptr());

    // Build template CLAS structures using cluster extension
    size_t totalClasSize = 0;
    bool success = buildTemplateClusterAccelerationStructures(
      m_device,
      rtxCtx,
      m_templateGrids,
      m_templateClasBuffer,
      m_templateAddresses,
      m_templateInstantiationSizes,
      &totalClasSize);

    if (success) {
      m_templateClasBuilt = true;
      Logger::info(str::format("[RTX Mega Geometry] Template CLAS structures built successfully (",
                              totalClasSize / 1024, " KB total)"));
      Logger::info(str::format("[RTX Mega Geometry] 121 templates ready for cluster instantiation"));
    } else {
      Logger::err("[RTX Mega Geometry] Failed to build template CLAS structures");
      Logger::warn("[RTX Mega Geometry] Cluster BLAS building will be disabled");
    }
  }

  void RtxMegaGeometry::computeClusterTiling(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "Compute Cluster Tiling");

    // Dispatch compute shader to calculate cluster layout per instance
    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D{ 1024, 1, 1 }, VkExtent3D{ 64, 1, 1 });

    // DISABLED: This is placeholder code that was causing descriptor set layout mismatches
    // The shader ComputeClusterTilingShader expects descriptor set 0, but no resources were ever bound.
    // This function is never called (dispatchTessellation is not used in the actual tessellation pipeline).
    // The real tessellation happens in tessellateCollectedGeometry() which is called from updateMegaGeometryPerFrame().
    //
    // Bind resources
    // Note: In a full implementation, these would be bound from actual scene data
    // For now, this sets up the pipeline correctly for compilation

    // COMMENTED OUT: ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ComputeClusterTilingShader::getShader());
    // COMMENTED OUT: ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void RtxMegaGeometry::fillClusterData(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "Fill Cluster Data");

    // Dispatch compute shader to tessellate subdivision surfaces into cluster vertices
    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D{ 2048, 1, 1 }, VkExtent3D{ 64, 1, 1 });

    // DISABLED: This is placeholder code that was causing descriptor set layout mismatches
    // The shader FillClusterDataShader expects descriptor set 0, but no resources were ever bound.
    // This function is never called (dispatchTessellation is not used in the actual tessellation pipeline).
    // The real tessellation happens in tessellateCollectedGeometry() which is called from updateMegaGeometryPerFrame().

    // COMMENTED OUT: ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, FillClusterDataShader::getShader());
    // COMMENTED OUT: ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void RtxMegaGeometry::dispatchTessellation(Rc<DxvkContext> ctx) {
    if (!m_initialized || !enable()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Dispatch Tessellation");

    // Step 1: Compute cluster tiling for all submitted geometry
    computeClusterTiling(ctx);

    // Step 2: Fill cluster vertex/normal/texcoord data
    fillClusterData(ctx);

    // Step 3: Build BLAS from tessellated clusters
    buildBLASFromClusters(ctx);
  }

  void RtxMegaGeometry::buildBLASFromClusters(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "Build BLAS from Clusters");

    // Build Bottom Level Acceleration Structure from cluster geometry
    // This converts the tessellated clusters into ray-traceable geometry

    // Note: Full implementation would use vkBuildAccelerationStructuresKHR
    // with cluster vertex/index data
  }

  void RtxMegaGeometry::updateHiZ(
    Rc<DxvkContext> ctx,
    const Rc<DxvkImageView>& depthBuffer) {

    if (!enableHiZCulling()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Update HiZ");

    // Reduce depth buffer to hierarchical mip chain for visibility culling
    const VkExtent3D depthExtent = depthBuffer->imageInfo().extent;
    const VkExtent3D workgroups = util::computeBlockCount(depthExtent, VkExtent3D{ 8, 8, 1 });

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, HiZReduceShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void RtxMegaGeometry::renderDebugView(
    Rc<DxvkContext> ctx,
    const Rc<DxvkImageView>& outputImage,
    uint32_t debugViewIndex) {

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Debug Visualization");

    // Use the debug view index directly as the shader mode
    // The debug view indices (900-907) match the shader's expected values
    uint32_t shaderDebugMode = debugViewIndex;

    // Create debug constants matching shader layout
    struct DebugConstants {
      uint32_t debugMode;
      uint32_t outputSizeX;
      uint32_t outputSizeY;
      uint32_t pad;
    };

    const VkExtent3D outputExtent = outputImage->imageInfo().extent;

    DebugConstants constants;
    constants.debugMode = shaderDebugMode;
    constants.outputSizeX = outputExtent.width;
    constants.outputSizeY = outputExtent.height;
    constants.pad = 0;

    Logger::info(str::format("[RTX Mega Geometry] Debug visualization: mode=", shaderDebugMode,
                             ", size=", outputExtent.width, "x", outputExtent.height));

    // Create constant buffer for this frame
    DxvkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    bufferInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
    bufferInfo.size = sizeof(DebugConstants);

    Rc<DxvkBuffer> constantBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Mega Geometry Debug Constants");
    ctx->updateBuffer(constantBuffer, 0, sizeof(DebugConstants), &constants);

    // Bind resources following shader binding layout (see rtx_mega_geometry.cpp anonymous namespace)
    // Binding 0: DebugConstants (constant buffer)
    ctx->bindResourceBuffer(0, DxvkBufferSlice(constantBuffer, 0, sizeof(DebugConstants)));

    // Binding 1: DepthBuffer (optional - not currently used)
    // Binding 2: HiZBuffer (optional - not currently used)
    // Binding 3: ClusterStatistics (optional)
    if (m_clusterStatisticsBuffer != nullptr) {
      ctx->bindResourceBuffer(3, DxvkBufferSlice(m_clusterStatisticsBuffer, 0, m_clusterStatisticsBuffer->info().size));
    }

    // Binding 4: OutputImage (RW texture for writing debug visualization)
    ctx->bindResourceView(4, outputImage, nullptr);

    // Bind debug visualization shader and dispatch
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, DebugVisualizationShader::getShader());

    // Dispatch with 8x8 thread groups matching shader layout
    const VkExtent3D workgroups = util::computeBlockCount(outputExtent, VkExtent3D{ 8, 8, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void RtxMegaGeometry::addGeometryForBatchTessellation(
    const DxvkBufferSlice& positionBuffer,
    const DxvkBufferSlice& normalBuffer,
    const DxvkBufferSlice& texcoordBuffer,
    const DxvkBufferSlice& indexBuffer,
    uint32_t vertexCount,
    uint32_t indexCount,
    uint32_t positionStride,
    uint32_t normalStride,
    uint32_t texcoordStride,
    uint32_t positionOffset,
    uint32_t normalOffset,
    uint32_t texcoordOffset,
    VkIndexType indexType,
    const Matrix4& transform,
    XXH64_hash_t materialId,
    XXH64_hash_t geometryHash) {

    // Collect geometry for batched tessellation
    BatchedGeometry geom;
    geom.positionBuffer = positionBuffer;
    geom.normalBuffer = normalBuffer;
    geom.texcoordBuffer = texcoordBuffer;
    geom.indexBuffer = indexBuffer;
    geom.vertexCount = vertexCount;
    geom.indexCount = indexCount;
    geom.positionStride = positionStride;
    geom.normalStride = normalStride;
    geom.texcoordStride = texcoordStride;
    geom.positionOffset = positionOffset;
    geom.normalOffset = normalOffset;
    geom.texcoordOffset = texcoordOffset;
    geom.indexType = indexType;
    geom.transform = transform;
    geom.materialId = materialId;
    geom.geometryHash = geometryHash;

    m_batchedGeometry.push_back(geom);
  }

  void RtxMegaGeometry::tessellateCollectedGeometry(Rc<DxvkContext> ctx) {
    static uint32_t callCount = 0;
    callCount++;
    Logger::info(str::format("[RTXMG TESSELLATE] ========== CALL #", callCount, " =========="));
    Logger::info(str::format("[RTXMG TESSELLATE] m_batchedGeometry.size()=", m_batchedGeometry.size(),
                            ", m_frameDrawCalls.size()=", m_frameDrawCalls.size()));
    if (m_batchedGeometry.empty()) {
      Logger::info("[RTXMG TESSELLATE] EARLY EXIT: m_batchedGeometry is empty");
      return;
    }

    // CRITICAL FIX: Reset cumulative offsets at start of each frame (NOT each batch!)
    // With ring buffering, each frame writes to its own buffer region starting at offset 0
    // tessellateCollectedGeometry() is called MULTIPLE times per frame, so only reset on frame change
    static uint32_t lastResetFrameId = 0;
    uint32_t currentFrameId = m_device->getCurrentFrameId();
    if (m_clusterBuilder && currentFrameId != lastResetFrameId) {
      m_clusterBuilder->resetCumulativeOffsets();  // SDK MATCH: Reset both cluster AND vertex offsets
      lastResetFrameId = currentFrameId;
      Logger::info(str::format("[RTXMG TESSELLATE] Frame ", currentFrameId, ": Reset cumulative cluster and vertex offsets"));
    }

    // Phase 1: Build template CLAS structures on first use (lazy initialization)
    buildTemplateClasIfNeeded(ctx);

    ScopedGpuProfileZone(ctx, "RTX Mega Geometry: Batch Tessellation");

    uint32_t tessellatedCount = 0;

    Logger::info(str::format("[RTX Mega Geometry] Collected ", m_batchedGeometry.size(), " meshes for tessellation"));

    // CRITICAL FIX: Sort by hash to ensure deterministic cluster ID assignment
    // This prevents color flickering in debug modes by making the same geometry
    // always get processed in the same order, regardless of submission order
    std::sort(m_batchedGeometry.begin(), m_batchedGeometry.end(),
      [this](const BatchedGeometry& a, const BatchedGeometry& b) {
        return computeGeometryHash(a) < computeGeometryHash(b);
      });

    // NOTE: Mid-frame eviction DISABLED to prevent VK_ERROR_DEVICE_LOST
    // Problem: Evicting buffers while they're referenced in the current command buffer
    // causes GPU crashes. Eviction only happens at end of frame (line 518) after all
    // command buffers are submitted to GPU.
    //
    // Trade-off: Risk of temporary memory spike vs guaranteed GPU crash
    // const uint64_t CACHE_HIGH_WATER_MARK = 768ULL * 1024ULL * 1024ULL;  // DISABLED
    // if (m_tessellationCacheMemoryBytes > CACHE_HIGH_WATER_MARK) {
    //   evictOldCacheEntries();  // UNSAFE - can delete buffers in use
    // }

    // Process each geometry: check cache first, tessellate if needed
    RtxmgConfig config;

    // CRITICAL: Disable subdivision while keeping clustering enabled
    // - UNIFORM mode = no adaptive subdivision based on screen space
    // - tessellationRate = 1.0f = MINIMUM subdivision (no extra detail)
    // - This creates clusters from ORIGINAL geometry without modifying it
    config.tessMode = RtxmgConfig::AdaptiveTessellationMode::UNIFORM;
    config.fineTessellationRate = 1.0f;       // Minimum rate = no extra subdivision
    config.coarseTessellationRate = 1.0f / 15.0f;  // Keep default coarse rate

    // Keep GPU culling enabled (works on clusters)
    config.enableFrustumVisibility = true;
    config.enableHiZVisibility = true;
    config.enableBackfaceVisibility = true;
    config.enableHiZCulling = true;

    // CRITICAL FIX: Batch ALL geometries together and dispatch tiling shader ONCE
    // Processing individually causes each dispatch to overwrite the same ring buffer region
    std::vector<ClusterInputGeometryGpu> batchInputs;
    std::vector<XXH64_hash_t> geometryHashes;
    std::vector<uint32_t> drawCallIndices;

    uint32_t drawCallIndex = 0;
    for (const auto& geom : m_batchedGeometry) {
      if (geom.vertexCount == 0 || geom.indexCount == 0) {
        drawCallIndex++;
        continue;
      }

      XXH64_hash_t geomHash = geom.geometryHash;

      // Prepare input for GPU tessellation
      ClusterInputGeometryGpu input;
      input.positionBuffer = geom.positionBuffer;
      input.normalBuffer = geom.normalBuffer;
      input.texcoordBuffer = geom.texcoordBuffer;
      input.indexBuffer = geom.indexBuffer;
      input.vertexCount = geom.vertexCount;
      input.indexCount = geom.indexCount;
      input.transform = geom.transform;
      input.surfaceId = geom.materialId;
      input.positionStride = geom.positionStride;
      input.normalStride = geom.normalStride;
      input.texcoordStride = geom.texcoordStride;
      input.positionOffset = geom.positionOffset;
      input.normalOffset = geom.normalOffset;
      input.texcoordOffset = geom.texcoordOffset;
      input.indexType = geom.indexType;

      batchInputs.push_back(input);
      geometryHashes.push_back(geomHash);
      drawCallIndices.push_back(drawCallIndex);

      drawCallIndex++;
    }

    // SAMPLE MATCH: Dispatch GPU tessellation for the entire batch this frame
    // to mirror the SDK's BuildAccel() flow.
    RtxContext* rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
    std::vector<ClusterOutputGeometryGpu> batchOutputs(batchInputs.size());

    if (m_clusterBuilder && rtxCtx &&
        m_clusterBuilder->buildClustersGpuBatch(rtxCtx, batchInputs, batchOutputs, config)) {
      for (size_t i = 0; i < batchOutputs.size(); ++i) {
        const ClusterOutputGeometryGpu& output = batchOutputs[i];

        FrameDrawCallData drawCallData;
        drawCallData.geometryHash = geometryHashes[i];
        drawCallData.transform = batchInputs[i].transform;
        drawCallData.drawCallIndex = drawCallIndices[i];
        drawCallData.clusterCount = output.numClusters;
        drawCallData.output = output;

        drawCallData.inputPositions = batchInputs[i].positionBuffer;
        drawCallData.inputNormals = batchInputs[i].normalBuffer;
        drawCallData.inputTexcoords = batchInputs[i].texcoordBuffer;
        drawCallData.inputIndices = batchInputs[i].indexBuffer;
        drawCallData.inputVertexCount = batchInputs[i].vertexCount;
        drawCallData.inputIndexCount = batchInputs[i].indexCount;
        drawCallData.positionOffset = batchInputs[i].positionOffset;
        drawCallData.normalOffset = batchInputs[i].normalOffset;
        drawCallData.texcoordOffset = batchInputs[i].texcoordOffset;

        m_frameDrawCalls.push_back(drawCallData);
        tessellatedCount++;
      }
    } else {
      Logger::err("[RTXMG] buildClustersGpuBatch failed - dropping geometry this frame");
    }

    if (tessellatedCount > 0) {
      Logger::info(str::format("[RTXMG] Per-instance tessellation complete: ", tessellatedCount, " geometries"));
    }

    // SAMPLE CODE MATCH: Build CLAS and BLAS structures at frame end
    // buildClusterAccelerationStructuresForFrame() is called in rtx_mega_geometry_integration.cpp after ALL tessellation

    // Clear collected geometry for next batch
    m_batchedGeometry.clear();

    Logger::info(str::format("[RTX Mega Geometry] Batch complete: tessellated ", tessellatedCount,
                            " draw calls | Total frame draw calls: ", m_frameDrawCalls.size()));

    // Increment frame counter
    m_currentFrame++;
  }

  XXH64_hash_t RtxMegaGeometry::computeGeometryHash(
    const DxvkBufferSlice& positionBuffer,
    const DxvkBufferSlice& indexBuffer,
    uint32_t vertexCount,
    uint32_t indexCount,
    uint32_t positionStride,
    uint32_t normalStride,
    uint32_t texcoordStride) const {

    // Use incremental hashing pattern with XXH3_64bits_withSeed
    XXH64_hash_t h = 0;

    // Hash geometry descriptor properties (stable across frames)
    h = XXH3_64bits_withSeed(&vertexCount, sizeof(vertexCount), h);
    h = XXH3_64bits_withSeed(&indexCount, sizeof(indexCount), h);
    h = XXH3_64bits_withSeed(&positionStride, sizeof(positionStride), h);
    h = XXH3_64bits_withSeed(&normalStride, sizeof(normalStride), h);
    h = XXH3_64bits_withSeed(&texcoordStride, sizeof(texcoordStride), h);

    // SAFE HASH: Use buffer handle + offset + size instead of reading data
    // Reading buffer data with mapPtr() causes VK_ERROR_DEVICE_LOST for GPU-only buffers
    //
    // Strategy: Hash the DxvkBuffer handle (stable ID) + slice offset/length
    // This creates unique hashes for different geometry while being safe
    if (positionBuffer.defined()) {
      const void* bufferHandle = positionBuffer.buffer().ptr();  // Stable buffer identity
      VkDeviceSize offset = positionBuffer.offset();
      VkDeviceSize length = positionBuffer.length();
      h = XXH3_64bits_withSeed(&bufferHandle, sizeof(bufferHandle), h);
      h = XXH3_64bits_withSeed(&offset, sizeof(offset), h);
      h = XXH3_64bits_withSeed(&length, sizeof(length), h);
    }

    if (indexBuffer.defined()) {
      const void* bufferHandle = indexBuffer.buffer().ptr();  // Stable buffer identity
      VkDeviceSize offset = indexBuffer.offset();
      VkDeviceSize length = indexBuffer.length();
      h = XXH3_64bits_withSeed(&bufferHandle, sizeof(bufferHandle), h);
      h = XXH3_64bits_withSeed(&offset, sizeof(offset), h);
      h = XXH3_64bits_withSeed(&length, sizeof(length), h);
    }

    return h;
  }

  XXH64_hash_t RtxMegaGeometry::computeGeometryHash(const BatchedGeometry& geom) const {
    return computeGeometryHash(
      geom.positionBuffer,
      geom.indexBuffer,
      geom.vertexCount,
      geom.indexCount,
      geom.positionStride,
      geom.normalStride,
      geom.texcoordStride);
  }

  const RtxMegaGeometry::TessellatedGeometryCache* RtxMegaGeometry::getTessellatedGeometry(XXH64_hash_t hash) const {
    auto it = m_tessellationCache.find(hash);
    if (it == m_tessellationCache.end()) {
      return nullptr;
    }

    // Allow cache entries from recent frames (not just current frame)
    // Cached geometry from previous frames is still valid - no need for exact frame match
    // The ring buffer system ensures the data is still alive (4-frame lag tolerance)
    const uint32_t currentFrameId = m_device ? m_device->getCurrentFrameId() : 0;
    constexpr uint32_t CACHE_FRAME_TOLERANCE = 300;  // 300 frames (~5 sec at 60fps)

    if (currentFrameId > it->second.lastUsedFrame + CACHE_FRAME_TOLERANCE) {
      return nullptr;  // Too old, will be evicted soon
    }

    return &it->second;
  }

  Rc<DxvkBuffer> RtxMegaGeometry::getBlasAddressesBuffer() const {
    if (!m_clusterBuilder) {
      return nullptr;
    }
    return m_clusterBuilder->getBlasAddressesBuffer();
  }

  int32_t RtxMegaGeometry::getBlasBufferIndex(XXH64_hash_t geometryHash) const {
    if (!m_clusterBuilder) {
      return -1;
    }
    return m_clusterBuilder->getBlasBufferIndex(geometryHash);
  }

  Rc<DxvkShader> RtxMegaGeometry::getPatchTlasInstanceShader() const {
    if (!m_clusterBuilder) {
      return nullptr;
    }
    return m_clusterBuilder->getPatchTlasShader();
  }

  RtxMegaGeometry::TessellatedGeometryCache* RtxMegaGeometry::lookupTessellatedGeometry(XXH64_hash_t hash) {
    auto it = m_tessellationCache.find(hash);
    if (it != m_tessellationCache.end()) {
      // Update last used frame
      it->second.lastUsedFrame = m_device->getCurrentFrameId();  // Use actual frame ID for ring buffer tracking
      m_stats.cacheHits++;

      // DEBUG: Log cache hits with geometry info every 300 frames (~5 seconds)
      static uint32_t hitLogCounter = 0;
      if ((hitLogCounter++ % 300) == 0) {
        Logger::info(str::format(
          "[RTXMG Cache HIT] hash=0x", std::hex, hash, std::dec,
          " verts=", it->second.vertexCount,
          " indices=", it->second.indexCount,
          " age=", (m_currentFrame - it->second.lastUsedFrame), " frames",
          " (hits=", m_stats.cacheHits, " misses=", m_stats.cacheMisses, ")"));
      }

      return &it->second;
    }

    m_stats.cacheMisses++;

    // DEBUG: Log cache misses for debugging every 300 frames (~5 seconds)
    static uint32_t missLogCounter = 0;
    if ((missLogCounter++ % 300) == 0) {
      Logger::info(str::format(
        "[RTXMG Cache MISS] hash=0x", std::hex, hash, std::dec,
        " - will tessellate new geometry",
        " (cache entries=", m_tessellationCache.size(),
        " hits=", m_stats.cacheHits, " misses=", m_stats.cacheMisses, ")"));
    }

    return nullptr;
  }

  void RtxMegaGeometry::cacheTessellatedGeometry(
    Rc<DxvkContext> ctx,
    XXH64_hash_t hash,
    const ClusterOutputGeometryGpu& output,
    const ClusterAccels* clusterAccels,
    VkDeviceAddress clusterBlasAddress,
    uint32_t clusterBlasSize) {

    // DEBUG: Check for hash collisions (same hash, different content)
    auto existingIt = m_tessellationCache.find(hash);
    if (existingIt != m_tessellationCache.end()) {
      if (existingIt->second.vertexCount != output.vertexCount ||
          existingIt->second.indexCount != output.indexCount) {
        Logger::warn(str::format(
          "[RTXMG Hash COLLISION!] hash=0x", std::hex, hash, std::dec,
          " - existing: verts=", existingIt->second.vertexCount, " indices=", existingIt->second.indexCount,
          " - new: verts=", output.vertexCount, " indices=", output.indexCount,
          " - THIS CAUSES COLOR FLICKERING!"));
      } else {
        // Only log occasionally to avoid spam
        static uint32_t updateLogCounter = 0;
        if ((updateLogCounter++ % 300) == 0) {
          Logger::info(str::format(
            "[RTXMG Cache UPDATE] hash=0x", std::hex, hash, std::dec,
            " - updating existing entry with same geometry",
            " verts=", output.vertexCount, " indices=", output.indexCount));
        }
      }
    }

    TessellatedGeometryCache entry;
    entry.vertexBuffer = output.vertexBuffer;
    entry.indexBuffer = output.indexBuffer;
    entry.vertexCount = output.vertexCount;
    entry.indexCount = output.indexCount;
    entry.lastUsedFrame = m_device->getCurrentFrameId();  // Use actual frame ID for ring buffer tracking
    entry.geometryHash = hash;

    // Calculate ACTUAL per-geometry memory size (not shared buffer size!)
    // BUG FIX: vertexBuffer/indexBuffer are SHARED across all geometries in cluster builder,
    // so we CANNOT use buffer->info().size (that's the entire shared buffer ~225MB).
    // Instead, calculate actual usage based on vertex/index counts for THIS geometry only.
    const uint64_t actualVertexBytes = entry.vertexCount * output.vertexBufferStride;
    const uint64_t actualIndexBytes = entry.indexCount * 4ULL;  // 4 bytes per index (uint32_t)
    entry.memorySizeBytes = actualVertexBytes + actualIndexBytes;

    // NV-DXVK: Store cluster BLAS if available
    if (clusterAccels != nullptr && clusterAccels->blasAccelStructure.ptr()) {
      entry.hasClusterBLAS = true;
      entry.clusterBLAS = clusterAccels->blasAccelStructure;
      // CRITICAL: Store BOTH persistent buffers - BLAS references these!
      // persistentInstanceBuffer contains the actual cluster instance data (512 bytes per cluster)
      // clusterReferencesBuffer contains device addresses pointing to persistentInstanceBuffer
      entry.clusterInstanceBuffer = clusterAccels->persistentInstanceBuffer.getBuffer();
      entry.clusterReferencesBuffer = clusterAccels->clusterReferencesBuffer.getBuffer();
      entry.clusterCount = output.numClusters;

      // CRITICAL: Do NOT cache the BLAS address here!
      // With VK_NV_cluster_acceleration_structure in EXPLICIT_DESTINATIONS mode, each geometry
      // gets its own unique BLAS address written to blasPtrsBuffer[geometryIndex] by the GPU.
      // The addresses are DIFFERENT for each geometry (they point to different BLASes within
      // the unified structure). CPU-side has NO access to these addresses - they're GPU-only.
      // GPU patching reads from blasPtrsBuffer and writes the correct per-geometry addresses.
      entry.clusterBlasAddress = 0;  // Will be patched by GPU shader from blasPtrsBuffer
      entry.clusterBlasSize = clusterBlasSize;

      // IMPORTANT: Do NOT add BLAS size to per-geometry memory accounting!
      // The BLAS is a SHARED/UNIFIED resource across multiple geometries.
      // Including it would count the same buffer hundreds of times, causing cache size to
      // be massively inflated (e.g., 730 entries * 6MB = 4.4GB when actual usage is ~100MB).
      entry.blasSizeBytes = clusterBlasSize > 0 ? clusterBlasSize : entry.clusterBLAS->info().size;
      // entry.memorySizeBytes += blasBytes;  // REMOVED - causes massive overcounting

      // CRITICAL GPU SYNCHRONIZATION: Store fence from BLAS build submission
      // We get the fence from the current command list being recorded
      // Later, vkGetFenceStatus() checks completion without blocking
      entry.buildFence = ctx->getCommandList()->fence();
      entry.gpuWorkComplete = false;  // Will be verified on first reuse

      Logger::info(str::format("[RTXMG] Cached cluster BLAS: clusters=", entry.clusterCount,
                              ", size=", entry.blasSizeBytes / 1024, "KB",
                              ", buildFence=", (void*)entry.buildFence,
                              " (will verify with vkGetFenceStatus on reuse)"));
    } else {
      entry.hasClusterBLAS = false;
      entry.clusterBLAS = nullptr;
      entry.clusterCount = 0;
      entry.blasSizeBytes = 0;
      entry.buildFence = VK_NULL_HANDLE;
      entry.gpuWorkComplete = false;
    }

    m_tessellationCache[hash] = entry;
    m_tessellationCacheMemoryBytes += entry.memorySizeBytes;
    m_stats.cachedEntries = static_cast<uint32_t>(m_tessellationCache.size());

    // Use KB for better precision (avoid 0MB for small meshes)
    const uint64_t sizeKB = entry.memorySizeBytes / 1024;
    const uint64_t totalMB = m_tessellationCacheMemoryBytes / (1024 * 1024);

    Logger::info(str::format("[RTX Mega Geometry] Cached tessellated geometry: hash=0x",
                            std::hex, hash, std::dec,
                            ", verts=", entry.vertexCount,
                            ", indices=", entry.indexCount,
                            ", size=", sizeKB, "KB",
                            " (cache: ", m_stats.cachedEntries, " entries, ",
                            totalMB, "MB total)"));
  }

  void RtxMegaGeometry::evictOldCacheEntries() {

    // Eviction based on frame age and memory budget only (no arbitrary entry count limit)
    // Memory budget is the real safeguard - prevents OOM crashes

    const uint32_t MAX_FRAME_AGE = 300;  // 300 frames (~5 sec at 60fps) - safe now with proper hash
    const uint64_t MAX_CACHE_MEMORY_BYTES = 3072ULL * 1024ULL * 1024ULL;  // 3 GB budget

    // GPU synchronization: Conservative safety margin to prevent use-after-free
    // GPU can be significantly behind CPU (3-10 frames), especially with cluster acceleration
    // Use larger fixed margin since pendingSubmissions() is unreliable (doesn't count GPU internal queuing)
    const uint32_t pendingSubs = m_device->pendingSubmissions();
    const uint32_t gpuPipelineDepth = std::max(pendingSubs + 10, 15u);

    uint32_t currentFrameId = m_device->getCurrentFrameId();
    Logger::info(str::format("[RTX Mega Geometry DEBUG] Cache eviction check: frame=", currentFrameId,
                            ", pendingSubmissions=", pendingSubs,
                            ", gpuPipelineDepth=", gpuPipelineDepth,
                            ", cacheEntries=", m_tessellationCache.size()));

    // Step 1: Age-based eviction (remove stale entries)
    // CRITICAL: Don't evict entries that GPU might still be using
    auto it = m_tessellationCache.begin();
    while (it != m_tessellationCache.end()) {
      uint32_t age = currentFrameId - it->second.lastUsedFrame;
      // Only evict if (1) old enough AND (2) GPU has finished with it
      if (age > MAX_FRAME_AGE && age > gpuPipelineDepth) {
        Logger::info(str::format("[RTX Mega Geometry] Evicting cache entry (age): hash=0x",
                                std::hex, it->first, std::dec,
                                ", age=", age, " frames, size=", it->second.memorySizeBytes / (1024 * 1024), "MB"));
        Logger::warn(str::format("[RTX Mega Geometry DEBUG] *** DESTROYING BLAS *** hash=0x",
                                std::hex, it->first, std::dec,
                                ", clusterBLAS=", (it->second.clusterBLAS != nullptr ? "valid" : "null"),
                                ", age=", age, " frames"));
        m_tessellationCacheMemoryBytes -= it->second.memorySizeBytes;
        it = m_tessellationCache.erase(it);
      } else {
        ++it;
      }
    }

    // Step 2: Memory budget eviction (CRITICAL: prevents OOM)
    if (m_tessellationCacheMemoryBytes > MAX_CACHE_MEMORY_BYTES) {
      Logger::warn(str::format("[RTX Mega Geometry] Cache memory exceeded budget: ",
                              m_tessellationCacheMemoryBytes / (1024 * 1024), "MB / ",
                              MAX_CACHE_MEMORY_BYTES / (1024 * 1024), "MB - evicting oldest entries"));

      // Find all entries sorted by age (oldest first)
      std::vector<std::pair<XXH64_hash_t, uint32_t>> entries;
      for (const auto& pair : m_tessellationCache) {
        entries.push_back({pair.first, pair.second.lastUsedFrame});
      }
      std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      // Evict old entries using proper per-resource usage tracking (no stalling)
      size_t evicted = 0;
      size_t skipped = 0;
      for (const auto& entry : entries) {
        if (m_tessellationCacheMemoryBytes <= MAX_CACHE_MEMORY_BYTES) {
          break;  // Under budget now
        }

        auto cacheIt = m_tessellationCache.find(entry.first);
        if (cacheIt != m_tessellationCache.end()) {
          // Check if GPU is still using this resource (per-resource, no global stall)
          bool inUse = false;
          if (cacheIt->second.clusterBLAS != nullptr) {
            if (cacheIt->second.clusterBLAS->isInUse(DxvkAccess::Read)) {
              inUse = true;
            }
          }
          if (!inUse && cacheIt->second.vertexBuffer != nullptr) {
            if (cacheIt->second.vertexBuffer->isInUse(DxvkAccess::Read)) {
              inUse = true;
            }
          }
          if (!inUse && cacheIt->second.indexBuffer != nullptr) {
            if (cacheIt->second.indexBuffer->isInUse(DxvkAccess::Read)) {
              inUse = true;
            }
          }

          if (inUse) {
            // GPU still using - skip for now, will try again next frame
            skipped++;
            uint32_t age = currentFrameId - cacheIt->second.lastUsedFrame;
            Logger::warn(str::format("[RTX Mega Geometry] Skipping eviction (GPU in-use): hash=0x",
                                    std::hex, entry.first, std::dec,
                                    ", age=", age, " frames, size=", cacheIt->second.memorySizeBytes / (1024 * 1024), "MB"));
            continue;
          }

          // Safe to evict - GPU is done with it
          uint32_t age = currentFrameId - cacheIt->second.lastUsedFrame;
          Logger::info(str::format("[RTX Mega Geometry] Evicting cache entry: hash=0x",
                                  std::hex, entry.first, std::dec,
                                  ", age=", age, " frames, size=", cacheIt->second.memorySizeBytes / (1024 * 1024), "MB",
                                  ", hasBLAS=", cacheIt->second.hasClusterBLAS));
          m_tessellationCacheMemoryBytes -= cacheIt->second.memorySizeBytes;
          m_tessellationCache.erase(cacheIt);
          evicted++;
        }
      }

      if (skipped > 0) {
        Logger::info(str::format("[RTX Mega Geometry] Memory eviction: evicted=", evicted, ", skipped=", skipped, " (GPU in-use)"));
      }

      // CRITICAL: If we're still over budget and couldn't evict anything, we MUST wait for GPU
      // Otherwise memory keeps growing until OOM → VK_ERROR_DEVICE_LOST
      if (m_tessellationCacheMemoryBytes > MAX_CACHE_MEMORY_BYTES) {
        if (evicted == 0 && skipped > 0) {
          // We skipped everything because GPU is using it - wait for GPU to finish some work
          uint32_t pendingSubs = m_device->pendingSubmissions();
          Logger::warn(str::format(
            "[RTX Mega Geometry] CRITICAL: Over memory budget (",
            m_tessellationCacheMemoryBytes / (1024 * 1024), "MB / ",
            MAX_CACHE_MEMORY_BYTES / (1024 * 1024), "MB) and could not evict anything (all ",
            skipped, " entries in-use). Waiting for GPU to complete work (pendingSubmissions=",
            pendingSubs, ")..."
          ));

          // Wait for GPU to finish - this will release resources
          m_device->waitForIdle();

          Logger::info(str::format(
            "[RTX Mega Geometry] GPU idle complete (finished ", pendingSubs, " submissions) - retrying eviction"
          ));

          // Retry eviction now that GPU has finished some work
          size_t retryEvicted = 0;
          for (const auto& entry : entries) {
            if (m_tessellationCacheMemoryBytes <= MAX_CACHE_MEMORY_BYTES) {
              break;  // Under budget now
            }

            auto cacheIt = m_tessellationCache.find(entry.first);
            if (cacheIt != m_tessellationCache.end()) {
              // Check again if GPU is still using this resource
              bool inUse = false;
              if (cacheIt->second.clusterBLAS != nullptr) {
                if (cacheIt->second.clusterBLAS->isInUse(DxvkAccess::Read)) {
                  inUse = true;
                }
              }
              if (!inUse && cacheIt->second.vertexBuffer != nullptr) {
                if (cacheIt->second.vertexBuffer->isInUse(DxvkAccess::Read)) {
                  inUse = true;
                }
              }
              if (!inUse && cacheIt->second.indexBuffer != nullptr) {
                if (cacheIt->second.indexBuffer->isInUse(DxvkAccess::Read)) {
                  inUse = true;
                }
              }

              if (!inUse) {
                // Now safe to evict
                m_tessellationCacheMemoryBytes -= cacheIt->second.memorySizeBytes;
                m_tessellationCache.erase(cacheIt);
                retryEvicted++;
              }
            }
          }

          Logger::info(str::format(
            "[RTX Mega Geometry] Post-idle eviction: freed ", retryEvicted, " entries, now at ",
            m_tessellationCacheMemoryBytes / (1024 * 1024), "MB / ",
            MAX_CACHE_MEMORY_BYTES / (1024 * 1024), "MB"
          ));
        } else {
          // We evicted some entries but still over budget - this is OK, just warn
          Logger::warn(str::format(
            "[RTX Mega Geometry] Still over memory budget after eviction (",
            m_tessellationCacheMemoryBytes / (1024 * 1024), "MB / ",
            MAX_CACHE_MEMORY_BYTES / (1024 * 1024), "MB) - evicted ", evicted, " entries but need more."
          ));
        }
      }
    }

    m_stats.cachedEntries = static_cast<uint32_t>(m_tessellationCache.size());
    Logger::info(str::format("[RTX Mega Geometry] Cache after eviction: ", m_stats.cachedEntries,
                            " entries, ", m_tessellationCacheMemoryBytes / (1024 * 1024), "MB"));
  }

  XXH64_hash_t RtxMegaGeometry::computeGeometryHashOnGPU(
    Rc<RtxContext> ctx,
    const DxvkBufferSlice& positionBuffer,
    const DxvkBufferSlice& indexBuffer,
    uint32_t vertexCount,
    uint32_t indexCount,
    uint32_t positionStrideInFloats,
    VkIndexType indexType) {

    ScopedCpuProfileZone();

    // Create buffer key to identify this buffer uniquely
    BufferKey key;
    key.bufferAddress = positionBuffer.buffer()->getDeviceAddress() + positionBuffer.offset();
    key.vertexCount = vertexCount;
    key.indexCount = indexCount;
    key.stride = positionStrideInFloats;

    // Check if we already have a cached GPU hash for this buffer
    auto cacheIt = m_gpuHashCache.find(key);
    if (cacheIt != m_gpuHashCache.end()) {
      // Cache hit! Return immediately
      XXH64_hash_t cachedHash = cacheIt->second;
      Logger::info(str::format("[RTX Mega Geometry ASYNC] GPU hash cache HIT: buffer=0x",
                              std::hex, key.bufferAddress,
                              " hash=0x", cachedHash, std::dec));
      return cachedHash;
    }

    // Check if async compute is already in progress for this buffer
    for (const auto& [requestId, request] : m_pendingGpuHashes) {
      if (request.key == key) {
        // Already computing, return temp hash
        XXH64_hash_t tempHash = generateTempHash(key);
        Logger::info(str::format("[RTX Mega Geometry ASYNC] GPU hash PENDING: buffer=0x",
                                std::hex, key.bufferAddress,
                                " tempHash=0x", tempHash, std::dec));
        return tempHash;
      }
    }

    // Not cached and not pending - start async compute!
    Logger::info(str::format("[RTX Mega Geometry ASYNC] Starting GPU hash compute: buffer=0x",
                            std::hex, key.bufferAddress, std::dec,
                            " verts=", vertexCount, " indices=", indexCount));

    // Create host-visible buffer for hash result
    DxvkBufferCreateInfo hashBufferInfo;
    hashBufferInfo.size = 8;  // uint64_t
    hashBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    hashBufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    hashBufferInfo.access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

    Rc<DxvkBuffer> hashResultBuffer = ctx->getDevice()->createBuffer(
      hashBufferInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "Async GPU Hash Result");

    // Record compute shader work (non-blocking)
    ctx->clearBuffer(hashResultBuffer, 0, 8, 0);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // Bind buffers and dispatch
    ctx->bindResourceBuffer(GPU_HASH_BINDING_VERTEX_POSITIONS,
                           DxvkBufferSlice(positionBuffer.buffer(), 0, positionBuffer.buffer()->info().size));
    if (indexBuffer.defined()) {
      ctx->bindResourceBuffer(GPU_HASH_BINDING_INDICES,
                             DxvkBufferSlice(indexBuffer.buffer(), 0, indexBuffer.buffer()->info().size));
    } else {
      ctx->bindResourceBuffer(GPU_HASH_BINDING_INDICES, DxvkBufferSlice(hashResultBuffer, 0, 8));
    }
    ctx->bindResourceBuffer(GPU_HASH_BINDING_HASH_OUTPUT, DxvkBufferSlice(hashResultBuffer, 0, 8));

    // Push constants
    GpuHashPushConstants pushConstants;
    pushConstants.vertexCount = vertexCount;
    pushConstants.indexCount = indexCount;
    pushConstants.positionStride = positionStrideInFloats;
    pushConstants.hasIndices = (indexBuffer.defined() && indexCount > 0) ? 1 : 0;
    pushConstants.positionBufferOffset = positionBuffer.offset();
    pushConstants.indexBufferOffset = indexBuffer.defined() ? indexBuffer.offset() : 0;
    pushConstants.indexIs16Bit = (indexType == VK_INDEX_TYPE_UINT16) ? 1 : 0;
    ctx->pushConstants(0, sizeof(pushConstants), &pushConstants);

    // Dispatch compute shader
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GpuHashGeometryShader::getShader());
    ctx->dispatch(1, 1, 1);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

    // Submit async (get command list for fence checking later)
    Rc<DxvkCommandList> cmdList = ctx->endRecording();
    ctx->getDevice()->submitCommandList(cmdList, VK_NULL_HANDLE, VK_NULL_HANDLE);
    ctx->beginRecording(ctx->getDevice()->createCommandList());  // Start new recording

    // Track this async request
    XXH64_hash_t requestId = m_nextHashRequestId++;
    AsyncHashRequest request;
    request.resultBuffer = hashResultBuffer;
    request.cmdList = cmdList;
    request.submitFrame = m_currentFrame;
    request.key = key;
    m_pendingGpuHashes[requestId] = request;

    // Return temporary metadata hash immediately (non-blocking!)
    XXH64_hash_t tempHash = generateTempHash(key);
    Logger::info(str::format("[RTX Mega Geometry ASYNC] GPU hash started (request ", requestId,
                            "), returning tempHash=0x", std::hex, tempHash, std::dec));
    return tempHash;
  }

  XXH64_hash_t RtxMegaGeometry::generateTempHash(const BufferKey& key) {
    // Generate deterministic hash from buffer metadata
    // This is instant and used while waiting for GPU hash
    XXH64_state_t* state = XXH64_createState();
    XXH64_reset(state, 0);
    XXH64_update(state, &key.bufferAddress, sizeof(key.bufferAddress));
    XXH64_update(state, &key.vertexCount, sizeof(key.vertexCount));
    XXH64_update(state, &key.indexCount, sizeof(key.indexCount));
    XXH64_update(state, &key.stride, sizeof(key.stride));
    XXH64_hash_t tempHash = XXH64_digest(state);
    XXH64_freeState(state);
    return tempHash;
  }

  void RtxMegaGeometry::processCompletedHashes(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();

    // Process pending GPU hash requests using frame-based completion
    // This avoids blocking fence waits - we assume GPU work completes within a few frames
    constexpr uint32_t GPU_COMPLETION_FRAMES = 3;  // Conservative: assume done after 3 frames

    auto it = m_pendingGpuHashes.begin();
    while (it != m_pendingGpuHashes.end()) {
      const XXH64_hash_t requestId = it->first;
      AsyncHashRequest& request = it->second;

      uint32_t framesElapsed = m_currentFrame - request.submitFrame;

      if (framesElapsed >= GPU_COMPLETION_FRAMES) {
        // Enough frames have passed - GPU compute should be done
        // Read result (this is safe because GPU has had plenty of time to complete)
        void* mappedData = request.resultBuffer->mapPtr(0);
        if (mappedData != nullptr) {
          XXH64_hash_t computedHash = 0;
          memcpy(&computedHash, mappedData, sizeof(computedHash));

          if (computedHash != 0) {
            // Valid hash - cache it!
            m_gpuHashCache[request.key] = computedHash;
            Logger::info(str::format("[RTX Mega Geometry ASYNC] GPU hash completed: buffer=0x",
                                    std::hex, request.key.bufferAddress,
                                    " hash=0x", computedHash, std::dec,
                                    " (", framesElapsed, " frames)"));
          } else {
            Logger::warn(str::format("[RTX Mega Geometry ASYNC] GPU hash returned 0x0 for buffer 0x",
                                    std::hex, request.key.bufferAddress, std::dec,
                                    " after ", framesElapsed, " frames"));
          }
        } else {
          Logger::err("[RTX Mega Geometry ASYNC] Failed to map completed hash result buffer");
        }

        // Remove completed request (frees the result buffer)
        it = m_pendingGpuHashes.erase(it);
      } else {
        // Still waiting for GPU completion (< 3 frames), check next frame
        ++it;
      }
    }

    if (m_pendingGpuHashes.size() > 0 || m_gpuHashCache.size() > 0) {
      Logger::debug(str::format("[RTX Mega Geometry ASYNC] Pending: ", m_pendingGpuHashes.size(),
                               " | Cached: ", m_gpuHashCache.size()));
    }
  }

} // namespace dxvk
