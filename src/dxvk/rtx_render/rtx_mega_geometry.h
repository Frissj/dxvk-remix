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
#pragma once

#include "rtx_resources.h"
#include "rtx_option.h"
#include "dxvk_buffer.h"
#include "dxvk_image.h"
#include "rtx_mg_cluster.h"
#include "rtxmg/rtxmg_cluster_builder.h"
#include "../vulkan/vulkan_loader.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;
  class RtxMegaGeometryAutoTune;
  class RtxmgClusterBuilder;
  struct MegaGeometryStatistics;
  struct ClusterAccels;

  /**
   * \brief RTX Mega Geometry Manager
   *
   * Implements NVIDIA RTX Mega Geometry (RTXMG) cluster-based tessellation
   * and structured acceleration structures for subdivision surfaces.
   *
   * This system provides:
   * - Cluster-based subdivision surface tessellation
   * - Structured Cluster Acceleration Structures (CLAS) via NVAPI
   * - Adaptive tessellation with visibility culling
   * - Hierarchical Z-buffer (HiZ) optimization
   * - Debug visualization modes
   *
   * Always-on by design with no fallback path.
   */
  class RtxMegaGeometry : public RtxPass {

  public:
    // Forward declare cache structure (defined later)
    struct TessellatedGeometryCache;

    RtxMegaGeometry(DxvkDevice* device);
    ~RtxMegaGeometry();

    /**
     * \brief Initialize mega geometry system
     *
     * Sets up cluster templates, shaders, and buffers.
     */
    void initialize();

    /**
     * \brief Submit geometry for cluster tessellation
     *
     * \param [in] ctx The DXVK context
     * \param [in] positions Vertex positions
     * \param [in] normals Vertex normals
     * \param [in] texCoords Texture coordinates
     * \param [in] vertexCount Number of vertices
     * \param [in] indices Triangle indices
     * \param [in] indexCount Number of indices
     * \param [in] transform Object-to-world transform
     * \param [in] materialId Material ID
     */
    void submitGeometry(
      Rc<DxvkContext> ctx,
      const Vector3* positions,
      const Vector3* normals,
      const Vector2* texCoords,
      uint32_t vertexCount,
      const uint32_t* indices,
      uint32_t indexCount,
      const Matrix4& transform,
      uint32_t materialId);

    /**
     * \brief Submit geometry using GPU buffers (zero-copy)
     *
     * Submits geometry for BLAS building using GPU buffer references.
     * No CPU readback required - keeps data on GPU for pooling and culling.
     *
     * \param [in] ctx The DXVK context
     * \param [in] positionBuffer GPU buffer containing vertex positions
     * \param [in] normalBuffer GPU buffer containing vertex normals (optional)
     * \param [in] texcoordBuffer GPU buffer containing texture coordinates (optional)
     * \param [in] indexBuffer GPU buffer containing indices (optional)
     * \param [in] vertexCount Number of vertices
     * \param [in] indexCount Number of indices
     * \param [in] transform Object-to-world transform
     * \param [in] materialId Material ID
     * \param [in] positionStride Stride of position buffer
     * \param [in] normalStride Stride of normal buffer
     * \param [in] texcoordStride Stride of texcoord buffer
     * \param [in] indexType Type of indices (16-bit or 32-bit)
     */
    void submitGeometryGpu(
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
      XXH64_hash_t geometryHash = 0);

    /**
     * \brief Build cluster acceleration structures for current frame
     *
     * Processes all submitted geometry and builds BLAS.
     *
     * \param [in] ctx The DXVK context
     */
    void buildClusterAccelerationStructuresForFrame(Rc<DxvkContext> ctx);

    /**
     * \brief Update HiZ buffer for visibility culling
     *
     * \param [in] ctx The DXVK context
     * \param [in] depthBuffer Input depth buffer
     */
    void updateHiZ(
      Rc<DxvkContext> ctx,
      const Rc<DxvkImageView>& depthBuffer);

    /**
     * \brief Dispatch compute shaders for cluster tessellation
     *
     * \param [in] ctx The DXVKK context
     */
    void dispatchTessellation(Rc<DxvkContext> ctx);

    /**
     * \brief Render debug visualization
     *
     * \param [in] ctx The DXVK context
     * \param [in] outputImage Target image for debug output
     * \param [in] debugViewIndex The RTX debug view index (900-907)
     */
    void renderDebugView(
      Rc<DxvkContext> ctx,
      const Rc<DxvkImageView>& outputImage,
      uint32_t debugViewIndex);

    /**
     * \brief Read back statistics from GPU
     *
     * \param [in] ctx The DXVK context
     */
    void readbackStatistics(Rc<DxvkContext> ctx);

    /**
     * \brief Get current statistics
     *
     * \param [out] outStats Statistics structure to fill
     */
    void getStatistics(MegaGeometryStatistics& outStats) const;

    /**
     * \brief Check if mega geometry is initialized
     */
    bool isInitialized() const {
      return m_initialized;
    }

    /**
     * \brief Get cluster builder for direct access
     */
    RtxmgClusterBuilder* getClusterBuilder() const {
      return m_clusterBuilder.get();
    }

    /**
     * \brief Add geometry to batch for end-of-frame tessellation
     *
     * Collects geometry without immediate tessellation. All collected geometry
     * will be tessellated in ONE dispatch at end of frame.
     */
    void addGeometryForBatchTessellation(
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
      XXH64_hash_t geometryHash = 0);

    /**
     * \brief Tessellate all collected geometry in ONE dispatch
     *
     * Called once per frame in updateMegaGeometryPerFrame.
     * Dispatches single compute shader for entire frame's geometry.
     */
    void tessellateCollectedGeometry(Rc<DxvkContext> ctx);

    /**
     * \brief Compute hash for geometry (public for integration)
     */
    XXH64_hash_t computeGeometryHash(
      const DxvkBufferSlice& positionBuffer,
      const DxvkBufferSlice& indexBuffer,
      uint32_t vertexCount,
      uint32_t indexCount,
      uint32_t positionStride,
      uint32_t normalStride,
      uint32_t texcoordStride) const;

    /**
     * \brief Compute hash for GPU-only geometry using compute shader
     * This dispatches a GPU compute shader to hash vertex data directly on GPU
     * Returns the hash without needing to read back vertex data to CPU
     */
    XXH64_hash_t computeGeometryHashOnGPU(
      Rc<RtxContext> ctx,
      const DxvkBufferSlice& positionBuffer,
      const DxvkBufferSlice& indexBuffer,
      uint32_t vertexCount,
      uint32_t indexCount,
      uint32_t positionStrideInFloats,
      VkIndexType indexType);

    /**
     * \brief Get tessellated geometry from cache (returns nullptr if not cached)
     */
    const TessellatedGeometryCache* getTessellatedGeometry(XXH64_hash_t hash) const;

    /**
     * \brief Check if pass is enabled (required by RtxPass)
     */
    virtual bool isEnabled() const override {
      return enable();
    }

    /**
     * \brief Get BLAS addresses buffer (populated by cluster extension on GPU)
     *
     * Used for GPU-side patching of TLAS instance descriptors.
     * Matches NVIDIA sample blasPtrsBuffer.
     */
    Rc<DxvkBuffer> getBlasAddressesBuffer() const;

    /**
     * \brief Get blasPtrsBuffer index for a geometry hash
     *
     * Returns -1 if not found. Used for GPU-side TLAS patching.
     */
    int32_t getBlasBufferIndex(XXH64_hash_t geometryHash) const;

    /**
     * \brief Get shader for patching TLAS instance descriptors on GPU
     *
     * Matches NVIDIA sample FillInstanceDescs shader.
     */
    Rc<DxvkShader> getPatchTlasInstanceShader() const;

  private:
    // Forward declarations for private types
    struct SubmittedMesh;
    struct GpuSubmittedMesh;

    /**
     * \brief Initialize structured cluster templates
     *
     * Creates 11x11 grid patterns for cluster tessellation (CPU-side).
     */
    void initializeClusterTemplates();

    /**
     * \brief Build template CLAS structures (GPU-side, lazy initialization)
     *
     * Called on first tessellation to build the 121 template CLAS structures.
     * Requires RtxContext for GPU commands.
     */
    void buildTemplateClasIfNeeded(Rc<DxvkContext> ctx);

    /**
     * \brief Build BLAS from cluster geometry
     */
    void buildBLASFromClusters(Rc<DxvkContext> ctx);

    /**
     * \brief Build BLAS for a single mesh
     */
    void buildBLASForMesh(Rc<DxvkContext> ctx, const SubmittedMesh& mesh);

    /**
     * \brief Build BLAS for a GPU-submitted mesh (zero-copy)
     */
    void buildBLASForGpuMesh(Rc<DxvkContext> ctx, const GpuSubmittedMesh& mesh);

    /**
     * \brief Compute cluster tiling layout
     */
    void computeClusterTiling(Rc<DxvkContext> ctx);

    /**
     * \brief Fill cluster vertex data
     */
    void fillClusterData(Rc<DxvkContext> ctx);

    /**
     * \brief Resize buffers based on auto-tune recommendations
     */
    void resizeBuffersIfNeeded(Rc<DxvkContext> ctx);

  public:
    // BLAS cache entry (public for RTX Remix integration)
    struct CachedBLAS {
      Rc<DxvkAccelStructure> blasBuffer;            // Acceleration structure buffer (changed from DxvkBuffer)
      VkAccelerationStructureKHR accelStructure;    // Acceleration structure handle
      uint32_t vertexCount;
      uint32_t triangleCount;
      uint32_t lastUsedFrame;                       // For LRU eviction
      uint64_t blasSize;                            // For memory tracking
      bool isCompacted;                             // Whether BLAS was compacted
    };

    /**
     * \brief Lookup BLAS in cache by geometry hash (public for RTX Remix integration)
     */
    CachedBLAS* lookupBLAS(XXH64_hash_t geometryHash);

    /**
     * \brief Store BLAS in cache (public for RTX Remix integration)
     */
    void cacheBLAS(XXH64_hash_t geometryHash, const CachedBLAS& blas);

    /**
     * \brief Evict old BLAS entries to free memory
     */
    void evictOldBLASEntries();

    /**
     * \brief Cleanup destroyed BLAS handles
     */
    void cleanupBLASCache();

    // Tessellation cache entry (public for integration access)
    struct TessellatedGeometryCache {
      Rc<DxvkBuffer> vertexBuffer;
      Rc<DxvkBuffer> indexBuffer;
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t lastUsedFrame;
      XXH64_hash_t geometryHash;
      uint64_t memorySizeBytes;  // Track actual memory usage for budget enforcement

      // NV-DXVK: Cluster acceleration structures (if built)
      bool hasClusterBLAS = false;
      Rc<DxvkAccelStructure> clusterBLAS;  // Production: use DxvkAccelStructure directly
      Rc<DxvkBuffer> clusterInstanceBuffer;  // PERSISTENT cluster instance data (not in ring buffer!)
      Rc<DxvkBuffer> clusterReferencesBuffer;  // PERSISTENT cluster references (addresses pointing to instance data)
      uint32_t clusterCount = 0;
      uint64_t blasSizeBytes = 0;  // BLAS memory usage
      VkDeviceAddress clusterBlasAddress = 0;
      uint32_t clusterBlasSize = 0;

      // GPU synchronization: Track command list that built this BLAS
      // CRITICAL: Must wait for this fence before reusing BLAS addresses
      Rc<DxvkCommandList> buildCommandList;  // Command list used to build this cluster BLAS
      bool gpuWorkComplete = false;          // Cached fence status (avoid repeated checks)
    };

    // Debug visualization modes
    enum class MegaGeometryDebugMode : uint32_t {
      None = 0,
      ClusterVisualization,
      TessellationDensity,
      SurfaceUV,
      SurfaceIndex,
      ClusterID,
      VertexNormals,
      WireframeOverlay,
      HiZVisualization,
      Count
    };

    // RtxOptions integration
    RTX_OPTION("rtx.megaGeometry", bool, enable, true, "Enable RTX Mega Geometry BLAS caching and cluster templates.");
    RTX_OPTION("rtx.megaGeometry", bool, enableTessellation, true, "Enable mesh tessellation (if false, only BLAS pooling and culling are used).");
    RTX_OPTION("rtx.megaGeometry", MegaGeometryDebugMode, debugMode, MegaGeometryDebugMode::None, "Debug visualization mode for mega geometry.");
    RTX_OPTION("rtx.megaGeometry", bool, showStatistics, false, "Show mega geometry statistics overlay.");
    RTX_OPTION("rtx.megaGeometry", float, tessellationDensity, 1.0f, "Global tessellation density multiplier (0.1 to 10.0).");
    RTX_OPTION("rtx.megaGeometry", uint32_t, maxClusters, 2097152, "Maximum number of clusters (default: 2M).");
    RTX_OPTION("rtx.megaGeometry", bool, enableHiZCulling, true, "Enable hierarchical Z-buffer visibility culling.");
    RTX_OPTION("rtx.megaGeometry", bool, enableFrustumCulling, true, "Enable view frustum culling.");
    RTX_OPTION("rtx.megaGeometry", bool, enableBackfaceCulling, true, "Enable backface culling for clusters.");
    RTX_OPTION("rtx.megaGeometry", uint32_t, edgeSegments, 11, "Target edge segments per patch (1-11, higher = finer tessellation).");
    RTX_OPTION("rtx.megaGeometry", float, pixelError, 1.0f, "Target pixel error for adaptive tessellation.");
  private:
    DxvkDevice* m_device = nullptr;

    // Auto-tuning system (handles memory, BLAS pooling, and parameter adaptation)
    class RtxMegaGeometryAutoTune* m_autoTune = nullptr;

    // RTXMG cluster builder (Phase 1-5 implementation)
    std::unique_ptr<RtxmgClusterBuilder> m_clusterBuilder;

    // Cluster templates (121 different grid patterns)
    struct ClusterTemplate {
      uint32_t gridSizeX;
      uint32_t gridSizeY;
      Rc<DxvkBuffer> vertexBuffer;
      Rc<DxvkBuffer> indexBuffer;
    };
    std::array<ClusterTemplate, 121> m_clusterTemplates;

    // Phase 1: Template CLAS infrastructure
    TemplateGrids m_templateGrids;                    // Generated template grid geometry (CPU-side)
    RtxmgBuffer<uint8_t> m_templateClasBuffer;        // Pre-built template CLAS structures (one-time, 121 templates)
    std::vector<VkDeviceAddress> m_templateAddresses; // Device addresses of each template CLAS
    std::vector<uint32_t> m_templateInstantiationSizes; // Size needed to instantiate each template
    bool m_templatesInitialized = false;              // Flag to prevent re-initialization
    bool m_templateClasBuilt = false;                 // Flag for lazy CLAS building

    // GPU buffers
    Rc<DxvkBuffer> m_clusterDataBuffer;        // Cluster vertex/normal data
    Rc<DxvkBuffer> m_clusterInfoBuffer;        // Cluster metadata
    Rc<DxvkBuffer> m_clusterTilingBuffer;      // Tiling layout per instance
    Rc<DxvkBuffer> m_clusterStatisticsBuffer;  // GPU statistics
    Rc<DxvkBuffer> m_hizBuffer;                // Hierarchical Z-buffer

    // Fence-tracked buffer release system for safe shrinking
    struct BuffersWithFence {
      Rc<DxvkBuffer> clusterDataBuffer;
      Rc<DxvkBuffer> clusterInfoBuffer;
      VkFence lastUsageFence;  // Fence from command buffer that last used these buffers
    };
    std::vector<BuffersWithFence> m_pendingClusterBufferReleases;

    std::unordered_map<XXH64_hash_t, CachedBLAS> m_blasCache;

    // Frame geometry submission
    struct SubmittedMesh {
      std::vector<ClusterVertex> vertices;
      std::vector<uint32_t> indices;
      Matrix4 transform;
      uint32_t materialId;
      XXH64_hash_t geometryHash;  // Hash for cache lookup (includes positions for deform detection)
    };
    std::vector<SubmittedMesh> m_pendingMeshes;

    // GPU geometry submission (zero-copy for BLAS pooling without tessellation)
    struct GpuSubmittedMesh {
      DxvkBufferSlice positionBuffer;
      DxvkBufferSlice normalBuffer;
      DxvkBufferSlice texcoordBuffer;
      DxvkBufferSlice indexBuffer;
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t positionStride;
      uint32_t normalStride;
      uint32_t texcoordStride;
      VkIndexType indexType;
      Matrix4 transform;
      uint32_t materialId;
      XXH64_hash_t geometryHash;  // For BLAS pooling/reuse
    };
    std::vector<GpuSubmittedMesh> m_pendingGpuMeshes;

    // Batched geometry collection (for end-of-frame tessellation)
    struct BatchedGeometry {
      DxvkBufferSlice positionBuffer;
      DxvkBufferSlice normalBuffer;
      DxvkBufferSlice texcoordBuffer;
      DxvkBufferSlice indexBuffer;
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t positionStride;
      uint32_t normalStride;
      uint32_t texcoordStride;
      uint32_t positionOffset;
      uint32_t normalOffset;
      uint32_t texcoordOffset;
      VkIndexType indexType;
      Matrix4 transform;
      XXH64_hash_t materialId;
      XXH64_hash_t geometryHash;  // Hash for cache lookup (includes positions)
    };
    std::vector<BatchedGeometry> m_batchedGeometry;  // Collected this frame

    // Tessellation cache (stores tessellated results for reuse)
    std::unordered_map<XXH64_hash_t, TessellatedGeometryCache> m_tessellationCache;
    uint32_t m_currentFrame = 0;
    uint64_t m_tessellationCacheMemoryBytes = 0;  // Track total cache memory usage

    // Async GPU hash infrastructure (for GPU-only buffers)
    struct BufferKey {
      VkDeviceAddress bufferAddress;
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t stride;

      bool operator==(const BufferKey& other) const {
        return bufferAddress == other.bufferAddress &&
               vertexCount == other.vertexCount &&
               indexCount == other.indexCount &&
               stride == other.stride;
      }
    };

    struct BufferKeyHash {
      size_t operator()(const BufferKey& key) const {
        size_t h1 = std::hash<VkDeviceAddress>()(key.bufferAddress);
        size_t h2 = std::hash<uint32_t>()(key.vertexCount);
        size_t h3 = std::hash<uint32_t>()(key.indexCount);
        size_t h4 = std::hash<uint32_t>()(key.stride);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
      }
    };

    struct AsyncHashRequest {
      Rc<DxvkBuffer> resultBuffer;      // GPU buffer containing hash result
      Rc<DxvkCommandList> cmdList;      // Command list for fence checking
      uint32_t submitFrame;              // Frame when compute was submitted
      BufferKey key;                     // Buffer being hashed
    };

    std::unordered_map<XXH64_hash_t, AsyncHashRequest> m_pendingGpuHashes;  // Request ID -> Request
    std::unordered_map<BufferKey, XXH64_hash_t, BufferKeyHash> m_gpuHashCache;  // Buffer -> Cached Hash
    uint64_t m_nextHashRequestId = 1;  // Monotonic counter for request IDs

   public:
    /**
     * \brief Process completed async hash requests (call once per frame)
     * Checks for completed GPU hashes and updates cache
     */
    void processCompletedHashes(Rc<DxvkContext> ctx);

    /**
     * \brief Generate temporary metadata hash for buffer
     * Used while waiting for GPU hash to complete
     */
    static XXH64_hash_t generateTempHash(const BufferKey& key);

   private:
    // TRUE BATCHING: Per-frame unified BLAS (like sample code)
    // Collect all cluster instances for the frame, build ONE BLAS at end
    // SAMPLE CODE MATCH: Per-draw-call data (not per-geometry-hash)
    // Sample processes each instance separately with its own transform
    struct FrameDrawCallData {
      XXH64_hash_t geometryHash;     // Geometry hash (for cache lookup)
      Matrix4 transform;              // Instance-specific transform
      uint32_t drawCallIndex;         // Draw call index in frame
      uint32_t clusterCount;          // Number of clusters for this instance
      ClusterOutputGeometryGpu output;  // GPU tessellation output

      // CRITICAL FIX: Store input geometry references for fill_clusters shader
      // fill_clusters needs to read ALL instances' input geometry from shared buffers
      // We must reconstruct these buffers in buildStructuredCLASes before running fill_clusters
    DxvkBufferSlice inputPositions;
    DxvkBufferSlice inputNormals;
    DxvkBufferSlice inputTexcoords;
    DxvkBufferSlice inputIndices;
    uint32_t inputVertexCount = 0;
    uint32_t inputIndexCount = 0;
    uint32_t positionOffset = 0;
    uint32_t normalOffset = 0;
    uint32_t texcoordOffset = 0;
    VkDeviceAddress clusterBlasAddress = 0;
    uint32_t clusterBlasSize = 0;
  };
    std::vector<FrameDrawCallData> m_frameDrawCalls;  // Per-draw-call data (matches sample's per-instance)

    /**
     * \\brief Compute hash for geometry to use as cache key
     */
    XXH64_hash_t computeGeometryHash(const BatchedGeometry& geom) const;

    /**
     * \\brief Lookup tessellated geometry in cache
     */
    TessellatedGeometryCache* lookupTessellatedGeometry(XXH64_hash_t hash);

    /**
     * \\brief Store tessellated geometry in cache
     */
    void cacheTessellatedGeometry(
      Rc<DxvkContext> ctx,
      XXH64_hash_t hash,
      const ClusterOutputGeometryGpu& output,
      const ClusterAccels* clusterAccels = nullptr,
      VkDeviceAddress clusterBlasAddress = 0,
      uint32_t clusterBlasSize = 0);

    /**
     * \\brief Evict old cache entries
     */
    void evictOldCacheEntries();

    // Statistics (CPU-side cache)
    struct Statistics {
      uint32_t totalClusters = 0;
      uint32_t visibleClusters = 0;
      uint32_t culledClusters = 0;
      uint32_t totalVertices = 0;
      uint32_t totalTriangles = 0;
      float memoryUsedMB = 0.0f;
      uint32_t cacheHits = 0;
      uint32_t cacheMisses = 0;
      uint32_t cachedEntries = 0;

      // BLAS pooling statistics
      uint32_t blasCacheHits = 0;
      uint32_t blasCacheMisses = 0;
      uint32_t blasCachedEntries = 0;
      float blasMemoryUsedMB = 0.0f;
    };
    Statistics m_stats;

    bool m_initialized = false;
  };

} // namespace dxvk
