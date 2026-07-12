/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// C++17-clean interface between dxvk (C++17, legacy glm at include/glm) and the
// lodclusters static library (C++20, modern glm). This header must NOT include
// any lodclusters / nvpro / glm headers and must NOT expose glm or std::span
// types: only plain std value types (and raw Vulkan handles/structs, which are
// ABI-identical on both sides) cross the boundary.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lodclusters_remix {

  // CPU snapshot of one unique geometry, taken while Remix's draw-submission
  // window guarantees the source data is alive. Owns all of its data.
  struct GeometrySnapshot {
    std::string name;

    // stable identity (Remix geometry asset hash) - keys the .nvsngeo cache file
    uint64_t geometryHash = 0;

    // content hashes stored in the cache entry for validation
    uint64_t indicesHash = 0;
    uint64_t verticesHash = 0;

    // triangle list, 0-based uint32 indices
    std::vector<uint32_t> indices;

    uint32_t vertexCount = 0;

    // tightly packed streams; empty vector means attribute not present
    std::vector<float> positions;   // 3 floats per vertex
    std::vector<float> normals;     // 3 floats per vertex
    std::vector<float> texcoords0;  // 2 floats per vertex
    std::vector<float> tangents;    // 4 floats per vertex

    // material state baked into cluster state bits
    bool twoSided = false;
    bool alphaMasked = false;
    float alphaCutOff = 0.5f;

    // true if this is the bind-pose snapshot of a deforming (skinned) mesh:
    // routed to Path B (cluster templates, P4b) instead of the LOD pipeline
    bool isDeforming = false;

    // P4b: true if this snapshot was taken because the geometry's vertex data
    // mutates in place per frame (CPU rewrites, no skinning). Routed to Path B
    // like skinned geometry: the topology is clusterized once, per-frame CLAS
    // consume the live GPU buffer.
    bool isMutating = false;

    // P4c (plan 7.7): true for vertex-captured, non-skinned, NON-mutating
    // snapshots - the rigid-capture promotion candidates. They register Path B
    // templates AND run the LOD pipeline at first sight so the .nvsngeo cache
    // and generation residency are ready when the promotion verdict lands.
    // Mutating meshes are excluded: their frozen snapshot goes stale by
    // definition (they can never promote), and their asset-rule hash is
    // STABLE across mutation states while the snapshot content hashes change,
    // so Path A processing produced a fresh cache-mismatch + overwrite of the
    // same .nvsngeo every session (observed 2026-07-04 17:30:48, x3).
    bool isCaptured = false;

    // P4b: topology-stable identity for Path B (combines the indices hash and
    // the vertex count; unlike geometryHash it does NOT include positions, so
    // it stays stable while a mesh deforms). Keys the template sets.
    uint64_t topologyKey = 0;

    // chrono: steady-clock microseconds at enqueue time - the worker reports
    // how long the snapshot waited in the processing queue (a growing wait
    // means discovery outpaces the single processing worker)
    uint64_t queuedAtUs = 0;

    size_t approximateSizeBytes() const {
      return indices.size() * sizeof(uint32_t)
        + (positions.size() + normals.size() + texcoords0.size() + tangents.size()) * sizeof(float);
    }
  };

  // Plain mirror of lodclusters::SceneConfig + SceneLoaderConfig (the fields Remix
  // exposes as RTX_OPTIONs). Mapped 1:1 onto NVIDIA's structs inside the library.
  struct ProcessorConfig {
    // SceneConfig
    uint32_t clusterVertices = 128;
    uint32_t clusterTriangles = 128;
    uint32_t clusterGroupSize = 32;
    uint32_t preferredNodeWidth = 8;
    bool meshoptPreferRayTracing = true;
    bool useCompressedData = false;
    uint32_t enabledAttributes = 0;  // shaderio::CLUSTER_ATTRIBUTE_* bits
    float meshoptFillWeight = 0.5f;
    float meshoptSplitFactor = 2.0f;
    float lodLevelDecimationFactor = 0.5f;
    float lodErrorMergePrevious = 1.5f;
    float lodErrorMergeAdditive = 0.0f;
    float simplifyNormalWeight = 0.5f;
    float simplifyTangentWeight = 0.01f;
    float simplifyTangentSignWeight = 0.5f;
    float simplifyTexCoordWeight = 0.0f;
    float simplifyMaterialWeight = 0.5f;
    uint32_t compressionPosDropBits = 7;
    uint32_t compressionTexDropBits = 7;

    // SceneLoaderConfig
    float processingThreadsPct = 0.5f;
    // P4c item 7: provider worker threads draining the intake queue in
    // parallel (0 = auto: hardware threads / 4, clamped to [1, 4])
    uint32_t processingWorkerCount = 0;
    bool autoSaveCache = true;
    bool autoLoadCache = true;
    bool memoryMappedCache = false;
    uint64_t forcePreprocessMiB = 2048;

    // P4c routing (plan 7.7): whether captured (isCaptured) snapshots also run
    // the LOD pipeline at first sight - promotion's process-at-first-sight
    // policy, carried here so the worker reads one config source
    bool processCapturedGeometry = true;

    // provider cache policy: one <geometryHash>.nvsngeo per geometry, grouped in a
    // per-config-digest subdirectory so config changes never load stale clusters
    std::string cacheDirectoryUtf8;
  };

  struct ProcessStats {
    bool success = false;
    bool loadedFromCache = false;
    bool memoryMapped = false;

    uint32_t lodLevelsCount = 0;
    uint64_t totalClusters = 0;
    uint64_t totalTriangles = 0;
    uint64_t totalVertices = 0;
    uint64_t hiClusters = 0;
    uint64_t hiTriangles = 0;

    uint32_t clusterTrianglesMax = 0;
    uint32_t clusterVerticesMax = 0;
    uint32_t groupClustersMax = 0;
    uint32_t lodLevelsMax = 0;

    uint64_t cacheFileSizeBytes = 0;
    double processingMs = 0.0;
  };

  // level: 0 = info, 1 = warning, 2 = error
  using LogSink = void (*)(int level, const char* message);

  // Routes the nvpro logger (all of NVIDIA's processing stats/progress output) into
  // the given sink and disables its debug-break-on-error behavior. Call once.
  void installLogSink(LogSink sink);

  // Sizes the shared nvutils processing thread pool from the pct - idempotent,
  // thread-safe, reapplies only when the resolved thread count changes (so a
  // runtime option change costs one reset, not one per geometry). Callers then
  // pass processingThreadsPct = 0 ("use the pool as-is") into per-geometry
  // processing configs: NVIDIA's ProcessingInfo::init/deinit otherwise reset
  // the ENTIRE pool down and back up around every geometry - two full thread
  // teardown/spawn cycles, the measured ~20 ms fixed floor per registration
  // (2026-07-04).
  void configureProcessingThreadPool(float threadsPct);

  // UTF-8 path of the .nvsngeo cache file a processed geometry hash resolves to
  // under the given config (same layout GeometryProcessor writes).
  std::string getGeometryCacheFileUtf8(uint64_t geometryHash, const ProcessorConfig& config);

  // Whether that .nvsngeo already exists on disk. Used by the P4c interim-
  // template skip: on a cache hit Path A lands within the cache-hit cooldown,
  // so interim templates would be pure waste.
  bool geometryCacheFileExists(uint64_t geometryHash, const ProcessorConfig& config);

  // P2.5: the SceneConfig digest (cache subdirectory name) the given config
  // resolves to. A render generation may only be appended to with cache files
  // from the digest it was built with; on a digest change the manager performs
  // a full rebuild instead (config-change invalidation).
  std::string getConfigCacheDigestUtf8(const ProcessorConfig& config);

  // Runs NVIDIA's full cluster-LOD processing for single geometries.
  // CPU-only; safe to use from one background thread at a time (the nvpro
  // thread pool underneath is a process-global singleton).
  class GeometryProcessor {
  public:
    GeometryProcessor();
    ~GeometryProcessor();

    GeometryProcessor(const GeometryProcessor&) = delete;
    GeometryProcessor& operator=(const GeometryProcessor&) = delete;

    // Process one geometry (or load it from its .nvsngeo cache if present and valid).
    // On success the cache file exists on disk afterwards (autoSaveCache).
    bool processGeometry(const GeometrySnapshot& snapshot, const ProcessorConfig& config, ProcessStats& outStats);

    // P1 verification: re-load the geometry's cache entry twice (system RAM copy and
    // memory-mapped) and compare both against the given reference stats.
    bool verifyCacheRoundTrip(const GeometrySnapshot& snapshot,
                              const ProcessorConfig& config,
                              const ProcessStats& referenceStats,
                              std::string& outMessage);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
  };

  //////////////////////////////////////////////////////////////////////////
  // P2: preloaded cluster rendering (RenderScene + RendererRayTraceClustersLod)

  // Device hookup for the cluster GPU systems. The library runs its own volk
  // loader and VMA allocator on these handles (exactly like the sample).
  struct RenderDeviceInfo {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t graphicsQueueFamilyIndex = 0;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t transferQueueFamilyIndex = 0;
    VkQueue transferQueue = VK_NULL_HANDLE;
  };

  // Plain mirror of the RendererConfig subset P2 exposes plus the P3
  // streaming configuration (defaults = sample defaults).
  struct RenderConfig {
    bool useSorting = false;
    // P4: frustum + HiZ occlusion culling (the HiZ feed comes from Remix's
    // previous-frame primary depth, see FrameParams depth fields)
    bool useCulling = true;
    bool useBlasSharing = true;
    // requires streaming (the renderer force-disables it without)
    bool useBlasMerging = true;
    // P4: BLAS caching for shared geometries (requires streaming + sharing);
    // sample default off
    bool useBlasCaching = false;
    // device budget for cached BLAS memory (MiB, StreamingConfig mirror)
    uint64_t maxBlasCachingMegaBytes = 1024;
    bool usePersistentTraversal = true;
    bool useRenderStats = false;
    // P4: culled instances are REMOVED from the TLAS (mask 0) instead of kept
    // at scaled LOD error - breaks shadows/reflections of off-screen geometry,
    // perf/debug mode only (sample default off)
    bool useForcedInvisibleCulling = false;

    // ---- P3: streaming (lodclusters::StreamingConfig mirror) ----
    // renders through SceneStreaming (on-demand cluster group residency)
    // instead of ScenePreloaded (everything resident)
    bool preferStreaming = false;
    // upload new groups on the dedicated transfer queue instead of the
    // graphics queue command buffer. The caller must externally synchronize
    // the transfer queue while recordFrame runs (dxvk submission lock).
    bool useAsyncTransfer = false;
    // async transfers may span multiple frames (update task deferred until
    // the transfer's semaphore completes) instead of same-frame waits
    bool useDecoupledAsyncTransfer = false;
    // GPU-driven persistent CLAS allocator vs simple compaction
    bool usePersistentClasAllocator = true;
    uint32_t maxPerFrameLoadRequests = 128;
    uint32_t maxPerFrameUnloadRequests = 1024;
    // resident cluster-group table size (0 maxClusters = derived)
    uint32_t streamingMaxGroups = 1u << 16;
    uint32_t streamingMaxClusters = 0;
    uint64_t maxTransferMegaBytes = 32;
    uint64_t maxGeometryMegaBytes = 2048;
    uint64_t maxClasMegaBytes = 2048;
    uint32_t clasAllocatorSectorSizeShift = 10;
    uint32_t clasAllocatorGranularityShift = 0;

    uint32_t numRenderClusterBits = 20;
    uint32_t numTraversalTaskBits = 20;

    // VkBuildAccelerationStructureFlagsKHR for the cluster BLAS builds
    uint32_t clusterBlasFlags = 0;
    // CLAS build flags + position truncation (ScenePreloaded::Config)
    uint32_t clasBuildFlags = 0x00000004;  // VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
    uint32_t clasPositionTruncateBits = 0;

    // instance capacity the renderer's buffers are sized for (grows via
    // generation rebuild when exceeded)
    uint32_t maxRenderInstances = 4096;

    // P2.5: geometry-slot capacity of a generation (geometry table +
    // BLAS-sharing buffers). Appends beyond this trigger a full rebuild with
    // a grown capacity; buildGeneration raises it to the next power of two
    // above the initial count when that is larger.
    uint32_t maxGeometries = 4096;
  };

  // Per-geometry constants Remix needs to build render instances and to
  // pre-fill TlasInstance::blasReference. Indexed by geometryID (== index into
  // the cache file list a generation was built from).
  struct GeometryRenderInfo {
    uint64_t geometryHash = 0;
    uint32_t lodLevelsCount = 0;
    uint32_t lowDetailClusterStateBits = 0;
    uint64_t lowDetailBlasAddress = 0;
    uint64_t totalClusters = 0;
  };

  // One render instance for the current frame, in TLAS-slot order.
  // Matches shaderio::RenderInstance content; converted inside the library.
  struct InstanceInput {
    // object-to-world, column-major 4x4 (last row assumed 0,0,0,1)
    float worldMatrix[16] = {};

    uint32_t geometryID = 0;
    bool twoSided = false;
    // 1 = opaque, 2 = alpha-masked, 3 = mixed (SHADERIO_OPAQUE_STATUS_*)
    uint32_t opaqueStatus = 1;
  };

  // Per-frame camera/traversal parameters (from Remix's main RtCamera).
  struct FrameParams {
    // column-major 4x4 matrices
    float viewMatrix[16] = {};
    float projMatrix[16] = {};
    float viewProjMatrix[16] = {};
    float prevViewProjMatrix[16] = {};

    float viewPos[3] = {};
    float fovRadians = 1.0f;
    uint32_t viewportWidth = 1;
    uint32_t viewportHeight = 1;
    float nearPlane = 0.1f;
    float farPlane = 10000.0f;

    float lodPixelError = 1.0f;
    float culledErrorScale = 2.0f;
    uint32_t traversalPersistentThreads = 2048;
    // P4 debug: keep last frame's cull matrices + HiZ content (sample parity:
    // the app-side latch skips the HiZ build and matrix updates while frozen)
    bool freezeCulling = false;
    // P4 debug: keep last frame's LOD traversal view (error metric input)
    bool freezeLoD = false;

    // P3: frames until an unused streamed cluster group is scheduled for unload
    uint32_t streamingAgeThreshold = 16;

    // ---- P4: HiZ occlusion feed ----
    // Remix's previous-frame primary depth (R32F color image at render
    // resolution, GENERAL layout, post-projection z/w in the game's own depth
    // convention). VK_NULL_HANDLE when unavailable (first frames) - occlusion
    // tests then pass everything (cleared far pyramid). The caller keeps the
    // view alive for the frames in flight (dxvk resource tracking).
    VkImageView depthView = VK_NULL_HANDLE;
    uint32_t depthWidth = 0;
    uint32_t depthHeight = 0;

    // ---- P4: per-frame BLAS sharing / caching tuning (FrameConfig mirror,
    //      defaults = sample defaults) ----
    bool sharingPushCulled = true;
    uint32_t sharingTolerantLevels = 7;
    uint32_t sharingEnabledLevels = 8;
    uint32_t cachingAgeThreshold = 16;
    uint32_t cachingEnabledLevels = 8;

    // ---- P4c rigid-capture promotion (plan 7.7 spec) ----
    // Per-frame promotion work items. The render system stages them, runs the
    // promotion_solve kernel between the instance upload and the first
    // consuming kernel (solve M from probe vs capture, update state, patch
    // promoted slots' RenderInstance.worldMatrix/TlasInstance transform), and
    // copies the compact per-slot states into the host readback ring.
    const struct PromotionEntry* promotionEntries = nullptr;
    uint32_t promotionEntryCount = 0;
    float promotionResidualEpsilon = 0.005f;
    // false (default) = every promoted slot full-solves each frame; the per-frame
    // re-solve skip (reuse last frame's cached M) freezes placement + zeroes the
    // motion vector under camera motion for camera-relative captured geometry.
    bool promotionAllowResolveSkip = false;
    // Max solve-frame gap for which a promoted slot's retained M is still trusted as its
    // previous transform. A slot re-rendering Path A after a longer gap (off-screen / Path A<->B
    // flip) has an N-frames-stale retained M; beyond this many frames prevM falls back to curM
    // (zero motion) instead of the stale-M motion-vector spike (the moving-cinematic smear).
    uint32_t promotionGapMaxFrames = 2;
  };

  // P4c: one promotion work item (48 bytes, mirrored by promotion_solve.comp
  // with scalar layout - keep field order/size in exact sync).
  struct PromotionEntry {
    uint64_t probeVa = 0;             // probe blob (template-system owned)
    uint64_t captureVa = 0;           // instance's live position buffer
    uint32_t captureStrideBytes = 0;
    uint32_t stateSlot = 0;           // persistent promo state slot
    uint32_t patchSlot = 0xFFFFFFFFu; // RenderInstance/TlasInstance index; ~0 = probe-only
    uint32_t mode = 0;                // 0 = solve (+patch), 1 = full-mesh gate
    uint32_t vertexCount = 0;         // gate dispatch sizing (mode 1)
    uint32_t pad0 = 0;
    // Previous-frame captured positions (modifiedGeometryData.previousPositionBuffer =
    // historyBuffer[1], same stride as captureVa). 0 = unavailable. Used ONLY to seed a
    // first-frame promoted slot's prevM (a B->A flip whose fresh slot has no history), so
    // the motion vector is continuous instead of the one-frame prevM = curM (zero motion)
    // pop the denoiser reprojects wrong. Warm slots ignore it (they retain last frame's M).
    uint64_t prevCaptureVa = 0;
  };
  static_assert(sizeof(PromotionEntry) == 48, "kernel mirrors this layout");

  // P4c: compact per-slot state the manager reads back (3-frame lag);
  // converted from the kernel's 32-byte PromoStatus.
  struct PromotionStateView {
    float residualRel = 0.0f;         // sparse validation residual, last solve
    float gateResidualRel = 0.0f;     // full-mesh sweep max residual
    uint32_t rigidStreak = 0;
    uint32_t flags = 0;               // bit0 rigid, bit2 demoted (last solve non-rigid)
    uint32_t lastFrame = 0;           // renderer frameIndex of the last solve
    float motionDelta = 0.0f;         // [MotionProbe] |M.t - prevM.t| this frame (world units) - CENTROID only
    uint32_t coldFrame = 0;           // [ColdPromo] last frame this promoted slot solved non-contiguously (prevM=curM)
    float maxVertMotion = 0.0f;       // [SmearProbe] max over solve samples of |(M-prevM)*vertex| (world units) - the
                                      // per-VERTEX manufactured motion. Rotation mismatch between M and prevM flings the
                                      // extremities while motionDelta (centroid) stays ~0 = the MV smear no centroid probe sees.
    float curT[3] = {0.f, 0.f, 0.f};  // [PromoDump] solved curM translation column (world units)
    float prevT[3] = {0.f, 0.f, 0.f}; // [PromoDump] solved prevM translation column (world units)
    float placed[3] = {0.f, 0.f, 0.f};// [PromoGap] world position the object's centroid is drawn at (M*centroid)
    float capture[3] = {0.f, 0.f, 0.f};// [PromoGap] mean of the actual capture samples (reliable "true position")
    float sampleIso = 0.0f;           // [ConditionProbe] solve-sample isotropy [0,1]; ~0 = planar/degenerate =
                                      // rotation underdetermined; ~1 = well-constrained; -1 = skip path (not computed)
  };

  // P3: semaphores the caller must attach to the queue submission that
  // executes the command buffer recordFrame recorded into (streaming mode).
  // The signal tells the streaming task queues when this frame's request
  // readback / uploads / patches completed on the GPU; the waits gate the
  // submission on async transfer-queue uploads it consumes.
  struct FrameSubmitSync {
    struct Entry {
      VkSemaphore semaphore = VK_NULL_HANDLE;  // library-owned timeline semaphore
      uint64_t value = 0;
    };
    Entry signal;                // semaphore == VK_NULL_HANDLE when nothing to signal
    std::vector<Entry> waits;    // empty unless async transfers ran this frame
  };

  // Delayed GPU statistics (from the sample's Readback, a few frames old).
  struct FrameStats {
    uint32_t numRenderClusters = 0;
    uint32_t numTraversalTasks = 0;
    uint32_t numBlasBuilds = 0;
    uint64_t blasActualSizeBytes = 0;

    uint64_t reservedClasBytes = 0;
    uint64_t reservedBlasBytes = 0;
    uint64_t reservedGeometryBytes = 0;
    uint64_t reservedOperationsBytes = 0;

    // ---- P3: streaming statistics (lodclusters::StreamingStats mirror);
    //      valid when `streaming` is true ----
    bool streaming = false;

    uint32_t residentGroups = 0;
    uint32_t residentClusters = 0;
    uint32_t maxGroups = 0;
    uint32_t maxClusters = 0;
    uint32_t persistentGroups = 0;

    uint64_t usedDataBytes = 0;
    uint64_t reservedDataBytes = 0;
    uint64_t maxDataBytes = 0;
    uint64_t persistentDataBytes = 0;

    uint64_t usedClasBytes = 0;
    uint64_t wastedClasBytes = 0;

    // last load batch
    uint64_t transferBytes = 0;
    uint32_t transferCount = 0;
    uint32_t loadCount = 0;
    uint32_t unloadCount = 0;
    uint32_t uncompletedLoadCount = 0;

    // soft saturation counters - should stay 0 in healthy operation. (The
    // hard allocator error counters of the verification plan are checked by
    // NVIDIA's own fatal guard in handleCompletedRequest: any nonzero logs
    // "streaming: fatal error" and terminates, so they can never read nonzero
    // here.)
    uint32_t couldNotAllocateGroup = 0;
    uint32_t couldNotAllocateClas = 0;
    uint32_t couldNotTransfer = 0;
    uint32_t couldNotStore = 0;
  };

  // Owns the GPU side of the cluster LOD system: Resources (volk/VMA), the
  // generation's combined Scene + RenderScene (preloaded) and the cluster
  // renderer. All methods that submit to the graphics queue (init,
  // buildGeneration, deinit) must be called with Remix's queue submission
  // externally synchronized; recordFrame only records into the given command
  // buffer.
  class ClusterRenderSystem {
  public:
    ClusterRenderSystem();
    ~ClusterRenderSystem();

    ClusterRenderSystem(const ClusterRenderSystem&) = delete;
    ClusterRenderSystem& operator=(const ClusterRenderSystem&) = delete;

    // Initializes volk + Resources on Remix's device. Submits (HiZ bootstrap
    // transitions/clears) and waits for idle.
    bool init(const RenderDeviceInfo& deviceInfo, const RenderConfig& config);
    // Destroys everything; waits for device idle.
    void deinit();

    // Builds a new generation from per-geometry .nvsngeo cache files: assembles
    // the combined Scene (memory-mapped), uploads it (ScenePreloaded - or
    // seeds the persistent low-detail set when preferStreaming, with higher
    // detail streamed on demand), builds the CLAS + low-detail BLAS, and
    // (re)initializes the renderer.
    // geometryHashes[i] keys geometryID i. maxRenderInstances may grow the
    // configured capacity. Waits for device idle before replacing the previous
    // generation (P5 refines this into fully overlapped swaps).
    bool buildGeneration(const std::vector<std::string>& cacheFilesUtf8,
                         const std::vector<uint64_t>& geometryHashes,
                         uint32_t maxRenderInstances);

    // P2.5: outcome of an incremental append attempt.
    enum class AppendResult : uint32_t {
      Ok = 0,          // geometries joined the generation; geometryIDs extend the existing table
      NeedsRebuild,    // capacity/maxima exceeded (or no generation) - perform a full buildGeneration
      Failed,          // a cache file was unreadable/invalid - generation unchanged, drop the entries
    };

    // P2.5: appends newly processed geometries to the current generation in
    // O(new): only the new geometries' buffers are created, uploaded and
    // CLAS/low-detail-BLAS built; resident geometry is untouched and no
    // device-wait-idle barrier runs (uploads wait on their own submission
    // fence). Existing geometryIDs are stable; the new geometries get the
    // next IDs in getGeometryRenderInfos() order. Caller must guarantee the
    // cache files share the SceneConfig digest of the generation's initial
    // set (see getConfigCacheDigestUtf8). Same external-synchronization
    // requirement as buildGeneration (Remix queue submission locked).
    AppendResult appendToGeneration(const std::vector<std::string>& cacheFilesUtf8,
                                    const std::vector<uint64_t>& geometryHashes);

    bool hasGeneration() const;

    // ---- P4: HiZ occlusion feed sizing ----
    // True when the HiZ source/pyramid do not match the given render
    // resolution. The caller then invokes updateHizResolution under Remix's
    // submission lock (it waits for device idle while recreating) BEFORE
    // recordFrame; recordFrame skips the HiZ build on any residual mismatch.
    bool hizResolutionDiffers(uint32_t width, uint32_t height) const;
    void updateHizResolution(uint32_t width, uint32_t height);

    // valid after a successful buildGeneration, indexed by geometryID
    const std::vector<GeometryRenderInfo>& getGeometryRenderInfos() const;

    uint32_t getMaxRenderInstances() const;

    // Records the full per-frame cluster build (traversal -> CLAS/BLAS ->
    // instance_assign_blas; streaming mode adds request handling, uploads and
    // scene patching) into cmd. instances and tlasInstances are parallel
    // arrays in TLAS-slot order; tlasInstances carries the CPU-known fields
    // (transform/mask/customIndex/sbt/flags) with blasReference pre-filled to
    // the geometry's low-detail BLAS. After execution the patched TlasInstances
    // are in getTlasInstancesBuffer() at [0, count).
    // P3: in streaming mode outSubmitSync (when given) receives the semaphores
    // the caller MUST attach to cmd's queue submission - without the signal
    // the streaming task queues never see completion and the next frames
    // deadlock. With useAsyncTransfer the caller must externally synchronize
    // the transfer queue for the duration of this call (dxvk submission lock),
    // as completed requests submit upload command buffers onto it directly.
    void recordFrame(VkCommandBuffer cmd,
                     const FrameParams& frame,
                     const InstanceInput* instances,
                     const VkAccelerationStructureInstanceKHR* tlasInstances,
                     uint32_t count,
                     FrameSubmitSync* outSubmitSync = nullptr);

    // buffer holding the patched TlasInstances after recordFrame's commands ran
    VkBuffer getTlasInstancesBuffer() const;

    // device address of the generation's shaderio::Geometry table (BDA), for
    // Remix's hit-side cluster fetch (raytrace_args)
    uint64_t getGeometriesTableAddress() const;

    // device address of the streaming-resident cluster address table
    // (SceneStreaming.resident.clusters) indexed by resident ClusterID; 0 unless
    // streaming is active. Needed because preloadedClusters is null while streaming
    uint64_t getResidentClustersAddress() const;

    // NV-DXVK: [ClasAlias] DIAGNOSTIC. This frame's Path A resident/low-detail CLAS
    // memory ranges (lo/hi pairs). Returns the count written (<= maxCount).
    uint32_t getPathAClasRanges(uint64_t* lo, uint64_t* hi, uint32_t maxCount) const;

    // ---- P4c rigid-capture promotion (plan 7.7 spec) ----

    // fixed number of persistent promotion state slots (system scope -
    // survives generation swaps); slot ids are managed by the caller
    static constexpr uint32_t kPromotionSlotCapacity = 8192;

    // device address of the promotion state array (128 B per slot), for the
    // hit side's prevM motion-vector fetch (raytrace_args)
    uint64_t getPromotionStateAddress() const;

    // Drains the newest complete host readback of the per-slot promotion
    // states (written by recordFrame with a ring of 4, read here with the
    // frame lag baked in). Returns false while nothing has been read back
    // yet. outStates must hold kPromotionSlotCapacity entries.
    bool readPromotionStates(PromotionStateView* outStates);

    // delayed statistics (a few frames old); false while nothing rendered yet
    bool getFrameStats(FrameStats& outStats) const;

    // chrono report of the per-frame GPU/CPU section timers NVIDIA's renderer
    // records around every kernel phase (Traversal Run, Blas Build, streaming,
    // HiZ, ...): one line per section ('\n'-separated), values averaged over
    // the profiler's last frames. False while nothing was timed yet.
    bool getProfilerReportUtf8(std::string& outReport) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
  };

  //////////////////////////////////////////////////////////////////////////
  // P4b: PATH B - deforming geometry through cluster templates
  // (vk_animated_clusters). The mesh TOPOLOGY is clusterized once (bind
  // pose); every frame the CLAS are instantiated on-GPU directly from
  // Remix's live (gpu_skinning output / per-frame-updated) vertex buffers
  // and one cluster BLAS per pose (BlasEntry) is rebuilt from the cluster
  // references. No LOD, no streaming - NVIDIA's own design boundary for
  // deforming content.

  // Plain mirror of the vk_animated_clusters SceneConfig/RendererConfig
  // subset Remix exposes (defaults = the sample's effective defaults).
  struct AnimatedConfig {
    // one-time clusterization of the bind-pose topology (meshoptimizer)
    uint32_t clusterVertices = 64;
    uint32_t clusterTriangles = 64;

    // cluster templates (topology baked once, CLAS instantiated per frame)
    // vs direct per-frame triangle-cluster builds - the sample exposes both
    bool useTemplates = true;
    // implicit template build + move compaction vs explicit build with a
    // COMPUTE_SIZES pre-query (sample effective default: explicit)
    bool useImplicitTemplates = false;

    // instantiationBoundingBoxLimit bloat as a fraction of the geometry bbox
    // diagonal; negative disables the limit
    float templateBboxBloatPercentage = 0.5f;
    // mantissa bits truncated from CLAS vertex positions
    uint32_t positionTruncateBits = 0;

    // VkBuildAccelerationStructureFlagsKHR (sample defaults for animated
    // content: templates fast-trace, everything per-frame fast-build)
    uint32_t templateBuildFlags = 0x00000004;       // PREFER_FAST_TRACE
    uint32_t templateInstantiateFlags = 0x00000008; // PREFER_FAST_BUILD
    uint32_t clusterBuildFlags = 0x00000008;        // PREFER_FAST_BUILD
    uint32_t clusterBlasFlags = 0x00000008;         // PREFER_FAST_BUILD

    // fraction of hardware threads the one-time clusterization may use
    // (runs on the provider's background worker)
    float processingThreadsPct = 0.5f;

    // per-frame budget: clusters instantiated across all poses; the caller
    // (ClusterLodManager) keeps instances classic once a frame would exceed
    // it (risk R15 - degrade, never corrupt)
    uint32_t maxPerFrameClusters = 1u << 20;
  };

  struct AnimatedStats {
    uint32_t registeredGeometries = 0;
    uint32_t activePoseSets = 0;
    uint64_t totalClusters = 0;

    uint64_t templateBytes = 0;   // built template memory (resident)
    uint64_t geometryBytes = 0;   // resident topology buffers (index remap)
    uint64_t clasBytes = 0;       // persistent per-pose CLAS destination memory
    uint64_t blasReservedBytes = 0;
    uint64_t operationsBytes = 0; // rings, sizes, scratch

    // delayed GPU readback (a few frames old): actual built sizes
    uint64_t clasActualBytes = 0;
    uint64_t blasActualBytes = 0;
  };

  // Owns the GPU side of Path B: its own Resources (volk/VMA) so deforming
  // geometry clusters independently of whether a Path A render generation
  // exists. Geometry registration runs on the provider's background worker
  // (CPU clusterization lock-free, GPU template builds under Remix's
  // submission lock); per-frame recording happens on the main thread.
  class ClusterTemplateSystem {
  public:
    ClusterTemplateSystem();
    ~ClusterTemplateSystem();

    ClusterTemplateSystem(const ClusterTemplateSystem&) = delete;
    ClusterTemplateSystem& operator=(const ClusterTemplateSystem&) = delete;

    bool init(const RenderDeviceInfo& deviceInfo, const AnimatedConfig& config);
    void deinit();

    // P4c: installs submission-lock callbacks on the system's Resources. Once
    // set, temp submissions lock ONLY around their raw vkQueueSubmit - fence
    // waits run unlocked, so per-geometry registrations no longer block the
    // render threads' submissions for the GPU duration of a template build.
    // Pass nullptrs to restore the external-lock contract (used around deinit,
    // which the manager runs while already holding the lock).
    void setSubmitLockCallbacks(std::function<void()> lockFn, std::function<void()> unlockFn);

    // ---- background worker (one thread at a time) ----

    // CPU-only: clusterizes the snapshot's topology (meshopt, bind-pose
    // reference positions). Returns a token for buildGeometryTemplates, 0 on
    // failure. No Vulkan usage.
    uint64_t clusterizeGeometry(const GeometrySnapshot& snapshot);

    // GPU part of the registration: uploads the cluster-ordered index
    // topology, builds + compacts the cluster templates (or, in direct build
    // mode, only queries the worst-case CLAS sizes) and appends the cluster
    // table records. Submits temporary command buffers and waits for them.
    // Queue synchronization: either the caller holds Remix's submission lock,
    // or setSubmitLockCallbacks is installed (preferred - see above).
    bool buildGeometryTemplates(uint64_t token);

    // P4c (plan 7.7): uploads a promotion probe blob (device-local, BDA) and
    // returns its device address; 0 on failure. Worker thread - same queue
    // synchronization contract as buildGeometryTemplates.
    uint64_t uploadPromotionProbe(const void* data, size_t bytes);

    // P4c: deferred-frees one probe blob once its geometry's verdict is
    // terminal (REJECTED - nothing references it again). Main thread; the
    // actual destroy waits out the in-flight frames (trash queue). Probes of
    // PROMOTED geometries stay resident - the periodic full-mesh sweeps (risk
    // R20) read their full-reference tails.
    void freePromotionProbe(uint64_t probeVa);

    // ---- main thread ----

    struct ReadyGeometry {
      uint64_t topologyKey = 0;
      uint32_t geometryIndex = 0;
    };

    // adopts geometries whose registration completed since the last drain;
    // their geometryIndex becomes valid for createPoseSet
    std::vector<ReadyGeometry> drainReadyGeometries();

    // One pose set per Remix BlasEntry: persistent explicit-destination CLAS
    // memory for one deforming mesh instance's per-frame instantiations.
    // Returns ~0u on failure (out of memory).
    uint32_t createPoseSet(uint32_t geometryIndex);
    // deferred destruction (safe against frames in flight)
    void releasePoseSet(uint32_t poseSetId);

    // clusters a pose set instantiates per frame (budget accounting)
    uint32_t getPoseSetClusterCount(uint32_t poseSetId) const;

    // NV-DXVK: [ClusterDecodeProbe] for a pose set, the [base, base+count) global cluster-id range
    // its clusters were baked with (globalClusterBase+c). Lets the readback compare a failing hit's
    // committed clusterId to what THIS geometry should have produced: committed inside the range but
    // a null table entry -> publish-ordering race; committed outside -> the cluster carries a
    // foreign id (bake/routing divergence). Returns false for an inactive/invalid pose set.
    bool getPoseSetClusterIdRange(uint32_t poseSetId, uint32_t& outBase, uint32_t& outCount) const;

    // NV-DXVK: [ClusterDecodeProbe] total populated animated cluster-table records (== sum of all
    // registered geometries' numClusters). A committed clusterId >= this is beyond EVERY animated
    // cluster -> foreign id; < this with a null entry -> unpublished (publish race).
    uint32_t getAnimatedClusterTableCount() const;

    // frame tick: processes deferred destruction
    void beginFrame(uint32_t frameId);

    struct PoseInput {
      uint32_t poseSetId = ~0u;
      // current-frame vertex data (skinned output / live geometry buffer)
      uint64_t positionsAddress = 0;
      uint32_t positionsStrideBytes = 0;
    };

    // Records the per-frame Path B build into cmd: CLAS instantiation from
    // the live vertex buffers (template mode) or direct triangle-cluster
    // builds, indirect cluster BLAS build (implicit destinations, one BLAS
    // per pose), and the cluster_blas_instances kernel that patches each
    // TLAS slot's blasReference (slotPoseIndex[slot] -> pose -> BLAS) and
    // sums sizes for statistics. tlasInstances carries the CPU-known fields;
    // after execution the patched entries are in getTlasInstancesBuffer()
    // at [0, slotCount).
    bool recordFrame(VkCommandBuffer cmd,
                     const PoseInput* poses, uint32_t poseCount,
                     const uint32_t* slotPoseIndex,
                     const VkAccelerationStructureInstanceKHR* tlasInstances,
                     uint32_t slotCount);

    // buffer holding the patched TlasInstances after recordFrame's commands ran
    VkBuffer getTlasInstancesBuffer() const;

    // device address of the global animated cluster table (8 bytes per
    // cluster: index-topology address of the cluster's triangles). The
    // ClusterID baked into templates indexes it; consumed by the hit-side
    // index remap via raytrace_args. Thread-safe (may grow on the worker).
    uint64_t getClusterTableAddress() const;

    // NV-DXVK: [SceneAnimInstScan] current pose-BLAS ring pool ranges (lo/hi per pool,
    // up to maxCount, ring order); returns the count written. outFrameCounter (optional)
    // receives the completed-recordFrame count - the last recorded frame's pose BLASes
    // live in pool (count-1) % kRingSlots. Diagnostic - revert.
    uint32_t getPoseBlasPools(uint64_t* lo, uint64_t* hi, uint32_t maxCount, uint32_t* outFrameCounter = nullptr) const;

    // NV-DXVK: [ClasAlias] ranges of every live pose-set CLAS buffer (across all ring
    // slots) - the memory the instantiated Path B CLAS (with baked clusterID) live in,
    // and the memory a pose BLAS references. Overlap with Path A CLAS memory = the
    // foreign clusterId 4096+ root. Returns the count written. Diagnostic - revert.
    uint32_t getPoseClasRanges(uint64_t* lo, uint64_t* hi, uint32_t maxCount) const;

    // NV-DXVK: [TplAlias] ranges of every registered geometry's TEMPLATE buffer - the memory
    // the instantiate reads clusterTemplateAddress from (with clusterIdOffset=0, the committed
    // clusterId IS the template's baked id). If a template buffer OVERLAPS Path A resident CLAS
    // memory, the instantiate copies a resident cluster's baked id (4096+) into the pose CLAS -
    // the last unchecked path for a genuine Path B instance to commit a resident clusterId
    // (pose CLAS buffers were already cleared by [ClasAlias]; templates never were). Returns the
    // count written. Diagnostic - revert.
    uint32_t getTemplateBufferRanges(uint64_t* lo, uint64_t* hi, uint32_t maxCount) const;

    // NV-DXVK: DIAGNOSTIC (revert) - constant added to the per-frame Path B instantiate
    // clusterIdOffset (normally 0). Pushed from rtx.clusterLod.animated.dbgClusterIdOffsetSentinel
    // each frame to test whether the committed resident clusterId 4096+ originates from this
    // instantiate. See the option doc in rtx_cluster_lod_manager.h.
    void setDbgClusterIdOffsetSentinel(uint32_t sentinel);

    bool getStats(AnimatedStats& outStats) const;

    // chrono report of the per-frame GPU/CPU section timers around the Path B
    // build phases (input fill, CLAS instantiation, BLAS build, slot patch):
    // one line per section ('\n'-separated), values averaged over the
    // profiler's last frames. False while nothing was timed yet.
    bool getProfilerReportUtf8(std::string& outReport) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
  };

}  // namespace lodclusters_remix
