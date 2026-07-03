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

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtx_option.h"
#include "rtx_types.h"
#include "lodclusters/lodclusters_remix.h"

namespace dxvk {

  class DxvkDevice;
  class DxvkContext;
  struct DrawCallState;
  class ClusterLodGeometryProvider;
  class AccelManager;
  class CameraManager;
  class InstanceManager;
  class RtInstance;

  // RTX Mega Geometry (VK_NV_cluster_acceleration_structure) cluster LOD options.
  // Defaults mirror the vk_lod_clusters sample's SceneConfig / SceneLoaderConfig.
  struct ClusterLodOptions {
    friend class ClusterLodManager;

    RTX_OPTION("rtx.clusterLod", bool, enable, false,
               "Enables the RTX Mega Geometry cluster LOD system.\n"
               "Every unique geometry is snapshotted on first sight, processed into a continuous-LOD cluster\n"
               "hierarchy on a background thread and disk-cached by geometry hash (rtx-remix/cache/geometry).\n"
               "Rendering through cluster BLASes arrives with the following integration phases; until then this\n"
               "covers processing and caching.");

    RTX_OPTION("rtx.clusterLod", bool, verifyCacheRoundTrip, false,
               "Debug: after processing a geometry, re-load its .nvsngeo cache entry twice (system RAM and\n"
               "memory-mapped) and verify both against the freshly processed statistics.");

    RTX_OPTION("rtx.clusterLod", int, logStatsIntervalFrames, 600,
               "Logs the [ClusterLOD] stats lines (intake/skip/processing counts and render/streaming state)\n"
               "every N frames whenever the counts changed since the last log, so the log always carries\n"
               "totals - the per-geometry skip messages only print their first occurrence per reason.\n"
               "0 disables periodic logging (taking a screenshot capture still logs the stats).");

    // lodclusters::SceneConfig mirror
    struct SceneConfig {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.sceneConfig", int, clusterVertices, 128,
                 "Max vertices per cluster. Must match a compiled shader variant: 64 or 128.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, clusterTriangles, 128,
                 "Max triangles per cluster. Must match a compiled shader variant: 64 or 128.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, clusterGroupSize, 32,
                 "Clusters per cluster group (the streaming and LOD-decimation unit).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, preferredNodeWidth, 8,
                 "Preferred child count per LOD hierarchy node.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", bool, meshoptPreferRayTracing, true,
                 "Configures meshoptimizer clusterization to favor ray tracing (fill weight) over rasterization.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", bool, useCompressedData, false,
                 "Stores cluster groups compressed (decompressed on upload/stream-in).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, enabledAttributes, 5,
                 "Bitmask of vertex attributes carried by clusters: 1 = normals, 2 = tangents, 4 = texcoord0, 8 = texcoord1.\n"
                 "Default 5 (normals + texcoord0).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, meshoptFillWeight, 0.5f,
                 "meshoptimizer cluster fill weight (used when preferring ray tracing).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, meshoptSplitFactor, 2.0f,
                 "meshoptimizer cluster split factor (used when not preferring ray tracing).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, lodLevelDecimationFactor, 0.5f,
                 "Per-LOD-step triangle reduction factor for cluster groups.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, lodErrorMergePrevious, 1.5f,
                 "LOD error propagation: previous-level error multiplier.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, lodErrorMergeAdditive, 0.0f,
                 "LOD error propagation: additive current-error term.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, simplifyNormalWeight, 0.5f,
                 "Simplification attribute weight for normals (0 disables).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, simplifyTangentWeight, 0.01f,
                 "Simplification attribute weight for tangents (0 disables).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, simplifyTangentSignWeight, 0.5f,
                 "Simplification attribute weight for the tangent sign (0 disables).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, simplifyTexCoordWeight, 0.0f,
                 "Simplification attribute weight for texcoords (0 disables).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", float, simplifyMaterialWeight, 0.5f,
                 "Simplification attribute weight for per-vertex material ids (0 disables).");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, compressionPosDropBits, 7,
                 "Mantissa bits dropped from positions when useCompressedData is on.");
      RTX_OPTION("rtx.clusterLod.sceneConfig", int, compressionTexDropBits, 7,
                 "Mantissa bits dropped from texcoords when useCompressedData is on.");
    };

    // lodclusters::SceneLoaderConfig mirror
    struct Processing {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.processing", float, threadsPct, 0.5f,
                 "Fraction of hardware threads the background cluster processing may use.");
      RTX_OPTION("rtx.clusterLod.processing", bool, autoSaveCache, true,
                 "Saves a .nvsngeo cache file after processing a geometry.");
      RTX_OPTION("rtx.clusterLod.processing", bool, autoLoadCache, true,
                 "Loads geometries from their .nvsngeo cache file when present.");
      RTX_OPTION("rtx.clusterLod.processing", bool, memoryMappedCache, false,
                 "Memory-maps cache files instead of loading them into system RAM.");
      RTX_OPTION("rtx.clusterLod.processing", int, forcePreprocessMiB, 2048,
                 "Geometry exceeding this raw size gets a dedicated preprocess pass writing straight to the cache.");
    };

    // lodclusters::RendererConfig / FrameConfig mirror (P2, preloaded rendering)
    struct Render {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.render", float, lodPixelError, 1.0f,
                 "Continuous-LOD error threshold in pixels; lower is more detail. The GPU traversal picks cluster\n"
                 "LODs whose projected error stays below this many pixels (0 converges to the source mesh).");
      RTX_OPTION("rtx.clusterLod.render", float, culledErrorScale, 2.0f,
                 "LOD error multiplier for instances without primary visibility.");
      RTX_OPTION("rtx.clusterLod.render", bool, useSorting, false,
                 "Sorts instances by traversal priority (vulkan_radix_sort) before LOD traversal.\n"
                 "Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, useCulling, true,
                 "Frustum + HiZ occlusion culling during LOD traversal (P4). The HiZ pyramid is built each frame\n"
                 "from Remix's previous-frame primary depth; instances/clusters without primary visibility keep\n"
                 "rendering (rays still need them) at culledErrorScale-reduced detail.\n"
                 "Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, useBlasSharing, true,
                 "Instances of the same geometry at compatible LOD ranges share one cluster BLAS.\n"
                 "Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, useBlasMerging, true,
                 "Merges the leftover per-instance BLAS builds of shared geometries into one merged traversal.\n"
                 "NVIDIA sample default. Requires streaming mode (self-disables without it).\n"
                 "Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, useBlasCaching, false,
                 "Caches the built BLAS of shared geometries in a dedicated memory region so stable geometry stops\n"
                 "being rebuilt every frame (P4). Requires streaming mode + BLAS sharing. Sample default off;\n"
                 "budgeted by rtx.clusterLod.streaming.maxBlasCachingMegaBytes.\n"
                 "Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, useForcedInvisibleCulling, false,
                 "Removes occlusion/frustum-culled instances from the TLAS entirely (mask 0) instead of keeping\n"
                 "them at reduced detail. Off-screen geometry then stops casting shadows/reflections - perf and\n"
                 "debug mode only. Read when the cluster render system starts (kernel variant selection).");
      RTX_OPTION("rtx.clusterLod.render", bool, freezeCulling, false,
                 "Debug: freezes the culling matrices and the HiZ content at their current state so culling\n"
                 "results can be inspected from a different camera position.");
      RTX_OPTION("rtx.clusterLod.render", bool, freezeLoD, false,
                 "Debug: freezes the LOD traversal view (error metric input) at its current state.");
      RTX_OPTION("rtx.clusterLod.render", bool, sharingPushCulled, true,
                 "BLAS sharing also pushes instances without primary visibility onto shared BLASes.");
      RTX_OPTION("rtx.clusterLod.render", int, sharingTolerantLevels, 7,
                 "LOD levels above which BLAS-sharing election tolerates a coarser shared BLAS.");
      RTX_OPTION("rtx.clusterLod.render", int, sharingEnabledLevels, 8,
                 "Number of (coarse) LOD levels eligible for BLAS sharing.");
      RTX_OPTION("rtx.clusterLod.render", bool, usePersistentTraversal, true,
                 "Uses the persistent-threads traversal kernel instead of the multi-pass variant.");
      RTX_OPTION("rtx.clusterLod.render", int, numRenderClusterBits, 20,
                 "log2 of the maximum renderable clusters per frame.");
      RTX_OPTION("rtx.clusterLod.render", int, numTraversalTaskBits, 20,
                 "log2 of the maximum intermediate LOD traversal tasks.");
      RTX_OPTION("rtx.clusterLod.render", int, maxRenderInstances, 4096,
                 "Initial render-instance capacity of the cluster renderer (grows automatically via generation rebuild).");
      RTX_OPTION("rtx.clusterLod.render", int, maxGeometries, 4096,
                 "Geometry-slot capacity of a render generation (geometry table + BLAS-sharing buffers).\n"
                 "Newly processed geometries append to the running generation in O(new) while they fit;\n"
                 "exceeding the capacity triggers a full generation rebuild with a grown table.");
      RTX_OPTION("rtx.clusterLod.render", int, positionTruncateBits, 0,
                 "Mantissa bits truncated from CLAS vertex positions (smaller CLAS, faster builds).");
      RTX_OPTION("rtx.clusterLod.render", bool, blasFastTrace, true,
                 "Builds per-frame cluster BLAS with PREFER_FAST_TRACE (else PREFER_FAST_BUILD).");
      RTX_OPTION("rtx.clusterLod.render", int, generationCooldownFrames, 30,
                 "Minimum frames between render-generation updates. Newly processed geometries are batched and\n"
                 "join the running generation incrementally (O(new) upload, no re-upload of resident geometry);\n"
                 "a full rebuild (GPU-idle swap) only happens for the first generation, capacity growth or a\n"
                 "SceneConfig change.");
      RTX_OPTION("rtx.clusterLod.render", int, traversalPersistentThreads, 2048,
                 "Thread count used by the persistent traversal kernel.");
    };

    // lodclusters::StreamingConfig mirror (P3). Read when the cluster render
    // system starts (first generation); changes require a game restart.
    struct Streaming {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.streaming", bool, preferStreaming, false,
                 "Renders through the on-demand streaming path (SceneStreaming) instead of keeping every\n"
                 "processed geometry fully resident (ScenePreloaded). Only the lowest-detail clusters stay\n"
                 "permanently on the GPU; higher detail cluster groups stream in when the LOD traversal asks\n"
                 "for them and age out when unused, bounded by the budgets below.");
      RTX_OPTION("rtx.clusterLod.streaming", int, ageThreshold, 16,
                 "Frames until an unused streamed cluster group is scheduled for unloading.");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxPerFrameLoadRequests, 128,
                 "Maximum cluster-group loads requested per frame.");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxPerFrameUnloadRequests, 1024,
                 "Maximum cluster-group unloads requested per frame.");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxGroups, 65536,
                 "Resident cluster-group table size (streaming residency capacity).");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxClusters, 0,
                 "Resident cluster table size; 0 derives it from maxGroups and the cluster group size.");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxTransferMegaBytes, 32,
                 "Per-frame host-to-device upload budget for streamed cluster groups (MiB).");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxGeometryMegaBytes, 2048,
                 "Device memory budget for streamed cluster-group geometry data (MiB).");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxClasMegaBytes, 2048,
                 "Device memory budget for the CLAS of streamed cluster groups (MiB).");
      RTX_OPTION("rtx.clusterLod.streaming", bool, useAsyncTransfer, false,
                 "Uploads streamed cluster groups on the dedicated transfer queue instead of the graphics\n"
                 "queue command buffer (sample default off; per-frame submission takes the dxvk queue lock).");
      RTX_OPTION("rtx.clusterLod.streaming", bool, useDecoupledAsyncTransfer, false,
                 "Async transfers may span multiple frames (scene update deferred until the transfer\n"
                 "completes) instead of the frame waiting on the transfer queue. Requires useAsyncTransfer.");
      RTX_OPTION("rtx.clusterLod.streaming", bool, usePersistentClasAllocator, true,
                 "GPU-driven persistent CLAS memory allocator (bit array + free-gap lists, CLAS moved once\n"
                 "after building). Off = simple compaction (packs all resident CLAS before appending new\n"
                 "ones; bursts of memory movement - NVIDIA does not recommend it).");
      RTX_OPTION("rtx.clusterLod.streaming", int, clasAllocatorSectorSizeShift, 10,
                 "Persistent CLAS allocator: log2 of the free-gap scan sector size.");
      RTX_OPTION("rtx.clusterLod.streaming", int, clasAllocatorGranularityShift, 0,
                 "Persistent CLAS allocator: allocation granularity as a shift on the CLAS alignment.");
      RTX_OPTION("rtx.clusterLod.streaming", int, maxBlasCachingMegaBytes, 1024,
                 "Device memory budget for cached BLAS of shared geometries (MiB); with\n"
                 "rtx.clusterLod.render.useBlasCaching. Read when the cluster render system starts.");
      RTX_OPTION("rtx.clusterLod.streaming", int, cachingAgeThreshold, 16,
                 "Frames a shared BLAS must stay stable before it is cached (per-frame tunable).");
      RTX_OPTION("rtx.clusterLod.streaming", int, cachingEnabledLevels, 8,
                 "Number of (coarse) LOD levels eligible for BLAS caching (per-frame tunable).");
    };

    // P4b PATH B: deforming geometry through cluster templates
    // (vk_animated_clusters mirror; defaults = the sample's effective
    // defaults for animated content). Read when the template system starts;
    // changes require a game restart.
    struct Animated {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.animated", bool, enable, true,
                 "Routes deforming geometry (GPU-skinned meshes and meshes whose vertex data mutates in place\n"
                 "every frame) through the cluster-template path: the topology is clusterized once, then every\n"
                 "frame the CLAS are instantiated on-GPU directly from the live vertex buffers and one cluster\n"
                 "BLAS per mesh instance is rebuilt from the cluster references (vk_animated_clusters design).\n"
                 "Gated by rtx.clusterLod.enable. Off = deforming geometry stays on the classic BLAS path.");
      RTX_OPTION("rtx.clusterLod.animated", int, clusterVertices, 64,
                 "Max vertices per cluster for deforming geometry (sample recommendation: 64).");
      RTX_OPTION("rtx.clusterLod.animated", int, clusterTriangles, 64,
                 "Max triangles per cluster for deforming geometry (sample recommendation: 64).");
      RTX_OPTION("rtx.clusterLod.animated", bool, useTemplates, true,
                 "Uses cluster templates (topology encoded once, per-frame CLAS instantiation) instead of\n"
                 "direct per-frame triangle-cluster builds. Templates instantiate faster and their worst-case\n"
                 "CLAS memory is queried per geometry instead of assumed (the sample exposes both).");
      RTX_OPTION("rtx.clusterLod.animated", bool, useImplicitTemplates, false,
                 "Builds templates in implicit-destination mode followed by a move compaction, instead of an\n"
                 "explicit build with a COMPUTE_SIZES pre-query (sample effective default: explicit).");
      RTX_OPTION("rtx.clusterLod.animated", float, templateBboxBloatPercentage, 0.5f,
                 "instantiationBoundingBoxLimit bloat as a fraction of the geometry's bind-pose bbox diagonal -\n"
                 "the animation may move vertices this far outside the reference bbox. Negative disables the limit.");
      RTX_OPTION("rtx.clusterLod.animated", int, positionTruncateBits, 0,
                 "Mantissa bits truncated from CLAS vertex positions (smaller CLAS, faster instantiation).");
      RTX_OPTION("rtx.clusterLod.animated", bool, templateBuildFastTrace, true,
                 "Builds cluster templates with PREFER_FAST_TRACE (sample default; templates build once).");
      RTX_OPTION("rtx.clusterLod.animated", bool, instantiateFastTrace, false,
                 "Instantiates per-frame CLAS with PREFER_FAST_TRACE instead of PREFER_FAST_BUILD\n"
                 "(sample default for animated content: fast build - the CLAS are rebuilt every frame).");
      RTX_OPTION("rtx.clusterLod.animated", bool, blasFastTrace, false,
                 "Builds the per-frame cluster BLAS with PREFER_FAST_TRACE instead of PREFER_FAST_BUILD\n"
                 "(sample default for animated content: fast build).");
      RTX_OPTION("rtx.clusterLod.animated", int, maxPerFrameClusters, 1048576,
                 "Per-frame budget of instantiated clusters across all deforming meshes. Instances that would\n"
                 "exceed it render classic that frame (degrade, never corrupt - risk R15).");
    };
  };

  // Owner of the cluster LOD subsystems on the Remix side:
  //  - P1: geometry intake (provider), background processing through NVIDIA's
  //    lodclusters pipeline, per-geometry-hash disk cache.
  //  - P2: preloaded rendering. Processed geometries join a render "generation"
  //    (combined Scene + RenderScene + renderer, see lodclusters_remix.h);
  //    eligible RtInstances are diverted from the classic BLAS path in
  //    AccelManager::mergeInstancesIntoBlas into reserved TLAS slots that the
  //    GPU traversal + instance_assign_blas patch each frame.
  class ClusterLodManager {
  public:
    explicit ClusterLodManager(DxvkDevice* device);
    ~ClusterLodManager();

    // VK_NV_cluster_acceleration_structure + required subgroup size. Gates the GPU
    // rendering path (P2+); CPU-side processing and caching run regardless so caches
    // can be prebuilt on any machine.
    static bool checkIsSupported(DxvkDevice* device);

    // CS thread, called for every draw call that reached the object cache.
    // vertexDataUpdated = the existing BlasEntry's vertex data changed in
    // place this draw (kUpdateBVH) - the Path B mutation signal.
    void onDrawCallGeometry(const DrawCallState& drawCallState, bool vertexDataUpdated);

    void logStatistics() const;

    // ---- P2 frame integration (main thread, SceneManager::prepareSceneData) ----

    // Before mergeInstancesIntoBlas: drains completed geometries, performs the
    // render-generation swap when due, resets the per-frame slot state.
    void onFrameBegin(Rc<DxvkContext> ctx, AccelManager& accelManager, InstanceManager& instanceManager);

    // True while a render generation exists: mergeInstancesIntoBlas must run its
    // full pass every frame (LOD traversal + cluster BLAS builds are per-frame
    // work even when the classic scene is unchanged; risk R8).
    bool needsFullMergePass() const;

    // mergeInstancesIntoBlas main loop: returns true (and the geometryID) if this
    // instance renders through the cluster path this frame and must skip the
    // classic BLAS routing. Path A (static LOD) and Path B (deforming,
    // cluster templates) both route through here; outGeometryId carries a tag
    // bit distinguishing them (consumed by recordClusterInstance).
    bool isClusterInstance(const RtInstance* instance, uint32_t& outGeometryId);

    // mergeInstancesIntoBlas divert branch: records the instance into the given
    // TLAS type's cluster region (in arrival order) with its CPU-known
    // VkAccelerationStructureInstanceKHR content (blasReference gets pre-filled
    // with the geometry's low-detail BLAS here). isSssDuplicate additionally
    // reserves a slot in the SSS region that receives a copy of this (Opaque)
    // instance's patched entry.
    void recordClusterInstance(RtInstance* instance,
                               uint32_t geometryId,
                               size_t tlasType,
                               bool isSssDuplicate,
                               const VkAccelerationStructureInstanceKHR& blasInstance);

    // reserved cluster TLAS slots per type this frame (AccelManager buffer sizing)
    uint32_t getClusterSlotCount(size_t tlasType) const;

    // After AccelManager::prepareSceneData + dispatchPointInstancerCulling and
    // before buildTlas: records the per-frame cluster build (traversal, CLAS/BLAS,
    // instance_assign_blas) and copies the patched TlasInstances into
    // AccelManager's instance buffer regions.
    void dispatchBuild(Rc<DxvkContext> ctx, const CameraManager& cameraManager, AccelManager& accelManager);

    // device address of the generation's shaderio::Geometry table (0 if none);
    // consumed by the path tracer's hit-side cluster fetch via raytrace_args
    uint64_t getGeometriesTableAddress() const;

    // P4b: device address of the global animated cluster table (0 if none);
    // consumed by the hit-side Path B primitive remap via raytrace_args
    uint64_t getAnimatedClusterTableAddress() const;

  private:
    struct ClusterSlot {
      RtInstance* instance = nullptr;
      // Path A: cluster geometry table index; Path B slot lists: frame pose index
      uint32_t geometryId = 0;
    };

    struct SssDuplicate {
      // flat kernel-array index of the Opaque-region entry this copies
      uint32_t sourceFlatIndex = 0;
    };

    bool ensureRenderSystem();
    // returns the milliseconds spent (0 when nothing was due) so onFrameBegin
    // can report generation events separately from its steady per-frame cost
    double buildGenerationIfDue(AccelManager& accelManager, InstanceManager& instanceManager);

    // chrono: logs the accumulated per-frame CPU section times and the GPU
    // section reports of both cluster systems, then resets the accumulators.
    // Rides the periodic stats interval; silent while nothing dispatched
    // (menus/loading record no cluster work).
    void logFrameTimes();

    // ---- P4b Path B (cluster templates) ----

    // worker thread (provider handler): full registration of one deforming
    // geometry - CPU clusterization, then the GPU template build under the
    // dxvk submission lock
    bool processAnimatedGeometry(lodclusters_remix::GeometrySnapshot&& snapshot);
    // worker + main thread; lazily creates the template system (mutex-guarded)
    bool ensureTemplateSystem();
    // Path B side of isClusterInstance
    bool isClusterTemplateInstance(const RtInstance* instance, const BlasEntry* blasEntry, uint32_t& outGeometryId);
    // Path B side of dispatchBuild: records instantiation + BLAS build +
    // TLAS-slot patch and copies the patched entries into the cluster regions
    void dispatchAnimated(Rc<DxvkContext> ctx, AccelManager& accelManager, VkCommandBuffer cmd);

    DxvkDevice* m_device = nullptr;

    std::unique_ptr<ClusterLodGeometryProvider> m_provider;

    // ---- P2 rendering state ----
    std::unique_ptr<lodclusters_remix::ClusterRenderSystem> m_renderSystem;
    bool m_renderSystemFailed = false;

    // ---- P4b Path B state ----
    std::unique_ptr<lodclusters_remix::ClusterTemplateSystem> m_templateSystem;
    std::mutex m_templateSystemMutex;  // creation happens on the provider worker
    bool m_templateSystemFailed = false;
    // main-thread view of m_templateSystem (published under the mutex once per
    // frame in onFrameBegin; the pointer never changes once set until teardown)
    lodclusters_remix::ClusterTemplateSystem* m_templateSystemMT = nullptr;

    // topologyKey -> template-system geometry index (main thread; filled from
    // drainReadyGeometries in onFrameBegin)
    std::unordered_map<uint64_t, uint32_t> m_animatedGeometryByKey;

    // one pose set per live BlasEntry (validated against frameCreated and the
    // geometry index to survive BlasEntry reuse); aged out when unseen
    struct PoseEntry {
      uint32_t poseSetId = ~0u;
      uint32_t geometryIndex = ~0u;
      uint32_t blasFrameCreated = 0;
      uint32_t lastSeenFrame = 0;
    };
    std::unordered_map<const BlasEntry*, PoseEntry> m_poseByBlas;
    static constexpr uint32_t kPoseSetKeepFrames = 60;

    // per-frame Path B state (rebuilt by every full mergeInstancesIntoBlas pass)
    struct FramePose {
      uint32_t poseSetId = ~0u;
      uint64_t positionsAddress = 0;
      uint32_t positionsStrideBytes = 0;
      Rc<DxvkBuffer> positionsBuffer;  // lifetime tracking on the cmd list
    };
    std::vector<FramePose> m_framePoses;
    std::unordered_map<const BlasEntry*, uint32_t> m_framePoseIndexByBlas;
    uint32_t m_frameClusterBudgetUsed = 0;
    std::vector<ClusterSlot> m_slotsB[Tlas::Count];  // geometryId = frame pose index
    std::vector<VkAccelerationStructureInstanceKHR> m_slotInstanceDataB[Tlas::Count];
    std::vector<SssDuplicate> m_sssDuplicatesB;      // flat index into the B Opaque block

    // P3/P4: captured from the options when the render system starts (the
    // streaming configuration and kernel variant selection are init-time;
    // the options document this)
    bool m_streamingActive = false;
    bool m_asyncTransferActive = false;
    bool m_cullingActive = false;

    // geometries in the active generation (geometryID order)
    std::vector<uint64_t> m_residentGeometryHashes;
    // processed geometries waiting to join (batched by the cooldown, then
    // appended in O(new) or - on capacity/config change - folded into a full
    // rebuild)
    std::vector<uint64_t> m_pendingGeometryHashes;
    // SceneConfig cache digest the generation was built from; appends require
    // the current config to still resolve to it
    std::string m_generationConfigDigest;
    uint32_t m_lastGenerationFrame = 0;
    uint32_t m_generationCount = 0;

    std::unordered_map<uint64_t, uint32_t> m_geometryIdByHash;

    // P4: primary-depth images already fed through one full frame cycle -
    // a first-sighted (freshly cleared) target must not feed the HiZ build
    std::unordered_set<uint64_t> m_hizDepthImagesSeen;

    // per-frame slot state (rebuilt by every full mergeInstancesIntoBlas pass)
    std::vector<ClusterSlot> m_slots[Tlas::Count];
    std::vector<VkAccelerationStructureInstanceKHR> m_slotInstanceData[Tlas::Count];
    std::vector<SssDuplicate> m_sssDuplicates;
    uint32_t m_frameOverflowCount = 0;
    uint32_t m_peakInstanceCount = 0;

    // periodic stats logging (rtx.clusterLod.logStatsIntervalFrames): last
    // frame a stats line was considered, and the counters it printed - a new
    // line is only emitted when they changed
    uint32_t m_lastStatsLogFrame = 0;
    uint64_t m_lastLoggedStatsDigest = 0;

    // ---- chrono: per-frame CPU section times, accumulated between periodic
    //      logs (avg = steady cost, max = the hitches an avg hides) ----
    struct SectionTimes {
      double totalMs = 0.0;
      double maxMs = 0.0;
      uint32_t samples = 0;
      void add(double ms) {
        totalMs += ms;
        maxMs = std::max(maxMs, ms);
        samples++;
      }
      double avgMs() const { return samples > 0 ? totalMs / samples : 0.0; }
    };
    struct FrameTimes {
      SectionTimes frameBegin;   // onFrameBegin minus generation events
      SectionTimes classify;     // isClusterInstance total across the merge pass
      SectionTimes dispatchA;    // dispatchBuild minus dispatchAnimated
      SectionTimes hizFeed;      // HiZ depth handoff (excl. resize events)
      SectionTimes lockWaitA;    // async-transfer submission-lock wait
      SectionTimes recordA;      // ClusterRenderSystem::recordFrame
      SectionTimes dispatchB;    // dispatchAnimated total
      SectionTimes recordB;      // ClusterTemplateSystem::recordFrame
    };
    FrameTimes m_frameTimes;
    // isClusterInstance is called per instance during mergeInstancesIntoBlas;
    // accumulated here and folded into m_frameTimes.classify at the next
    // onFrameBegin (one sample per frame)
    double m_frameClassifyMs = 0.0;
  };

}  // namespace dxvk
