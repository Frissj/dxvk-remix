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
#include <chrono>
#include <cstdint>
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
      RTX_OPTION("rtx.clusterLod.processing", int, workerCount, 0,
                 "Provider worker threads draining the geometry intake queue in parallel (P4c item 7:\n"
                 "discovery floods outpaced the single worker - queue waits peaked at seconds). Each worker\n"
                 "runs the CPU LOD pipeline independently; template/probe GPU work stays serialized across\n"
                 "workers. 0 = auto (hardware threads / 4, clamped to [1, 4]). Read at startup.");
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
      RTX_OPTION("rtx.clusterLod.render", bool, useBlasCaching, true,
                 "Caches the built BLAS of shared geometries in a dedicated memory region so stable geometry stops\n"
                 "being rebuilt every frame (P4). Requires streaming mode + BLAS sharing. Default on (F1,\n"
                 "2026-07-04, NVIDIA's game-world guidance: 3.9->1.6 ms frame, 62->9 MB BLAS in their tests);\n"
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
      RTX_OPTION("rtx.clusterLod.render", bool, usePersistentTraversal, false,
                 "Uses the persistent-threads traversal kernel instead of the multi-pass variant.\n"
                 "Default off (F2, 2026-07-04): NVIDIA's own render() comment calls multi-pass typically\n"
                 "faster and the README calls the persistent thread heuristic crude.");
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
      RTX_OPTION("rtx.clusterLod.render", int, positionTruncateBits, 4,
                 "Mantissa bits truncated from CLAS vertex positions (smaller CLAS, faster builds).\n"
                 "Default 4 (F3, 2026-07-04, NVIDIA-recommended): relative position error ~2^-19, far below\n"
                 "visibility. Set 0 to disable.");
      RTX_OPTION("rtx.clusterLod.render", bool, blasFastTrace, true,
                 "Builds per-frame cluster BLAS with PREFER_FAST_TRACE (else PREFER_FAST_BUILD).");
      RTX_OPTION("rtx.clusterLod.render", bool, routeTrivialToClassic, true,
                 "Renders geometries whose LOD build produced a single LOD level through the classic triangle\n"
                 "BLAS path at render time (F7, user-approved 2026-07-04; NVIDIA guidance: for purely static\n"
                 "data without LOD benefit the traditional BLAS is still recommended - it also rejoins Remix's\n"
                 "merged bucket BLASes). The geometry is still processed and cached, so it upgrades to clusters\n"
                 "automatically if a config change ever gives it more LOD levels.");
      RTX_OPTION("rtx.clusterLod.render", int, generationCooldownFrames, 30,
                 "Minimum frames between render-generation updates. Newly processed geometries are batched and\n"
                 "join the running generation incrementally (O(new) upload, no re-upload of resident geometry);\n"
                 "a full rebuild (GPU-idle swap) only happens for the first generation, capacity growth or a\n"
                 "SceneConfig change.");
      RTX_OPTION("rtx.clusterLod.render", int, cacheHitCooldownFrames, 4,
                 "Cooldown used instead of generationCooldownFrames while the pending batch contains a geometry\n"
                 "served from its .nvsngeo cache (fast lane, plan 7.7 first-frame guarantee). Cache loads cost\n"
                 "milliseconds, so the full cooldown would delay their classic->cluster flip with nothing to\n"
                 "amortize; a few frames of micro-batching still bounds append frequency during session-start\n"
                 "cache-hit bursts.");
      RTX_OPTION("rtx.clusterLod.render", int, traversalPersistentThreads, 2048,
                 "Thread count used by the persistent traversal kernel.");
    };

    // lodclusters::StreamingConfig mirror (P3). Read when the cluster render
    // system starts (first generation); changes require a game restart.
    struct Streaming {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.streaming", bool, preferStreaming, true,
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
      RTX_OPTION("rtx.clusterLod.streaming", bool, useAsyncTransfer, true,
                 "Uploads streamed cluster groups on the dedicated transfer queue instead of the graphics\n"
                 "queue command buffer. Default on (F4, 2026-07-04): Remix's transfer queue is mostly idle.\n"
                 "Set False (sample default) if streaming instability appears.");
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
                 "Mantissa bits truncated from CLAS vertex positions (smaller CLAS, faster instantiation).\n"
                 "MUST be 0 for deforming geometry: the template is built from the bind pose and then\n"
                 "instantiated against live deformed positions, so truncation quantizes close vertices\n"
                 "together and collapses triangles into flickering razor slivers (root cause of the Path B\n"
                 "razor-triangle flicker; the F3 default of 4 introduced it, 2026-07-05). The animated\n"
                 "sample recommends truncation only because it never deforms the reference pose. Raise\n"
                 "above 0 only after verifying the specific content shows no razor slivers.");
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
      RTX_OPTION("rtx.clusterLod.animated", bool, interimTemplates, false,
                 "P4c ladder (plan 7.7): static geometry the LOD pipeline has not finished processing renders\n"
                 "through cluster templates in the meantime instead of classic. DEFAULT OFF (user decision\n"
                 "2026-07-04, minimal frame cost wins): the classic BLAS is built for the first 1-2 frames\n"
                 "either way and then costs nothing per frame, while interim templates re-instantiate CLAS and\n"
                 "rebuild BLAS every frame until Path A lands. Enable for cluster-pipeline uniformity testing\n"
                 "or when AS memory matters more than frame time (clusters ~ half a classic BLAS on huge\n"
                 "meshes). Skipped when the geometry's .nvsngeo cache already exists; interim pose sets age\n"
                 "out via the normal 60-frame pose GC after the flip.");
    };

    // 7.7 rigid-capture promotion (P4c).
    struct Promotion {
      friend class ClusterLodManager;

      RTX_OPTION("rtx.clusterLod.promotion", bool, enable, true,
                 "Rigid-capture promotion (plan 7.7): per frame, a GPU kernel solves the affine transform that\n"
                 "maps a captured mesh's input-space snapshot onto its vertex-capture output. When the fit is\n"
                 "rigid (sparse residual streak + one full-mesh sweep), the instance flips from Path B cluster\n"
                 "templates to the Path A LOD pipeline with the GPU-recovered transform patched into its\n"
                 "render instance every frame - captured static world geometry gets real LOD/streaming.\n"
                 "Universal: no per-game assumptions, the verdict is a property of the mesh data itself.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, processAtFirstSight, true,
                 "Runs the LOD pipeline (clusterization, LOD DAG, .nvsngeo cache, generation residency) for\n"
                 "vertex-captured static-topology geometry at first sight, in parallel with its Path B\n"
                 "registration (plan 7.7, rule 3: spend background work to minimize flip latency).");
      RTX_OPTION("rtx.clusterLod.promotion", int, rigidFrames, 2,
                 "Consecutive rigid solve frames before the full-mesh gate is scheduled (hysteresis only - M is\n"
                 "re-solved every frame, so streak length adds no correctness; see plan 7.7).");
      RTX_OPTION("rtx.clusterLod.promotion", float, residualEpsilon, 0.005f,
                 "Maximum solve/gate residual relative to the geometry's bounding radius for a frame to count\n"
                 "as rigid. Non-rigid VS output (skinning in shader, foliage sway, billboards) fails this and\n"
                 "keeps the mesh on Path B.");
      RTX_OPTION("rtx.clusterLod.promotion", int, gateLagFrames, 6,
                 "Frames between dispatching the full-mesh gate sweep and reading its verdict (covers the\n"
                 "readback ring lag).");
      RTX_OPTION("rtx.clusterLod.promotion", int, restRejectRetryFrames, 600,
                 "Frames after which a REST-rejected instance re-enters probing. Rejection cannot be terminal\n"
                 "on games whose captured-draw hash churns: the BlasEntry<->capture-content binding can swap\n"
                 "when the draw set changes (verified: a gating slot's capture changed scale mid-gate), so a\n"
                 "terminal verdict condemns a slot for content it no longer holds. 0 keeps rejection terminal.");
      RTX_OPTION("rtx.clusterLod.promotion", int, fullSweepIntervalFrames, 32,
                 "Steady-state full-mesh residual sweep cadence per PROMOTED instance (plan 7.7, risk R20):\n"
                 "the sparse per-frame solve can miss a VS animating a small vertex subset, so every promoted\n"
                 "instance re-runs the every-vertex sweep on this interval (staggered by state slot). A failing\n"
                 "sweep demotes that instance to Path B. 0 disables the sweeps (not recommended).");
      RTX_OPTION("rtx.clusterLod.promotion", float, temporalEpsilon, 0.01f,
                 "Maximum inter-frame drift of the solve samples' pairwise distances for a frame to count as\n"
                 "rigid. A rigid transform (camera OR object motion) preserves pairwise distances, so this stays\n"
                 "~0 for genuinely rigid geometry - moving or still - and spikes only when the mesh itself\n"
                 "deforms. It is what rejects VS-animated captures (e.g. characters skinned in the vertex shader,\n"
                 "which have numBones==0 and so are not caught as skinned) that momentarily fit a rigid transform\n"
                 "on a single frame. Folded into the per-frame rigid verdict, so a candidate must be temporally\n"
                 "rigid across its whole streak to promote, and a promoted instance demotes the moment it deforms.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, restCaptureReference, true,
                 "Rest-capture reference for NON-AFFINE captured meshes: a candidate that stays temporally\n"
                 "static but never fits any single transform of its CPU input data (the VS assembles a\n"
                 "genuinely different shape) gets its capture buffer read back ONCE, is re-clusterized from\n"
                 "that true rendered shape under a space-tagged hash, and its promotion probe re-references\n"
                 "the rest capture - the per-frame solve then fits identity(+motion) and the mesh promotes.\n"
                 "Costs one small readback + one re-clusterization per such mesh per session.");
      RTX_OPTION("rtx.clusterLod.promotion", int, restCaptureStuckFrames, 120,
                 "Consecutive temporally-static frames a candidate must stay above residualEpsilon before the\n"
                 "rest-capture path triggers (avoids re-referencing meshes that are merely briefly stuck).");
      RTX_OPTION("rtx.clusterLod.promotion", int, restClassStuckFrames, 24,
                 "Misfit accruals (temporally-calm fresh solves) a CONTENT CLASS needs before requesting its\n"
                 "own rest reference. Much lower than restCaptureStuckFrames: candidate probes solve every\n"
                 "frame, but a class only accrues evidence during its content's dwell windows (bind-churn\n"
                 "games rotate content across slots). Low is safe - a class promotes only through its own\n"
                 "gate against the captured reference, so a premature request costs one readback, never a\n"
                 "wrong verdict.");
      RTX_OPTION("rtx.clusterLod.promotion", int, restClassMaxRefs, 4,
                 "Identity-by-fit sibling cap per content-class bucket. The spread signature cannot separate\n"
                 "non-affine-equivalent contents (different SHAPES share a bucket), so class membership is\n"
                 "decided by FIT: an instance persistently misfitting sibling N's own reference advances to\n"
                 "sibling N+1 (existing first, else a new capture from the misfitting member). This bounds\n"
                 "how many sibling references one bucket may hold; past the cap the chain REJECTS and retries\n"
                 "after restRejectRetryFrames as before.");
      RTX_OPTION("rtx.clusterLod.promotion", float, demoteHysteresis, 2.0f,
                 "Residual multiplier an ALREADY-PROMOTED instance is allowed before demoting to Path B\n"
                 "(candidates still need the strict residualEpsilon to promote). A rigid mesh whose residual\n"
                 "sits exactly on the epsilon boundary otherwise flaps promote/demote every few frames\n"
                 "(observed: residual 0.005-0.008 vs eps 0.005 with zero temporal drift). Applies to the\n"
                 "per-frame solve verdict and the periodic full-mesh sweep alike. 1.0 disables hysteresis.");
      RTX_OPTION("rtx.clusterLod.promotion", std::string, dumpGeometryHash, "",
                 "DIAGNOSTIC RAW DUMP (empty = off). Hex geometry hash (no 0x). At probe build the 64 solve\n"
                 "samples' REF positions are logged ([PromoDump] ref); each frame the same samples' CAPTURE\n"
                 "positions are read back and logged throttled ([PromoDump] cap). Diffing the two shows the\n"
                 "actual per-vertex displacement field - raw data, no interpretation.");
      RTX_OPTION("rtx.clusterLod.promotion", std::string, traceMaterialHash, "",
                 "DIAGNOSTIC DRAW TRACE (empty = off). One or more hex MATERIAL hashes (no 0x) as shown by the\n"
                 "Remix picker, separated by commas or spaces (e.g. 'a4e20a16f03bf6f8, 3857086b6625afcc').\n"
                 "Every draw whose material matches ANY listed hash is logged at three pipeline stages, throttled\n"
                 "per geometry: [DrawTrace/scene] at SceneManager::processDrawCallState (BEFORE any cluster\n"
                 "filtering - catches draws that are ignored/culled and never reach the cluster system at all),\n"
                 "[DrawTrace/intake] at the cluster manager entry (empty-hash filtering), and [DrawTrace/provider]\n"
                 "at the provider (the fast-path early-returns: already-known, mutating-skip, ineligible). Use it\n"
                 "to see exactly how the game draws a user-identified surface - captured vs not, which sub-meshes\n"
                 "reach the system. Read live per draw, so it can be changed at runtime without a rebuild.");
      RTX_OPTION("rtx.clusterLod.promotion", float, eigenEpsilon, 0.02f,
                 "Option 1 (permutation-invariant rigidity): max allowed mismatch between the capture\n"
                 "cloud's TRACE-NORMALIZED sorted covariance eigenvalues and the PREDICTION (last-RIGID M\n"
                 "applied to the reference's full-set covariance: eig(A*refCov*A^T)) before an eigen sweep\n"
                 "counts as drifting. The triple is computed over the FULL referenced capture set as a SUM\n"
                 "(order-free), so it is immune to the engine re-batching the vertex ORDER - the failure\n"
                 "that poisons every per-index signal (residual, temporal drift) and made perfectly static\n"
                 "buildings demote on phantom deformation and stick on Path B. Comparing against the\n"
                 "prediction (NOT the previous sweep) makes it immune to the slot's capture content\n"
                 "ALTERNATING between draws (measured: temporal sweep-to-sweep comparison false-demoted at\n"
                 "drift median 0.18 on static geometry); eig(R*X*R^T) = eig(X) keeps it immune to rigid\n"
                 "motion after the proven fit, trace-normalizing to scale levels, and A carries any\n"
                 "anisotropic placement bake. Per-index residual/temporal signals only SCHEDULE\n"
                 "sweeps (suspicion -> verify), never demote. 0 disables the tolerance (any drift counts).");
      RTX_OPTION("rtx.clusterLod.promotion", int, eigenDemoteSweeps, 3,
                 "Consecutive CLEARLY-DRIFTING eigen sweeps before a PROMOTED content class un-promotes to\n"
                 "Path B (direction 2: the demote verdict is per CONTENT CLASS, not per instance slot).\n"
                 "'Clearly drifting' = drift above eigenEpsilon * demoteHysteresis (the hysteresis band\n"
                 "between eigenEpsilon and that upper bound is a genuine minor content difference - a piece\n"
                 "measured static and rigid but a few % off the shared reference - and is HELD, not demoted;\n"
                 "any sweep at-or-below the band resets the streak). Default 3 keeps a static world promoted:\n"
                 "a steady small offset never demotes, and noise spikes that occasionally cross the upper\n"
                 "bound don't reach 3-in-a-row. Only sustained large drift (real deformation, or a genuinely\n"
                 "different shape) demotes. Suspicion arms the next sweep immediately, so 3 accrues at the\n"
                 "readback cadence (~3*gateLagFrames), not the periodic stagger. LOWER to 1 for games with\n"
                 "rigid->animated transitions (e.g. a character leaving a T-pose) where the fastest possible\n"
                 "demote matters more than tolerating a static variant's drift noise.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, correspondenceScan, false,
                 "DIAGNOSTIC PROBE (no fix, off by default). For every candidate, the solve kernel also runs a\n"
                 "transform-invariant pairwise-distance scan over a fixed table of ref->capture vertex-index\n"
                 "offsets and reports which offset best matches the reference point cloud. This reveals the\n"
                 "ref/cap index skew that scrambles shared-vertex-buffer meshes (the sEff~1 + high-residual\n"
                 "rejects, e.g. buildings) WITHOUT applying any correction. The verdict + offset are appended to\n"
                 "the 'gate REJECTED' log line as scanOff=<n> scanV=<none|impr|COLLAPSE>. Costs a per-candidate\n"
                 "scan while enabled; bounds-clamped so it can never read past the capture buffer.");
    };
  };

  // DIAG: true if matHash appears in rtx.clusterLod.promotion.traceMaterialHash,
  // which accepts one or more hex material hashes separated by commas or spaces.
  // Parses the option string once and re-parses only when it changes (so it stays
  // runtime-tunable without a per-draw allocation). Shared by every DrawTrace probe
  // site so they all match the same list. 0/empty option -> never matches.
  inline bool clusterLodPromoTraceMatchesMaterial(uint64_t matHash) {
    if (matHash == 0) {
      return false;
    }
    const std::string& s = ClusterLodOptions::Promotion::traceMaterialHash();
    if (s.empty()) {
      return false;
    }
    static thread_local std::string s_cachedStr;
    static thread_local std::vector<uint64_t> s_cachedHashes;
    if (s != s_cachedStr) {
      s_cachedStr = s;
      s_cachedHashes.clear();
      size_t i = 0;
      while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\t')) {
          i++;
        }
        size_t j = i;
        while (j < s.size() && s[j] != ',' && s[j] != ' ' && s[j] != '\t') {
          j++;
        }
        if (j > i) {
          try { s_cachedHashes.push_back(std::stoull(s.substr(i, j - i), nullptr, 16)); } catch (...) {}
        }
        i = j;
      }
    }
    for (const uint64_t h : s_cachedHashes) {
      if (h == matHash) {
        return true;
      }
    }
    return false;
  }

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

    // Loader threads (P4c, plan 7.1a): load-time intake for replacement
    // meshes - keys by the PURE geometry hash (same key the draw-time intake
    // derives) and snapshots straight from the replacement's host-visible
    // staging buffers, so cluster processing runs during the load window.
    void onReplacementGeometryLoaded(const RasterGeometry& geometryData);

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

    // REST-CAPTURE readbacks (promotion, non-affine leftovers): records staged
    // capture->host copies and drains retired ones into rest snapshots.
    void processRestCaptureRequests(Rc<DxvkContext> ctx);

    // device address of the generation's shaderio::Geometry table (0 if none);
    // consumed by the path tracer's hit-side cluster fetch via raytrace_args
    uint64_t getGeometriesTableAddress() const;

    // P4c: device address of the promotion matrices array (M/prevM per state
    // slot; 0 while inactive) - consumed by promoted surfaces via raytrace_args
    uint64_t getPromotionStateAddress() const;

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
    // dxvk submission lock. Const ref (P4c): static snapshots register interim
    // templates FIRST and then continue into Path A processing with the same
    // snapshot - the handler only reads.
    bool processAnimatedGeometry(const lodclusters_remix::GeometrySnapshot& snapshot);
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
      uint32_t positionsCount = 0;     // DIAG: valid capture slots from positionsAddress
                                       // (bounds the correspondence scan; 0 = unknown)
      VkDeviceSize positionsBufferOffset = 0;  // byte offset of positionsAddress in positionsBuffer
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
    // P4c fast lane: true while any pending geometry came from its .nvsngeo
    // cache - the batch then uses cacheHitCooldownFrames instead of the full
    // cooldown (cleared whenever the pending batch is consumed)
    bool m_pendingHasCacheHit = false;

    // stats latches: slot lists are reset every onFrameBegin, so the periodic
    // digest (which runs there) must report the counts captured at dispatch
    // time instead of reading the just-cleared lists
    uint32_t m_statsSlotsOpaque = 0;
    uint32_t m_statsSlotsUnordered = 0;
    uint32_t m_statsSlotsPathB = 0;

    // F7: geometryIds whose LOD build produced a single LOD level - routed to
    // the classic (bucket-merged) BLAS path at render time; rebuilt alongside
    // m_geometryIdByHash
    std::unordered_set<uint32_t> m_trivialGeometryIds;

    // ---- P4c rigid-capture promotion (plan 7.7 spec) ----
    struct PromotionCandidate {
      uint64_t probeVa = 0;
      uint32_t vertexCount = 0;
      uint32_t stateSlot = 0;
      enum class Phase : uint32_t { Probing, GateScheduled, GateRunning, Promoted, Rejected } phase = Phase::Probing;
      uint32_t gateFrames = 0;
      bool loggedTemporalHold = false;  // DIAG: one-shot "held by temporal gate" log

      // [PromoLat] latency-breakdown instrumentation: decomposes first-sight ->
      // Path A into "waiting to be solved" (instance not drawn) vs "drawn but
      // grinding the gate". createdFrame/Time stamp candidate adoption;
      // solveCount counts frames the GPU actually advanced this slot's solve
      // (state.lastFrame stepped) - if solveCount << framesElapsed the instance
      // simply was not on screen, which is inherent and not a pipeline fix.
      // gateScheduledFrame marks the Probing->Gate handoff; streakResets counts
      // how often a non-rigid spike knocked the rigid streak back to 0.
      uint32_t createdFrame = 0;
      std::chrono::steady_clock::time_point createdTime{};
      uint32_t solveCount = 0;
      uint32_t lastCountedSolveFrame = 0;  // last state.lastFrame we tallied
      uint32_t gateScheduledFrame = 0;
      uint32_t streakResets = 0;
      uint32_t prevRigidStreak = 0;
      // ---- REST-CAPTURE reference (non-affine leftovers) ----
      // routeHash: residency hash instances route to when promoted (0 = the
      // candidate's own key). Set to the space-tagged rest hash once the probe
      // was rebuilt from the captured rest pose.
      uint64_t routeHash = 0;
      // static-stuck detector: consecutive updatePromotionStates passes with
      // residual > eps while temporally static; resets when the mesh moves.
      uint32_t stuckFrames = 0;
      enum class RestState : uint32_t { None, Requested, Referenced } restState = RestState::None;
      // ---- pinned temporal-probe instance ----
      // The candidate's geometry-level solve compares this frame's capture samples
      // to last frame's on the SAME state slot to measure temporalDeform. The
      // capture buffer is per-INSTANCE (a specific placement's live positions), and
      // the emit used to pick "first Path B slot in arrival order", which is NOT
      // stable frame-to-frame when a geometry has many placed instances. Consecutive
      // frames then sampled DIFFERENT placements, so tDeform spiked every frame and
      // reset the rigid streak - a perfectly-rigid building (residual ~3e-6) sat in
      // Probing for 20+ seconds instead of promoting in a frame or two. Pin the probe
      // to one instance so tDeform reflects that instance's real motion (~0 for a
      // static building). probeBlas == nullptr until first adoption; frameCreated
      // guards BlasEntry* address reuse. Single-instance geometries pin trivially
      // (the fix is a no-op there).
      const BlasEntry* probeBlas = nullptr;
      uint32_t probeBlasFrameCreated = 0;
      // ---- STABLE topology identity (churning-hash immunity) ----
      // This game's captured draws churn their geometry ASSET hash every frame (the
      // capture buffer content feeds the hash), so the SAME pillar mesh appears under
      // a dozen different hashes. The provider dedups by TOPOLOGY key, so only one
      // hash per mesh becomes a candidate - but the per-frame draw carries a DIFFERENT
      // (churned) hash, so a candidate keyed only by geometryHash almost never matches
      // the live draw and is solved only on the rare frames its exact registered hash
      // recurs (observed: pillar promoted after ~2 minutes instead of ~2 seconds).
      // topologyKey is stable across the churn (indicesHash + counts), so the draw
      // side resolves the candidate through m_promoCandidateByTopology every frame.
      uint64_t topologyKey = 0;
    };
    // main-thread after adoption in onFrameBegin. Keyed by the registered geometry
    // hash (residency lookups use it); the draw side resolves through the topology
    // index below because the live draw's hash churns.
    std::unordered_map<uint64_t, PromotionCandidate> m_promoCandidates;
    // stable-topology -> candidate key (registered geometryHash). Lets a churned-hash
    // draw find its candidate every frame instead of only when the hash recurs.
    std::unordered_map<uint64_t, uint64_t> m_promoCandidateByTopology;

    // ---- [DrawCoverage] genuine-offscreen vs over-culling probe ----
    // The promotion pipeline only advances a candidate on frames its geometry is
    // DRAWN (routed Path B and solved). Most candidates promote OFF-SCREEN-BOUND
    // (coverage < 0.2) - but that "off-screen" is measured as solveCount/elapsed, so
    // it cannot tell "the game never submitted the draw" (genuinely off-screen) from
    // "the game drew it but the cluster path/culling dropped it before it could solve"
    // (a fixable over-cull). onDrawCallGeometry (CS thread, BEFORE any Remix visibility
    // culling) stamps every captured draw's TOPOLOGY key here; updatePromotionStates
    // joins it by candidate.topologyKey and logs drawnFrames vs solveCount. drawn ~
    // solved => genuinely off-screen (game culls); drawn >> solved => a cluster-side
    // drop worth chasing. Mutex-guarded: written on the CS thread, read on main.
    std::mutex m_promoDrawMutex;
    std::unordered_map<uint64_t, uint32_t> m_promoDrawnFrameByTopo;  // topo -> last frame the game drew it
    std::unordered_map<uint64_t, uint32_t> m_promoDrawnCountByTopo;  // topo -> # distinct frames drawn

    // ---- Path-A timing (churn-proof): first sight -> first ACTUAL Path A render ----
    // Keyed by stable topology key, NOT the churning geometry hash. Measures wall-
    // clock from the first frame a captured mesh is seen to the first frame it really
    // routes Path A (established or pinned). pathAFrame==0 => STILL WAITING (never
    // promoted to Path A yet). materialHash is stored so the report can name a
    // pickable material for the worst offender. onFrameBegin reports the worst
    // reached latency and the longest-still-waiting mesh.
    struct PromoPathATiming {
      uint32_t firstFrame = 0;
      std::chrono::steady_clock::time_point firstSeen{};
      uint32_t pathAFrame = 0;        // 0 = not yet on Path A
      float secondsToPathA = -1.0f;   // <0 = still waiting
      uint64_t materialHash = 0;
      uint64_t lastGeomHash = 0;
    };
    std::unordered_map<uint64_t, PromoPathATiming> m_promoPathATiming;
    // worker -> main handoff of uploaded probes
    struct PendingProbe {
      uint64_t geometryHash = 0;  // candidate key (ORIGINAL hash for rest probes)
      uint64_t probeVa = 0;
      uint32_t vertexCount = 0;
      uint64_t routeHash = 0;     // rest probes: space-tagged residency hash; else 0
      uint64_t topologyKey = 0;   // stable identity for the churning-hash draw side
      int32_t classQ = INT32_MIN; // [ShapeClass] class-scoped rest probe target; INT32_MIN = candidate-level
      int32_t classSubId = 0;     // identity-by-fit sibling the reference belongs to
      bool restored = false;      // [PromoRefs] sidecar restore: adoption skips the class-wipe
    };
    std::mutex m_promoPendingMutex;
    std::vector<PendingProbe> m_promoPendingProbes;
    uint32_t m_promoNextStateSlot = 0;

    // ---- REST-CAPTURE reference machinery (non-affine leftovers) ----
    // Topology retained at probe build so a rest snapshot can be assembled from a
    // capture readback without touching (possibly dead) draw data. Small: indices
    // of captured candidates only.
    struct RetainedTopology {
      std::vector<uint32_t> indices;
      uint64_t indicesHash = 0;
      uint64_t topologyKey = 0;
      uint32_t vertexCount = 0;
      std::string name;
      // [RestCapProbe] weld baseline measured from the BASE (candidate-level)
      // rest capture: counts of exact-duplicate position triples. Welds are
      // affine-invariant, so ANY faithful single-pose capture of this content
      // must reproduce them - a class capture with fewer welded verts is
      // per-vertex corrupt or mid-deform, never a scaled/rotated sibling.
      uint32_t weldGroups = ~0u;  // ~0u = baseline not measured yet
      uint32_t weldVerts = 0;
    };
    std::mutex m_promoTopologyMutex;
    std::unordered_map<uint64_t, RetainedTopology> m_promoTopologyByHash;
    // In-flight capture readbacks: staged in buildPromotionEntries (frame-pose
    // buffer at hand), copied in dispatchBuild (ctx at hand), drained after the
    // frames-in-flight window into a rest snapshot for the provider.
    struct RestCaptureRequest {
      uint64_t geometryHash = 0;       // original candidate key
      Rc<DxvkBuffer> source;           // capture buffer (lifetime held)
      VkDeviceSize sourceOffset = 0;
      uint32_t strideBytes = 0;
      uint32_t vertexCount = 0;
      Rc<DxvkBuffer> staging;          // host-visible readback target
      uint32_t copyFrame = ~0u;        // frame the copy was recorded (~0 = not yet)
      int32_t classQ = INT32_MIN;      // [ShapeClass] class-scoped request; INT32_MIN = candidate-level
      int32_t classSubId = 0;          // identity-by-fit sibling this capture is for
      // [RestCapProbe] the state slot whose solve reads the SAME capture buffer
      // this request copies. When the copy is recorded, a same-frame snapshot of
      // that slot's solve-sample view (promoLastSampleBuffer region) is taken
      // into sampleStaging; the drain bit-compares the two. ~0u = no probe.
      uint32_t stateSlot = ~0u;
      Rc<DxvkBuffer> sampleStaging;    // 64*vec3 solve-view snapshot (null = not probed)
    };
    std::vector<RestCaptureRequest> m_restCaptureRequests;
    // [RestCapProbe] one probe in flight at a time; the pending fields carry the
    // armed request's slot + staging handle from processRestCaptureRequests to
    // the same-frame frameParams fill later in dispatchBuild.
    bool m_restCapProbeInFlight = false;
    uint32_t m_restCapProbeSlotPending = ~0u;
    VkBuffer m_restCapProbeTargetPending = VK_NULL_HANDLE;
    VkDeviceSize m_restCapProbeTargetOffsetPending = 0;

    // ---- [ShapeClass] per-CONTENT-CLASS rest verdicts ----
    // The verdict layer is keyed by WHAT the capture holds (content class =
    // quantized log2 capture-spread), not by WHICH BlasEntry/buffer holds it -
    // the binding between the two is unstable on churning-hash games (verified:
    // a slot's capture content changed scale mid-gate, globally at one frame).
    // Each class of a rest-referenced candidate earns its own verdict, and a
    // class the shared reference cannot fit earns its OWN rest reference
    // (probeVa/routeHash) - so every scale/shape variant can promote, not just
    // the placement the first rest capture happened to be read from.
    struct RestClassState {
      // Content identity = the capture's trace-normalized eigenvalue pair (lam1,lam2)
      // quantized to an eigenEpsilon grid (see quantizeEigClass). It is a rigid- and
      // uniform-scale-INVARIANT descriptor, so the same piece at any placement lands
      // on the same cell EXACTLY - no nearest-merge tolerance needed (the old capSig
      // key was scale-variant and noisy, which is why it needed the 1.5/16 hack).
      int32_t classQ = INT32_MIN;   // quantized eigen-key cell (== match key, exact)
      // Identity-by-FIT sibling id within the bucket. The spread signature cannot
      // separate non-affine-equivalent contents (verified: faithful captures +
      // static content still misfit their class's own reference by a per-content
      // constant - two different SHAPES share one bucket). Membership is therefore
      // decided by fit: an instance that persistently misfits sibling N's
      // reference advances to sibling N+1 (existing first, else freshly minted,
      // each running the full ladder: shared probe -> own capture). subId is the
      // instance cursor's target; references/residency are salted per sibling.
      int32_t subId = 0;
      enum class Phase : uint32_t { Probing, GateScheduled, GateRunning, Promoted, Rejected };
      Phase phase = Phase::Probing;
      // which reference this class solves against: the candidate's shared rest
      // probe, or (after the shared one failed it) its own class-scoped one
      enum class Ref : uint32_t { CandidateProbe, Requested, Own };
      Ref ref = Ref::CandidateProbe;
      uint64_t probeVa = 0;         // own reference (Ref::Own); 0 = candidate's
      uint64_t routeHash = 0;       // own residency hash (Ref::Own); 0 = candidate's
      uint32_t vertexCount = 0;     // gate range for the own probe
      uint32_t gateFrames = 0;
      uint32_t stuckFrames = 0;
      uint32_t gateStateSlot = ~0u; // instance slot whose capture the in-flight gate reads
      uint32_t rejectedFrame = 0;   // retry cooldown (restRejectRetryFrames)
      bool captureStaged = false;   // Ref::Requested: readback already staged
      // wedge guard: frame of the last gate schedule/tick. A gate whose owning
      // slot's content swapped away (or left the scene) would wait forever -
      // stale Gate* phases reset to Probing after 4*gateLagFrames.
      uint32_t lastGateTickFrame = 0;
      // Option 1: the judging slot's eigFrame AT GATE EMISSION. The slot's
      // status also receives per-instance eigen sweeps, so the gate verdict is
      // only valid once eigFrame has ADVANCED past this mark (else a stale
      // instance-sweep verdict would be misread as the gate's).
      uint32_t gateEigMark = 0;
      // ---- direction 2 (content governs routing): per-CLASS demote streak ----
      // The demote verdict is a property of the CONTENT, not the slot. This game
      // multiplexes many distinct pieces through one BlasEntry slot, so a per-slot
      // demote flag thrashed every time a slot's content changed piece (the 122:2
      // Path_B:Path_A split). Instead, every slot that currently holds this class
      // contributes its eigen-sweep verdict here: a drifting sweep (capture shape
      // vs the last-RIGID-M prediction, permutation-invariant) increments the
      // streak, a matching sweep resets it. When a PROMOTED class accrues 3
      // CONSECUTIVE drifting sweeps (from any of its members) it un-promotes -
      // demoting ALL its slots to Path B together - which is genuine deformation
      // of that content, never a slot simply rebinding to a different piece.
      uint32_t eigenDriftStreak = 0;
      // driftDemoted: the class's members currently route Path B because the
      // content DEFORMED (3 consecutive drifting sweeps), NOT because it hasn't
      // proven itself yet. This is the demote authority for NON-rest candidates:
      // their content routes Path A by DEFAULT (the candidate is already proven
      // rigid against its object-space probe - forcing every piece back through a
      // fresh per-class gate before routing A cost tens of seconds at this game's
      // frame rate and effectively promoted nothing). Only a class that eigen-
      // drifts flips this and drops to Path B; a clean sweep clears it and the
      // content routes A again next frame. Rest candidates ignore this and gate
      // on `phase == Promoted` (they need their own captured reference first).
      bool driftDemoted = false;
    };
    // candidate key -> its content classes (small vectors, linear exact-match)
    std::unordered_map<uint64_t, std::vector<RestClassState>> m_restClassesByCandidate;
    // Quantize the capture's trace-normalized eigenvalue pair to an eigenEpsilon
    // grid cell = the content-class id. Stable invariant -> the same shape always
    // maps to the same cell, so classes match EXACTLY (no tolerance). INT32_MIN
    // when the eigen key is not yet available (degenerate / no sweep landed).
    static int32_t quantizeEigClass(float lam1Hat, float lam2Hat);
    // find (or create) the class with this EXACT quantized eigen-key cell. subId
    // selects the identity-by-fit sibling for the rare case where two genuinely
    // different shapes share an eigen cell (the pair is necessary, the gate fit
    // is sufficient). Returned pointer is invalidated by the next creating call
    // for the same candidate - resolve, use, drop.
    RestClassState* resolveRestClass(uint64_t candidateHash, int32_t classQ, int32_t subId, bool createIfMissing);
    // per-INSTANCE state slots for PROMOTED instances (plan risk R21: M is per
    // instance - every captured instance's buffer carries its own transform -
    // so patch/prevM state must never alias across instances; the candidate's
    // own slot serves only the geometry-level probe/gate verdict).
    // sweepPending/sweepLagFrames track this slot's in-flight eigen sweep; the
    // verdict feeds the slot's CONTENT CLASS, which is the routing+demote
    // authority (direction 2 - see the field comments below).
    struct PromoInstance {
      uint32_t stateSlot = 0;
      uint32_t sweepLagFrames = 0;
      bool sweepPending = false;
      // ---- direction 2: no per-slot demote flag ----
      // Routing and demotion are now a property of the slot's CONTENT CLASS
      // (RestClassState), not the slot. A slot routes Path A iff the cell it
      // currently classifies into is Promoted; the per-class eigenDriftStreak
      // carries the demote verdict. The old per-slot `demoted` flag thrashed
      // whenever a slot rebound to a different piece and is gone.
      // ---- REST verdicts moved to RestClassState ([ShapeClass]) ----
      // Formerly per-instance (RestPhase et al.), which keyed verdicts by
      // BlasEntry - an identity the churning-hash draw matching does NOT keep
      // stable (a slot's capture content can be rebound to a different
      // placement mid-verdict). Verdicts now live on the candidate's content
      // classes (m_restClassesByCandidate); this slot contributes evidence to
      // whichever class its CURRENT content classifies into (contentClassQ).
      // isRestWorld: this slot serves a rest-referenced candidate - set at
      // emit time; the verdict pass only reads class evidence from these.
      bool isRestWorld = false;
      // NV-DXVK: pinned Path A residency. Once an instance PROMOTES, its identity
      // is this stable BlasEntry* (the draw-call cache keeps the same BlasEntry
      // across camera moves), NOT the asset hash. This game's captured draws
      // produce an asset hash that is unstable frame-to-frame under camera motion,
      // so re-deriving residency from it every frame is exactly what dropped
      // promoted meshes back to Path B on any camera move. Cached at establish
      // time so the per-frame route reads the id straight off the slot instead of
      // an m_geometryIdByHash lookup by the churning hash. residentGeometryId == ~0u
      // means "not yet pinned"; geometryHash is the ingest-time key; blasFrameCreated
      // guards against BlasEntry* address reuse.
      uint32_t residentGeometryId = ~0u;
      uint64_t geometryHash = 0;
      uint32_t blasFrameCreated = 0;
      // [RestGateTrace] the capture buffer address this slot's solve/gate read
      // on the most recent emit (diagnostic only - address is NOT identity).
      uint64_t lastCaptureVa = 0;

      // ---- [ShapeClass] content-class identity (from the capture's SHAPE) ----
      // The eigen sweep readback carries the capture's trace-normalized eigenvalue
      // pair (eigLam1Hat, eigLam2Hat) - a rigid- and uniform-scale-INVARIANT shape
      // descriptor. quantizeEigClass() maps it to a stable cell id = contentClassQ,
      // which identifies WHICH content this slot's capture holds independent of the
      // unstable BlasEntry/buffer binding. Unlike the old scale-variant capSig it is
      // stable across placements, so classes match EXACTLY (no merge tolerance).
      // INT32_MIN = not yet classified.
      int32_t contentClassQ = INT32_MIN;
      // identity-by-fit cursor: which SIBLING reference of the cell this instance
      // is currently judged against (see RestClassState::subId). Resets with
      // contentClassQ - a cell move re-enters at sibling 0.
      int32_t classSubId = 0;
      uint32_t lastClassifiedFrame = 0;  // last state.lastFrame consumed
      // swap confirmation: a single divergent read can be a transient garbage
      // capture (mid-upload rename), so a class change commits only after 2
      // CONSECUTIVE eigen sweeps land on the same new cell.
      int32_t pendingClassQ = INT32_MIN;
      uint32_t pendingClassCount = 0;
      // ---- Option 1 (eigen verdict) ----
      // lastEigenFrame: the eigFrame of the last CONSUMED eigen sweep readback
      // (a new eigFrame = a fresh verdict landed). eigenSuspect: the per-index
      // signals (residual/tDeform - permutation-POISONED, never demote directly)
      // flagged this slot; buildPromotionEntries schedules an eigen sweep to
      // verify. sweepPending doubles as the in-flight guard for eigen sweeps.
      // The demote streak itself lives on the CLASS now (RestClassState::
      // eigenDriftStreak), aggregated across whichever slots hold that content;
      // this slot only reports each sweep verdict into its current class.
      uint32_t lastEigenFrame = 0;
      bool eigenSuspect = false;
      // [EigSettle] VERIFY PROBE: was the source vertex buffer being rewritten in
      // place (updatedInPlace) on the frame the currently in-flight eigen sweep was
      // ENQUEUED? Stamped at enqueue, read when that sweep's result lands, to prove
      // whether unsettled captures are what produce the off-cell (phantom) eigen
      // readings that flap stable geometry to Path B. Diagnostic only (no gating yet).
      bool sweepSrcUnsettled = false;
    };
    std::unordered_map<const BlasEntry*, PromoInstance> m_promoSlotByBlas;
    // per-frame kernel work items (built in dispatchBuild, consumed by
    // recordFrame the same call) + readback scratch
    std::vector<lodclusters_remix::PromotionEntry> m_framePromoEntries;
    std::vector<lodclusters_remix::PromotionStateView> m_promoStates;
    bool m_promoStatesValid = false;
    uint32_t m_statsPromoted = 0;
    uint32_t m_statsPromoRejected = 0;
    // promotion-solve diagnostics, aggregated per updatePromotionStates pass and
    // reported by logFrameTimes (gameplay-gated + throttled). See PromoStatus.
    float m_diagMaxAffineNonRigid = 0.0f;  // worst affine shear/scale seen (0 == rigid)
    uint32_t m_diagProbeZeroSlots = 0;     // candidates that hit the probeVa==0 guard
    uint32_t m_diagDegenSlots = 0;         // candidates whose fit was rejected degenerate
    // forensics for the worst (max non-rigid) slot: names WHY it degenerates
    uint64_t m_diagWorstGeom = 0;          // geometry hash of the worst slot
    float m_diagWorstRefVar = 0.0f;        // its ref-sample spread (~0 == coincident refs)
    uint32_t m_diagWorstSampleN = 0;       // its probe.sampleCount
    // reject-reason histogram over degenerate slots (index = degenReason enum in
    // promotion_solve.comp: 1=coincident 2=rank-deficient 3=non-orthogonal
    // 4=refVar~0 5=scale-overflow 6=non-finite; [0] unused). Names which stage is
    // actually costing promotions instead of guessing from the single worst slot.
    uint32_t m_diagReasonHist[7] = {};

    // worker thread (P4c): probe precompute (samples + Gram pseudoinverse in
    // doubles) + upload through the template system's callback-locked path
    void buildAndUploadPromotionProbe(const lodclusters_remix::GeometrySnapshot& snapshot);
    // onFrameBegin: adopt pending probes, read verdicts, run the state machine
    void updatePromotionStates();
    // dispatchBuild: emit this frame's solve/gate/patch entries
    void buildPromotionEntries();
    // Resolve the m_promoCandidates key for a live draw's geometry. This game's
    // captured asset hash churns every frame, so try the direct hash (stable-hash
    // geometry) then fall back to the stable topology index. 0 = no candidate.
    uint64_t resolvePromoCandidateKey(const RasterGeometry& geometryData) const;
    // SceneConfig cache digest the generation was built from; appends require
    // the current config to still resolve to it
    std::string m_generationConfigDigest;
    uint32_t m_lastGenerationFrame = 0;
    uint32_t m_generationCount = 0;

    // [GenTrace] residency-stall instrumentation: per-frame diagnosis of WHY
    // pending geometries did or did not join the render generation this frame.
    // Wall-clock 1s throttle for the "still waiting" heartbeat + last-seen
    // state so transitions log immediately (idle<->pending<->appending).
    std::chrono::steady_clock::time_point m_lastGenTraceLog{};
    size_t m_lastGenTracePending = 0;
    uint64_t m_lastProviderQueueDepth = 0;
    uint64_t m_lastProviderProcessed = 0;
    uint64_t m_lastProviderSubmitted = 0;
    uint32_t m_genTraceEnqueuedFrame = 0;   // frame the oldest still-pending hash was drained
    uint32_t m_genTraceDeferrals = 0;       // consecutive frames pending>0 without an append

    // [ScanProbe] throttle for the periodic correspondence-scan readout of the
    // single targeted candidate (Promotion::dumpGeometryHash).
    uint32_t m_scanProbeLastFrame = 0;
    // [ShapeClass] 1s throttle for the per-candidate content-class histogram.
    std::chrono::steady_clock::time_point m_lastShapeClassLog{};
    // [SwapDebounce] verdict on whether single-solve transient reads exist:
    // abandoned = pending swaps that never confirmed (1-solve excursions -
    // proven transients); committed = swaps that held for 2+ solves (real
    // content changes). abandoned == 0 across runs => the debounce guards a
    // ghost and should be deleted; abandoned > 0 => a real transient-read
    // path exists and deserves a root-cause chase (buffer lifetime/rename).
    uint32_t m_swapPendingAbandoned = 0;
    uint32_t m_swapCommitted = 0;

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
    // promoDiag: wall-clock 1s throttle, emitted unconditionally every frame
    std::chrono::steady_clock::time_point m_lastPromoDiagLog{};

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
