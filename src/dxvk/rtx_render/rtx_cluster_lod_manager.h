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

    RTX_OPTION("rtx.clusterLod", bool, debugScanTlasInstanceRefs, true,
               "Debug ([TlasRefScan]): every frame, mirror the WHOLE Vulkan AS instance buffer to host memory\n"
               "and scan every VkAccelerationStructureInstanceKHR for a 0 accelerationStructureReference - the\n"
               "null BLAS reference that faults the full TLAS BUILD at GPU VA=0. Unlike the CPU-side merged/Path-A\n"
               "scans this covers ALL regions (merged / PointInstancer / cluster) of ALL three TLAS types\n"
               "(Opaque/Unordered/SSS) and maps each hit back to its region, local index and instanceCustomIndex.\n"
               "The scan is one frame lagged (it reads the previous frame's completed copy) so a persistent 0-ref\n"
               "shows in the frames leading up to the device-lost.");

    RTX_OPTION("rtx.clusterLod", int, debugScanTlasInstanceRefsHeartbeatFrames, 128,
               "[TlasRefScan]: when no null reference is found, log a 'clean' heartbeat with the per-region\n"
               "instance counts every N frames so the scanner's liveness and the layout evolution are visible.\n"
               "0 disables the heartbeat (null-reference hits are always logged).");

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
      RTX_OPTION("rtx.clusterLod.render", bool, animatedTopologyExcludesPathA, false,
                 "THE fix for the foreign-clusterId (4096+) symptom, proven by [DualRoute]: a mesh topology\n"
                 "that is animated-registered (a skinned/deforming instance renders it Path B) must not ALSO\n"
                 "have a static sibling instance rendered Path A resident - the resident CLAS (id 4096+) and\n"
                 "the Path B surface then coexist in the shared cluster TLAS region and a ray commits the\n"
                 "Path A id under the Path B surface. When true, a static instance whose topology is animated\n"
                 "is routed to the classic BLAS instead of Path A, making the topology single-path. Default\n"
                 "false so the symptom + [DualRoute] can be observed; set true to apply the fix.");
      RTX_OPTION("rtx.clusterLod.render", int, pathHysteresisFrames, 0,
                 "Routing hysteresis for the foreign-clusterId (4096+) fix. When >0, a static instance that\n"
                 "was Path A (resident) within this many frames is held on the classic BLAS instead of dropping\n"
                 "to the interim Path B template on a transient residency-lookup miss (generation swap /\n"
                 "streaming churn). That A->B flip is what lets a fresh Path B surface commit the lingering\n"
                 "Path A resident CLAS. 0 disables (default, so the [DualRoute] diagnostic can reproduce the\n"
                 "symptom). Skinned/captured geometry is genuinely deforming and is never subject to this.");
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
      RTX_OPTION("rtx.clusterLod.animated", int, dbgClusterIdOffsetSentinel, 0,
                 "DIAGNOSTIC (revert). Adds this constant to the per-frame Path B instantiate clusterIdOffset\n"
                 "(normally 0, so the committed clusterId == the template's baked globalClusterBase+c, <=1792).\n"
                 "Recommended value 32768 (0x8000): it shifts every committed id into a clearly-tagged band\n"
                 "while staying inside the >=65536-entry cluster table (the renderer clamps it to\n"
                 "clusterTableCapacity - clusterTableCount - 1 so it can never OOB-fault the hit-side lookup;\n"
                 "1000000 device-lost at frame 76 by reading +8MB of unmapped memory before this clamp).\n"
                 "Read the failing surfaces' [ClusterDecodeProbe] clusterId: if it becomes (base + sentinel)\n"
                 "with base <=1792, the resident 4096+ did NOT come from this instantiate; if it becomes\n"
                 "(4096 + sentinel), the driver baked a resident id despite the CPU input. Everything goes\n"
                 "black while non-zero (shifted ids miss the populated table) - diagnosis only.");
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
      RTX_OPTION("rtx.clusterLod.promotion", bool, atomicDemotion, false,
                 "SUPERSEDED by deformingPromotedToClassic (the collision is rigid-promoted + deforming siblings,\n"
                 "not promoted + demoted - demotions were zero). Kept for comparison. Per-instance demotion lets one instance of\n"
                 "a mesh stay promoted (Path A resident CLAS, id 4096+) while a sibling instance demotes to Path B\n"
                 "(deforming) - so the SAME topology sits on BOTH paths in one frame and a ray commits the Path A\n"
                 "id under the Path B surface ([DualRoute] proved this). When true, demotion is atomic per\n"
                 "GEOMETRY: if ANY instance of a mesh is demoted, NONE of its instances promote - the whole mesh\n"
                 "renders Path B - so each topology is single-path. Set false to restore per-instance demotion\n"
                 "and reproduce the symptom for comparison.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, deformingPromotedToClassic, false,
                 "Workaround (default OFF while the CommittedInstanceID/surfaceIndex probe diagnoses the\n"
                 "coexistence so deforming can stay on Path B). Routes deforming instances of a promoted topology\n"
                 "to classic. Set true to apply the workaround. proven by [DualRoute] (same\n"
                 "geomHash on both paths). When a mesh is promoted, a rigid instance renders its Path A resident\n"
                 "CLAS (id 4096+) while a DEFORMING instance of the SAME mesh renders Path B - the two coexist in\n"
                 "the cluster TLAS and a ray commits the Path A id under the Path B surface. When true, the\n"
                 "deforming instances of a promoted mesh render CLASSIC instead of Path B, so that mesh has no\n"
                 "Path B surface to mis-resolve - while its rigid instances KEEP their Path A promotion (unlike\n"
                 "the coarse atomicDemotion). Set false to reproduce the symptom.");
      RTX_OPTION("rtx.clusterLod.promotion", int, fullSweepIntervalFrames, 32,
                 "Steady-state full-mesh residual sweep cadence per PROMOTED instance (plan 7.7, risk R20):\n"
                 "the sparse per-frame solve can miss a VS animating a small vertex subset, so every promoted\n"
                 "instance re-runs the every-vertex sweep on this interval (staggered by state slot). A failing\n"
                 "sweep demotes that instance to Path B. 0 disables the sweeps (not recommended).");
      // [PromoSmear] DIAGNOSTIC probe for the moving-cinematic RAW smear (per-instance, Path-A-only).
      // Hypothesis: a promoted instance that is NON-RIGID this frame (a VS-animated prop) whose sparse
      // 32-sample rigidity check happens to pass gets a least-squares rigid M patched; Path A then draws
      // fitted = M*obj which disagrees with the true capture PER-VERTEX = a visible stretch/smear on that
      // one instance. The full-vertex gate WOULD catch it, but it only runs every fullSweepIntervalFrames
      // and demotes through the ~gateLag readback ring - tens of seconds at 1.6fps. When true this forces
      // the full gate EVERY frame for EVERY promoted instance (so gateResidualRel is fresh this frame) and
      // logs, per promoted slot, sparse residualRel vs full-mesh gateResidualRel and the placement gap.
      // PURE MEASUREMENT: it does NOT set sweepPending, so routing/demotion is unchanged (the smear stays
      // on screen while the log confirms which slots carry gate>>eps while residualRel<=eps). Off by default.
      // [RotStabilize] THE rotation-underdetermination fix. A planar/collinear/symmetric promoted mesh
      // gives the affine solve a sample cloud that does NOT constrain the rotation DOF: it fits every
      // sample yet lands on an arbitrary valid rotation that JUMPS frame-to-frame, flinging the mesh
      // extremities (the [SmearProbe] 34u spikes on iso~0 slots = the raw smear). When the solve-sample
      // isotropy (det(cov)/(trace/3)^3, [ConditionProbe]) is below this threshold, blend the newly-solved
      // linear part toward the previous frame's rotation, weighted by how degenerate it is (full anchor at
      // iso 0, none at iso >= threshold), and re-pin the centroid so placement is unchanged. Well-conditioned
      // meshes (iso above threshold) are untouched. 0 disables the fix entirely (A/B the raw smear).
      RTX_OPTION("rtx.clusterLod.promotion", float, rotStabilizeIso, 0.0f,
                 "[RotStabilize] anchor the solved rotation to the previous frame's when the solve-sample\n"
                 "isotropy is below this. DEFAULT OFF (0): the LEGO Batman cinematic disproved the premise -\n"
                 "the low-iso smearing meshes were GENUINELY rotating props, and anchoring lags their real\n"
                 "spin (visible pose error). Keep 0 unless a static-scene underdetermined-rotation case is\n"
                 "actually measured ([SmearProbe] fling with low iso on geometry PROVEN world-static).");
      // [TeleportClamp] THE cinematic-smear fix (root cause proven 2026-07-14). The game animates
      // cinematic props in DISCRETE STEPS; at streaming framerates (1.6fps) one step is 10-34 world
      // units ([SmearPix]: motionW 24-34u paired with |curT-prevT|=0 the NEXT frame = jump then hold).
      // The promoted MV honestly reports the jump and DLSS drags a full step of shading = the raw
      // smear. Path B looked clean only because captured geometry's CPU prev==cur matrices report
      // ZERO motion there, so DLSS treats the step as a disocclusion and rejects history. This clamp
      // gives Path A the same (correct) disocclusion semantics WITHOUT leaving Path A: when the
      // per-frame per-vertex motion exceeds the object's own radius x this factor, the solve writes
      // prevM = M (zero MV). Placement is untouched - the object renders at its exact new pose.
      // Radius-relative = scale-free and game-agnostic.
      RTX_OPTION("rtx.clusterLod.promotion", float, teleportClampRadii, 1.0f,
                 "[TeleportClamp] report zero motion (disocclusion) for a promoted instance whose per-frame\n"
                 "motion exceeds this multiple of its own bounding radius - a discrete animation step no\n"
                 "temporal accumulator can reuse (the stop-motion cinematic smear at low fps). 0 disables;\n"
                 "lower = clamp smaller jumps (more history rejection), higher = tolerate faster motion.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, probeGateEveryFrame, false,
                 "[PromoSmear] diagnostic: force the full-mesh residual gate every frame for every promoted\n"
                 "instance and log sparse-vs-full residual per slot, to confirm the moving-cinematic smear is\n"
                 "per-instance deformation the sparse solve misses. Measurement only (does not change routing).");
      // Stable-identity cell size (plan §2 Option L). The promo state slot (GPU M/prevM
      // continuity) is recovered per frame by quantizing the instance's reconstructed world
      // centroid into a cell of this size, instead of the churning RtInstance id / BlasEntry
      // pointer. A static prop lands in the same (geometryHash, cell) every frame -> it recovers
      // its WARM slot instead of a cold one -> no one-frame zero-motion snap ([ColdPromo]).
      // Constraint: >= the reconstruction wobble amplitude ([MotionProbe] motionDelta, ~0.30
      // world units observed) but < half the min spacing between two instances of the same
      // asset (else two distinct props share a cell/slot and swap motion). Size it from
      // [MotionProbe] data (Phase 0). Captured-geometry-scoped only; does NOT touch lights or
      // rtx.uniqueObjectDistance.
      // Churn reduction (keeps promoted geometry warm across off-screen gaps). SceneManager GC
      // keeps a promoted mesh's BlasEntry alive this many frames after it was last drawn (vs the
      // global numFramesToKeepBLAS=1), so a mesh that leaves and re-enters view reuses its SAME
      // BlasEntry -> its promo slot pin survives -> no cold re-promote snap. Bound it so truly-gone
      // geometry still frees eventually. 0 disables (revert to the global GC lifetime).
      RTX_OPTION("rtx.clusterLod.promotion", int, promotedKeepFrames, 300,
                 "Frames to keep a promoted mesh's BlasEntry alive after it leaves view, so re-entry\n"
                 "reuses the same BlasEntry and stays warm (no cold re-promote snap). 0 = use the\n"
                 "global numFramesToKeepBLAS lifetime.");
      RTX_OPTION("rtx.clusterLod.promotion", int, gapMaxFrames, 1,
                 "Max solve-frame gap for which a promoted slot's retained transform is still trusted\n"
                 "as its PREVIOUS transform (motion vector). A promoted instance that leaves Path A\n"
                 "(off-screen, or a Path A<->B flip) and re-renders N frames later re-solves with a\n"
                 "retained M from N frames ago; reusing it as prevM turns the motion vector into N\n"
                 "frames of camera drift in one frame - the smear that runs the whole moving cinematic\n"
                 "([ChurnDiag] gappedStale, worstGap 29-157f). Beyond this many frames, prevM falls back\n"
                 "to the current M (zero motion), correct for a reappearance/disocclusion and bounded\n"
                 "instead of the stale-M spike.\n"
                 "Default 1 = STRICT CONTIGUITY: prevM is trusted only when the slot actually solved the\n"
                 "immediately preceding frame (solveGap == 1), so retained M IS last frame's M and the\n"
                 "motion-vector lineage is exact. The old default 2 permitted the every-other-frame\n"
                 "(solveGap == 2) case to reuse a 2-frame-stale M as 'last frame's', which the LEGO Batman\n"
                 "cinematic proved is the residual raw smear: [PromoStale] logged VISIBLE prevM lineage\n"
                 "breaks (contErr up to 2.4u) on stationary geometry, dragging the DLSS-reprojected shading\n"
                 "across a correctly-placed surface (raw-visible because DLSS consumes the MV even with the\n"
                 "denoiser off). Set very high (e.g. 100000) to A/B the old stale-M behavior.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, churnDiag, true,
                 "[ChurnDiag] per-frame instrumentation of WHY promoted instances leave Path A during\n"
                 "streaming (the churn behind the residual gapped-prevM smear). Logs a routing census\n"
                 "(pin hit / rebuilt / capacity / establish-fail reasons), the per-slot render-gap split\n"
                 "(fresh-warmup vs gapped-everSolved = stale-prevM smear risk), and BlasEntry-rebuild and\n"
                 "capacity-overflow counts. Only logs on frames with promotion activity (skips menus/idle).\n"
                 "Diagnostic - default on for this investigation; revert with the rest of the [Churn*] probes.");
      RTX_OPTION("rtx.clusterLod.promotion", bool, warmup, true,
                 "1-frame promotion warmup (default ON). A freshly promoted instance's state slot has\n"
                 "no GPU transform history, so its first Path A solve seeds prevM = curM (zero motion) -\n"
                 "a Path B->A flip then SMEARS for one frame (correct position, but the denoiser blends\n"
                 "stale history onto a pixel it thinks did not move). Under streaming churn this fires on\n"
                 "many instances per frame. With warmup on, a genuinely fresh slot stays on Path B for one\n"
                 "more frame and is solved UNPATCHED, so it promotes the NEXT frame with a warm prevM =\n"
                 "last frame's solved M -> correct motion vector, no smear. Recovered (warm) slots are\n"
                 "unaffected. Set False to A/B the smear (off => [ColdPromo] events return with motionDelta 0).");
      RTX_OPTION("rtx.clusterLod.promotion", bool, resolveSkip, false,
                 "Per-frame re-solve skip: when a promoted slot's cached transform from last frame\n"
                 "still reproduces this frame's vertex-capture within residualEpsilon, reuse it and\n"
                 "skip the full affine solve (a perf shortcut that assumed most promoted geometry\n"
                 "sits still). DEFAULT OFF - THE motion-lag fix. The captured geometry is\n"
                 "reconstructed CAMERA-RELATIVE (inv(D3DTS)), so under camera motion every capture\n"
                 "drifts a little each frame; the skip's 'still fits' test is a rigidity test, not a\n"
                 "stillness test, so a slowly-drifting object passes it, the kernel FREEZES its M and\n"
                 "forces prevM = M (zero motion) - the promoted geometry then trails the camera and\n"
                 "snaps when motion stops, with a dead motion vector (smear). Proven by the game log\n"
                 "([PromoDump] SKIP/static slots 123/145 held ~0.9-2.6u behind their own drifting\n"
                 "capture while SOLVING slots tracked to <0.3u). Off = every promoted slot full-solves\n"
                 "each frame (a handful of 64-thread workgroups - negligible), placement tracks the\n"
                 "capture and prevM = last frame's M gives the correct motion vector. Set True only to\n"
                 "A/B the lag or measure the skip's perf saving.");
      RTX_OPTION("rtx.clusterLod.promotion", float, identityCellSize, 1.0f,
                 "World-space cell size (units) for stable promoted-instance identity (Option L). Quantizes the\n"
                 "reconstructed world centroid to recover a warm GPU transform slot across BlasEntry/RtInstance\n"
                 "churn. Set >= the wobble amplitude ([MotionProbe] motionDelta) and < half the min inter-instance\n"
                 "spacing. Runtime-tunable: sweep it while watching [ColdPromo] collapse to near-zero.");
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

    // Churn reduction: a promoted instance stays WARM only while its BlasEntry lives (the promo
    // slot is pinned by BlasEntry*). The default BlasEntry GC (numFramesToKeepBLAS=1) drops it one
    // frame after it goes off-screen, so re-entry gets a fresh BlasEntry -> cold re-promote (the
    // small snap jitter). isBlasPromoted lets SceneManager::garbageCollection keep promoted
    // BlasEntries alive for a longer bounded window (Promotion::promotedKeepFrames) so the pointer
    // pin survives camera cuts / brief occlusion -> warm re-entry, no cold, and NO aliasing (the
    // pin is pointer-exact, unlike Option L's position recovery). onBlasEvicted prunes the slot
    // when the entry is finally GC'd so nothing leaks past the window.
    bool isBlasPromoted(const BlasEntry* blasEntry) const {
      return m_promoSlotByBlas.find(blasEntry) != m_promoSlotByBlas.end();
    }
    void onBlasEvicted(const BlasEntry* blasEntry) {
      m_promoSlotByBlas.erase(blasEntry);
    }

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

    // NV-DXVK: [GhostSurface] previous-TLAS rays hit LAST frame's BLAS for an instance
    // but resolve it through THIS frame's surface (via surfaceMapping). When the
    // instance's cluster-path affiliation changed between the frames (Path A <-> Path B,
    // e.g. static geometry detected as deforming at load), the current surface's routing
    // flags misdecode the old hit's committed ClusterID - the root of the foreign
    // clusterId 4096+ resolves and the pre-guard VA=0 device-losts. The fix: for each
    // transitioned instance, AccelManager appends a GHOST surface record (copy of the
    // live surface with LAST frame's routing flags/geometryId) and points
    // surfaceMapping[prevIdx] at it, so previous-frame hits decode with the semantics of
    // the BLAS they actually hit. Ghosts are metadata only (no TLAS entry, one frame).
    struct GhostSurfaceRequest {
      RtInstance* instance = nullptr;
      uint32_t prevSurfaceIndex = 0;
      bool prevIsClusterLod = false;
      bool prevIsClusterTemplate = false;
      uint32_t prevClusterGeometryId = 0;
    };

    // clears this frame's ghost requests (call at the start of instance recording)
    void beginInstanceRecording();
    const std::vector<GhostSurfaceRequest>& getGhostSurfaceRequests() const { return m_ghostRequests; }

    // After AccelManager::prepareSceneData + dispatchPointInstancerCulling and
    // before buildTlas: records the per-frame cluster build (traversal, CLAS/BLAS,
    // instance_assign_blas) and copies the patched TlasInstances into
    // AccelManager's instance buffer regions.
    void dispatchBuild(Rc<DxvkContext> ctx, const CameraManager& cameraManager, AccelManager& accelManager);

    // device address of the generation's shaderio::Geometry table (0 if none);
    // consumed by the path tracer's hit-side cluster fetch via raytrace_args
    uint64_t getGeometriesTableAddress() const;

    // device address of the streaming-resident cluster address table
    // (SceneStreaming.resident.clusters), indexed by resident ClusterID; 0 unless
    // streaming is active. The hit-side cluster fetch needs it because the shaderio
    // Geometry's preloadedClusters array is null while streaming (raytrace_args)
    uint64_t getResidentClustersAddress() const;

    // P4c: device address of the promotion matrices array (M/prevM per state
    // slot; 0 while inactive) - consumed by promoted surfaces via raytrace_args
    uint64_t getPromotionStateAddress() const;

    // [SmearPix] device address of the promo status buffer (smear write-back; 0 while inactive)
    uint64_t getPromotionStatusAddress() const;

    // [UvXCheck] DIAGNOSTIC (remove with the smear probes): resolve a shader-side
    // cluster geometry table index (surface.clusterGeometryId & 0x3FFFF) to the
    // geometry's asset hash; 0 if out of range for the active generation
    uint64_t getResidentGeometryHash(uint32_t geometryId) const;

    // P4b: device address of the global animated cluster table (0 if none);
    // consumed by the hit-side Path B primitive remap via raytrace_args
    uint64_t getAnimatedClusterTableAddress() const;

    // NV-DXVK: [ClusterDecodeProbe] Path A (m_slots) vs Path B (m_slotsB) slot counts for a
    // TLAS type this frame, so the readback can classify a committed InstanceIndex into the
    // Path A block [start, start+numA) or Path B block [start+numA, +numB) of the cluster region.
    void getClusterPathSlotCounts(size_t tlasType, uint32_t& numPathA, uint32_t& numPathB) const {
      numPathA = uint32_t(m_slots[tlasType].size());
      numPathB = uint32_t(m_slotsB[tlasType].size());
    }

    // NV-DXVK: [ClusterDecodeProbe] for a committed region slot, return the CPU-written
    // instanceCustomIndex (what the CPU actually bound into that TLAS slot) plus the slot's
    // RtInstance*, so the readback can compare both against the shader-decoded surfaceIndex and
    // pin whether the crossing is in the customIndex WRITE (slotCustom == template surface) or in
    // the committedInstanceIndex->slot MAPPING (slotCustom != decoded surface). pathA selects the
    // m_slots/m_slotInstanceData (Path A) block vs m_slotsB/m_slotInstanceDataB (Path B) block;
    // localIndex is the region-local index within that block. Returns false if out of range.
    bool getRegionSlotBinding(size_t tlasType, uint32_t localIndex, bool pathA,
                              uint32_t& outCustomIndex, RtInstance*& outInstance) const {
      const auto& slots = pathA ? m_slots[tlasType] : m_slotsB[tlasType];
      const auto& data  = pathA ? m_slotInstanceData[tlasType] : m_slotInstanceDataB[tlasType];
      if (localIndex >= slots.size() || localIndex >= data.size()) {
        return false;
      }
      outCustomIndex = data[localIndex].instanceCustomIndex;
      outInstance = slots[localIndex].instance;
      return true;
    }

    // NV-DXVK: [ClusterDecodeProbe] reverse lookup - find which block actually OWNS a given
    // customIndex (== the shader-decoded surfaceIndex of the committed hit). This answers Path A
    // vs Path B for the REAL hit surface WITHOUT relying on committedInstanceIndex->slot mapping
    // (which the writtenCustom!=decoded result proved unreliable). Each instance carries a unique
    // surface index, so at most one block owns it. Returns the region-local index within m_slots
    // (Path A) via localIdxA and within m_slotsB (Path B) via localIdxB, or -1 if absent.
    void findRegionSlotByCustomIndex(size_t tlasType, uint32_t customIndex,
                                     int& localIdxA, int& localIdxB) const {
      localIdxA = -1;
      localIdxB = -1;
      const auto& da = m_slotInstanceData[tlasType];
      for (uint32_t i = 0; i < da.size(); i++) {
        if (da[i].instanceCustomIndex == customIndex) { localIdxA = int(i); break; }
      }
      const auto& db = m_slotInstanceDataB[tlasType];
      for (uint32_t i = 0; i < db.size(); i++) {
        if (db[i].instanceCustomIndex == customIndex) { localIdxB = int(i); break; }
      }
    }

    // NV-DXVK: [ClusterDecodeProbe] for a Path B region slot, resolve the [base, base+count) global
    // cluster-id range its geometry was baked with (via m_slotsB -> framePoseIndex -> m_framePoses ->
    // poseSet -> geometry.globalClusterBase). Lets the readback compare a failing hit's committed
    // clusterId to what THIS cluster should carry: inside range + null table = publish race; outside
    // = foreign id (bake/routing divergence). Also returns the total populated table count via
    // getAnimatedClusterTableCount() on the template system. Returns false if unresolvable.
    bool getPathBExpectedClusterRange(size_t tlasType, uint32_t localIdxB,
                                      uint32_t& outBase, uint32_t& outCount) const;
    uint32_t getAnimatedClusterTableTotal() const;

    // NV-DXVK: [PosBufDual] the mechanism test. For a failing hit's positionBufferIndex, count how
    // many Path A (m_slots: promoted/resident cluster instances) vs Path B (m_slotsB: deforming
    // template instances) live slots carry the SAME posBuf this frame. Both > 0 => the same mesh is
    // dual-routed - a resident instance (its CLAS baked with a 4096+ resident id) coexists with a
    // Path B instance, and a ray on the Path B surface can commit the resident CLAS. This is the
    // coexistence the "committed resident id under a Path B surface" data implies; counting it
    // directly confirms/refutes it at the moment of the failing hit. Also reports whether ANY Path A
    // slot shares the posBuf's geometry surface index (surface sharing vs distinct instances).
    void countSlotsByPosBuf(uint32_t posBuf, uint32_t& outPathA, uint32_t& outPathB) const;

    // NV-DXVK: [PatchRef] read the PATCHED blasReference the GPU wrote for a Path B region slot
    // (from the scene-anim mirror, flat order = m_slotsB Opaque then Unordered), and test whether
    // it lands inside a pose-BLAS pool (its correct home) or outside (foreign - the per-frame
    // cluster_blas_instances patch wrote a non-pose BLAS onto the slot, so a ray traversing this
    // Path B instance reaches a resident CLAS -> committed clusterId 4096+). outRef=0 => unpatched.
    // Returns false if the slot isn't in the mirror (SSS Path B, or count/lag mismatch).
    bool readPathBSlotPatchedBlasRef(size_t tlasType, uint32_t localIdxB,
                                     uint64_t& outRef, bool& outInPosePool, uint32_t& outPoolCount) const;

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
      Rc<DxvkBuffer> positionsBuffer;  // lifetime tracking on the cmd list
    };
    std::vector<FramePose> m_framePoses;
    std::unordered_map<const BlasEntry*, uint32_t> m_framePoseIndexByBlas;
    uint32_t m_frameClusterBudgetUsed = 0;
    std::vector<ClusterSlot> m_slotsB[Tlas::Count];  // geometryId = frame pose index
    std::vector<VkAccelerationStructureInstanceKHR> m_slotInstanceDataB[Tlas::Count];
    std::vector<SssDuplicate> m_sssDuplicatesB;      // flat index into the B Opaque block

    // NV-DXVK: [GhostSurface] per-instance cluster-path affiliation of the last recorded
    // frame (1 = Path A, 2 = Path B) + the surface's clusterGeometryId at that time;
    // recordClusterInstance compares against it to detect transitions. Entries pruned
    // when stale (instance not cluster-recorded for a while).
    struct PathAffiliation {
      uint8_t path = 0;
      uint32_t clusterGeometryId = 0;
      uint32_t frame = 0;
    };
    std::unordered_map<const RtInstance*, PathAffiliation> m_pathAffiliation;
    std::vector<GhostSurfaceRequest> m_ghostRequests;

    // NV-DXVK: [PathCollision] per-frame posBufferIndex -> path (1=A,2=B). The persistent
    // foreign-clusterId misroutes are a Path B surface committing a Path A resident
    // ClusterID (4096+, impossible from Path B whose ids are <1794): the SAME geometry
    // is present as BOTH a Path A resident instance and a Path B template instance this
    // frame, colliding in the shared cluster TLAS region. If a posBufferIndex appears on
    // both paths in one frame, that is the collision. Cleared each beginInstanceRecording.
    std::unordered_map<uint32_t, uint8_t> m_posBufPathThisFrame;

    // NV-DXVK: [DualRoute] THE decisive per-frame diagnostic for the foreign-clusterId
    // (4096+) symptom. Keyed by TOPOLOGY - the common identity of a mesh across its Path A
    // resident CLAS and its Path B pose (posBufferIndex is a WEAK cross-path key: the
    // skinned B-form and static A-form use different position buffers). Records which
    // path(s) each topology took this frame; a topology on BOTH paths in one frame is the
    // resident Path A CLAS (id 4096+) coexisting with a fresh Path B surface, so a ray can
    // commit the Path A id under the Path B surface. Cleared each beginInstanceRecording.
    struct TopoRoute {
      const RtInstance* aInstance = nullptr;   // an instance routed Path A (resident)
      const RtInstance* bInstance = nullptr;   // an instance routed Path B (template/pose)
      uint32_t residentGeometryId = 0;         // Path A cluster-generation geometry id
      uint32_t bOutGeometryId = 0;             // Path B tagged id (pose set)
      uint32_t aPosBuf = 0;
      uint32_t bPosBuf = 0;
      uint8_t aSource = 0;                     // 1=resident-static, 2=promoted
      uint8_t bSource = 0;                     // 3=deforming(skinned/captured), 4=interim
      uint64_t aGeomHash = 0;                  // asset hash of the A instance (unique per mesh)
      uint64_t bGeomHash = 0;                  // asset hash of the B instance
      bool loggedDual = false;                 // one detection per topology per frame
    };
    std::unordered_map<uint64_t, TopoRoute> m_topoRouteThisFrame;
    // NV-DXVK: atomicDemotion fix (real signal) - last frame each topology had a DEFORMING
    // (Path B) instance. If a topology deforms, promotion of its rigid siblings is suppressed
    // so the topology stays single-path (no promoted Path A CLAS + Path B surface coexistence).
    // Persistent; pruned periodically. [DualRoute] proved every collision is PROMOTED + deforming.
    std::unordered_map<uint64_t, uint32_t> m_topoDeformingFrame;
    // NV-DXVK: deformingPromotedToClassic (topology-keyed) - last frame each topology had a
    // Path A (promoted OR resident) instance. If a topology is on Path A, its deforming
    // instances render classic - catches SAME-mesh splits AND different meshes conflated by
    // the deformation-invariant topology key (both proven by [DualRoute]). Persistent; pruned.
    std::unordered_map<uint64_t, uint32_t> m_topoPathAFrame;
    // persistent across frames: distinct dual-routed topologies already logged in full
    // (bounds the log to one line per NEW offender) + cumulative event count.
    std::unordered_set<uint64_t> m_dualRouteSeenKeys;
    uint64_t m_dualRouteEvents = 0;
    // records this frame's path decision for a topology and logs [DualRoute] on the first
    // frame a topology is seen on both paths. path: 1 = Path A, 2 = Path B.
    void recordTopoRoute(uint64_t topologyKey, uint8_t path, const RtInstance* instance, uint32_t outGeometryId, uint8_t source);

    // NV-DXVK: [SceneAnimInstScan] crash-safe mirror of the animated OPAQUE cluster
    // instances AS FED INTO THE SCENE TLAS (patch-kernel output copied into
    // AccelManager's instance buffer, on the main render cmd). This is the ONLY
    // capture on the fatal frame's path: buildTlas/[TlasRefScan] runs once early with
    // a null instance buffer and never sees these; [AnimTlasCapture] mirrors the
    // template cmd which (ring-slot-1 first use) may never execute. Dumped from the
    // device-lost instance hook (armed at first use). Names the null/stale
    // accelerationStructureReference the reflection-PSR ray traversal VA=0's on.
    // Diagnostic - revert with the rest.
    // 2-slot ring (frame parity): slot being GPU-written this frame vs slot completed
    // last frame; the live scan reads the completed one (CPU throttled >=1 frame behind
    // the GPU, same lag assumption as [TlasRefScan]).
    Rc<DxvkBuffer> m_dbgSceneAnimInstHost;
    VkDeviceSize m_dbgSceneAnimInstStride = 0;
    uint32_t m_dbgSceneAnimInstCount[2] = {};
    uint32_t m_dbgSceneAnimInstFrame[2] = {};
    // ring pool the capture's pose BLASes were built into (recorded at capture time so
    // skipped recordFrames cannot skew the expected-pool comparison at scan time)
    uint32_t m_dbgSceneAnimInstExpectedPool[2] = {};
    // NV-DXVK: pose-BLAS pool ranges SNAPSHOTTED at capture time (per ring slot), so the
    // scan classifies refs against the pools that were live WHEN the refs were captured -
    // not the (possibly grown) current pools. This makes the OUTSIDE-all-pools verdict
    // reliable: an OUTSIDE ref genuinely points at a pool freed before capture = dangling.
    uint64_t m_dbgSceneAnimInstPoolLo[2][8] = {};
    uint64_t m_dbgSceneAnimInstPoolHi[2][8] = {};
    uint32_t m_dbgSceneAnimInstPoolCount[2] = {};
    uint32_t m_dbgSceneAnimInstLastSlot = 0;
    bool m_dbgSceneAnimInstArmed = false;
    void dumpSceneAnimInstOnDeviceLost();
    void scanSceneAnimInstMirror(uint32_t slot);

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
      // NV-DXVK: atomicDemotion fix - true this frame if ANY instance of this geometry is
      // demoted; recomputed each updatePromotionStates. When set, no instance promotes.
      bool anyInstanceDemoted = false;
    };
    // main-thread after adoption in onFrameBegin
    std::unordered_map<uint64_t, PromotionCandidate> m_promoCandidates;
    // worker -> main handoff of uploaded probes
    struct PendingProbe {
      uint64_t geometryHash = 0;
      uint64_t probeVa = 0;
      uint32_t vertexCount = 0;
    };
    std::mutex m_promoPendingMutex;
    std::vector<PendingProbe> m_promoPendingProbes;
    uint32_t m_promoNextStateSlot = 0;
    // per-INSTANCE state slots for PROMOTED instances (plan risk R21: M is per
    // instance - every captured instance's buffer carries its own transform -
    // so patch/prevM state must never alias across instances; the candidate's
    // own slot serves only the geometry-level probe/gate verdict).
    // demoted: instance-level demotion (former V1 limitation was geometry-
    // level) - this instance renders Path B while its solves stay non-rigid;
    // a fresh rigid streak re-promotes it. sweepPending/sweepLagFrames track
    // the periodic full-mesh sweep verdict (risk R20).
    struct PromoInstance {
      uint32_t stateSlot = 0;
      uint32_t sweepLagFrames = 0;
      bool sweepPending = false;
      bool demoted = false;
      // NV-DXVK: pinned Path A residency. Once an instance PROMOTES, its identity
      // is this stable BlasEntry* (the draw-call cache keeps the same BlasEntry
      // across camera moves - topological bucket + material-match reuse), NOT the
      // asset hash. This game's captured draws produce an asset hash that is
      // unstable frame-to-frame under camera motion, so re-deriving residency from
      // it every frame is exactly what dropped promoted meshes back to Path B on
      // any camera move. Cached at establish time so the per-frame route reads the
      // id straight off the slot instead of an m_geometryIdByHash lookup by the
      // churning hash. residentGeometryId == ~0u means "not yet pinned"; geometryHash
      // is the ingest-time key (stable) for the atomicDemotion candidate lookup;
      // blasFrameCreated guards against BlasEntry* address reuse.
      uint32_t residentGeometryId = ~0u;
      uint64_t geometryHash = 0;
      uint32_t blasFrameCreated = 0;
      // NV-DXVK: 1-frame promotion WARMUP (kills the cold-promote motion-vector smear). A
      // freshly allocated slot has no GPU M history, so its first patched solve seeds prevM = curM
      // (zero motion) - a Path B->A flip then smears for one frame (the denoiser blends stale
      // history onto a pixel it thinks did not move). Under streaming churn this fires on many
      // instances per frame. Instead, on the establish frame the instance stays on Path B and its
      // slot is solved UNPATCHED (buildPromotionEntries), so the NEXT frame it promotes to Path A
      // with a warm prevM = last frame's solved M -> correct motion vector, no smear. Only genuinely
      // fresh slots warm up; a recovered (id/cell) slot already has history. warmupFrame is the
      // frame the warmup solve was emitted; the route promotes once currentFrame passes it.
      bool warmupPending = false;
      uint32_t warmupFrame = 0;
    };
    std::unordered_map<const BlasEntry*, PromoInstance> m_promoSlotByBlas;
    // NV-DXVK: recover a promoted instance's state slot across BlasEntry churn. Streaming and
    // [BlasWave] generation rebuilds destroy/recreate BlasEntry objects, orphaning the
    // pointer-keyed slot above; a fresh slot has no GPU M/prevM history, so its first solve
    // reports zero motion (one frame of the reconstruction wobble => promotion smear - proven
    // by [ColdPromo] waves coinciding with [BufVa] DESTROY + [BlasWave]). Keyed by the STABLE
    // RtInstance id so the rebuilt instance inherits its warm slot instead. Also caps the slot
    // leak: the same instance stops consuming a new slot on every rebuild.
    std::unordered_map<uint64_t, uint32_t> m_promoSlotByInstanceId;
    // NV-DXVK (plan §2 Option L): the STABLE identity that actually kills the cold-slot snap.
    // Keyed by hash(geometryHash, quantize(worldCentroid, identityCellSize)) - a static prop's
    // reconstructed world centroid lands in the same cell every frame despite the wobble, so it
    // recovers its warm M/prevM slot even when both the BlasEntry pointer AND the RtInstance id
    // churn (the m_promoSlotByInstanceId tier above is dormant because the id churns too). The
    // stored centroid + frameClaimed drive the 26-neighbor hysteresis that catches a prop
    // wobbling across a cell boundary. geometryHash distinguishes different assets that share a
    // cell. Captured-geometry-scoped, so lights / uniqueObjectDistance are untouched.
    struct CellSlot {
      uint32_t stateSlot = 0;
      uint64_t residentGeometryId = 0;  // STABLE per-asset id (not the churning geometryHash)
      Vector3 centroid = Vector3(0.f);
      uint32_t frameClaimed = 0;
    };
    std::unordered_map<XXH64_hash_t, CellSlot> m_promoSlotByCell;
    // [StableId] diagnostic (Option L effectiveness): cumulative slot-recovery outcomes. warmCell
    // + warmNeighbor recovered a warm GPU M/prevM slot (good - no snap); coldAlloc took a fresh
    // slot (a genuine first sight, OR the cell key failed to recover -> a [ColdPromo] snap). A
    // healthy run: coldAlloc plateaus near the live-instance count while warm* keeps climbing.
    uint32_t m_stableIdWarmCell = 0;      // exact-cell hit
    uint32_t m_stableIdWarmNeighbor = 0;  // 26-neighbor hysteresis hit (boundary wobble)
    uint32_t m_stableIdWarmId = 0;        // dormant instance-id tier hit (rare)
    uint32_t m_stableIdColdAlloc = 0;     // fresh slot allocated (first sight or recovery miss)
    // [PromoStale] diagnostic: per-slot last-seen solve frame (from the readback) + how many CPU
    // frames its solve frame has NOT advanced while the slot is still promoted. A stalled slot
    // means the GPU M/prevM froze -> stale placement (lag) + dead motion vector (smear).
    std::unordered_map<uint32_t, uint32_t> m_dbgSlotLastSolveFrame;
    std::unordered_map<uint32_t, uint32_t> m_dbgSlotStallFrames;
    // [PromoCont] previous-frame curM translation per slot (CPU-tracked) to verify prevM lineage:
    // this frame's prevT should equal last frame's curT. A break = the smear's motion-vector bug.
    std::unordered_map<uint32_t, Vector3> m_dbgSlotPrevCurT;
    // [PromoRender] the set of promo state slots actually RENDERED Path A (got a solve+patch) each
    // frame, so diagnostics can isolate the VISIBLE instances from the ~90% non-rendered pool. A
    // slot dropping out then re-appearing = Path A<->B flip = flicker/smear; a slot re-appearing
    // after a gap re-promotes COLD = the snap.
    std::unordered_set<uint32_t> m_dbgRenderedSlots;
    std::unordered_set<uint32_t> m_dbgRenderedSlotsPrev;
    // [PromoGap] per-slot EXPECTED world position (this frame's live bbox centroid, i.e. where
    // Path B draws it), recorded at route time. Compared against the kernel's PLACED position
    // (M*centroid, from the readback) to catch an object drawn far from where it should be.
    std::unordered_map<uint32_t, Vector3> m_dbgSlotExpectedPos;
    // Quantize a world centroid to an integer cell coord + fold it with the geometry hash into a
    // stable 64-bit cell key. Static so the neighbor-cell hysteresis can reuse it.
    static XXH64_hash_t makePromoCellKey(uint64_t geometryHash, int64_t cx, int64_t cy, int64_t cz);
    // per-frame kernel work items (built in dispatchBuild, consumed by
    // recordFrame the same call) + readback scratch
    std::vector<lodclusters_remix::PromotionEntry> m_framePromoEntries;
    std::vector<lodclusters_remix::PromotionStateView> m_promoStates;
    bool m_promoStatesValid = false;
    uint32_t m_statsPromoted = 0;
    uint32_t m_statsPromoRejected = 0;

    // worker thread (P4c): probe precompute (samples + Gram pseudoinverse in
    // doubles) + upload through the template system's callback-locked path
    void buildAndUploadPromotionProbe(const lodclusters_remix::GeometrySnapshot& snapshot);
    // onFrameBegin: adopt pending probes, read verdicts, run the state machine
    void updatePromotionStates();
    // dispatchBuild: emit this frame's solve/gate/patch entries
    void buildPromotionEntries();
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

    // NV-DXVK: [SwapProbe] per-frame routing census to LOCATE the 1-frame visible
    // gap at a full generation rebuild (the flicker). Cheap always-on counters
    // (2 int adds per instance), logged ONLY across a rebuild window. A dip in
    // pathA on the swap frame vs the frame before = instances that were resident
    // last frame but routed classic / Path B / dropped this frame = the gap. The
    // classic column shows where they went. Diagnostic - revert with the P5 fix.
    uint32_t m_swapProbeClassified = 0;  // isClusterInstance calls this frame
    uint32_t m_swapProbePathA = 0;       // recorded Path A (resident/promoted) this frame
    uint32_t m_swapProbePathB = 0;       // recorded Path B (deforming) this frame
    struct SwapProbeSample {
      uint32_t frame = 0, classified = 0, pathA = 0, pathB = 0, classic = 0;
      bool swapFrame = false;            // a full rebuild activated this frame
    };
    static constexpr uint32_t kSwapProbeRing = 8;
    SwapProbeSample m_swapProbeRing[kSwapProbeRing] = {};
    uint32_t m_swapProbeRingHead = 0;
    uint32_t m_swapProbeSwapFrame = 0;   // frame the last full rebuild activated
    uint32_t m_swapProbeDumpAtFrame = 0; // dump the ring once currentFrame reaches this (0 = idle)

    // ---- [ChurnDiag] streaming-churn instrumentation (Promotion::churnDiag) ----
    // Per-frame routing-outcome census for captured promotion candidates: WHY an instance is or
    // is not on Path A this frame. Reset in beginInstanceRecording, accumulated across the
    // isClusterInstance calls of mergeInstancesIntoBlas, logged in buildPromotionEntries.
    struct ChurnCensus {
      uint32_t capturedCandidates = 0;   // captured, non-skinned, promotion-eligible instances seen
      uint32_t pinHit = 0;               // routed Path A via the pinned fast-path
      uint32_t establishPromote = 0;     // routed Path A via (re)establish
      uint32_t warmupHold = 0;           // held on Path B one frame by the warmup (intentional)
      uint32_t pinMissNoEntry = 0;       // no pin entry for this BlasEntry* (new/rebuilt pointer, or first sight)
      uint32_t pinMissRebuilt = 0;       // pin entry present but blasFrameCreated changed (BlasEntry rebuilt in place)
      uint32_t pinMissNotResident = 0;   // pin present, residentGeometryId == ~0 (still warming / not yet pinned)
      uint32_t pinMissDemoted = 0;       // pin present but slot demoted (non-rigid solve)
      uint32_t pinBlockedAtomic = 0;     // pin conditions met but atomicDemotion suppressed
      uint32_t pinBlockedCapacity = 0;   // pin conditions met but render-instance capacity full
      uint32_t pinBlockedTrivial = 0;    // pin conditions met but routeTrivialToClassic
      uint32_t estNoCandidate = 0;       // no promotion candidate for the (churning) asset hash
      uint32_t estNotPromoted = 0;       // candidate exists but not yet in Promoted phase
      uint32_t estNoGeometryId = 0;      // m_geometryIdByHash miss (generation table churn)
      uint32_t estCapacity = 0;          // establish blocked by render-instance capacity
      uint32_t estNoSlot = 0;            // slot allocation failed (promo slot capacity)
      void reset() { *this = ChurnCensus(); }
    };
    ChurnCensus m_churn;
    // Per promoted stateSlot: the last frame it actually rendered Path A (got a solve+patch entry),
    // so buildPromotionEntries can split this frame's rendered slots into contiguous / fresh (first
    // ever render -> warmup, clean) / gapped-everSolved (re-render after a gap -> stale prevM = smear).
    std::unordered_map<uint32_t, uint32_t> m_dbgSlotLastRenderFrame;
    uint32_t m_dbgSlotGapFresh = 0;      // rendered this frame, never rendered before (fresh/warmup)
    uint32_t m_dbgSlotGapStale = 0;      // rendered this frame, rendered before but with a gap (SMEAR)
    uint32_t m_dbgSlotGapWorst = 0;      // largest such gap (frames) this frame
    // Per promoted stateSlot: the instance's blasFrameCreated last time it rendered; a change means
    // the BlasEntry was rebuilt under it (streaming / generation swap) - the prime churn suspect.
    std::unordered_map<uint32_t, uint32_t> m_dbgSlotBlasFrame;
    uint32_t m_dbgBlasRebuilds = 0;      // slots whose BlasEntry frameCreated changed this frame
    uint32_t m_dbgChurnOverflowLast = 0; // m_frameOverflowCount at the previous [ChurnDiag] log
    // [SolveCadence] the displayed-vs-artifact test. getCurrentFrameId is the PRESENT counter; the
    // shader's contiguity check uses the render system's frameIndex which only advances when the
    // cluster solve actually dispatches. So a present-gap here (delta > 1) means displayed frames
    // advanced with NO promotion solve. Paired with the main-camera worldToView delta across the
    // gap: gap + camera moved => the promoted geometry (absent those frames) reappears with a
    // whole-gap-stale prevM = a REAL reappearance smear; gap + camera still => a hang, no smear.
    uint32_t m_dbgLastSolveFrame = 0;    // getCurrentFrameId at the previous dispatchBuild
    Matrix4 m_dbgLastCamW2V;             // main-camera worldToView at the previous dispatchBuild

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
