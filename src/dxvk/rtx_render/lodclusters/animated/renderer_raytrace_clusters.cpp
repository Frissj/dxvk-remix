/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

// NV-DXVK: This file originates from nvpro-samples/vk_animated_clusters
// (src/renderer_raytrace_clusters.cpp). Per the RTX Remix integration plan
// (section 4) it becomes the Path B cluster-template system: the template
// build / move-compaction / size-query machinery, the per-frame CLAS
// instantiation + cluster BLAS build sequences and their barriers are kept
// from the sample; the sample's viewer sections (its own TLAS, ray tracing
// pipeline/SBT, rgen/rchit shaders, render target) are dropped - Remix's
// AccelManager owns the TLAS and the path tracer consumes it.
//
// Structural adaptations for Remix (each marked REMIX below):
//  * The sample sizes everything once for a fixed scene; Remix discovers
//    deforming geometry at runtime. Geometries register incrementally
//    (background worker: CPU clusterization via animatedclusters::Scene,
//    then the sample's per-geometry template build under Remix's submission
//    lock) and per-frame capacities grow on demand.
//  * The sample's per-instance vertex buffers are stable, so instantiation
//    inputs upload once; Remix's skinned output buffers ping-pong every
//    frame (BlasEntry history buffers), so the instantiation/build inputs
//    are (re)written each frame into a host-visible ring.
//  * One "pose set" per Remix BlasEntry owns the persistent explicit-
//    destination CLAS memory (the sample's per-render-instance
//    clusterBuffer); one cluster BLAS per pose set is rebuilt per frame.
//  * Template clusterIDs are globalClusterBase + c: they index the global
//    animated cluster table (8 bytes per cluster: device address of the
//    cluster's triangles in the resident cluster-ordered index topology) so
//    the hit side can remap the cluster-local primitiveIndex back to the
//    original triangle (risk R14) with the classic vertex fetch.
//  * cluster_blas_instances.comp is used verbatim through its "static"
//    branch (animated=0): instances.d[slot].geometryID carries the
//    slot -> pose(BLAS) index, because multiple Remix instances may share
//    one BlasEntry pose.

#include <volk.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <nvvk/check_error.hpp>
#include <nvutils/profiler.hpp>
#include <nvvk/profiler_vk.hpp>

#include "../resources.hpp"
#include "../lodclusters_remix.h"
#include "scene.hpp"
#include "shaderio_animated_host.h"

namespace lodclusters_remix {

// defined in lodclusters_remix_render.cpp: shared chrono-report formatting of
// one profiler timeline snapshot ('\n'-separated section lines, milliseconds)
bool formatProfilerReportUtf8(const nvutils::ProfilerTimeline* timeline, std::string& outReport);

namespace {

// the library's own function pointers (volk); Remix's dxvk loader is separate.
// Safe to run after Path A's loadVolk (volk reload is idempotent).
void loadVolkAnimated(VkInstance instance, VkDevice device)
{
  static std::once_flag s_volkOnce;
  std::call_once(s_volkOnce, [&] {
    // may already be initialized by the Path A render system - volkInitialize
    // only resolves the loader entry points again
    if(volkInitialize() != VK_SUCCESS)
    {
      LOGE("ClusterTemplateSystem: volkInitialize failed\n");
      return;
    }
    volkLoadInstance(instance);
  });

  volkLoadDevice(device);
}

uint32_t nextPowerOfTwo(uint32_t value)
{
  uint32_t result = 1;
  while(result < value)
  {
    result <<= 1;
  }
  return result;
}

// 32 byte alignment requirement for bbox (sample: initRayTracingTemplates)
struct TemplateBbox
{
  animatedclusters::shaderio::BBox bbox;
  uint32_t                         _pad[2];
};

// device+host readback of the statistics sums produced by
// cluster_blas_instances.comp (sample: shaderio::Readback clustersSize /
// blasesSize)
struct AnimatedReadback
{
  uint64_t clustersSize;
  uint64_t blasesSize;
};

}  // namespace

struct ClusterTemplateSystem::Impl
{
  using Resources = lodclusters::Resources;

  // Frames-in-flight depth for the per-frame resource rings. The ray trace of a
  // frame keeps reading this frame's CLAS + cluster BLAS while the next frames
  // already re-instantiate/rebuild them, so every per-frame-written GPU resource
  // the trace CONSUMES (the per-pose CLAS destinations and the implicit BLAS
  // pool) is cycled across this many physical buffers - frame N writes slot
  // N % kRingSlots, so it never overwrites the slot a still-in-flight earlier
  // frame is tracing. The AS memory is raw VMA that dxvk does not frame-track
  // (risk R17), so without this the overwrite races the previous frame's trace -
  // invisible at debug's ~2 FPS (GPU idle between frames), visible as deforming
  // objects lagging/jittering once the release build lets frames overlap. Must
  // be >= Remix's frames in flight PLUS ONE: with kMaxFramesInFlight submitted
  // and incomplete (N..N+3), frame N+4 is already recording - at 4 slots it
  // writes slot N%4 while frame N's trace still reads it (the flickering
  // garbage-triangle clumps under deep GPU queueing: spike frames hit the
  // boundary exactly). 6 = 4 in flight + recording + margin.
  static constexpr uint32_t kRingSlots = 6;

  bool initialized = false;

  RenderDeviceInfo deviceInfo;
  AnimatedConfig   config;

  VkBuildAccelerationStructureFlagsKHR templateBuildFlags       = 0;
  VkBuildAccelerationStructureFlagsKHR templateInstantiateFlags = 0;
  VkBuildAccelerationStructureFlagsKHR clusterBuildFlags        = 0;
  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags         = 0;

  Resources res;

  // chrono: per-frame GPU/CPU section timers around the Path B build phases
  // (same nvpro profiler Path A's renderer records into; read out through
  // getProfilerReportUtf8)
  nvutils::ProfilerManager   profilerManager;
  nvutils::ProfilerTimeline* profilerTimeline = nullptr;
  nvvk::ProfilerGpuTimer     profilerGpuTimer;

  // cluster_blas_instances.comp (prebuilt SPIR-V, sample's compute pipeline)
  shaderc::SpvCompilationResult blasInstancesShader;
  VkPipelineLayout              computePipelineLayout = VK_NULL_HANDLE;
  VkPipeline                    blasInstancesPipeline = VK_NULL_HANDLE;
  // per-pose vertex gather (global -> cluster-local order; razor-triangle fix).
  // Shares computePipelineLayout (GatherConstants fits its push range).
  shaderc::SpvCompilationResult gatherShader;
  VkPipeline                    gatherPipeline = VK_NULL_HANDLE;

  //////////////////////////////////////////////////////////////////////////
  // registered geometries (template sets)

  struct GeometryData
  {
    std::string name;

    uint32_t numClusters  = 0;
    uint32_t numVertices  = 0;
    uint32_t numTriangles = 0;

    // range [globalClusterBase, globalClusterBase + numClusters) in the
    // global animated cluster table; baked into the template clusterIDs
    uint32_t globalClusterBase = 0;

    bool opaque = true;

    // CPU cluster headers (drive the per-frame instantiation/build fills)
    std::vector<animatedclusters::shaderio::Cluster> clusters;

    // resident cluster-ordered index topology (uvec3 global vertex indices):
    // the hit-side primitive remap target (cluster table records). NOT a build
    // input - the NV cluster build indexes vertices in [0, vertexCount) per
    // cluster, so it consumes the LOCAL topology below (razor-triangle fix).
    nvvk::Buffer trianglesBuffer;

    // cluster-LOCAL topology (build inputs):
    //  - localTrianglesBuffer: uint8 indices in [0, cluster.numVertices) per
    //    cluster at Cluster::firstLocalTriangle (3 per triangle, same triangle
    //    ORDER as trianglesBuffer, so primitiveIndex remap is unchanged)
    //  - localVerticesBuffer: uint32 local slot -> global vertex index at
    //    Cluster::firstLocalVertex (the per-frame gather kernel's map)
    nvvk::Buffer localTrianglesBuffer;
    nvvk::Buffer localVerticesBuffer;
    // total local vertex slots (sum of cluster.numVertices; the pose's gather
    // output size and the build's per-frame vertex total)
    uint32_t     numClusterVertices = 0;

    // template mode
    nvvk::Buffer          templatesBuffer;
    std::vector<uint64_t> templateAddresses;
    std::vector<uint32_t> instantiationOffsets;
    uint32_t              sumInstantiationSizes = 0;
  };

  // main-thread-owned after adoption via drainReadyGeometries
  std::vector<std::unique_ptr<GeometryData>> geometries;

  // worker -> main handoff
  struct ReadyEntry
  {
    uint64_t                      topologyKey = 0;
    uint32_t                      geometryIndex = 0;
    std::unique_ptr<GeometryData> data;
  };
  std::mutex              readyMutex;
  std::deque<ReadyEntry>  readyGeometries;
  uint32_t                geometryIndexNext = 0;  // worker-side

  // clusterize tokens (single background worker; no locking needed)
  struct PendingGeometry
  {
    uint64_t                             topologyKey = 0;
    std::string                          name;
    bool                                 opaque = true;
    // REMIX: mutating/captured snapshots carry reference positions that are NOT
    // in the space the live vertex buffers use (capture space / per-frame CPU
    // rewrites), so their bind-pose cluster bboxes must not become
    // instantiationBoundingBoxLimits - a wrong-space limit clips real geometry.
    bool                                 useBboxLimit = true;
    animatedclusters::Scene              scene;
    animatedclusters::Scene::Geometry    geometry;
  };
  uint64_t                                                    tokenNext = 1;
  std::unordered_map<uint64_t, std::unique_ptr<PendingGeometry>> pendingGeometries;

  // P4c: promotion probe blobs (uploadPromotionProbe) - device-local, read by
  // the render system's promotion_solve kernel via BDA. Freed individually
  // when their geometry's verdict goes terminal (freePromotionProbe, deferred
  // via trash) or at deinit. Mutex: uploads run on the provider worker, frees
  // on the main thread.
  std::mutex                promotionProbeMutex;
  std::vector<nvvk::Buffer> promotionProbes;

  //////////////////////////////////////////////////////////////////////////
  // global animated cluster table (REMIX, hit-side primitive remap)

  nvvk::Buffer          clusterTableBuffer;
  uint32_t              clusterTableCapacity = 0;  // records
  uint32_t              clusterTableCount    = 0;  // records used (worker-side)
  std::atomic<uint64_t> clusterTableAddress { 0 };

  //////////////////////////////////////////////////////////////////////////
  // pose sets (one per Remix BlasEntry)

  struct PoseSet
  {
    uint32_t     geometryIndex = ~0u;
    // persistent explicit CLAS destinations, one per frame-in-flight slot (see
    // kRingSlots): the trace reads slot N%kRingSlots while frame N+1 writes the
    // next slot, so a pose's geometry is never overwritten mid-trace
    nvvk::Buffer clasBuffers[kRingSlots];
    bool         active = false;
  };
  std::vector<PoseSet>  poseSets;
  std::vector<uint32_t> poseSetFreeList;
  uint32_t              activePoseSets = 0;
  uint64_t              poseClasBytes = 0;

  //////////////////////////////////////////////////////////////////////////
  // per-frame device buffers, grown on demand

  uint32_t clusterCapacity  = 0;  // per-frame clusters across all poses
  uint32_t triangleCapacity = 0;  // per-frame triangle total
  uint32_t vertexCapacity   = 0;  // per-frame cluster-vertex total
  uint32_t poseCapacity     = 0;  // per-frame BLAS count
  uint32_t slotCapacity     = 0;  // per-frame TLAS slots

  // maxima across registered geometries (BLAS size query inputs)
  uint32_t maxGeometryClusters = 0;
  // the maxGeometryClusters the BLAS scratch + implicit pool were last sized
  // for. maxGeometryClusters grows in drainReadyGeometries (a newly-ready,
  // larger geometry) independently of the per-frame totals, so it needs its own
  // grow trigger - otherwise the BLAS build (recordFrame) runs with a bigger
  // maxClusterCountPerAccelerationStructure than the scratch/pool were queried
  // for and the driver over-reads them (illegal-access AS-build fault)
  uint32_t sizedMaxGeometryClusters = 0;

  // worst-case size of one explicitly built CLAS (direct build mode)
  VkDeviceSize singleExplicitClusterSize = 0;

  nvvk::Buffer dstSizesBuffer;       // u32 per cluster (built CLAS sizes, stats only - never dereferenced, single is safe)
  nvvk::Buffer blasSizesBuffer;      // u32 per pose (built BLAS sizes, stats only - never dereferenced, single is safe)
  // VkAccelerationStructureInstanceKHR per slot. RINGED: the frame's copy-in /
  // patch / copy-out are same-frame barriered, but these are raw vkCmd accesses
  // dxvk's barrier tracking never sees - frame N+1's copy-in would WAW/WAR race
  // frame N's copy-out on a single buffer once submissions overlap
  nvvk::Buffer tlasInstancesBuffers[kRingSlots];
  uint32_t     lastRecordedRingIndex = 0;  // getTlasInstancesBuffer (the caller's copy-out runs after recordFrame)
  // per-frame-in-flight rings (kRingSlots): a LATER stage dereferences these
  // across the frame boundary, so overlapping frames must not share one
  // instance (same reasoning as blasImplicitBuffers / risk R17):
  //  - blasAddresses: frame N patches these BLAS addresses into its TLAS
  //    instance references; single-buffered, frame N+1's BLAS build overwrites
  //    them before frame N's TLAS build consumes them.
  //  - scratch: the cluster ops' internal working memory. Single-buffered, two
  //    overlapping AS builds corrupt each other's build-internal pointers and
  //    the driver dereferences an unmapped address -> device-lost, unit
  //    "AS Build or Refit", fault VA owned by no allocation. This transient
  //    scratch was the one per-frame GPU buffer left shared when the input
  //    rings / CLAS dst / implicit-BLAS pools were ringed (the device-loss root
  //    cause).
  nvvk::Buffer blasAddressesBuffer[kRingSlots];  // u64 per pose (built BLAS addresses)
  // implicit-destination BLAS pool, one per frame-in-flight slot (kRingSlots):
  // the TLAS of frame N references BLASes in slot N%kRingSlots which the trace
  // dereferences, so frame N+1 must build into a different slot (risk R17)
  nvvk::Buffer blasImplicitBuffers[kRingSlots];
  nvvk::Buffer scratchBuffers[kRingSlots];
  VkDeviceSize scratchSize = 0;
  // per-frame gathered positions (vec3 per local vertex slot, all poses
  // concatenated): output of anim_gather_positions.comp, vertex input of the
  // CLAS instantiate/build. Ringed - the build of frame N reads slot N%4 while
  // frame N+1 gathers into its own slot (razor-triangle fix)
  nvvk::Buffer gatherBuffers[kRingSlots];

  nvvk::Buffer readbackBuffer;      // device AnimatedReadback
  nvvk::Buffer readbackHostBuffer;  // host ring of AnimatedReadback

  // host-visible per-frame input ring: instantiation/build infos, CLAS dst
  // addresses, BLAS build infos, RenderInstances (slot -> pose index) and the
  // TlasInstance staging. kRingSlots (declared at the top of Impl) covers
  // Remix's frames in flight (same reasoning as the P2 staging ring).
  nvvk::Buffer ringBuffers[kRingSlots];
  size_t       ringSrcInfosOffset       = 0;
  size_t       ringDstAddressesOffset   = 0;
  size_t       ringBlasInfosOffset      = 0;
  size_t       ringRenderInstancesOffset = 0;
  size_t       ringTlasOffset           = 0;

  uint32_t frameCounter = 0;
  bool     anyFrameRecorded = false;

  // deferred destruction: buffers a recorded frame may still reference are
  // destroyed only after every conceivable in-flight frame completed
  static constexpr uint32_t kDestroyDelayFrames = 8;
  struct Trash
  {
    nvvk::Buffer buffer;
    uint32_t     frameId;
    // DIAG (2026-07-05): device-address range + call-site tag so a post-crash
    // log can match the Aftermath page-fault address to the exact buffer and
    // see the queue/free frames (whether the 8-frame delay was actually enough)
    uint64_t     address;
    uint64_t     size;
    const char*  tag;
  };
  std::mutex            trashMutex;
  std::deque<Trash>     trash;
  std::atomic<uint32_t> currentFrameId { 0 };

  //////////////////////////////////////////////////////////////////////////

  void deferDestroy(nvvk::Buffer& buffer, const char* tag = "?")
  {
    if(!buffer.buffer)
    {
      return;
    }
    std::lock_guard<std::mutex> lock(trashMutex);
    const uint32_t fid = currentFrameId.load();
    // DIAG: log the range being deferred. Match the Aftermath fault address
    // (e.g. 0xa60bb000) against these [addr, addr+size) ranges post-crash.
    LOGI("[ClusterLOD] deferDestroy '%s': range 0x%llx..0x%llx (%llu bytes), queued frame %u\n",
         tag, (unsigned long long)buffer.address,
         (unsigned long long)(buffer.address + buffer.bufferSize),
         (unsigned long long)buffer.bufferSize, fid);
    trash.push_back({buffer, fid, buffer.address, buffer.bufferSize, tag});
    buffer = {};
  }

  void processTrash(uint32_t frameId, bool force)
  {
    std::lock_guard<std::mutex> lock(trashMutex);
    while(!trash.empty() && (force || frameId - trash.front().frameId > kDestroyDelayFrames))
    {
      const Trash& t = trash.front();
      // DIAG: log the actual free. If a page fault hits an address in this
      // range shortly AFTER this line, the buffer was freed while still GPU-live
      // (delay too short) - the queue->free frame gap shows the real headroom.
      LOGI("[ClusterLOD] FREE '%s': range 0x%llx..0x%llx, queued frame %u, freed frame %u (delay %u)\n",
           t.tag, (unsigned long long)t.address, (unsigned long long)(t.address + t.size),
           t.frameId, frameId, frameId - t.frameId);
      res.m_allocator.destroyBuffer(trash.front().buffer);
      trash.pop_front();
    }
  }

  // NV-DXVK: device-lost forensics (Path B). All per-frame build inputs live
  // in the host-visible ring, so no GPU copies are needed: pre/post stamps
  // bracket the instantiation + BLAS build, and the dump reads the faulting
  // frame's ring directly (the ring survives - deferDestroy holds buffers 8
  // frames and the device dies well within that window).
  nvvk::Buffer dbgStampHost;
  struct DbgFrameInfo
  {
    uint32_t       stamp         = 0;
    uint32_t       ringIndex     = 0;
    uint32_t       poseCount     = 0;
    uint32_t       totalClusters = 0;
    uint32_t       slotCount     = 0;
    const uint8_t* ringMapping   = nullptr;
    uint64_t       ringVa        = 0;
    size_t         dstOff        = 0;
    size_t         blasOff       = 0;
    // every pose's full vertex read range [addr, addr + stride*numVertices) +
    // pose set id - the builds' only reads outside cluster-owned memory are
    // these, so the Aftermath fault address must match one of them (or the
    // vertex-read theory is dead)
    std::vector<std::array<uint64_t, 3>> poseRanges;
  };
  DbgFrameInfo dbgFrames[2];

  void debugDump()
  {
    static std::mutex s_dumpMutex;
    static bool       s_dumped = false;
    std::lock_guard<std::mutex> lock(s_dumpMutex);
    if(s_dumped || !dbgStampHost.mapping)
    {
      return;
    }
    s_dumped = true;

    const uint32_t* stamps = reinterpret_cast<const uint32_t*>(dbgStampHost.mapping);

    LOGE("[AnimCapture] ==== device lost - dumping Path B (animated) build inputs ====\n");
    for(uint32_t slot = 0; slot < 2; slot++)
    {
      const DbgFrameInfo& info = dbgFrames[slot];
      const uint32_t      pre  = stamps[slot * 2 + 0];
      const uint32_t      post = stamps[slot * 2 + 1];

      LOGE("[AnimCapture] slot %u: pre %u post %u -> %s (ring %u, poses %u, clusters %u, slots %u, ringVa 0x%llx)\n",
           slot, pre, post,
           pre == 0 ? "never used" : (pre == post ? "builds COMPLETED" : "builds DID NOT COMPLETE (faulting frame)"),
           info.ringIndex, info.poseCount, info.totalClusters, info.slotCount, (unsigned long long)info.ringVa);

      if(pre == 0 || pre == post || info.ringMapping == nullptr)
      {
        continue;
      }

      uint32_t loggedLines = 0;

      // BLAS build inputs: cluster reference list pointers must sit inside
      // this ring slot's dstAddresses array
      const auto* dumpBlasInfos = reinterpret_cast<const VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV*>(
          info.ringMapping + info.blasOff);
      const uint64_t dstVaBase = info.ringVa + info.dstOff;
      const uint64_t dstVaEnd  = dstVaBase + sizeof(uint64_t) * uint64_t(info.totalClusters);
      uint32_t       badBlas   = 0;
      for(uint32_t p = 0; p < info.poseCount; p++)
      {
        const auto& b     = dumpBlasInfos[p];
        const bool  ptrOk = b.clusterReferences >= dstVaBase
                           && b.clusterReferences + uint64_t(b.clusterReferencesCount) * 8 <= dstVaEnd;
        if(!ptrOk || b.clusterReferencesStride != 8 || b.clusterReferencesCount == 0)
        {
          badBlas++;
          if(loggedLines < 32)
          {
            LOGE("[AnimCapture]  BAD BLAS %u: count %u stride %u references 0x%llx (dst array 0x%llx..0x%llx)\n", p,
                 b.clusterReferencesCount, b.clusterReferencesStride, (unsigned long long)b.clusterReferences,
                 (unsigned long long)dstVaBase, (unsigned long long)dstVaEnd);
            loggedLines++;
          }
        }
      }

      // CLAS destinations (CPU-written): raw sample + null count
      const uint64_t* dumpDst  = reinterpret_cast<const uint64_t*>(info.ringMapping + info.dstOff);
      uint32_t        nullDst  = 0;
      for(uint32_t c = 0; c < info.totalClusters; c++)
      {
        if(dumpDst[c] == 0)
        {
          nullDst++;
        }
      }
      for(uint32_t c = 0; c < std::min(info.totalClusters, 8u); c++)
      {
        LOGE("[AnimCapture]  dst[%u] = 0x%llx\n", c, (unsigned long long)dumpDst[c]);
      }

      // instantiation inputs: template + live vertex buffer addresses
      uint32_t nullTemplates = 0, nullVertices = 0;
      if(config.useTemplates)
      {
        const auto* dumpInst =
            reinterpret_cast<const VkClusterAccelerationStructureInstantiateClusterInfoNV*>(info.ringMapping);
        for(uint32_t c = 0; c < info.totalClusters; c++)
        {
          const auto& inst = dumpInst[c];
          if(inst.clusterTemplateAddress == 0)
          {
            nullTemplates++;
          }
          if(inst.vertexBuffer.startAddress == 0)
          {
            nullVertices++;
          }
          if((inst.clusterTemplateAddress == 0 || inst.vertexBuffer.startAddress == 0) && loggedLines < 64)
          {
            LOGE("[AnimCapture]  BAD INST %u: template 0x%llx vtx 0x%llx stride %llu\n", c,
                 (unsigned long long)inst.clusterTemplateAddress, (unsigned long long)inst.vertexBuffer.startAddress,
                 (unsigned long long)inst.vertexBuffer.strideInBytes);
            loggedLines++;
          }
        }
        for(uint32_t c = 0; c < std::min(info.totalClusters, 8u); c++)
        {
          const auto& inst = dumpInst[c];
          LOGE("[AnimCapture]  inst[%u]: template 0x%llx vtx 0x%llx stride %llu\n", c,
               (unsigned long long)inst.clusterTemplateAddress, (unsigned long long)inst.vertexBuffer.startAddress,
               (unsigned long long)inst.vertexBuffer.strideInBytes);
        }
      }

      // every pose's vertex read range - match the Aftermath fault address here
      for(size_t p = 0; p < info.poseRanges.size() && p < 192; p++)
      {
        LOGE("[AnimCapture]  pose %zu set %llu vtx 0x%llx..0x%llx\n", p, (unsigned long long)info.poseRanges[p][2],
             (unsigned long long)info.poseRanges[p][0], (unsigned long long)info.poseRanges[p][1]);
      }

      LOGE("[AnimCapture] slot %u summary: %u bad blas infos, %u null dst, %u null templates, %u null vertex buffers\n",
           slot, badBlas, nullDst, nullTemplates, nullVertices);
    }
    LOGE("[AnimCapture] ==== dump end ====\n");
  }

  VkClusterAccelerationStructureTriangleClusterInputNV makeTriangleClusterInput(uint32_t maxTotalTriangles, uint32_t maxTotalVertices) const
  {
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    triangleInput.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
    triangleInput.maxClusterTriangleCount     = config.clusterTriangles;
    triangleInput.maxClusterVertexCount       = config.clusterVertices;
    triangleInput.maxTotalTriangleCount       = maxTotalTriangles;
    triangleInput.maxTotalVertexCount         = maxTotalVertices;
    triangleInput.minPositionTruncateBitCount = config.positionTruncateBits;
    return triangleInput;
  }

  // (re)creates the per-frame buffers when the requested frame totals exceed
  // the current capacities; re-queries the CLAS/BLAS scratch and the implicit
  // BLAS pool sizes for the grown capacities. Old buffers are defer-destroyed
  // (in-flight frames may still reference them).
  bool ensureFrameCapacities(uint32_t totalClusters, uint32_t totalTriangles, uint32_t totalVertices, uint32_t poseCount, uint32_t slotCount)
  {
    const bool growClusters = totalClusters > clusterCapacity || totalTriangles > triangleCapacity || totalVertices > vertexCapacity;
    const bool growPoses    = poseCount > poseCapacity;
    const bool growSlots    = slotCount > slotCapacity;
    // the BLAS scratch + implicit pool are sized from maxGeometryClusters (line
    // ~411 / recordFrame) - a newly-ready larger geometry must force a re-query
    // even when the per-frame totals are unchanged (else the build over-reads them)
    const bool growBlas     = maxGeometryClusters > sizedMaxGeometryClusters;

    if(!growClusters && !growPoses && !growSlots && !growBlas && scratchBuffers[0].buffer)
    {
      return true;
    }

    clusterCapacity  = std::max(clusterCapacity, nextPowerOfTwo(std::max(totalClusters, 1024u)));
    triangleCapacity = std::max(triangleCapacity, nextPowerOfTwo(std::max(totalTriangles, 1024u)));
    vertexCapacity   = std::max(vertexCapacity, nextPowerOfTwo(std::max(totalVertices, 1024u)));
    poseCapacity     = std::max(poseCapacity, nextPowerOfTwo(std::max(poseCount, 16u)));
    slotCapacity     = std::max(slotCapacity, nextPowerOfTwo(std::max(slotCount, 16u)));
    // committed here so the BLAS size query below (and every subsequent build)
    // matches the scratch/pool we are about to allocate
    sizedMaxGeometryClusters = maxGeometryClusters;

    // ---- size queries (sample: initRayTracingScene / initRayTracingBlas) ----

    VkDeviceSize newScratchSize = 0;

    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput =
        makeTriangleClusterInput(triangleCapacity, vertexCapacity);

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    // per-frame CLAS op: template instantiation or direct build, explicit destinations
    inputs.maxAccelerationStructureCount = clusterCapacity;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType = config.useTemplates ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV :
                                          VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opInput.pTriangleClusters = &triangleInput;
    inputs.flags = config.useTemplates ? templateInstantiateFlags : clusterBuildFlags;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    newScratchSize = std::max(newScratchSize, sizesInfo.buildScratchSize);

    // per-frame BLAS build (implicit destinations, one per pose)
    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    blasInput.maxClusterCountPerAccelerationStructure = std::max(maxGeometryClusters, 1u);
    blasInput.maxTotalClusterCount                    = clusterCapacity;

    inputs                              = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount = poseCapacity;
    inputs.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel = &blasInput;
    inputs.flags                        = clusterBlasFlags;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    newScratchSize = std::max(newScratchSize, sizesInfo.buildScratchSize);

    const VkDeviceSize blasPoolSize = sizesInfo.accelerationStructureSize;

    // ---- (re)create the capacity-sized buffers ----

    const VkBufferUsageFlags2 kDeviceUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    deferDestroy(dstSizesBuffer, "dstSizes");
    NVVK_CHECK(res.createBuffer(dstSizesBuffer, sizeof(uint32_t) * clusterCapacity, kDeviceUsage));

    for(nvvk::Buffer& blasAddresses : blasAddressesBuffer)
    {
      deferDestroy(blasAddresses, "blasAddresses");
      NVVK_CHECK(res.createBuffer(blasAddresses, sizeof(uint64_t) * poseCapacity, kDeviceUsage));
    }

    deferDestroy(blasSizesBuffer, "blasSizes");
    NVVK_CHECK(res.createBuffer(blasSizesBuffer, sizeof(uint32_t) * poseCapacity, kDeviceUsage));

    for(nvvk::Buffer& tlasInstances : tlasInstancesBuffers)
    {
      deferDestroy(tlasInstances, "tlasInstances");
      NVVK_CHECK(res.createBuffer(tlasInstances, sizeof(VkAccelerationStructureInstanceKHR) * slotCapacity,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                      | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
    }

    // one implicit BLAS pool per frame-in-flight slot (see kRingSlots): the
    // trace reads the previous frames' BLASes, so each frame builds into its own
    for(nvvk::Buffer& blasPool : blasImplicitBuffers)
    {
      deferDestroy(blasPool, "blasImplicit");
      NVVK_CHECK(res.createBuffer(blasPool, blasPoolSize,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                      | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));
    }

    scratchSize = newScratchSize;
    for(nvvk::Buffer& scratch : scratchBuffers)
    {
      deferDestroy(scratch, "scratch");
      NVVK_CHECK(res.createBuffer(scratch, scratchSize,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    }

    // gathered positions: vec3 per local vertex slot (vertexCapacity counts
    // cluster-local slots - recordFrame sums geometry.numClusterVertices).
    // Written by the gather kernel, read by the CLAS instantiate/build.
    for(nvvk::Buffer& gather : gatherBuffers)
    {
      deferDestroy(gather, "gather");
      NVVK_CHECK(res.createBuffer(gather, sizeof(glm::vec3) * vertexCapacity,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                      | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    }

    // ---- host-visible per-frame input ring ----

    const size_t srcInfoStride = config.useTemplates ? sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) :
                                                       sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);

    size_t offset = 0;
    auto alignUp = [](size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); };

    ringSrcInfosOffset = 0;
    offset             = srcInfoStride * clusterCapacity;

    offset                 = alignUp(offset, 16);
    ringDstAddressesOffset = offset;
    offset += sizeof(uint64_t) * clusterCapacity;

    offset              = alignUp(offset, 16);
    ringBlasInfosOffset = offset;
    offset += sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * poseCapacity;

    offset                    = alignUp(offset, 16);
    ringRenderInstancesOffset = offset;
    offset += sizeof(animatedclusters::shaderio::RenderInstance) * slotCapacity;

    offset         = alignUp(offset, 16);
    ringTlasOffset = offset;
    offset += sizeof(VkAccelerationStructureInstanceKHR) * slotCapacity;

    const size_t ringSlotSize = offset;

    // host-coherent mapped ring readable through device addresses by the
    // cluster ops (sample precedent: its host-visible infosBuffer feeds
    // vkCmdBuildClusterAccelerationStructureIndirectNV directly)
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size               = ringSlotSize;
    bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                       | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                       | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo vmaInfo{};
    vmaInfo.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    vmaInfo.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    vmaInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    for(nvvk::Buffer& ring : ringBuffers)
    {
      deferDestroy(ring, "ring");
      NVVK_CHECK(res.m_allocator.createBuffer(ring, bufferInfo, vmaInfo));
    }

    LOGI("ClusterTemplateSystem: frame capacities - clusters %u, poses %u, slots %u, blas pool %s, scratch %s\n",
         clusterCapacity, poseCapacity, slotCapacity, lodclusters::formatMemorySize(blasPoolSize).c_str(),
         lodclusters::formatMemorySize(scratchSize).c_str());

    // DIAG (2026-07-05): new per-frame buffer ranges, so a page-fault address
    // can be matched to the newly-grown (post-deferDestroy) allocations
    LOGI("[ClusterLOD] grow ranges: dstSizes 0x%llx..0x%llx\n",
         (unsigned long long)dstSizesBuffer.address, (unsigned long long)(dstSizesBuffer.address + dstSizesBuffer.bufferSize));
    for(uint32_t s = 0; s < kRingSlots; s++)
    {
      LOGI("[ClusterLOD] grow ranges: tlasInst[%u] 0x%llx..0x%llx\n",
           s, (unsigned long long)tlasInstancesBuffers[s].address,
           (unsigned long long)(tlasInstancesBuffers[s].address + tlasInstancesBuffers[s].bufferSize));
    }
    // scratch + blasAddresses are now ringed (kRingSlots) - log every slot so a
    // fault address still matches (see the scratch-overlap fix)
    for(uint32_t s = 0; s < kRingSlots; s++)
    {
      LOGI("[ClusterLOD] grow ranges: ring[%u] 0x%llx..0x%llx, blasImplicit[%u] 0x%llx..0x%llx, scratch[%u] 0x%llx..0x%llx, blasAddr[%u] 0x%llx..0x%llx, gather[%u] 0x%llx..0x%llx\n",
           s, (unsigned long long)ringBuffers[s].address, (unsigned long long)(ringBuffers[s].address + ringBuffers[s].bufferSize),
           s, (unsigned long long)blasImplicitBuffers[s].address, (unsigned long long)(blasImplicitBuffers[s].address + blasImplicitBuffers[s].bufferSize),
           s, (unsigned long long)scratchBuffers[s].address, (unsigned long long)(scratchBuffers[s].address + scratchBuffers[s].bufferSize),
           s, (unsigned long long)blasAddressesBuffer[s].address, (unsigned long long)(blasAddressesBuffer[s].address + blasAddressesBuffer[s].bufferSize),
           s, (unsigned long long)gatherBuffers[s].address, (unsigned long long)(gatherBuffers[s].address + gatherBuffers[s].bufferSize));
    }

    return true;
  }

  // grows the global cluster table, copying resident records (worker thread,
  // submission externally synchronized)
  bool ensureClusterTableCapacity(uint32_t requiredRecords)
  {
    if(requiredRecords <= clusterTableCapacity && clusterTableBuffer.buffer)
    {
      return true;
    }

    const uint32_t newCapacity = std::max(nextPowerOfTwo(requiredRecords), 1u << 16);

    nvvk::Buffer newBuffer;
    NVVK_CHECK(res.createBuffer(newBuffer, sizeof(uint64_t) * newCapacity,
                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
    if(!newBuffer.buffer)
    {
      return false;
    }

    if(clusterTableBuffer.buffer && clusterTableCount > 0)
    {
      VkCommandBuffer cmd = res.createTempCmdBuffer();
      VkBufferCopy    region{0, 0, sizeof(uint64_t) * clusterTableCount};
      vkCmdCopyBuffer(cmd, clusterTableBuffer.buffer, newBuffer.buffer, 1, &region);
      res.tempSyncSubmit(cmd);
    }

    // frames in flight may still read the old table through raytrace_args
    deferDestroy(clusterTableBuffer, "clusterTable");

    clusterTableBuffer   = newBuffer;
    clusterTableCapacity = newCapacity;
    clusterTableAddress.store(newBuffer.address);

    return true;
  }
};

ClusterTemplateSystem::ClusterTemplateSystem()
    : m_impl(std::make_unique<Impl>())
{
}

ClusterTemplateSystem::~ClusterTemplateSystem()
{
  deinit();
}

bool ClusterTemplateSystem::init(const RenderDeviceInfo& deviceInfo, const AnimatedConfig& config)
{
  Impl& impl = *m_impl;

  if(impl.initialized)
  {
    return true;
  }

  impl.deviceInfo = deviceInfo;
  impl.config     = config;

  impl.templateBuildFlags       = VkBuildAccelerationStructureFlagsKHR(config.templateBuildFlags);
  impl.templateInstantiateFlags = VkBuildAccelerationStructureFlagsKHR(config.templateInstantiateFlags);
  impl.clusterBuildFlags        = VkBuildAccelerationStructureFlagsKHR(config.clusterBuildFlags);
  impl.clusterBlasFlags         = VkBuildAccelerationStructureFlagsKHR(config.clusterBlasFlags);

  loadVolkAnimated(deviceInfo.instance, deviceInfo.device);
  if(!vkGetDeviceProcAddr || !vkCmdBuildClusterAccelerationStructureIndirectNV)
  {
    LOGE("ClusterTemplateSystem: volk did not resolve VK_NV_cluster_acceleration_structure entry points\n");
    return false;
  }

  nvvk::QueueInfo queueGraphics;
  queueGraphics.familyIndex = deviceInfo.graphicsQueueFamilyIndex;
  queueGraphics.queueIndex  = 0;
  queueGraphics.queue       = deviceInfo.graphicsQueue;

  nvvk::QueueInfo queueTransfer;
  queueTransfer.familyIndex = deviceInfo.transferQueueFamilyIndex;
  queueTransfer.queueIndex  = 0;
  queueTransfer.queue       = deviceInfo.transferQueue;

  impl.res.init(deviceInfo.device, deviceInfo.physicalDevice, deviceInfo.instance, queueGraphics, queueTransfer);

  // ---- cluster_blas_instances compute pipeline (sample: init lines 183-204,
  //      kept verbatim; the shader resolves to the prebuilt SPIR-V variant) ----
  {
    impl.res.compileShader(impl.blasInstancesShader, VK_SHADER_STAGE_COMPUTE_BIT, "cluster_blas_instances.comp.glsl");
    if(!impl.res.verifyShaders(1, &impl.blasInstancesShader))
    {
      LOGE("ClusterTemplateSystem: cluster_blas_instances shader lookup failed\n");
      impl.res.deinit();
      return false;
    }

    VkPushConstantRange pushRange;
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(animatedclusters::shaderio::ClusterBlasConstants);

    VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pPushConstantRanges        = &pushRange;
    layoutInfo.pushConstantRangeCount     = 1;
    vkCreatePipelineLayout(impl.res.m_device, &layoutInfo, nullptr, &impl.computePipelineLayout);

    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = impl.computePipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(impl.blasInstancesShader);
    vkCreateComputePipelines(impl.res.m_device, nullptr, 1, &compInfo, nullptr, &impl.blasInstancesPipeline);

    // per-pose vertex gather kernel (global -> cluster-local order; the
    // razor-triangle fix). Same descriptor-free layout - GatherConstants fits
    // the ClusterBlasConstants push range (static_asserted in shaderio).
    impl.res.compileShader(impl.gatherShader, VK_SHADER_STAGE_COMPUTE_BIT, "anim_gather_positions.comp.glsl");
    if(!impl.res.verifyShaders(1, &impl.gatherShader))
    {
      LOGE("ClusterTemplateSystem: anim_gather_positions shader lookup failed\n");
      vkDestroyPipeline(impl.res.m_device, impl.blasInstancesPipeline, nullptr);
      impl.blasInstancesPipeline = VK_NULL_HANDLE;
      vkDestroyPipelineLayout(impl.res.m_device, impl.computePipelineLayout, nullptr);
      impl.computePipelineLayout = VK_NULL_HANDLE;
      impl.res.deinit();
      return false;
    }
    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(impl.gatherShader);
    vkCreateComputePipelines(impl.res.m_device, nullptr, 1, &compInfo, nullptr, &impl.gatherPipeline);
  }

  // direct (non-template) mode: worst-case size of one explicitly built CLAS.
  // (sample: initRayTracingClusters - "in explicit the returned size is that
  // of one element")
  if(!config.useTemplates)
  {
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput =
        impl.makeTriangleClusterInput(config.clusterTriangles, config.clusterVertices);

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = 1;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opInput.pTriangleClusters = &triangleInput;
    inputs.flags                     = impl.clusterBuildFlags;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(impl.res.m_device, &inputs, &sizesInfo);
    impl.singleExplicitClusterSize = sizesInfo.accelerationStructureSize;
  }

  // statistics readback (device sums + host ring)
  NVVK_CHECK(impl.res.createBuffer(impl.readbackBuffer, sizeof(AnimatedReadback),
                                   VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
  NVVK_CHECK(impl.res.m_allocator.createBuffer(
      impl.readbackHostBuffer, sizeof(AnimatedReadback) * Impl::kRingSlots, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));

  // chrono: GPU section timers for the per-frame recordFrame phases (Path A
  // parity: ClusterRenderSystem's init creates the identical setup)
  nvutils::ProfilerTimeline::CreateInfo timelineInfo;
  timelineInfo.name     = "clusterlod-animated";
  impl.profilerTimeline = impl.profilerManager.createTimeline(timelineInfo);
  impl.profilerGpuTimer.init(impl.profilerTimeline, deviceInfo.device, deviceInfo.physicalDevice,
                             int(deviceInfo.graphicsQueueFamilyIndex), false);

  // NV-DXVK: Path B device-lost forensics (see Impl::debugDump). Registered on
  // this system's own Resources (its tempSyncSubmit noticed past losses first)
  // and on the shared aux hook the LOD renderer's primary dump chains into.
  NVVK_CHECK(impl.res.m_allocator.createBuffer(impl.dbgStampHost, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                               VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
  if(impl.dbgStampHost.mapping)
  {
    memset(impl.dbgStampHost.mapping, 0, sizeof(uint32_t) * 4);
  }
  {
    Impl* implPtr                       = m_impl.get();
    impl.res.deviceLostDumpFn           = [implPtr]() { implPtr->debugDump(); };
    lodclusters::deviceLostAuxDumpFn()  = [implPtr]() { implPtr->debugDump(); };
  }

  impl.initialized = true;

  LOGI("ClusterTemplateSystem: initialized (%s, cluster %u/%u)\n", config.useTemplates ? "templates" : "direct builds",
       config.clusterVertices, config.clusterTriangles);

  return true;
}

void ClusterTemplateSystem::deinit()
{
  Impl& impl = *m_impl;

  if(!impl.initialized)
  {
    return;
  }

  vkDeviceWaitIdle(impl.deviceInfo.device);

  // NV-DXVK: forensic teardown
  impl.res.deviceLostDumpFn          = nullptr;
  lodclusters::deviceLostAuxDumpFn() = nullptr;
  impl.res.m_allocator.destroyBuffer(impl.dbgStampHost);

  impl.processTrash(0, true);

  // P4c: promotion probe blobs (uploadPromotionProbe)
  for(nvvk::Buffer& probe : impl.promotionProbes)
  {
    if(probe.buffer)
    {
      impl.res.m_allocator.destroyBuffer(probe);
    }
  }
  impl.promotionProbes.clear();

  for(auto& geometry : impl.geometries)
  {
    if(geometry)
    {
      impl.res.m_allocator.destroyBuffer(geometry->trianglesBuffer);
      impl.res.m_allocator.destroyBuffer(geometry->localTrianglesBuffer);
      impl.res.m_allocator.destroyBuffer(geometry->localVerticesBuffer);
      impl.res.m_allocator.destroyBuffer(geometry->templatesBuffer);
    }
  }
  impl.geometries.clear();

  {
    std::lock_guard<std::mutex> lock(impl.readyMutex);
    for(auto& entry : impl.readyGeometries)
    {
      impl.res.m_allocator.destroyBuffer(entry.data->trianglesBuffer);
      impl.res.m_allocator.destroyBuffer(entry.data->localTrianglesBuffer);
      impl.res.m_allocator.destroyBuffer(entry.data->localVerticesBuffer);
      impl.res.m_allocator.destroyBuffer(entry.data->templatesBuffer);
    }
    impl.readyGeometries.clear();
  }
  impl.pendingGeometries.clear();

  for(auto& poseSet : impl.poseSets)
  {
    for(nvvk::Buffer& clasBuffer : poseSet.clasBuffers)
    {
      impl.res.m_allocator.destroyBuffer(clasBuffer);
    }
  }
  impl.poseSets.clear();
  impl.poseSetFreeList.clear();

  impl.res.m_allocator.destroyBuffer(impl.clusterTableBuffer);
  impl.res.m_allocator.destroyBuffer(impl.dstSizesBuffer);
  impl.res.m_allocator.destroyBuffer(impl.blasSizesBuffer);
  for(nvvk::Buffer& tlasInstances : impl.tlasInstancesBuffers)
  {
    impl.res.m_allocator.destroyBuffer(tlasInstances);
  }
  for(nvvk::Buffer& blasAddresses : impl.blasAddressesBuffer)
  {
    impl.res.m_allocator.destroyBuffer(blasAddresses);
  }
  for(nvvk::Buffer& blasPool : impl.blasImplicitBuffers)
  {
    impl.res.m_allocator.destroyBuffer(blasPool);
  }
  for(nvvk::Buffer& scratch : impl.scratchBuffers)
  {
    impl.res.m_allocator.destroyBuffer(scratch);
  }
  for(nvvk::Buffer& gather : impl.gatherBuffers)
  {
    impl.res.m_allocator.destroyBuffer(gather);
  }
  impl.res.m_allocator.destroyBuffer(impl.readbackBuffer);
  impl.res.m_allocator.destroyBuffer(impl.readbackHostBuffer);
  for(nvvk::Buffer& ring : impl.ringBuffers)
  {
    impl.res.m_allocator.destroyBuffer(ring);
  }

  if(impl.blasInstancesPipeline)
  {
    vkDestroyPipeline(impl.res.m_device, impl.blasInstancesPipeline, nullptr);
    impl.blasInstancesPipeline = VK_NULL_HANDLE;
  }
  if(impl.gatherPipeline)
  {
    vkDestroyPipeline(impl.res.m_device, impl.gatherPipeline, nullptr);
    impl.gatherPipeline = VK_NULL_HANDLE;
  }
  if(impl.computePipelineLayout)
  {
    vkDestroyPipelineLayout(impl.res.m_device, impl.computePipelineLayout, nullptr);
    impl.computePipelineLayout = VK_NULL_HANDLE;
  }

  impl.profilerGpuTimer.deinit();
  if(impl.profilerTimeline)
  {
    impl.profilerManager.destroyTimeline(impl.profilerTimeline);
    impl.profilerTimeline = nullptr;
  }

  impl.res.deinit();
  impl.initialized = false;
}

void ClusterTemplateSystem::setSubmitLockCallbacks(std::function<void()> lockFn, std::function<void()> unlockFn)
{
  Impl& impl = *m_impl;

  impl.res.submitLockFn   = std::move(lockFn);
  impl.res.submitUnlockFn = std::move(unlockFn);
}

uint64_t ClusterTemplateSystem::uploadPromotionProbe(const void* data, size_t bytes)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || data == nullptr || bytes == 0)
  {
    return 0;
  }

  nvvk::Buffer buffer;
  NVVK_CHECK(impl.res.createBuffer(buffer, bytes,
                                   VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
  if(!buffer.buffer)
  {
    return 0;
  }

  lodclusters::Resources::BatchedUploader uploader(impl.res);
  uploader.uploadBuffer<uint8_t>(buffer, 0, bytes, reinterpret_cast<const uint8_t*>(data));
  uploader.flush();

  {
    std::lock_guard<std::mutex> lock(impl.promotionProbeMutex);
    impl.promotionProbes.push_back(buffer);
  }
  return buffer.address;
}

void ClusterTemplateSystem::freePromotionProbe(uint64_t probeVa)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || probeVa == 0)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(impl.promotionProbeMutex);
  for(size_t i = 0; i < impl.promotionProbes.size(); i++)
  {
    if(impl.promotionProbes[i].address == probeVa)
    {
      // an in-flight frame's gate/solve may still read the blob - deferred
      // destruction only (trash queue, kDestroyDelayFrames)
      impl.deferDestroy(impl.promotionProbes[i], "promoProbe");
      impl.promotionProbes.erase(impl.promotionProbes.begin() + ptrdiff_t(i));
      return;
    }
  }
}

uint64_t ClusterTemplateSystem::clusterizeGeometry(const GeometrySnapshot& snapshot)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || snapshot.indices.empty() || snapshot.vertexCount < 3 || snapshot.positions.size() < size_t(snapshot.vertexCount) * 3)
  {
    return 0;
  }

  auto pending         = std::make_unique<Impl::PendingGeometry>();
  pending->topologyKey = snapshot.topologyKey;
  pending->name        = snapshot.name;
  pending->opaque      = !snapshot.alphaMasked;
  // Only skinned bind poses may bound instantiation: their reference positions
  // ARE the live buffers' space. Mutating/captured snapshots are wrong-space,
  // and P4c interim statics can SHARE a template set with a mesh of identical
  // topology but different positions (template sets are topology-keyed) - a
  // foreign bbox limit would clip real geometry.
  pending->useBboxLimit = snapshot.isDeforming;

  animatedclusters::Scene::Geometry& geometry = pending->geometry;
  geometry.numTriangles                       = uint32_t(snapshot.indices.size() / 3);
  geometry.numVertices                        = snapshot.vertexCount;

  geometry.positions.resize(snapshot.vertexCount);
  std::memcpy(geometry.positions.data(), snapshot.positions.data(), sizeof(glm::vec3) * snapshot.vertexCount);

  geometry.triangles.resize(geometry.numTriangles);
  std::memcpy(geometry.triangles.data(), snapshot.indices.data(), sizeof(glm::uvec3) * geometry.numTriangles);

  animatedclusters::SceneConfig sceneConfig;
  sceneConfig.clusterVertices          = impl.config.clusterVertices;
  sceneConfig.clusterTriangles         = impl.config.clusterTriangles;
  sceneConfig.clusterDedicatedVertices = false;  // Remix consumes the global-index topology (hit-side remap)
  // pool sized once from the pct; 0 = "use the pool as-is" so NVIDIA's
  // per-geometry ProcessingInfo::init/deinit pool resets never run (they were
  // the ~20 ms fixed floor measured on every Path B registration, 2026-07-04)
  configureProcessingThreadPool(impl.config.processingThreadsPct);
  sceneConfig.processingThreadsPct     = 0.0f;

  if(!pending->scene.processSingleGeometry(geometry, sceneConfig))
  {
    LOGW("ClusterTemplateSystem: %s: clusterization produced no clusters\n", pending->name.c_str());
    return 0;
  }

  const uint64_t token = impl.tokenNext++;
  impl.pendingGeometries.emplace(token, std::move(pending));
  return token;
}

bool ClusterTemplateSystem::buildGeometryTemplates(uint64_t token)
{
  Impl& impl = *m_impl;

  auto found = impl.pendingGeometries.find(token);
  if(!impl.initialized || found == impl.pendingGeometries.end())
  {
    return false;
  }

  std::unique_ptr<Impl::PendingGeometry> pending = std::move(found->second);
  impl.pendingGeometries.erase(found);

  animatedclusters::Scene::Geometry& geometry = pending->geometry;
  const uint32_t                     numClusters = geometry.numClusters;

  auto data          = std::make_unique<Impl::GeometryData>();
  data->name         = pending->name;
  data->numClusters  = numClusters;
  data->numVertices  = geometry.numVertices;
  data->numTriangles = geometry.numTriangles;
  data->opaque       = pending->opaque;
  data->clusters     = std::move(geometry.clusters);

  // [ClusterLOD] AnimIdx probe (2026-07-05, repurposed post-fix): the build now
  // consumes the cluster-LOCAL topology, whose indices MUST be < each cluster's
  // numVertices and whose local->global map MUST be < the mesh vertex count.
  // A hit here means clusterizer output corruption - expect silence.
  {
    uint32_t badLocal = 0, badMap = 0;
    for (uint32_t c = 0; c < numClusters; c++) {
      const animatedclusters::shaderio::Cluster& cl = data->clusters[c];
      for (uint32_t e = 0; e < uint32_t(cl.numTriangles) * 3; e++) {
        const uint32_t entry = cl.firstLocalTriangle + e;
        if (entry >= geometry.clusterLocalTriangles.size()
            || geometry.clusterLocalTriangles[entry] >= cl.numVertices) {
          badLocal++;
        }
      }
      for (uint32_t v = 0; v < cl.numVertices; v++) {
        const uint32_t slot = cl.firstLocalVertex + v;
        if (slot >= geometry.clusterLocalVertices.size()
            || geometry.clusterLocalVertices[slot] >= geometry.numVertices) {
          badMap++;
        }
      }
    }
    if (badLocal > 0 || badMap > 0) {
      LOGW("[ClusterLOD] AnimIdx %s: clusters=%u LOCAL topology invalid - badLocalIndices=%u badVertexMap=%u (clusterizer output corrupt)\n",
           data->name.c_str(), numClusters, badLocal, badMap);
      return false;
    }
  }

  // global cluster table range (REMIX: baked into template clusterIDs)
  data->globalClusterBase = impl.clusterTableCount;
  if(!impl.ensureClusterTableCapacity(impl.clusterTableCount + numClusters))
  {
    return false;
  }
  impl.clusterTableCount += numClusters;

  // ---- resident topology + temporary reference positions ----
  //  - trianglesBuffer: GLOBAL uvec3 indices, hit-side primitive remap only
  //  - localTrianglesBuffer / localVerticesBuffer: cluster-LOCAL build topology
  //    + local->global gather map (razor-triangle fix)
  //  - positionsBuffer (temp, template build only): reference positions in
  //    cluster-LOCAL order - the local indices the template bakes must address
  //    the same layout the per-frame gathered live positions will use

  data->numClusterVertices = uint32_t(geometry.clusterLocalVertices.size());

  nvvk::Buffer positionsBuffer;

  {
    NVVK_CHECK(impl.res.createBuffer(data->trianglesBuffer, sizeof(glm::uvec3) * geometry.triangles.size(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                         | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(impl.res.createBuffer(data->localTrianglesBuffer, sizeof(uint8_t) * geometry.clusterLocalTriangles.size(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                         | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(impl.res.createBuffer(data->localVerticesBuffer, sizeof(uint32_t) * geometry.clusterLocalVertices.size(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(impl.res.createBuffer(positionsBuffer, sizeof(glm::vec3) * data->numClusterVertices,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                         | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));

    if(!data->trianglesBuffer.buffer || !data->localTrianglesBuffer.buffer || !data->localVerticesBuffer.buffer
       || !positionsBuffer.buffer)
    {
      impl.res.m_allocator.destroyBuffer(data->trianglesBuffer);
      impl.res.m_allocator.destroyBuffer(data->localTrianglesBuffer);
      impl.res.m_allocator.destroyBuffer(data->localVerticesBuffer);
      impl.res.m_allocator.destroyBuffer(positionsBuffer);
      return false;
    }

    // reference positions gathered into cluster-local order (CPU-side; the
    // per-frame GPU gather kernel does the same reorder on the live buffers)
    std::vector<glm::vec3> localReferencePositions(data->numClusterVertices);
    for(uint32_t slot = 0; slot < data->numClusterVertices; slot++)
    {
      localReferencePositions[slot] = geometry.positions[geometry.clusterLocalVertices[slot]];
    }

    lodclusters::Resources::BatchedUploader uploader(impl.res);
    uploader.uploadBuffer(data->trianglesBuffer, geometry.triangles.data());
    uploader.uploadBuffer(data->localTrianglesBuffer, geometry.clusterLocalTriangles.data());
    uploader.uploadBuffer(data->localVerticesBuffer, geometry.clusterLocalVertices.data());
    uploader.uploadBuffer(positionsBuffer, localReferencePositions.data());
    uploader.flush();
  }

  bool success = true;

  if(impl.config.useTemplates)
  {
    // ---- template build + instantiation-size query -------------------------
    // Ported from the sample's initRayTracingTemplates, single geometry (the
    // sample's per-geometry loop body). Both the implicit-build+move-compaction
    // and the explicit COMPUTE_SIZES paths are kept.

    const bool useImplicitTemplates = impl.config.useImplicitTemplates;

    nvvk::Buffer implicitBuffer;

    VkDeviceSize tempScratchSize = 0;

    VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput =
        impl.makeTriangleClusterInput(geometry.numTriangles, geometry.numClusterVertices);

    VkClusterAccelerationStructureMoveObjectsInputNV moveInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = numClusters;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
    inputs.opMode = useImplicitTemplates ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV :
                                           VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters                   = &templateTriangleInput;
    inputs.flags                                       = impl.templateBuildFlags;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(impl.res.m_device, &inputs, &sizesInfo);
    tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);

    if(useImplicitTemplates)
    {
      impl.res.m_allocator.createBuffer(implicitBuffer, sizesInfo.accelerationStructureSize,
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

      // implicit builds are not guaranteed to be perfectly compact either, we run an extra compaction step after the implicit build.
      moveInput.type              = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_TEMPLATE_NV;
      moveInput.noMoveOverlap     = VK_TRUE;  // we move/copy from implicitBuffer to final per-geometry buffer
      moveInput.maxMovedBytes     = sizesInfo.accelerationStructureSize;  // worst case everything is moved
      inputs.opType               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
      inputs.opMode               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
      inputs.flags                = 0;
      inputs.opInput.pMoveObjects = &moveInput;
      vkGetClusterAccelerationStructureBuildSizesNV(impl.res.m_device, &inputs, &sizesInfo);
      tempScratchSize = std::max(tempScratchSize, sizesInfo.updateScratchSize);
    }
    else
    {
      // when not doing implicit build, we want to query the sizes in advance.
      inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
      inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
      inputs.flags  = impl.templateBuildFlags;
      vkGetClusterAccelerationStructureBuildSizesNV(impl.res.m_device, &inputs, &sizesInfo);
      tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);
    }

    // to know how big the clusters will be after instantiation we query their sizes
    inputs.opInput.pTriangleClusters = &templateTriangleInput;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.flags                     = impl.templateInstantiateFlags;
    vkGetClusterAccelerationStructureBuildSizesNV(impl.res.m_device, &inputs, &sizesInfo);
    tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);

    // let's setup temporary resources

    nvvk::Buffer scratchBuffer;
    impl.res.m_allocator.createBuffer(scratchBuffer, tempScratchSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    size_t infoSize = std::max(std::max(sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV),
                                        sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV)),
                               sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV));

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo vmaInfo{};
    vmaInfo.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    vmaInfo.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    vmaInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    nvvk::Buffer infosBuffer;
    bufferInfo.size = infoSize * numClusters;
    impl.res.m_allocator.createBuffer(infosBuffer, bufferInfo, vmaInfo);

    nvvk::Buffer sizesBuffer;
    bufferInfo.size = sizeof(uint32_t) * numClusters;
    impl.res.m_allocator.createBuffer(sizesBuffer, bufferInfo, vmaInfo);

    nvvk::Buffer dstAddressesBuffer;
    bufferInfo.size = sizeof(uint64_t) * numClusters;
    impl.res.m_allocator.createBuffer(dstAddressesBuffer, bufferInfo, vmaInfo);

    nvvk::Buffer bboxesBuffer;
    impl.res.m_allocator.createBuffer(bboxesBuffer, sizeof(TemplateBbox) * numClusters,
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

    if(!scratchBuffer.buffer || !infosBuffer.buffer || !sizesBuffer.buffer || !dstAddressesBuffer.buffer || !bboxesBuffer.buffer)
    {
      success = false;
    }

    if(success)
    {
      const float bloatSize = glm::length(geometry.bbox.hi - geometry.bbox.lo) * impl.config.templateBboxBloatPercentage;

      auto* templateInfosMapping =
          reinterpret_cast<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV*>(infosBuffer.mapping);

      for(uint32_t c = 0; c < numClusters; c++)
      {
        const animatedclusters::shaderio::Cluster&                        cluster      = data->clusters[c];
        VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& templateInfo = templateInfosMapping[c];

        // add bloat to original bbox

        TemplateBbox& tempBbox = ((TemplateBbox*)bboxesBuffer.mapping)[c];

        animatedclusters::shaderio::BBox clusterBbox = geometry.clusterBboxes[c];
        clusterBbox.lo -= bloatSize;
        clusterBbox.hi += bloatSize;

        tempBbox.bbox = clusterBbox;

        templateInfo = {0};

        // REMIX: clusterID indexes the global animated cluster table (the
        // sample used the geometry-local index c)
        templateInfo.clusterID     = data->globalClusterBase + c;
        templateInfo.vertexCount   = cluster.numVertices;
        templateInfo.triangleCount = cluster.numTriangles;
        templateInfo.baseGeometryIndexAndGeometryFlags.geometryFlags =
            data->opaque ? VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV : 0;

        // cluster-LOCAL topology (razor-triangle fix): the template build
        // indexes vertices in [0, vertexCount), so it gets the uint8 local
        // indices + the reference positions in cluster-local order. The
        // GLOBAL trianglesBuffer stays hit-side only (primitive remap).
        templateInfo.indexBuffer       = data->localTrianglesBuffer.address + cluster.firstLocalTriangle;
        templateInfo.indexBufferStride = sizeof(uint8_t);
        templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

        templateInfo.vertexBuffer       = positionsBuffer.address + sizeof(glm::vec3) * cluster.firstLocalVertex;
        templateInfo.vertexBufferStride = sizeof(glm::vec3);

        templateInfo.positionTruncateBitCount = impl.config.positionTruncateBits;

        templateInfo.instantiationBoundingBoxLimit =
            (impl.config.templateBboxBloatPercentage < 0 || !pending->useBboxLimit) ?
                0 :
                bboxesBuffer.address + sizeof(TemplateBbox) * c;
      }

      // actual count of current geometry
      inputs.maxAccelerationStructureCount = numClusters;

      VkCommandBuffer cmd;
      VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
      cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
      cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
      cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
      cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
      cmdInfo.dstSizesArray.size              = sizesBuffer.bufferSize;
      cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
      cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
      cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.bufferSize;
      cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
      cmdInfo.scratchData                     = scratchBuffer.address;

      if(useImplicitTemplates)
      {
        inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        inputs.flags  = impl.templateBuildFlags;

        cmdInfo.dstImplicitData = implicitBuffer.address;
      }
      else
      {
        // query size of templates
        inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        inputs.flags  = impl.templateBuildFlags;
      }

      cmd = impl.res.createTempCmdBuffer();

      cmdInfo.input = inputs;
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      impl.res.tempSyncSubmit(cmd);

      // compute template buffer sizes

      uint32_t buildSum = 0;
      for(uint32_t c = 0; c < numClusters; c++)
      {
        buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
      }
      // allocate outputs and setup dst addresses
      impl.res.m_allocator.createBuffer(data->templatesBuffer, buildSum, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);

      data->templateAddresses.resize(numClusters);

      if(!data->templatesBuffer.buffer)
      {
        success = false;
      }
      else if(useImplicitTemplates)
      {
        // after the implicit build, let's move from the scratch implicit buffer
        // to the final per-geometry buffer in a compacted fashion.

        // compute address / offset for each template
        uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);
        buildSum               = 0;

        auto* moveInfosMapping = reinterpret_cast<VkClusterAccelerationStructureMoveObjectsInfoNV*>(infosBuffer.mapping);

        for(uint32_t c = 0; c < numClusters; c++)
        {
          data->templateAddresses[c] = data->templatesBuffer.address + buildSum;
          uint32_t templateSize      = ((const uint32_t*)sizesBuffer.mapping)[c];

          // read from old address
          moveInfosMapping[c].srcAccelerationStructure = dstAddresses[c];
          // setup new dst address
          dstAddresses[c] = data->templateAddresses[c];

          assert(templateSize);

          buildSum += templateSize;
        }

        cmd = impl.res.createTempCmdBuffer();

        inputs.opType               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
        inputs.opMode               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        inputs.flags                = 0;
        inputs.opInput.pMoveObjects = &moveInput;

        cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
        cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
        cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);
        cmdInfo.dstSizesArray.deviceAddress     = 0;
        cmdInfo.dstSizesArray.size              = 0;
        cmdInfo.dstSizesArray.stride            = 0;
        cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
        cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.bufferSize;
        cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

        cmdInfo.input = inputs;
        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        impl.res.tempSyncSubmit(cmd);
      }
      else
      {
        uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);
        buildSum               = 0;
        for(uint32_t c = 0; c < numClusters; c++)
        {
          dstAddresses[c]            = data->templatesBuffer.address + buildSum;
          data->templateAddresses[c] = data->templatesBuffer.address + buildSum;
          buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
        }

        // build explicit
        inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        inputs.flags  = impl.templateBuildFlags;

        cmd = impl.res.createTempCmdBuffer();

        cmdInfo.input = inputs;
        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        impl.res.tempSyncSubmit(cmd);
      }

      if(success)
      {
        // now compute instantiation sizes
        data->instantiationOffsets.resize(numClusters);

        auto* instantiationInfosMapping =
            reinterpret_cast<VkClusterAccelerationStructureInstantiateClusterInfoNV*>(infosBuffer.mapping);

        for(uint32_t c = 0; c < numClusters; c++)
        {
          VkClusterAccelerationStructureInstantiateClusterInfoNV& instantiationInfo = instantiationInfosMapping[c];

          instantiationInfo.clusterIdOffset        = 0;
          instantiationInfo.clusterTemplateAddress = data->templateAddresses[c];
          instantiationInfo.geometryIndexOffset    = 0;
          // leave vertices off given we are looking for worst case instantiation size, not actual
          instantiationInfo.vertexBuffer.startAddress  = 0;
          instantiationInfo.vertexBuffer.strideInBytes = 0;
        }

        // query size of instantiations
        inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        inputs.flags  = impl.templateInstantiateFlags;
        inputs.opInput.pTriangleClusters = &templateTriangleInput;

        cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
        cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
        cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
        cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
        cmdInfo.dstSizesArray.size              = sizesBuffer.bufferSize;
        cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
        cmdInfo.dstAddressesArray.deviceAddress = 0;
        cmdInfo.dstAddressesArray.size          = 0;
        cmdInfo.dstAddressesArray.stride        = 0;

        cmd = impl.res.createTempCmdBuffer();

        cmdInfo.input = inputs;
        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        impl.res.tempSyncSubmit(cmd);

        // compute output offsets for instantiations, and total sum
        // this is later used for building the per-pose clusters
        uint32_t instantiationSum = 0;
        for(uint32_t c = 0; c < numClusters; c++)
        {
          data->instantiationOffsets[c] = instantiationSum;
          uint32_t instantiationSize    = ((const uint32_t*)sizesBuffer.mapping)[c];
          assert(instantiationSize);
          instantiationSum += instantiationSize;
        }

        data->sumInstantiationSizes = instantiationSum;
      }
    }

    // delete temp resources
    impl.res.m_allocator.destroyBuffer(scratchBuffer);
    impl.res.m_allocator.destroyBuffer(infosBuffer);
    impl.res.m_allocator.destroyBuffer(sizesBuffer);
    impl.res.m_allocator.destroyBuffer(dstAddressesBuffer);
    impl.res.m_allocator.destroyBuffer(bboxesBuffer);
    impl.res.m_allocator.destroyBuffer(implicitBuffer);
  }

  // bind-pose reference positions only feed the template build
  impl.res.m_allocator.destroyBuffer(positionsBuffer);

  if(!success)
  {
    LOGE("ClusterTemplateSystem: %s: template build FAILED\n", data->name.c_str());
    impl.res.m_allocator.destroyBuffer(data->trianglesBuffer);
    impl.res.m_allocator.destroyBuffer(data->localTrianglesBuffer);
    impl.res.m_allocator.destroyBuffer(data->localVerticesBuffer);
    impl.res.m_allocator.destroyBuffer(data->templatesBuffer);
    return false;
  }

  // ---- global cluster table records (REMIX: hit-side primitive remap) ----
  {
    std::vector<uint64_t> records(numClusters);
    for(uint32_t c = 0; c < numClusters; c++)
    {
      records[c] = data->trianglesBuffer.address + uint64_t(data->clusters[c].firstTriangle) * sizeof(glm::uvec3);
    }
    impl.res.simpleUploadBuffer(impl.clusterTableBuffer, sizeof(uint64_t) * data->globalClusterBase,
                                sizeof(uint64_t) * numClusters, records.data());
  }

  LOGI("ClusterTemplateSystem: %s: %u clusters, %u tris, %u verts%s\n", data->name.c_str(), numClusters,
       data->numTriangles, data->numVertices,
       impl.config.useTemplates ? lodclusters::formatMemorySize(data->templatesBuffer.bufferSize).insert(0, ", templates ").c_str() : "");

  // DIAG (2026-07-05): persistent geometry buffer ranges (append-only, freed
  // only at deinit; the CLAS build reads templateAddresses into templatesBuffer,
  // the index into trianglesBuffer). Match a fault address to rule these in/out.
  LOGI("[ClusterLOD] geom create: %s templates 0x%llx..0x%llx, triangles 0x%llx..0x%llx\n",
       data->name.c_str(),
       (unsigned long long)data->templatesBuffer.address, (unsigned long long)(data->templatesBuffer.address + data->templatesBuffer.bufferSize),
       (unsigned long long)data->trianglesBuffer.address, (unsigned long long)(data->trianglesBuffer.address + data->trianglesBuffer.bufferSize));

  // publish for main-thread adoption
  {
    std::lock_guard<std::mutex> lock(impl.readyMutex);

    Impl::ReadyEntry entry;
    entry.topologyKey   = pending->topologyKey;
    entry.geometryIndex = impl.geometryIndexNext++;
    entry.data          = std::move(data);
    impl.readyGeometries.push_back(std::move(entry));
  }

  return true;
}

std::vector<ClusterTemplateSystem::ReadyGeometry> ClusterTemplateSystem::drainReadyGeometries()
{
  Impl& impl = *m_impl;

  std::vector<ReadyGeometry> result;

  std::lock_guard<std::mutex> lock(impl.readyMutex);
  while(!impl.readyGeometries.empty())
  {
    Impl::ReadyEntry& entry = impl.readyGeometries.front();

    if(impl.geometries.size() <= entry.geometryIndex)
    {
      impl.geometries.resize(entry.geometryIndex + 1);
    }
    impl.maxGeometryClusters = std::max(impl.maxGeometryClusters, entry.data->numClusters);
    impl.geometries[entry.geometryIndex] = std::move(entry.data);

    result.push_back({entry.topologyKey, entry.geometryIndex});
    impl.readyGeometries.pop_front();
  }

  return result;
}

uint32_t ClusterTemplateSystem::createPoseSet(uint32_t geometryIndex)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || geometryIndex >= impl.geometries.size() || !impl.geometries[geometryIndex])
  {
    return ~0u;
  }

  const Impl::GeometryData& geometry = *impl.geometries[geometryIndex];

  const VkDeviceSize clasSize = impl.config.useTemplates ?
                                    VkDeviceSize(geometry.sumInstantiationSizes) :
                                    impl.singleExplicitClusterSize * geometry.numClusters;

  // one CLAS destination buffer per frame-in-flight slot (sample:
  // per-render-instance clusterBuffer usage, here cycled to avoid the
  // cross-frame overwrite race - see kRingSlots). All-or-nothing: on any
  // failure destroy the ones already made so we never leave a partial pose.
  nvvk::Buffer clasBuffers[Impl::kRingSlots];
  uint64_t     poseClasBytes = 0;
  for(uint32_t slot = 0; slot < Impl::kRingSlots; slot++)
  {
    if(impl.res.createBuffer(clasBuffers[slot], clasSize,
                             VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                 | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                 | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
           != VK_SUCCESS
       || !clasBuffers[slot].buffer)
    {
      LOGW("ClusterTemplateSystem: pose CLAS allocation failed (%s)\n", lodclusters::formatMemorySize(clasSize).c_str());
      for(uint32_t made = 0; made < slot; made++)
      {
        impl.res.m_allocator.destroyBuffer(clasBuffers[made]);
      }
      return ~0u;
    }
    poseClasBytes += clasBuffers[slot].bufferSize;
  }

  uint32_t poseSetId;
  if(!impl.poseSetFreeList.empty())
  {
    poseSetId = impl.poseSetFreeList.back();
    impl.poseSetFreeList.pop_back();
  }
  else
  {
    poseSetId = uint32_t(impl.poseSets.size());
    impl.poseSets.emplace_back();
  }

  Impl::PoseSet& poseSet = impl.poseSets[poseSetId];
  poseSet.geometryIndex  = geometryIndex;
  for(uint32_t slot = 0; slot < Impl::kRingSlots; slot++)
  {
    poseSet.clasBuffers[slot] = clasBuffers[slot];
  }
  poseSet.active = true;

  impl.activePoseSets++;
  impl.poseClasBytes += poseClasBytes;

  // DIAG (2026-07-05): per-pose CLAS ranges (persistent until releasePoseSet;
  // the BLAS build reads these via the dst-address array). Match a fault address.
  LOGI("[ClusterLOD] poseClas create: set %u geom %u ranges 0x%llx..0x%llx / 0x%llx..0x%llx / 0x%llx..0x%llx / 0x%llx..0x%llx\n",
       poseSetId, geometryIndex,
       (unsigned long long)clasBuffers[0].address, (unsigned long long)(clasBuffers[0].address + clasBuffers[0].bufferSize),
       (unsigned long long)clasBuffers[1].address, (unsigned long long)(clasBuffers[1].address + clasBuffers[1].bufferSize),
       (unsigned long long)clasBuffers[2].address, (unsigned long long)(clasBuffers[2].address + clasBuffers[2].bufferSize),
       (unsigned long long)clasBuffers[3].address, (unsigned long long)(clasBuffers[3].address + clasBuffers[3].bufferSize));

  return poseSetId;
}

void ClusterTemplateSystem::releasePoseSet(uint32_t poseSetId)
{
  Impl& impl = *m_impl;

  if(poseSetId >= impl.poseSets.size() || !impl.poseSets[poseSetId].active)
  {
    return;
  }

  Impl::PoseSet& poseSet = impl.poseSets[poseSetId];

  for(nvvk::Buffer& clasBuffer : poseSet.clasBuffers)
  {
    impl.poseClasBytes -= clasBuffer.bufferSize;
    impl.deferDestroy(clasBuffer, "poseClas");  // in-flight frames may still trace it
  }
  impl.activePoseSets--;

  poseSet.geometryIndex = ~0u;
  poseSet.active        = false;

  impl.poseSetFreeList.push_back(poseSetId);
}

uint32_t ClusterTemplateSystem::getPoseSetClusterCount(uint32_t poseSetId) const
{
  const Impl& impl = *m_impl;

  if(poseSetId >= impl.poseSets.size() || !impl.poseSets[poseSetId].active)
  {
    return 0;
  }

  const Impl::PoseSet& poseSet = impl.poseSets[poseSetId];
  return impl.geometries[poseSet.geometryIndex] ? impl.geometries[poseSet.geometryIndex]->numClusters : 0;
}

void ClusterTemplateSystem::beginFrame(uint32_t frameId)
{
  Impl& impl = *m_impl;

  if(!impl.initialized)
  {
    return;
  }

  impl.currentFrameId.store(frameId);
  impl.processTrash(frameId, false);
}

bool ClusterTemplateSystem::recordFrame(VkCommandBuffer                            cmd,
                                        const PoseInput*                           poses,
                                        uint32_t                                   poseCount,
                                        const uint32_t*                            slotPoseIndex,
                                        const VkAccelerationStructureInstanceKHR*  tlasInstances,
                                        uint32_t                                   slotCount)
{
  Impl& impl = *m_impl;

  if(!impl.initialized || poseCount == 0 || slotCount == 0)
  {
    return false;
  }

  // frame totals
  uint32_t totalClusters  = 0;
  uint32_t totalTriangles = 0;
  uint32_t totalVertices  = 0;
  for(uint32_t p = 0; p < poseCount; p++)
  {
    const Impl::PoseSet& poseSet = impl.poseSets[poses[p].poseSetId];
    const Impl::GeometryData& geometry = *impl.geometries[poseSet.geometryIndex];
    totalClusters += geometry.numClusters;
    totalTriangles += geometry.numTriangles;
    // cluster-LOCAL vertex slots (razor-triangle fix): the build consumes the
    // gathered per-cluster layout, so both the op's maxTotalVertexCount and
    // the gather ring are sized by numClusterVertices, not the mesh count
    totalVertices += geometry.numClusterVertices;
  }

  if(!impl.ensureFrameCapacities(totalClusters, totalTriangles, totalVertices, poseCount, slotCount))
  {
    return false;
  }

  // chrono: advance the profiler frame, then bracket every phase below (Path A
  // parity - its renderer sections the same way in render())
  impl.profilerTimeline->frameAdvance();

  const uint32_t     ringIndex = impl.frameCounter % Impl::kRingSlots;
  impl.lastRecordedRingIndex   = ringIndex;  // getTlasInstancesBuffer's copy-out follows this recordFrame
  const nvvk::Buffer& ring     = impl.ringBuffers[ringIndex];
  uint8_t*            mapping  = ring.mapping;

  // ---- per-frame input fill (sample: initRayTracingTemplateInstantiations /
  //      initRayTracingClusters fill loops; per-frame here because Remix's
  //      skinned output buffers ping-pong every frame) ----

  // chrono (CPU-only section: the ring fill runs before any GPU commands)
  const nvutils::ProfilerTimeline::FrameSectionID inputFillSection = impl.profilerTimeline->frameBeginSection("Anim Input Fill");

  auto* dstAddresses = reinterpret_cast<uint64_t*>(mapping + impl.ringDstAddressesOffset);
  auto* blasInfos = reinterpret_cast<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV*>(mapping + impl.ringBlasInfosOffset);
  auto* renderInstances = reinterpret_cast<animatedclusters::shaderio::RenderInstance*>(mapping + impl.ringRenderInstancesOffset);

  const uint64_t ringDstAddressesVa = ring.address + impl.ringDstAddressesOffset;

  // razor-triangle fix: the CLAS build/instantiate indexes vertices in
  // [0, per-cluster vertexCount) - CLUSTER-LOCAL. The live buffers are in
  // GLOBAL vertex order, so the gather kernel reorders each pose's positions
  // into this frame's gather ring slot first; every per-cluster vertex input
  // below points into the gathered layout (gatherBase + firstLocalVertex).
  const nvvk::Buffer& gatherRing   = impl.gatherBuffers[ringIndex];
  const uint64_t      gatherBaseVa = gatherRing.address;
  uint32_t            gatherSlotOffset = 0;

  std::vector<animatedclusters::shaderio::GatherConstants> gatherDispatches(poseCount);

  uint32_t clusterOffset = 0;

  for(uint32_t p = 0; p < poseCount; p++)
  {
    const PoseInput&          pose     = poses[p];
    const Impl::PoseSet&      poseSet  = impl.poseSets[pose.poseSetId];
    const Impl::GeometryData& geometry = *impl.geometries[poseSet.geometryIndex];

    // this frame's ring slot - a different physical CLAS buffer than the slots
    // still being traced by earlier in-flight frames
    const uint64_t clasBase = poseSet.clasBuffers[ringIndex].address;

    // this pose's slice of the gathered positions (tightly packed vec3)
    const uint64_t poseGatherVa = gatherBaseVa + uint64_t(gatherSlotOffset) * sizeof(glm::vec3);

    animatedclusters::shaderio::GatherConstants& gather = gatherDispatches[p];
    gather.srcPositions   = pose.positionsAddress;
    gather.localVertexMap = geometry.localVerticesBuffer.address;
    gather.dstPositions   = poseGatherVa;
    gather.vertexCount    = geometry.numClusterVertices;
    gather.srcStrideBytes = pose.positionsStrideBytes;

    if(impl.config.useTemplates)
    {
      auto* instantiationInfos =
          reinterpret_cast<VkClusterAccelerationStructureInstantiateClusterInfoNV*>(mapping + impl.ringSrcInfosOffset);

      for(uint32_t c = 0; c < geometry.numClusters; c++)
      {
        const animatedclusters::shaderio::Cluster&               cluster  = geometry.clusters[c];
        VkClusterAccelerationStructureInstantiateClusterInfoNV& instInfo = instantiationInfos[clusterOffset + c];

        instInfo                        = {};
        instInfo.clusterIdOffset        = 0;  // stored in template
        instInfo.geometryIndexOffset    = 0;
        instInfo.clusterTemplateAddress = geometry.templateAddresses[c];

        // current-frame vertex data, gathered into the cluster-local layout
        // the template's baked (local) indices address
        instInfo.vertexBuffer.startAddress  = poseGatherVa + uint64_t(cluster.firstLocalVertex) * sizeof(glm::vec3);
        instInfo.vertexBuffer.strideInBytes = sizeof(glm::vec3);

        // destination (persistent per-pose CLAS memory)
        dstAddresses[clusterOffset + c] = clasBase + geometry.instantiationOffsets[c];
      }
    }
    else
    {
      auto* buildInfos =
          reinterpret_cast<VkClusterAccelerationStructureBuildTriangleClusterInfoNV*>(mapping + impl.ringSrcInfosOffset);

      for(uint32_t c = 0; c < geometry.numClusters; c++)
      {
        const animatedclusters::shaderio::Cluster&                 cluster   = geometry.clusters[c];
        VkClusterAccelerationStructureBuildTriangleClusterInfoNV&  buildInfo = buildInfos[clusterOffset + c];

        buildInfo = {0};

        // REMIX: clusterID indexes the global animated cluster table
        buildInfo.clusterID     = geometry.globalClusterBase + c;
        buildInfo.vertexCount   = cluster.numVertices;
        buildInfo.triangleCount = cluster.numTriangles;

        buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags =
            geometry.opaque ? VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV : 0;

        // cluster-LOCAL indices + gathered cluster-local vertex layout
        buildInfo.indexBuffer       = geometry.localTrianglesBuffer.address + cluster.firstLocalTriangle;
        buildInfo.indexBufferStride = sizeof(uint8_t);
        buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

        buildInfo.vertexBuffer       = poseGatherVa + uint64_t(cluster.firstLocalVertex) * sizeof(glm::vec3);
        buildInfo.vertexBufferStride = sizeof(glm::vec3);

        buildInfo.positionTruncateBitCount = impl.config.positionTruncateBits;

        // explicit worst-case destinations
        dstAddresses[clusterOffset + c] = clasBase + impl.singleExplicitClusterSize * c;
      }
    }

    gatherSlotOffset += geometry.numClusterVertices;

    // BLAS build input: this pose's cluster references (sample:
    // initRayTracingBlas - clusterReferences into the dst address array)
    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV& blasInfo = blasInfos[p];
    blasInfo                        = {0};
    blasInfo.clusterReferences      = ringDstAddressesVa + sizeof(uint64_t) * clusterOffset;
    blasInfo.clusterReferencesCount = geometry.numClusters;
    blasInfo.clusterReferencesStride = sizeof(uint64_t);

    clusterOffset += geometry.numClusters;
  }

  // slot -> pose(BLAS) index for the kernel's static branch
  for(uint32_t s = 0; s < slotCount; s++)
  {
    renderInstances[s]            = {};
    renderInstances[s].geometryID = slotPoseIndex[s];
  }

  // CPU-known TlasInstance fields
  std::memcpy(mapping + impl.ringTlasOffset, tlasInstances, sizeof(VkAccelerationStructureInstanceKHR) * slotCount);

  impl.profilerTimeline->frameEndSection(inputFillSection);

  //////////////////////////////////////////////////////////////////////////
  // command recording (sample: updateRayTracingScene sequence + barriers)

  // stage the TlasInstances into the kernel-patched device buffer
  {
    VkBufferCopy region;
    region.srcOffset = impl.ringTlasOffset;
    region.dstOffset = 0;
    region.size      = sizeof(VkAccelerationStructureInstanceKHR) * slotCount;
    vkCmdCopyBuffer(cmd, ring.buffer, impl.tlasInstancesBuffers[ringIndex].buffer, 1, &region);
  }

  vkCmdFillBuffer(cmd, impl.readbackBuffer.buffer, 0, sizeof(AnimatedReadback), 0);

  // NV-DXVK: Path B forensics - snapshot this frame's ring + pre-stamp (the
  // barrier below orders the fill before the builds, so a build fault cannot
  // lose it)
  const uint32_t dbgSlot  = impl.frameCounter & 1;
  const uint32_t dbgStamp = impl.frameCounter + 1;
  if(impl.dbgStampHost.buffer)
  {
    Impl::DbgFrameInfo& dbgInfo = impl.dbgFrames[dbgSlot];
    dbgInfo.stamp               = dbgStamp;
    dbgInfo.ringIndex           = ringIndex;
    dbgInfo.poseCount           = poseCount;
    dbgInfo.totalClusters       = totalClusters;
    dbgInfo.slotCount           = slotCount;
    dbgInfo.ringMapping         = mapping;
    dbgInfo.ringVa              = ring.address;
    dbgInfo.dstOff              = impl.ringDstAddressesOffset;
    dbgInfo.blasOff             = impl.ringBlasInfosOffset;
    dbgInfo.poseRanges.clear();
    dbgInfo.poseRanges.reserve(poseCount);
    for(uint32_t p = 0; p < poseCount; p++)
    {
      const Impl::PoseSet&      dbgPoseSet  = impl.poseSets[poses[p].poseSetId];
      const Impl::GeometryData& dbgGeometry = *impl.geometries[dbgPoseSet.geometryIndex];
      dbgInfo.poseRanges.push_back(
          {poses[p].positionsAddress,
           poses[p].positionsAddress + uint64_t(poses[p].positionsStrideBytes) * dbgGeometry.numVertices,
           uint64_t(poses[p].poseSetId)});
    }
    vkCmdFillBuffer(cmd, impl.dbgStampHost.buffer, sizeof(uint32_t) * (dbgSlot * 2 + 0), sizeof(uint32_t), dbgStamp);
  }

  // ---- per-pose vertex gather (razor-triangle fix) ----
  // Reorder each pose's live positions (global vertex order, gpu_skinning /
  // capture output) into this frame's gather ring slot in cluster-local
  // layout. Runs before the CLAS ops, which consume the gathered data.
  {
    auto timerSection = impl.profilerGpuTimer.cmdFrameSection(cmd, "Anim Vtx Gather");

    // wait for the live-buffer writes before the gather reads them: gpu_skinning
    // is COMPUTE, but mutating (kUpdateBVH) meshes' vertex data lands via
    // TRANSFER copies (cacheVertexDataOnGPU) - both sources must be covered
    VkMemoryBarrier gatherBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    gatherBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    gatherBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &gatherBarrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.gatherPipeline);
    for(uint32_t p = 0; p < poseCount; p++)
    {
      const animatedclusters::shaderio::GatherConstants& gather = gatherDispatches[p];
      if(gather.vertexCount == 0)
      {
        continue;
      }
      vkCmdPushConstants(cmd, impl.computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(animatedclusters::shaderio::GatherConstants), &gather);
      vkCmdDispatch(cmd, (gather.vertexCount + 63) / 64, 1, 1);
    }
  }

  // wait for the gather writes (and our staging writes) before the CLAS ops
  // read the gathered vertex data / build inputs
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run template instantiation or clas build (sample: updateRayTracingClusters,
  // explicit destinations into the persistent per-pose CLAS memory)
  {
    auto timerSection = impl.profilerGpuTimer.cmdFrameSection(cmd, "Anim Clas Instantiate");
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput =
        impl.makeTriangleClusterInput(impl.triangleCapacity, impl.vertexCapacity);

    inputs.maxAccelerationStructureCount = totalClusters;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType = impl.config.useTemplates ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV :
                                               VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opInput.pTriangleClusters = &triangleInput;
    inputs.flags = impl.config.useTemplates ? impl.templateInstantiateFlags : impl.clusterBuildFlags;

    cmdInfo.dstAddressesArray.deviceAddress = ringDstAddressesVa;
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * totalClusters;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmdInfo.dstSizesArray.deviceAddress = impl.dstSizesBuffer.address;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * totalClusters;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = ring.address + impl.ringSrcInfosOffset;
    cmdInfo.srcInfosArray.size = (impl.config.useTemplates ? sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) :
                                                             sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV))
                                 * totalClusters;
    cmdInfo.srcInfosArray.stride = impl.config.useTemplates ? sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) :
                                                              sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);

    cmdInfo.scratchData = impl.scratchBuffers[ringIndex].address;
    cmdInfo.input       = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run blas build (sample: updateRayTracingBlas - implicit destinations, the
  // generated blas addresses feed the patch kernel)
  {
    auto timerSection = impl.profilerGpuTimer.cmdFrameSection(cmd, "Anim Blas Build");

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    blasInput.maxClusterCountPerAccelerationStructure = std::max(impl.maxGeometryClusters, 1u);
    blasInput.maxTotalClusterCount                    = impl.clusterCapacity;

    inputs.maxAccelerationStructureCount = poseCount;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel  = &blasInput;
    inputs.flags                         = impl.clusterBlasFlags;

    // we feed the generated blas addresses directly into the patch kernel
    cmdInfo.dstAddressesArray.deviceAddress = impl.blasAddressesBuffer[ringIndex].address;
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * poseCount;
    cmdInfo.dstAddressesArray.stride        = sizeof(VkDeviceAddress);

    cmdInfo.dstSizesArray.deviceAddress = impl.blasSizesBuffer.address;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * poseCount;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = ring.address + impl.ringBlasInfosOffset;
    cmdInfo.srcInfosArray.size = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * poseCount;
    cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    // in implicit mode we provide one big chunk from which outputs are sub-allocated
    // (this frame's ring slot, distinct from the pools earlier frames are tracing)
    cmdInfo.dstImplicitData = impl.blasImplicitBuffers[ringIndex].address;

    cmdInfo.scratchData = impl.scratchBuffers[ringIndex].address;
    cmdInfo.input       = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
  }

  // NV-DXVK: Path B forensics post-stamp - executes only if the instantiation
  // and BLAS build above did not kill the device
  if(impl.dbgStampHost.buffer)
  {
    VkMemoryBarrier dbgBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    dbgBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    dbgBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &dbgBarrier, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd, impl.dbgStampHost.buffer, sizeof(uint32_t) * (dbgSlot * 2 + 1), sizeof(uint32_t), dbgStamp);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                             | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // fill in per slot blas addresses after the blas were built + statistics
  // (sample: updateRayTracingScene's cluster_blas_instances dispatches; the
  // "static" branch resolves slot -> pose through renderInstances.geometryID)
  {
    auto timerSection = impl.profilerGpuTimer.cmdFrameSection(cmd, "Anim Slot Patch");

    animatedclusters::shaderio::ClusterBlasConstants blasConstants = {};

    blasConstants.instanceCount = slotCount;
    blasConstants.sumCount      = poseCount;
    blasConstants.animated      = 0;
    blasConstants.instances     = ring.address + impl.ringRenderInstancesOffset;
    blasConstants.rayInstances  = impl.tlasInstancesBuffers[ringIndex].address;
    blasConstants.blasAddresses = impl.blasAddressesBuffer[ringIndex].address;
    blasConstants.sizes         = impl.blasSizesBuffer.address;
    blasConstants.sum           = impl.readbackBuffer.address + offsetof(AnimatedReadback, blasesSize);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.blasInstancesPipeline);

    vkCmdPushConstants(cmd, impl.computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(animatedclusters::shaderio::ClusterBlasConstants), &blasConstants);
    vkCmdDispatch(cmd, (std::max(blasConstants.instanceCount, blasConstants.sumCount) + CLUSTER_BLAS_WORKGROUP_SIZE - 1) / CLUSTER_BLAS_WORKGROUP_SIZE,
                  1, 1);

    // CLAS size statistics (sample: render() statistics dispatch)
    blasConstants.instanceCount = 0;
    blasConstants.sumCount      = totalClusters;
    blasConstants.sizes         = impl.dstSizesBuffer.address;
    blasConstants.sum           = impl.readbackBuffer.address + offsetof(AnimatedReadback, clustersSize);

    vkCmdPushConstants(cmd, impl.computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(animatedclusters::shaderio::ClusterBlasConstants), &blasConstants);
    vkCmdDispatch(cmd, (blasConstants.sumCount + CLUSTER_BLAS_WORKGROUP_SIZE - 1) / CLUSTER_BLAS_WORKGROUP_SIZE, 1, 1);
  }

  // patched TlasInstances are copied out by the caller; readback goes to host
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                             | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  {
    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = sizeof(AnimatedReadback) * ringIndex;
    region.size      = sizeof(AnimatedReadback);
    vkCmdCopyBuffer(cmd, impl.readbackBuffer.buffer, impl.readbackHostBuffer.buffer, 1, &region);
  }

  impl.frameCounter++;
  impl.anyFrameRecorded = true;

  return true;
}

VkBuffer ClusterTemplateSystem::getTlasInstancesBuffer() const
{
  return m_impl->tlasInstancesBuffers[m_impl->lastRecordedRingIndex].buffer;
}

uint64_t ClusterTemplateSystem::getClusterTableAddress() const
{
  return m_impl->clusterTableAddress.load();
}

bool ClusterTemplateSystem::getStats(AnimatedStats& outStats) const
{
  const Impl& impl = *m_impl;

  outStats = {};

  if(!impl.initialized)
  {
    return false;
  }

  uint64_t templateBytes = 0;
  uint64_t geometryBytes = 0;
  uint64_t totalClusters = 0;
  uint32_t registered    = 0;
  for(const auto& geometry : impl.geometries)
  {
    if(geometry)
    {
      registered++;
      totalClusters += geometry->numClusters;
      templateBytes += geometry->templatesBuffer.bufferSize;
      geometryBytes += geometry->trianglesBuffer.bufferSize;
    }
  }

  outStats.registeredGeometries = registered;
  outStats.activePoseSets       = impl.activePoseSets;
  outStats.totalClusters        = totalClusters;
  outStats.templateBytes        = templateBytes;
  outStats.geometryBytes        = geometryBytes + sizeof(uint64_t) * impl.clusterTableCapacity;
  outStats.clasBytes            = impl.poseClasBytes;
  outStats.blasReservedBytes    = impl.blasImplicitBuffers[0].bufferSize * Impl::kRingSlots;
  outStats.operationsBytes      = impl.scratchBuffers[0].bufferSize * Impl::kRingSlots + impl.dstSizesBuffer.bufferSize
                                  + impl.blasAddressesBuffer[0].bufferSize * Impl::kRingSlots + impl.blasSizesBuffer.bufferSize
                                  + impl.gatherBuffers[0].bufferSize * Impl::kRingSlots
                                  + impl.tlasInstancesBuffers[0].bufferSize * Impl::kRingSlots
                                  + (impl.ringBuffers[0].buffer ? impl.ringBuffers[0].bufferSize * Impl::kRingSlots : 0);

  if(impl.anyFrameRecorded && impl.readbackHostBuffer.mapping)
  {
    // oldest ring slot = most conservative (its GPU writes have certainly
    // landed once 4 frames passed)
    const uint32_t slot = impl.frameCounter % Impl::kRingSlots;
    const AnimatedReadback* readback =
        reinterpret_cast<const AnimatedReadback*>(impl.readbackHostBuffer.mapping) + slot;
    outStats.clasActualBytes = readback->clustersSize;
    outStats.blasActualBytes = readback->blasesSize;
  }

  return true;
}

bool ClusterTemplateSystem::getProfilerReportUtf8(std::string& outReport) const
{
  const Impl& impl = *m_impl;

  if(!impl.initialized || !impl.anyFrameRecorded)
  {
    outReport.clear();
    return false;
  }

  return formatProfilerReportUtf8(impl.profilerTimeline, outReport);
}

}  // namespace lodclusters_remix
