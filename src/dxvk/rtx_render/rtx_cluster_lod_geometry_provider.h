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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "../util/thread.h"
#include "lodclusters/lodclusters_remix.h"

namespace dxvk {

  struct DrawCallState;

  // RTX Mega Geometry: CPU-snapshot intake for the cluster LOD pipeline.
  //
  // Runs on Remix's draw-submission path (CS thread): the first time a geometry hash
  // is seen, the draw call's CPU-visible geometry data (the same staging copies Remix's
  // geometry hashing reads - guaranteed alive in this window, never a GPU readback) is
  // snapshotted and queued. A single background worker feeds the snapshots through
  // NVIDIA's cluster LOD processing (lodclusters::Scene) and the per-geometry-hash
  // .nvsngeo disk cache.
  class ClusterLodGeometryProvider {
  public:
    using ConfigProvider = std::function<lodclusters_remix::ProcessorConfig()>;
    using VerifyProvider = std::function<bool()>;
    // P4b: worker-thread handler for Path B (deforming/mutating) snapshots -
    // ClusterLodManager runs the cluster-template registration (CPU
    // clusterization + GPU template build) inside it. Returns success.
    using AnimatedHandler = std::function<bool(lodclusters_remix::GeometrySnapshot&&)>;

    ClusterLodGeometryProvider(ConfigProvider configProvider, VerifyProvider verifyProvider, AnimatedHandler animatedHandler);
    ~ClusterLodGeometryProvider();

    ClusterLodGeometryProvider(const ClusterLodGeometryProvider&) = delete;
    ClusterLodGeometryProvider& operator=(const ClusterLodGeometryProvider&) = delete;

    // CS thread. Snapshots and enqueues the geometry on first sight of its hash
    // (Path A) or of its topology key (Path B: skinned draws, or an existing
    // BlasEntry whose vertex data updated in place - vertexDataUpdated).
    void onDrawCallGeometry(const DrawCallState& drawCallState, uint64_t geometryHash, bool vertexDataUpdated);

    struct Stats {
      uint64_t submitted = 0;        // unique geometries snapshotted + queued
      uint64_t pending = 0;          // still in the queue
      uint64_t processed = 0;        // Path A geometries fully processed
      uint64_t cacheHits = 0;        // of processed: served from .nvsngeo cache
      uint64_t failed = 0;           // processing failures
      uint64_t deforming = 0;        // Path B captures (skinned bind pose + mutating topology)
      uint64_t animatedReady = 0;    // of deforming: template sets registered
      uint64_t animatedFailed = 0;   // of deforming: registration failures
      uint64_t ineligible = 0;       // structurally impossible to snapshot (no CPU data / topology)
      uint64_t verified = 0;         // cache round-trips verified
      uint64_t verifyFailed = 0;     // cache round-trips that mismatched
      uint64_t pendingBytes = 0;     // CPU memory held by queued snapshots
      uint64_t totalClusters = 0;    // sum over processed geometries
      uint64_t totalTriangles = 0;   // sum over processed geometries (all lods)
    };

    Stats getStats() const;

    // Main thread (P2). Returns the geometry hashes whose cluster processing
    // completed since the last drain; each has a valid .nvsngeo cache file on
    // disk and is ready to join the next render generation.
    std::vector<uint64_t> drainReadyGeometries();

  private:
    // returns false if the draw call cannot be snapshotted (no CPU-visible data,
    // unsupported topology/formats)
    static bool makeSnapshot(const DrawCallState& drawCallState,
                             uint64_t geometryHash,
                             lodclusters_remix::GeometrySnapshot& outSnapshot);

    void workerLoop();

    ConfigProvider m_configProvider;
    VerifyProvider m_verifyProvider;
    AnimatedHandler m_animatedHandler;

    lodclusters_remix::GeometryProcessor m_processor;

    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<lodclusters_remix::GeometrySnapshot> m_queue;
    std::unordered_set<uint64_t> m_knownHashes;
    // P4b: Path B dedup (topology-stable keys) and the topologies whose asset
    // hash churns every frame (CPU-mutating vertex data): their per-frame "new"
    // hashes must never enter the Path A pipeline or the known-hash set.
    std::unordered_set<uint64_t> m_knownTopologyKeys;
    std::unordered_set<uint64_t> m_mutatingTopologyKeys;
    std::vector<uint64_t> m_readyHashes;
    bool m_stopping = false;

    Stats m_stats;

    dxvk::thread m_workerThread;
  };

}  // namespace dxvk
