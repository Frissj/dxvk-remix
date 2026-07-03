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

#include "rtx_cluster_lod_geometry_provider.h"

#include <algorithm>
#include <utility>

#include "rtx_types.h"
#include "rtx_hashing.h"
#include "../util/log/log.h"
#include "../util/util_string.h"
#include "../util/util_env.h"
#include "../util/util_once.h"

namespace dxvk {

  namespace {

    // P4b: topology-stable Path B identity - indices content hash + vertex
    // count. Unlike the asset hash it excludes positions, so it survives
    // skinning bind poses being identical across characters AND per-frame
    // CPU vertex rewrites.
    uint64_t makeTopologyKey(const RasterGeometry& geometryData) {
      const XXH64_hash_t indicesHash = geometryData.hashes[HashComponents::Indices];
      return XXH64(&indicesHash, sizeof(indicesHash), geometryData.vertexCount);
    }

  }  // namespace

  ClusterLodGeometryProvider::ClusterLodGeometryProvider(ConfigProvider configProvider, VerifyProvider verifyProvider, AnimatedHandler animatedHandler)
    : m_configProvider(std::move(configProvider))
    , m_verifyProvider(std::move(verifyProvider))
    , m_animatedHandler(std::move(animatedHandler))
    , m_workerThread([this] {
        env::setThreadName("rtx-cluster-lod-process");
        workerLoop();
      }) {
  }

  ClusterLodGeometryProvider::~ClusterLodGeometryProvider() {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_stopping = true;
    }
    m_condition.notify_all();
    if (m_workerThread.joinable()) {
      m_workerThread.join();
    }
  }

  void ClusterLodGeometryProvider::onDrawCallGeometry(const DrawCallState& drawCallState, uint64_t geometryHash, bool vertexDataUpdated) {
    const RasterGeometry& geometryData = drawCallState.getGeometryData();

    // P4b routing (plan 7.1): deforming geometry goes to Path B, keyed by the
    // topology-stable key instead of the (position-including) asset hash.
    //  - skinned: bones on the draw; the CPU data IS the bind pose Path B needs
    //  - mutating: an existing BlasEntry's vertex data changed in place
    //    (kUpdateBVH) - its asset hash churns every frame, so it can never be
    //    a Path A geometry; its topology is stable and clusterizes once
    const bool skinned = drawCallState.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const bool deforming = skinned || vertexDataUpdated;
    const uint64_t topologyKey = makeTopologyKey(geometryData);

    // fast path: identity already known (snapshotted, queued, processed or ineligible)
    {
      std::unique_lock<std::mutex> lock(m_mutex);

      if (deforming && !skinned) {
        // remember the churn source so its per-frame "new" asset hashes stop
        // being treated as fresh static geometry below (and never bloat
        // m_knownHashes - one entry per frame would grow unboundedly)
        m_mutatingTopologyKeys.insert(topologyKey);
      }

      if (deforming) {
        if (m_knownTopologyKeys.find(topologyKey) != m_knownTopologyKeys.end()) {
          return;
        }
      } else {
        if (m_mutatingTopologyKeys.find(topologyKey) != m_mutatingTopologyKeys.end()) {
          // a known-mutating mesh on a frame without an update (or its very
          // first sighting order) - not a Path A candidate
          return;
        }
        if (m_knownHashes.find(geometryHash) != m_knownHashes.end()) {
          return;
        }
      }
    }

    // Snapshot outside the lock: this copies the geometry's CPU staging data. This is
    // the only window where that data is guaranteed alive, so the copy must happen
    // here on the submission thread (design rule: no GPU->CPU readbacks, ever).
    lodclusters_remix::GeometrySnapshot snapshot;
    const bool eligible = makeSnapshot(drawCallState, geometryHash, snapshot);
    snapshot.isDeforming = skinned;
    snapshot.isMutating = deforming && !skinned;
    snapshot.topologyKey = topologyKey;

    std::unique_lock<std::mutex> lock(m_mutex);

    if (deforming) {
      // re-check: another draw of the same topology may have won the race
      if (!m_knownTopologyKeys.insert(topologyKey).second) {
        return;
      }
      if (skinned) {
        // a skinned mesh's asset hash is stable (bind-pose input) - keep the
        // cheap hash fast path for its later draws
        m_knownHashes.insert(geometryHash);
      }
    } else {
      if (!m_knownHashes.insert(geometryHash).second) {
        return;
      }
    }

    if (!eligible) {
      m_stats.ineligible++;
      return;
    }

    m_stats.submitted++;
    m_stats.pending++;
    m_stats.pendingBytes += snapshot.approximateSizeBytes();
    m_queue.push_back(std::move(snapshot));

    lock.unlock();
    m_condition.notify_one();
  }

  ClusterLodGeometryProvider::Stats ClusterLodGeometryProvider::getStats() const {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_stats;
  }

  std::vector<uint64_t> ClusterLodGeometryProvider::drainReadyGeometries() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return std::exchange(m_readyHashes, {});
  }

  bool ClusterLodGeometryProvider::makeSnapshot(const DrawCallState& drawCallState,
                                                uint64_t geometryHash,
                                                lodclusters_remix::GeometrySnapshot& outSnapshot) {
    const RasterGeometry& geometryData = drawCallState.getGeometryData();

    // triangle-list, indexed geometry only - the same constraint the classic RT path
    // enforces via isTopologyRaytraceReady before geometry reaches the BLAS builders
    if (geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: non-triangle-list topology ", std::dec, geometryData.topology)));
      return false;
    }

    if (!geometryData.usesIndices() || geometryData.indexCount < 3 || geometryData.vertexCount < 3) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: no indices or too few vertices")));
      return false;
    }

    if (!geometryData.positionBuffer.defined()) {
      return false;
    }

    const VkFormat positionFormat = geometryData.positionBuffer.vertexFormat();
    if (positionFormat != VK_FORMAT_R32G32B32_SFLOAT && positionFormat != VK_FORMAT_R32G32B32A32_SFLOAT) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: unsupported position format ", std::dec, positionFormat)));
      return false;
    }

    // CPU-visible data required. Device-local-only sources (no mapPtr) stay on the
    // classic path and are surfaced in the log rather than silently read back.
    const void* indexPtr = geometryData.indexBuffer.mapPtr();
    const uint8_t* positionPtr =
      (const uint8_t*) geometryData.positionBuffer.mapPtr((size_t) geometryData.positionBuffer.offsetFromSlice());

    if (indexPtr == nullptr || positionPtr == nullptr) {
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: no CPU-visible vertex/index data (stays classic)")));
      return false;
    }

    const uint32_t indexCount = geometryData.indexCount;
    const uint32_t vertexCount = geometryData.vertexCount;
    const uint32_t triangleCount = indexCount / 3;

    outSnapshot = {};
    outSnapshot.name = str::format("draw_", std::hex, geometryHash);
    outSnapshot.geometryHash = geometryHash;
    outSnapshot.indicesHash = geometryData.hashes[HashComponents::Indices];
    outSnapshot.verticesHash = geometryData.hashes[HashComponents::VertexPosition];
    outSnapshot.vertexCount = vertexCount;

    // deforming (GPU-skinned) geometry: this CPU data is the bind-pose mesh the game
    // submits - exactly the one-time clusterization input Path B (cluster templates)
    // needs. Routed away from the LOD pipeline.
    outSnapshot.isDeforming = drawCallState.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;

    // indices: staged copies are rebased to 0 (D3D9Rtx::copyIndices subtracts the min
    // index), truncate only in the defensive out-of-range case below
    outSnapshot.indices.resize(triangleCount * size_t(3));

    uint32_t maxIndex = 0;
    if (geometryData.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) {
      const uint16_t* src = (const uint16_t*) indexPtr;
      for (uint32_t i = 0; i < triangleCount * 3; i++) {
        const uint32_t value = src[i];
        outSnapshot.indices[i] = value;
        maxIndex = std::max(maxIndex, value);
      }
    } else {
      const uint32_t* src = (const uint32_t*) indexPtr;
      for (uint32_t i = 0; i < triangleCount * 3; i++) {
        const uint32_t value = src[i];
        outSnapshot.indices[i] = value;
        maxIndex = std::max(maxIndex, value);
      }
    }

    if (maxIndex >= vertexCount) {
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: index range exceeds vertex count")));
      return false;
    }

    // positions -> tightly packed vec3
    {
      const size_t strideBytes = geometryData.positionBuffer.stride();

      outSnapshot.positions.resize(size_t(vertexCount) * 3);
      for (uint32_t v = 0; v < vertexCount; v++) {
        const float* src = (const float*) (positionPtr + strideBytes * v);
        outSnapshot.positions[size_t(v) * 3 + 0] = src[0];
        outSnapshot.positions[size_t(v) * 3 + 1] = src[1];
        outSnapshot.positions[size_t(v) * 3 + 2] = src[2];
      }
    }

    // normals -> tightly packed vec3 (float formats only; packed R32_UINT normals from
    // vertex capture cannot be read as floats and are treated as absent)
    if (geometryData.normalBuffer.defined()) {
      const VkFormat normalFormat = geometryData.normalBuffer.vertexFormat();
      if (normalFormat == VK_FORMAT_R32G32B32_SFLOAT || normalFormat == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const uint8_t* normalPtr =
          (const uint8_t*) geometryData.normalBuffer.mapPtr((size_t) geometryData.normalBuffer.offsetFromSlice());

        if (normalPtr != nullptr) {
          const size_t strideBytes = geometryData.normalBuffer.stride();

          outSnapshot.normals.resize(size_t(vertexCount) * 3);
          for (uint32_t v = 0; v < vertexCount; v++) {
            const float* src = (const float*) (normalPtr + strideBytes * v);
            outSnapshot.normals[size_t(v) * 3 + 0] = src[0];
            outSnapshot.normals[size_t(v) * 3 + 1] = src[1];
            outSnapshot.normals[size_t(v) * 3 + 2] = src[2];
          }
        }
      } else {
        ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " has non-float normal format ", std::dec, normalFormat, ", clustering without normals")));
      }
    }

    // texcoords -> tightly packed vec2 (same float32-only rule as GeometryBufferData)
    if (geometryData.texcoordBuffer.defined()) {
      const VkFormat texcoordFormat = geometryData.texcoordBuffer.vertexFormat();
      if (texcoordFormat == VK_FORMAT_R32G32_SFLOAT || texcoordFormat == VK_FORMAT_R32G32B32_SFLOAT
          || texcoordFormat == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const uint8_t* texcoordPtr =
          (const uint8_t*) geometryData.texcoordBuffer.mapPtr((size_t) geometryData.texcoordBuffer.offsetFromSlice());

        if (texcoordPtr != nullptr) {
          const size_t strideBytes = geometryData.texcoordBuffer.stride();

          outSnapshot.texcoords0.resize(size_t(vertexCount) * 2);
          for (uint32_t v = 0; v < vertexCount; v++) {
            const float* src = (const float*) (texcoordPtr + strideBytes * v);
            outSnapshot.texcoords0[size_t(v) * 2 + 0] = src[0];
            outSnapshot.texcoords0[size_t(v) * 2 + 1] = src[1];
          }
        }
      }
    }

    // material state baked into cluster state bits (shading stays Remix's)
    outSnapshot.twoSided = geometryData.cullMode == VK_CULL_MODE_NONE;
    outSnapshot.alphaMasked = drawCallState.getMaterialData().alphaTestEnabled;

    return true;
  }

  void ClusterLodGeometryProvider::workerLoop() {
    while (true) {
      lodclusters_remix::GeometrySnapshot snapshot;

      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock, [this] { return m_stopping || !m_queue.empty(); });

        if (m_stopping && m_queue.empty()) {
          return;
        }

        snapshot = std::move(m_queue.front());
        m_queue.pop_front();

        m_stats.pending--;
        m_stats.pendingBytes -= snapshot.approximateSizeBytes();
      }

      if (snapshot.isDeforming || snapshot.isMutating) {
        // P4b Path B: cluster templates (vk_animated_clusters). The manager's
        // handler runs the one-time registration on this worker thread: CPU
        // clusterization of the topology, then the GPU template build under
        // Remix's submission lock.
        const std::string name = snapshot.name;
        const bool skinned = snapshot.isDeforming;

        {
          std::unique_lock<std::mutex> lock(m_mutex);
          m_stats.deforming++;
        }

        const bool registered = m_animatedHandler && m_animatedHandler(std::move(snapshot));

        {
          std::unique_lock<std::mutex> lock(m_mutex);
          if (registered) {
            m_stats.animatedReady++;
          } else {
            m_stats.animatedFailed++;
          }
        }

        if (registered) {
          Logger::info(str::format("[ClusterLOD] ", name, ": ", skinned ? "skinned" : "mutating",
                                   " geometry registered for Path B (cluster templates)"));
        } else {
          Logger::info(str::format("[ClusterLOD] ", name, ": ", skinned ? "skinned" : "mutating",
                                   " geometry NOT registered for Path B - stays classic"));
        }
        continue;
      }

      const lodclusters_remix::ProcessorConfig config = m_configProvider();

      lodclusters_remix::ProcessStats stats;
      const bool success = m_processor.processGeometry(snapshot, config, stats);

      {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (success) {
          m_stats.processed++;
          if (stats.loadedFromCache) {
            m_stats.cacheHits++;
          }
          m_stats.totalClusters += stats.totalClusters;
          m_stats.totalTriangles += stats.totalTriangles;

          // ready for the next render generation (P2)
          m_readyHashes.push_back(snapshot.geometryHash);
        } else {
          m_stats.failed++;
        }
      }

      if (!success) {
        Logger::warn(str::format("[ClusterLOD] ", snapshot.name, ": processing FAILED"));
        continue;
      }

      Logger::info(str::format("[ClusterLOD] ", snapshot.name,
                               ": tris ", snapshot.indices.size() / 3,
                               " verts ", snapshot.vertexCount,
                               " -> clusters ", stats.totalClusters,
                               " lodLevels ", stats.lodLevelsCount,
                               " totalTris ", stats.totalTriangles,
                               " cache ", stats.cacheFileSizeBytes, " bytes",
                               stats.loadedFromCache ? (stats.memoryMapped ? " (cache hit, mapped)" : " (cache hit)") : " (processed)",
                               " in ", stats.processingMs, " ms"));

      if (m_verifyProvider && m_verifyProvider()) {
        std::string message;
        const bool verified = m_processor.verifyCacheRoundTrip(snapshot, config, stats, message);

        {
          std::unique_lock<std::mutex> lock(m_mutex);
          if (verified) {
            m_stats.verified++;
          } else {
            m_stats.verifyFailed++;
          }
        }

        if (verified) {
          Logger::info(str::format("[ClusterLOD] ", snapshot.name, ": ", message));
        } else {
          Logger::err(str::format("[ClusterLOD] ", snapshot.name, ": cache round-trip FAILED: ", message));
        }
      }
    }
  }

}  // namespace dxvk
