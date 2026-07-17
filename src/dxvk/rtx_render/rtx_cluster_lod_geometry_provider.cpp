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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <utility>

#include "rtx_types.h"
#include "rtx_hashing.h"
#include "rtx_options.h"
#include "../util/log/log.h"
#include "../util/util_string.h"
#include "../util/util_env.h"
#include "../util/util_once.h"

namespace dxvk {

  namespace {

    // chrono helpers (steady_clock: monotonic, ~41 ns per read - safe on the
    // per-draw CS-thread path)
    uint64_t elapsedUs(const std::chrono::steady_clock::time_point& since) {
      return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - since).count());
    }

    // ---- [PromoRefs] sidecar serialization ----
    // <candidateHash>.promorefs, little-endian:
    //   u32 magic 'PRF1' | u64 candidateHash | u64 topologyKey | u64 indicesHash
    //   u32 entryCount | entries
    // entry: i32 classQ | i32 subId | u64 restHash | u32 vertexCount
    //        | u32 nameLen | name bytes | f32 positions[vertexCount*3]
    // topologyKey + indicesHash gate restore: a content patch that changes the
    // mesh invalidates the whole file (it is rediscovered and rewritten once).
    constexpr uint32_t kPromoRefsMagic = 0x31465250u;  // 'PRF1'
    constexpr uint32_t kPromoRefsMaxEntries = 64u;     // sanity bound on read
    constexpr uint32_t kPromoRefsMaxVerts = 4u * 1024u * 1024u;

    struct PromoRefEntry {
      int32_t classQ = INT32_MIN;
      int32_t subId = 0;
      uint64_t restHash = 0;
      std::string name;
      std::vector<float> positions;  // 3 floats per vertex
    };

    struct PromoRefsFile {
      uint64_t candidateHash = 0;
      uint64_t topologyKey = 0;
      uint64_t indicesHash = 0;
      std::vector<PromoRefEntry> entries;
    };

    bool readPromoRefs(const std::string& pathUtf8, PromoRefsFile& out) {
      std::ifstream f(std::filesystem::u8path(pathUtf8), std::ios::binary);
      if (!f.is_open()) {
        return false;
      }
      auto rd = [&f](void* dst, size_t bytes) -> bool {
        f.read(reinterpret_cast<char*>(dst), std::streamsize(bytes));
        return f.good();
      };
      uint32_t magic = 0;
      uint32_t count = 0;
      if (!rd(&magic, 4) || magic != kPromoRefsMagic
          || !rd(&out.candidateHash, 8) || !rd(&out.topologyKey, 8) || !rd(&out.indicesHash, 8)
          || !rd(&count, 4) || count > kPromoRefsMaxEntries) {
        return false;
      }
      out.entries.resize(count);
      for (PromoRefEntry& e : out.entries) {
        uint32_t vertexCount = 0;
        uint32_t nameLen = 0;
        if (!rd(&e.classQ, 4) || !rd(&e.subId, 4) || !rd(&e.restHash, 8)
            || !rd(&vertexCount, 4) || vertexCount == 0 || vertexCount > kPromoRefsMaxVerts
            || !rd(&nameLen, 4) || nameLen > 1024u) {
          return false;
        }
        e.name.resize(nameLen);
        if (nameLen > 0 && !rd(e.name.data(), nameLen)) {
          return false;
        }
        e.positions.resize(size_t(vertexCount) * 3);
        if (!rd(e.positions.data(), e.positions.size() * sizeof(float))) {
          return false;
        }
      }
      return true;
    }

    bool writePromoRefs(const std::string& pathUtf8, const PromoRefsFile& in) {
      std::ofstream f(std::filesystem::u8path(pathUtf8), std::ios::binary | std::ios::trunc);
      if (!f.is_open()) {
        return false;
      }
      auto wr = [&f](const void* src, size_t bytes) {
        f.write(reinterpret_cast<const char*>(src), std::streamsize(bytes));
      };
      const uint32_t count = uint32_t(std::min<size_t>(in.entries.size(), kPromoRefsMaxEntries));
      wr(&kPromoRefsMagic, 4);
      wr(&in.candidateHash, 8);
      wr(&in.topologyKey, 8);
      wr(&in.indicesHash, 8);
      wr(&count, 4);
      for (uint32_t i = 0; i < count; i++) {
        const PromoRefEntry& e = in.entries[i];
        const uint32_t vertexCount = uint32_t(e.positions.size() / 3);
        const uint32_t nameLen = uint32_t(std::min<size_t>(e.name.size(), 1024u));
        wr(&e.classQ, 4);
        wr(&e.subId, 4);
        wr(&e.restHash, 8);
        wr(&vertexCount, 4);
        wr(&nameLen, 4);
        if (nameLen > 0) {
          wr(e.name.data(), nameLen);
        }
        wr(e.positions.data(), e.positions.size() * sizeof(float));
      }
      return f.good();
    }

    uint64_t nowUs() {
      return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    }

  }  // namespace

  // P4b topology identity (see the header comment). indexCount and topology are
  // included because non-indexed draws have no index content to hash - without
  // them any two non-indexed meshes with the same vertex count would collide
  // into one topology identity.
  uint64_t ClusterLodGeometryProvider::makeTopologyKey(const RasterGeometry& geometryData) {
    const XXH64_hash_t indicesHash = geometryData.hashes[HashComponents::Indices];
    const uint64_t key = XXH64(&indicesHash, sizeof(indicesHash), geometryData.vertexCount);
    const uint64_t streamSeed = (uint64_t(geometryData.topology) << 32) | geometryData.indexCount;
    return XXH64(&key, sizeof(key), streamSeed);
  }

  ClusterLodGeometryProvider::ClusterLodGeometryProvider(ConfigProvider configProvider, VerifyProvider verifyProvider,
                                                         AnimatedHandler animatedHandler, CapturedProcessedHandler capturedProcessedHandler)
    : m_configProvider(std::move(configProvider))
    , m_verifyProvider(std::move(verifyProvider))
    , m_animatedHandler(std::move(animatedHandler))
    , m_capturedProcessedHandler(std::move(capturedProcessedHandler)) {
    // P4c item 7: parallel queue drain. The single worker measured 3.4 s queue
    // waits during discovery floods (2026-07-04 17:30 log) once batch 5 sent
    // all captured geometry through the LOD pipeline.
    uint32_t workerCount = m_configProvider().processingWorkerCount;
    if (workerCount == 0) {
      const uint32_t hwThreads = std::max(1u, std::thread::hardware_concurrency());
      workerCount = std::min(4u, std::max(1u, hwThreads / 4));
    }
    m_workerThreads.reserve(workerCount);
    for (uint32_t i = 0; i < workerCount; i++) {
      m_workerThreads.emplace_back([this, i] {
        env::setThreadName(str::format("rtx-cluster-lod-process-", i).c_str());
        workerLoop();
      });
    }
    Logger::info(str::format("[ClusterLOD] geometry provider: ", workerCount, " processing worker(s)"));
  }

  ClusterLodGeometryProvider::~ClusterLodGeometryProvider() {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_stopping = true;
    }
    m_condition.notify_all();
    for (dxvk::thread& workerThread : m_workerThreads) {
      if (workerThread.joinable()) {
        workerThread.join();
      }
    }
  }

  void ClusterLodGeometryProvider::onDrawCallGeometry(const DrawCallState& drawCallState, uint64_t geometryHash, bool vertexDataUpdated, bool traceThis) {
    // DIAG (DrawTrace/provider): log this geometry's fate through the intake
    // fast-path. Throttled per geometry hash. Only when the manager flagged the
    // material match. Records which early-return (if any) drops it.
    auto traceProvider = [&](const char* stage, const char* fate) {
      if (!traceThis) {
        return;
      }
      static std::mutex s_mx;
      static std::unordered_map<uint64_t, uint32_t> s_last;
      std::lock_guard<std::mutex> lk(s_mx);
      uint32_t& last = s_last[geometryHash];
      // count-based throttle (no frame id here): log first sight + every ~256 hits
      if ((last++ % 256u) == 0u) {
        Logger::info(str::format("[DrawTrace/provider] geom 0x", std::hex, geometryHash, std::dec,
                                 " ", stage, ": ", fate));
      }
    };
    // chrono: record the CS-thread cost of this intake call on every exit path
    // (declared FIRST so it destructs LAST, after any scoped lock below has
    // released m_mutex - the destructor takes it again)
    struct IntakeChrono {
      ClusterLodGeometryProvider* self;
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      uint64_t snapshotUs = 0;
      bool tookSnapshot = false;
      ~IntakeChrono() {
        const uint64_t us = elapsedUs(start);
        std::unique_lock<std::mutex> lock(self->m_mutex);
        Stats& stats = self->m_stats;
        stats.intakeCalls++;
        stats.intakeUsTotal += us;
        stats.intakeUsMax = std::max(stats.intakeUsMax, us);
        if (tookSnapshot) {
          stats.snapshotCount++;
          stats.snapshotUsTotal += snapshotUs;
          stats.snapshotUsMax = std::max(stats.snapshotUsMax, snapshotUs);
        }
      }
    } intakeChrono { this };

    const RasterGeometry& geometryData = drawCallState.getGeometryData();

    // P4b routing (plan 7.1): deforming geometry goes to Path B, keyed by the
    // topology-stable key instead of the (position-including) asset hash.
    //  - skinned: bones on the draw; the CPU data IS the bind pose Path B needs
    //  - captured: vertex-captured draws (programmable-VS games) - the rendered
    //    mesh is the GPU capture buffer content, whose model->world transform
    //    exists only in the game's shader constants; the CPU input snapshot
    //    cannot be placed in that space, so Path A (static clusters in input
    //    space) would render it untransformed. Path B instantiates CLAS from
    //    the live capture-derived buffers, matching classic rendering by
    //    construction (user decision 2026-07-03).
    //  - mutating: an existing BlasEntry's vertex data changed in place
    //    (kUpdateBVH) - its asset hash churns every frame, so it can never be
    //    a Path A geometry; its topology is stable and clusterizes once
    const bool skinned = drawCallState.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const bool captured = drawCallState.preCaptureVertexData != nullptr;
    const bool deforming = skinned || captured || vertexDataUpdated;
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
          traceProvider("fastpath", "already-known deforming topology (dedup return; already registered)");
          return;
        }
      } else {
        if (m_mutatingTopologyKeys.find(topologyKey) != m_mutatingTopologyKeys.end()) {
          // a known-mutating mesh on a frame without an update (or its very
          // first sighting order) - not a Path A candidate
          traceProvider("fastpath", "topology seen mutating on another frame -> NOT a Path A candidate (return)");
          return;
        }
        if (m_knownHashes.find(geometryHash) != m_knownHashes.end()) {
          traceProvider("fastpath", "already-known static hash (dedup return; already processed)");
          return;
        }
      }
    }
    traceProvider("classify", str::format("skinned ", skinned, " captured ", captured,
                                          " vtxUpd ", vertexDataUpdated, " deforming ", deforming,
                                          " -> snapshotting").c_str());

    // Snapshot outside the lock: this copies the geometry's CPU staging data. This is
    // the only window where that data is guaranteed alive, so the copy must happen
    // here on the submission thread (design rule: no GPU->CPU readbacks, ever).
    const std::chrono::steady_clock::time_point snapshotStart = std::chrono::steady_clock::now();
    lodclusters_remix::GeometrySnapshot snapshot;
    const SnapshotResult snapshotResult = makeSnapshot(drawCallState, geometryHash, snapshot);
    intakeChrono.snapshotUs = elapsedUs(snapshotStart);
    intakeChrono.tookSnapshot = true;
    const bool eligible = snapshotResult == SnapshotResult::Eligible
                       || snapshotResult == SnapshotResult::EligibleConverted;
    snapshot.isDeforming = skinned;
    // 4a fix: isMutating means GENUINE CPU vertex-rewrite only, NOT capture. The
    // old `deforming && !skinned` folded captured draws in (deforming includes
    // captured), so every captured mesh became isMutating=true and isCaptured
    // below was mathematically unreachable -> 0 promotion candidates. A captured
    // mesh is content-stable (its "rewrite" is the GPU capture, not a CPU edit),
    // so it is a promotion candidate, not churn - separate the two here.
    const bool cpuMutating = vertexDataUpdated && !captured;
    snapshot.isMutating = cpuMutating && !skinned;
    // promotion candidates: captured AND content-stable. Mutating meshes stay
    // pure Path B - they can never promote (frozen snapshot vs live rewrites)
    // and their stable asset-rule hash + churning content hashes made every
    // session reprocess + overwrite the same .nvsngeo (see GeometrySnapshot)
    snapshot.isCaptured = captured && !skinned && !snapshot.isMutating;
    snapshot.topologyKey = topologyKey;

    // DIAG (trace): log each geometry's DRAW-CALL hash (geometry ^ material - what the
    // Remix object picker shows) beside its ClusterLOD classification, once per unique
    // geometry. ClusterLOD keys everything by the geometry-ONLY hash, which never
    // matches the picker, so this is the only way to locate a user-identified draw and
    // see whether it is a promotion candidate, mutating, skinned, or not eligible.
    {
      static std::mutex s_traceMutex;
      static std::unordered_set<uint64_t> s_traced;
      std::lock_guard<std::mutex> traceLock(s_traceMutex);
      if (s_traced.insert(geometryHash).second) {
        const auto& rule = RtxOptions::geometryAssetHashRule();
        // several hash variants so a picker/mod hash of ANY rule can be matched:
        // draw = geometry^material, geom = geometry-only, each in asset + legacy form.
        const XXH64_hash_t drawHash    = drawCallState.getHash(rule);
        const XXH64_hash_t drawLegacy  = drawCallState.getHashLegacy(rule);
        const XXH64_hash_t geomAsset   = geometryData.getHashForRule(rule);
        const XXH64_hash_t geomLegacy  = geometryData.getHashForRuleLegacy(rule);
        // material hash = the picker's "Material Hash" (cached; NO getImageHash() -
        // that derefs the live mip view and races texture streaming on this thread).
        const XXH64_hash_t matHash     = drawCallState.getMaterialData().getHash();
        Logger::info(str::format("[PromoTrace] draw=0x", std::hex, drawHash,
          " drawLegacy=0x", drawLegacy, " geom=0x", geomAsset, " geomLegacy=0x", geomLegacy,
          " mat=0x", matHash, " clHash=0x", geometryHash, std::dec,
          " skinned=", skinned, " captured=", captured, " vtxUpd=", vertexDataUpdated,
          " mutating=", snapshot.isMutating, " promoCandidate=", snapshot.isCaptured,
          " eligible=", eligible, " verts=", snapshot.vertexCount));
      }
    }

    std::unique_lock<std::mutex> lock(m_mutex);

    if (deforming) {
      // re-check: another draw of the same topology may have won the race
      if (!m_knownTopologyKeys.insert(topologyKey).second) {
        traceProvider("submit", "lost topology-key race (another draw registered it first; return)");
        return;
      }
      if (skinned || captured) {
        // skinned and vertex-captured meshes have stable asset hashes (input
        // data) - keep the cheap hash fast path for their later draws
        m_knownHashes.insert(geometryHash);
      }
    } else {
      if (!m_knownHashes.insert(geometryHash).second) {
        traceProvider("submit", "lost static-hash race (return)");
        return;
      }
    }

    if (!eligible) {
      m_stats.ineligible++;
      const char* why = snapshotResult == SnapshotResult::SkipTopology  ? "INELIGIBLE topology (unsupported primitive topology)"
                      : snapshotResult == SnapshotResult::SkipTooSmall  ? "INELIGIBLE tooSmall (below min triangle/vertex count)"
                      : snapshotResult == SnapshotResult::SkipFormat    ? "INELIGIBLE format (unsupported vertex/index format)"
                      : snapshotResult == SnapshotResult::SkipNoCpuData ? "INELIGIBLE noCpuData (no host-visible geometry to snapshot)"
                      : "INELIGIBLE (other)";
      traceProvider("eligibility", why);
      switch (snapshotResult) {
        case SnapshotResult::SkipTopology:  m_stats.skippedTopology++; break;
        case SnapshotResult::SkipTooSmall:  m_stats.skippedTooSmall++; break;
        case SnapshotResult::SkipFormat:    m_stats.skippedFormat++; break;
        case SnapshotResult::SkipNoCpuData: m_stats.skippedNoCpuData++; break;
        default: break;
      }
      return;
    }
    traceProvider("submit", captured ? "SUBMITTED as captured promotion candidate"
                                     : (skinned ? "SUBMITTED as skinned Path B"
                                                : (vertexDataUpdated ? "SUBMITTED as mutating Path B"
                                                                     : "SUBMITTED as static Path A")));

    if (snapshotResult == SnapshotResult::EligibleConverted) {
      m_stats.convertedTopology++;
    }

    m_stats.submitted++;
    m_stats.pending++;
    m_stats.pendingBytes += snapshot.approximateSizeBytes();
    snapshot.queuedAtUs = nowUs();  // chrono: worker reports the queue wait
    const size_t queueDepthAtEnqueue = m_queue.size() + 1;
    const uint64_t submittedTotal = m_stats.submitted;
    const uint64_t pendingTotal = m_stats.pending;
    m_queue.push_back(std::move(snapshot));

    // [GenTrace] discovery pulse: a NEW geometry hash was first-sighted this
    // draw and queued for processing. The cadence of these lines IS the intake
    // rate - a gap here (vs the render-generation "grew" cadence) proves a
    // residency stall is discovery-limited (game drew nothing new), while a
    // steady stream of these while residency is frozen proves the worker pool /
    // mesh optimiser is the bottleneck. queueDepth is the worker backlog.
    Logger::info(str::format("[GenTrace] ENQUEUE ", captured ? "captured" : (skinned ? "skinned"
                             : (vertexDataUpdated ? "mutating" : "static")),
                             " hash 0x", std::hex, geometryHash, std::dec,
                             " | queueDepth ", queueDepthAtEnqueue,
                             " | submitted(total) ", submittedTotal, " pending ", pendingTotal));

    lock.unlock();
    m_condition.notify_one();
  }

  void ClusterLodGeometryProvider::onReplacementGeometry(const RasterGeometry& geometryData, uint64_t geometryHash) {
    // P4c load-time intake (plan 7.1a): replacement meshes processed during the
    // load window, seconds before their first draw. Dedup shares m_knownHashes
    // with the draw-time route, so whichever sees a geometry first wins and the
    // other becomes a no-op.
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      if (!m_knownHashes.insert(geometryHash).second) {
        return;
      }
    }

    lodclusters_remix::GeometrySnapshot snapshot;
    const SnapshotResult result = makeReplacementSnapshot(geometryData, geometryHash, snapshot);

    std::unique_lock<std::mutex> lock(m_mutex);

    if (result != SnapshotResult::Eligible) {
      m_stats.ineligible++;
      switch (result) {
        case SnapshotResult::SkipTopology:  m_stats.skippedTopology++; break;
        case SnapshotResult::SkipTooSmall:  m_stats.skippedTooSmall++; break;
        case SnapshotResult::SkipFormat:    m_stats.skippedFormat++; break;
        case SnapshotResult::SkipNoCpuData: m_stats.skippedNoCpuData++; break;
        default: break;
      }
      return;
    }

    m_stats.submitted++;
    m_stats.pending++;
    m_stats.pendingBytes += snapshot.approximateSizeBytes();
    snapshot.queuedAtUs = nowUs();
    m_queue.push_back(std::move(snapshot));

    lock.unlock();
    m_condition.notify_one();
  }

  void ClusterLodGeometryProvider::savePromoRef(const lodclusters_remix::GeometrySnapshot& snapshot,
                                                const lodclusters_remix::ProcessorConfig& config) {
    // keyed by the ORIGINAL candidate hash; upserts this (classQ, subId) entry
    if (snapshot.promoKeyHash == 0 || snapshot.positions.empty()) {
      return;
    }
    const std::string path = lodclusters_remix::getPromoRefsFileUtf8(snapshot.promoKeyHash, config);
    std::lock_guard<std::mutex> lock(m_promoRefsMutex);
    PromoRefsFile file;
    if (!readPromoRefs(path, file)
        || file.candidateHash != snapshot.promoKeyHash
        || file.topologyKey != snapshot.topologyKey
        || file.indicesHash != snapshot.indicesHash) {
      // absent or stale (topology changed): start the sidecar fresh
      file = PromoRefsFile();
      file.candidateHash = snapshot.promoKeyHash;
      file.topologyKey = snapshot.topologyKey;
      file.indicesHash = snapshot.indicesHash;
    }
    PromoRefEntry* slot = nullptr;
    for (PromoRefEntry& e : file.entries) {
      if (e.classQ == snapshot.promoClassQ && e.subId == snapshot.promoClassSubId) {
        slot = &e;
        break;
      }
    }
    if (slot == nullptr) {
      if (file.entries.size() >= kPromoRefsMaxEntries) {
        return;
      }
      file.entries.emplace_back();
      slot = &file.entries.back();
    }
    slot->classQ = snapshot.promoClassQ;
    slot->subId = snapshot.promoClassSubId;
    slot->restHash = snapshot.geometryHash;
    slot->name = snapshot.name;
    slot->positions = snapshot.positions;
    if (writePromoRefs(path, file)) {
      Logger::info(str::format("[PromoRefs] saved rest reference: candidate 0x", std::hex,
                               snapshot.promoKeyHash, std::dec, " classQ ", snapshot.promoClassQ,
                               " sub ", snapshot.promoClassSubId, " (", file.entries.size(),
                               " entries, ", snapshot.positions.size() / 3, " verts)"));
    }
  }

  void ClusterLodGeometryProvider::restorePromoRefs(const lodclusters_remix::GeometrySnapshot& snapshot,
                                                    const lodclusters_remix::ProcessorConfig& config) {
    PromoRefsFile file;
    {
      std::lock_guard<std::mutex> lock(m_promoRefsMutex);
      if (!m_promoRefsRestored.insert(snapshot.geometryHash).second) {
        return;  // once per candidate per session
      }
      if (!readPromoRefs(lodclusters_remix::getPromoRefsFileUtf8(snapshot.geometryHash, config), file)) {
        return;  // no sidecar - nothing resolved for this candidate yet
      }
    }
    if (file.candidateHash != snapshot.geometryHash
        || file.topologyKey != snapshot.topologyKey
        || file.indicesHash != snapshot.indicesHash) {
      Logger::info(str::format("[PromoRefs] sidecar stale for candidate 0x", std::hex,
                               snapshot.geometryHash, std::dec, " (topology changed) - ignored"));
      return;
    }
    // candidate-level entry FIRST: its adoption resets the candidate's class
    // state, so it must not land after the sibling adoptions it would wipe
    std::stable_sort(file.entries.begin(), file.entries.end(),
                     [](const PromoRefEntry& a, const PromoRefEntry& b) {
                       return (a.classQ == INT32_MIN) > (b.classQ == INT32_MIN);
                     });
    uint32_t restored = 0;
    for (PromoRefEntry& e : file.entries) {
      lodclusters_remix::GeometrySnapshot rest;
      rest.geometryHash = e.restHash;
      rest.promoKeyHash = snapshot.geometryHash;
      rest.promoClassQ = e.classQ;
      rest.promoClassSubId = e.subId;
      rest.isRestCapture = true;
      rest.promoRestored = true;
      rest.name = e.name;
      rest.indices = snapshot.indices;  // same topology - indicesHash verified above
      rest.indicesHash = snapshot.indicesHash;
      rest.topologyKey = snapshot.topologyKey;
      rest.vertexCount = uint32_t(e.positions.size() / 3);
      rest.positions = std::move(e.positions);
      rest.verticesHash = XXH3_64bits(rest.positions.data(), rest.positions.size() * sizeof(float));
      enqueueRestSnapshot(std::move(rest));
      restored++;
    }
    if (restored > 0) {
      Logger::info(str::format("[PromoRefs] restored ", restored, " rest reference(s) for candidate 0x",
                               std::hex, snapshot.geometryHash, std::dec,
                               " - re-enqueued through the rest pipeline"));
    }
  }

  void ClusterLodGeometryProvider::enqueueRestSnapshot(lodclusters_remix::GeometrySnapshot&& snapshot) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_stopping) {
      return;
    }
    m_stats.submitted++;
    m_stats.pending++;
    m_stats.pendingBytes += snapshot.approximateSizeBytes();
    snapshot.queuedAtUs = nowUs();
    m_queue.push_back(std::move(snapshot));
    lock.unlock();
    m_condition.notify_one();
  }

  ClusterLodGeometryProvider::Stats ClusterLodGeometryProvider::getStats() const {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_stats;
  }

  std::vector<ClusterLodGeometryProvider::ReadyGeometry> ClusterLodGeometryProvider::drainReadyGeometries() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return std::exchange(m_readyGeometries, {});
  }

  ClusterLodGeometryProvider::SnapshotResult ClusterLodGeometryProvider::makeSnapshot(
      const DrawCallState& drawCallState,
      uint64_t geometryHash,
      lodclusters_remix::GeometrySnapshot& outSnapshot) {
    const RasterGeometry& geometryData = drawCallState.getGeometryData();

    // Vertex-captured draws (programmable-VS games): the RasterGeometry's vertex
    // buffers point at the GPU-only capture slice by the time the intake runs.
    // The CPU-visible INPUT data - the same staging copies the geometry hash was
    // computed from - travels in the pre-capture hold, so snapshot from there.
    // The index buffer is never replaced by capture and is read from the
    // geometry directly either way.
    const PreCaptureVertexData* preCapture = drawCallState.preCaptureVertexData.get();
    const RasterBuffer& positionBuffer = preCapture != nullptr ? preCapture->positionBuffer : geometryData.positionBuffer;
    const RasterBuffer& normalBuffer   = preCapture != nullptr ? preCapture->normalBuffer   : geometryData.normalBuffer;
    const RasterBuffer& texcoordBuffer = preCapture != nullptr ? preCapture->texcoordBuffer : geometryData.texcoordBuffer;

    // Accept the same triangle topologies the classic RT path accepts. Native lists
    // pass through; strips, fans and non-indexed draws are expanded to an indexed
    // triangle list on the CPU below - the exact conversion the classic path runs
    // on the GPU (gen_tri_list_index_buffer) because the BVH builders cannot consume
    // them either. The strip/fan index data is CPU-visible whenever list index data
    // is, so this involves no readback.
    const VkPrimitiveTopology topology = geometryData.topology;
    if (topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        && topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
        && topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: non-triangle topology ", std::dec, topology, " (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipTopology;
    }

    // primitive stream length: index count when indexed, vertex count otherwise
    // (same rules as RtxGeometryUtils::getOptimalTriangleListSize/generateTriangleList,
    // including a defined-but-empty index buffer counting as non-indexed)
    const bool usesIndices = geometryData.usesIndices() && geometryData.indexCount > 0;
    const uint32_t vertexCount = geometryData.vertexCount;
    const uint32_t primCount = usesIndices ? geometryData.indexCount : vertexCount;
    const uint32_t triangleCount = (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      ? primCount / 3
      : (primCount >= 3 ? primCount - 2 : 0);

    if (triangleCount == 0 || vertexCount < 3) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: too few vertices/primitives (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipTooSmall;
    }

    if (!positionBuffer.defined()) {
      // this was the ONLY silent skip - 18 replacement prims vanished into it on
      // 2026-07-04 before it logged (plan 7.1a); every skip reason must announce
      // its first occurrence and count in the stats line
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: no position buffer on the draw (stays classic; count of all such skips is in the stats log)")));
      return SnapshotResult::SkipNoCpuData;
    }

    const VkFormat positionFormat = positionBuffer.vertexFormat();
    if (positionFormat != VK_FORMAT_R32G32B32_SFLOAT && positionFormat != VK_FORMAT_R32G32B32A32_SFLOAT) {
      ONCE(Logger::info(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: unsupported position format ", std::dec, positionFormat, " (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipFormat;
    }

    // CPU-visible data required. Device-local-only sources (no mapPtr) stay on the
    // classic path and are surfaced in the log rather than silently read back.
    const void* indexPtr = usesIndices ? geometryData.indexBuffer.mapPtr() : nullptr;
    const uint8_t* positionPtr =
      (const uint8_t*) positionBuffer.mapPtr((size_t) positionBuffer.offsetFromSlice());

    if ((usesIndices && indexPtr == nullptr) || positionPtr == nullptr) {
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: no CPU-visible vertex/index data (stays classic; count of all such skips is in the stats log)")));
      return SnapshotResult::SkipNoCpuData;
    }

    // Bounds: never read past a source slice. The counts come from the draw call
    // but the buffers can legally be smaller than count x stride implies (and a
    // mismatched hold/override pairing MUST skip here, not overread the heap).
    if (usesIndices) {
      const size_t indexSizeBytes = geometryData.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16 ? 2 : 4;
      if (size_t(primCount) * indexSizeBytes > geometryData.indexBuffer.length()) {
        ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: index buffer smaller than indexCount implies (stays classic; count of all such skips is in the stats log)")));
        return SnapshotResult::SkipNoCpuData;
      }
    }
    if (size_t(positionBuffer.offsetFromSlice()) + size_t(positionBuffer.stride()) * (vertexCount - 1) + 3 * sizeof(float) > positionBuffer.length()) {
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: position buffer smaller than vertexCount x stride implies (stays classic; count of all such skips is in the stats log)")));
      return SnapshotResult::SkipNoCpuData;
    }

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

    // Indices: staged copies are rebased to 0 (D3D9Rtx::copyIndices subtracts the min
    // index). Strip/fan/non-indexed streams are expanded per triangle with the same
    // index derivation as generateIndices() in gen_tri_list_index_buffer.h. One
    // deliberate difference: the classic path must preserve the triangle COUNT (it
    // fills a pre-sized GPU buffer), so it collapses degenerate and out-of-range
    // triangles to a point; the snapshot has no such constraint, so those triangles
    // are dropped and the clusterizer gets clean input.
    const uint16_t* src16 = nullptr;
    const uint32_t* src32 = nullptr;
    if (usesIndices) {
      if (geometryData.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) {
        src16 = (const uint16_t*) indexPtr;
      } else {
        src32 = (const uint32_t*) indexPtr;
      }
    }
    // non-indexed draws use the identity mapping (vertex i is index i)
    const auto loadIndex = [&](uint32_t i) -> uint32_t {
      return src16 ? src16[i] : (src32 ? src32[i] : i);
    };

    outSnapshot.indices.reserve(size_t(triangleCount) * 3);

    for (uint32_t t = 0; t < triangleCount; t++) {
      uint32_t i0, i1, i2;
      switch (topology) {
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        // alternate winding correction, identical to gen_tri_list_index_buffer.h
        i0 = t;
        i1 = t + 1 + (t & 1);
        i2 = t + 2 - (t & 1);
        break;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
        i0 = 0;
        i1 = t + 1;
        i2 = t + 2;
        break;
      default:  // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        i0 = t * 3 + 0;
        i1 = t * 3 + 1;
        i2 = t * 3 + 2;
        break;
      }

      const uint32_t idx0 = loadIndex(i0);
      const uint32_t idx1 = loadIndex(i1);
      const uint32_t idx2 = loadIndex(i2);

      // degenerate (strips/fans connect adjacent primitives through them), or invalid
      if (idx0 == idx1 || idx0 == idx2 || idx1 == idx2 ||
          idx0 >= vertexCount || idx1 >= vertexCount || idx2 >= vertexCount) {
        continue;
      }

      outSnapshot.indices.push_back(idx0);
      outSnapshot.indices.push_back(idx1);
      outSnapshot.indices.push_back(idx2);
    }

    if (outSnapshot.indices.empty()) {
      ONCE(Logger::warn(str::format("[ClusterLOD] geometry 0x", std::hex, geometryHash, " skipped: all triangles degenerate or out of range (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipTooSmall;
    }

    // positions -> tightly packed vec3
    {
      const size_t strideBytes = positionBuffer.stride();

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
    if (normalBuffer.defined()) {
      const VkFormat normalFormat = normalBuffer.vertexFormat();
      if (normalFormat == VK_FORMAT_R32G32B32_SFLOAT || normalFormat == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const uint8_t* normalPtr =
          (const uint8_t*) normalBuffer.mapPtr((size_t) normalBuffer.offsetFromSlice());
        const size_t strideBytes = normalBuffer.stride();

        // optional attribute: on a short buffer just cluster without normals
        if (normalPtr != nullptr
            && size_t(normalBuffer.offsetFromSlice()) + strideBytes * (vertexCount - 1) + 3 * sizeof(float) <= normalBuffer.length()) {
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
    if (texcoordBuffer.defined()) {
      const VkFormat texcoordFormat = texcoordBuffer.vertexFormat();
      if (texcoordFormat == VK_FORMAT_R32G32_SFLOAT || texcoordFormat == VK_FORMAT_R32G32B32_SFLOAT
          || texcoordFormat == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const uint8_t* texcoordPtr =
          (const uint8_t*) texcoordBuffer.mapPtr((size_t) texcoordBuffer.offsetFromSlice());
        const size_t strideBytes = texcoordBuffer.stride();

        // optional attribute: on a short buffer just cluster without texcoords
        if (texcoordPtr != nullptr
            && size_t(texcoordBuffer.offsetFromSlice()) + strideBytes * (vertexCount - 1) + 2 * sizeof(float) <= texcoordBuffer.length()) {
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

    return (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && usesIndices)
      ? SnapshotResult::Eligible
      : SnapshotResult::EligibleConverted;
  }

  ClusterLodGeometryProvider::SnapshotResult ClusterLodGeometryProvider::makeReplacementSnapshot(
      const RasterGeometry& geometryData,
      uint64_t geometryHash,
      lodclusters_remix::GeometrySnapshot& outSnapshot) {
    // replacement meshes are authored as indexed uint32 triangle lists (the USD
    // importer's contract); anything else is unexpected and skips with a count
    if (geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        || !geometryData.usesIndices() || geometryData.indexCount == 0
        || geometryData.indexBuffer.indexType() != VK_INDEX_TYPE_UINT32) {
      ONCE(Logger::info(str::format("[ClusterLOD] replacement 0x", std::hex, geometryHash, " skipped at load: unexpected topology/index layout (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipTopology;
    }

    const uint32_t vertexCount = geometryData.vertexCount;
    const uint32_t triangleCount = geometryData.indexCount / 3;
    if (triangleCount == 0 || vertexCount < 3) {
      return SnapshotResult::SkipTooSmall;
    }

    const RasterBuffer& positionBuffer = geometryData.positionBuffer;
    if (!positionBuffer.defined()) {
      return SnapshotResult::SkipNoCpuData;
    }

    const VkFormat positionFormat = positionBuffer.vertexFormat();
    if (positionFormat != VK_FORMAT_R32G32B32_SFLOAT && positionFormat != VK_FORMAT_R32G32B32A32_SFLOAT) {
      ONCE(Logger::info(str::format("[ClusterLOD] replacement 0x", std::hex, geometryHash, " skipped at load: unsupported position format ", std::dec, positionFormat, " (count of all such skips is in the stats log)")));
      return SnapshotResult::SkipFormat;
    }

    const uint32_t* indexPtr = (const uint32_t*) geometryData.indexBuffer.mapPtr();
    const uint8_t* positionPtr =
      (const uint8_t*) positionBuffer.mapPtr((size_t) positionBuffer.offsetFromSlice());

    if (indexPtr == nullptr || positionPtr == nullptr) {
      // dynamic (skinned) replacements are device-local by design - they take
      // the draw-time Path B route; expected, so no warning
      return SnapshotResult::SkipNoCpuData;
    }

    // bounds (same guards as the draw-time snapshot)
    if (size_t(geometryData.indexCount) * sizeof(uint32_t) > geometryData.indexBuffer.length()) {
      ONCE(Logger::warn(str::format("[ClusterLOD] replacement 0x", std::hex, geometryHash, " skipped at load: index buffer smaller than indexCount implies")));
      return SnapshotResult::SkipNoCpuData;
    }
    if (size_t(positionBuffer.offsetFromSlice()) + size_t(positionBuffer.stride()) * (vertexCount - 1) + 3 * sizeof(float) > positionBuffer.length()) {
      ONCE(Logger::warn(str::format("[ClusterLOD] replacement 0x", std::hex, geometryHash, " skipped at load: position buffer smaller than vertexCount x stride implies")));
      return SnapshotResult::SkipNoCpuData;
    }

    outSnapshot = {};
    outSnapshot.name = str::format("asset_", std::hex, geometryHash);
    outSnapshot.geometryHash = geometryHash;
    outSnapshot.indicesHash = geometryData.hashes[HashComponents::Indices];
    outSnapshot.verticesHash = geometryData.hashes[HashComponents::VertexPosition];
    outSnapshot.vertexCount = vertexCount;
    outSnapshot.isDeforming = false;
    outSnapshot.isMutating = false;
    outSnapshot.topologyKey = makeTopologyKey(geometryData);

    // indices: drop degenerate/out-of-range triangles like the draw-time path
    outSnapshot.indices.reserve(size_t(triangleCount) * 3);
    for (uint32_t t = 0; t < triangleCount; t++) {
      const uint32_t idx0 = indexPtr[t * 3 + 0];
      const uint32_t idx1 = indexPtr[t * 3 + 1];
      const uint32_t idx2 = indexPtr[t * 3 + 2];

      if (idx0 == idx1 || idx0 == idx2 || idx1 == idx2 ||
          idx0 >= vertexCount || idx1 >= vertexCount || idx2 >= vertexCount) {
        continue;
      }

      outSnapshot.indices.push_back(idx0);
      outSnapshot.indices.push_back(idx1);
      outSnapshot.indices.push_back(idx2);
    }

    if (outSnapshot.indices.empty()) {
      return SnapshotResult::SkipTooSmall;
    }

    // positions -> tightly packed vec3
    {
      const size_t strideBytes = positionBuffer.stride();

      outSnapshot.positions.resize(size_t(vertexCount) * 3);
      for (uint32_t v = 0; v < vertexCount; v++) {
        const float* src = (const float*) (positionPtr + strideBytes * v);
        outSnapshot.positions[size_t(v) * 3 + 0] = src[0];
        outSnapshot.positions[size_t(v) * 3 + 1] = src[1];
        outSnapshot.positions[size_t(v) * 3 + 2] = src[2];
      }
    }

    // texcoords -> tightly packed vec2 (replacements interleave float2)
    const RasterBuffer& texcoordBuffer = geometryData.texcoordBuffer;
    if (texcoordBuffer.defined()) {
      const VkFormat texcoordFormat = texcoordBuffer.vertexFormat();
      if (texcoordFormat == VK_FORMAT_R32G32_SFLOAT || texcoordFormat == VK_FORMAT_R32G32B32_SFLOAT
          || texcoordFormat == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const uint8_t* texcoordPtr =
          (const uint8_t*) texcoordBuffer.mapPtr((size_t) texcoordBuffer.offsetFromSlice());
        const size_t strideBytes = texcoordBuffer.stride();

        if (texcoordPtr != nullptr
            && size_t(texcoordBuffer.offsetFromSlice()) + strideBytes * (vertexCount - 1) + 2 * sizeof(float) <= texcoordBuffer.length()) {
          outSnapshot.texcoords0.resize(size_t(vertexCount) * 2);
          for (uint32_t v = 0; v < vertexCount; v++) {
            const float* src = (const float*) (texcoordPtr + strideBytes * v);
            outSnapshot.texcoords0[size_t(v) * 2 + 0] = src[0];
            outSnapshot.texcoords0[size_t(v) * 2 + 1] = src[1];
          }
        }
      }
    }

    // normals: replacements store packed R32_UINT (octahedral) - same
    // "cluster without normals" behavior as the draw-time snapshot.
    // State bits: two-sided from the authored cull mode; alpha state is
    // per-draw and unknown at load (P2's per-instance opaqueStatus overrides
    // at render, sample parity).
    outSnapshot.twoSided = geometryData.cullMode == VK_CULL_MODE_NONE;
    outSnapshot.alphaMasked = false;

    return SnapshotResult::Eligible;
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

      // chrono: time spent waiting for the single worker - if this grows over
      // a session, discovery outpaces processing (raise processing.threadsPct)
      const double queuedMs = snapshot.queuedAtUs != 0 ? double(nowUs() - snapshot.queuedAtUs) * 1e-3 : 0.0;

      // 4a: isCaptured included. Captured meshes MUST register Path B every
      // session (unconditionally here, unlike the cache-gated interim path
      // below) - they render via cluster templates until promotion recovers
      // their transform and flips them to Path A. Without this they fall to the
      // static path and, on a cache hit, render as an UNTRANSFORMED Path A
      // render (see the routing comment at the top of onDrawCallGeometry).
      if (snapshot.isDeforming || snapshot.isMutating || snapshot.isCaptured) {
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

        const std::chrono::steady_clock::time_point registrationStart = std::chrono::steady_clock::now();
        bool registered = false;
        if (m_animatedHandler) {
          // template system + GPU work: single-caller by design, serialize
          // across the worker pool
          std::lock_guard<std::mutex> templateLock(m_templateSerialMutex);
          registered = m_animatedHandler(snapshot);
        }
        const double registrationMs = double(elapsedUs(registrationStart)) * 1e-3;

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
                                   " geometry registered for Path B (cluster templates)",
                                   " - queued ", queuedMs, " ms, registration ", registrationMs, " ms"));
        } else {
          Logger::info(str::format("[ClusterLOD] ", name, ": ", skinned ? "skinned" : "mutating",
                                   " geometry NOT registered for Path B - stays classic",
                                   " (queued ", queuedMs, " ms, attempt ", registrationMs, " ms)"));
        }

        // P4c promotion foundation (plan 7.7): captured static-topology
        // geometry ALSO runs the LOD pipeline at first sight - the .nvsngeo
        // cache and generation residency are then ready the moment the
        // promotion verdict machinery lands, and session 2+ is cache-instant.
        // Instances keep rendering Path B meanwhile: the captured branch of
        // isClusterInstance never consults Path A until promotion flips it.
        if (!(snapshot.isCaptured && m_configProvider().processCapturedGeometry)) {
          continue;
        }
      }

      const lodclusters_remix::ProcessorConfig config = m_configProvider();

      // P4c ladder: register interim cluster templates FIRST so the mesh renders
      // as cluster geometry within milliseconds while the LOD DAG build below
      // runs (seconds for large meshes). Pointless when the .nvsngeo already
      // exists - Path A then lands within the cache-hit cooldown - so cache hits
      // skip straight to processing. Template sets are TOPOLOGY-keyed and poses
      // read each instance's own buffers, so a topology someone already
      // registered (static or deforming) is reused, not duplicated. The handler
      // gates on the interimTemplates option and logs its own chrono.
      if (m_animatedHandler && !lodclusters_remix::geometryCacheFileExists(snapshot.geometryHash, config)) {
        // template-serial lock held across check + register: two workers with
        // different hashes but one topology must not both register a set
        std::lock_guard<std::mutex> templateLock(m_templateSerialMutex);

        bool topologyAlreadyRegistered = false;
        {
          std::unique_lock<std::mutex> lock(m_mutex);
          topologyAlreadyRegistered = m_knownTopologyKeys.find(snapshot.topologyKey) != m_knownTopologyKeys.end();
        }

        if (!topologyAlreadyRegistered) {
          const std::chrono::steady_clock::time_point interimStart = std::chrono::steady_clock::now();
          if (m_animatedHandler(snapshot)) {
            Logger::info(str::format("[ClusterLOD] ", snapshot.name,
                                     ": static geometry registered INTERIM cluster templates in ",
                                     double(elapsedUs(interimStart)) * 1e-3,
                                     " ms - renders as clusters until the LOD build lands (queued ",
                                     snapshot.queuedAtUs != 0 ? double(nowUs() - snapshot.queuedAtUs) * 1e-3 : 0.0, " ms)"));

            std::unique_lock<std::mutex> lock(m_mutex);
            m_knownTopologyKeys.insert(snapshot.topologyKey);
          }
        }
      }

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

          // ready for the next render generation (P2); cache hits ride the
          // manager's cooldown fast lane (P4c)
          m_readyGeometries.push_back(ReadyGeometry { snapshot.geometryHash, stats.loadedFromCache });
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
                               " in ", stats.processingMs, " ms (queued ", queuedMs, " ms)"));

      // P4c (plan 7.7): captured candidates get their promotion probe built +
      // uploaded here, on the worker, while the snapshot's CPU data is alive
      // (upload rides the template system's queue - serialize with it).
      // Rest-capture snapshots take the same handler: their probe references the
      // TRUE rendered shape, so the solve becomes identity for non-affine meshes.
      if ((snapshot.isCaptured || snapshot.isRestCapture) && m_capturedProcessedHandler) {
        std::lock_guard<std::mutex> templateLock(m_templateSerialMutex);
        m_capturedProcessedHandler(snapshot);
      }

      // [PromoRefs] cross-session persistence: a completed rest capture saves
      // its reference; a candidate's plain captured processing (which just
      // queued the candidate's probe pending above, so adoption ordering is
      // guaranteed) restores any references resolved in earlier sessions.
      if (snapshot.isRestCapture) {
        if (!snapshot.promoRestored) {  // restored entries came FROM the sidecar - nothing new to save
          savePromoRef(snapshot, config);
        }
      } else if (snapshot.isCaptured) {
        restorePromoRefs(snapshot, config);
      }

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
