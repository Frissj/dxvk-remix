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

#include <algorithm>
#include <cstring>
#include <mutex>

#include "rtx_cluster_lod_manager.h"
#include "rtx_cluster_lod_geometry_provider.h"
#include "rtx_hashing.h"
#include "rtx_accel_manager.h"
#include "rtx_camera_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_resources.h"
#include "rtx_types.h"
#include "rtx_options.h"
#include "../dxvk_device.h"
#include "../dxvk_context.h"
#include "../util/log/log.h"
#include "../util/util_string.h"
#include "../util/util_once.h"

namespace dxvk {

  namespace {

    // routes all of NVIDIA's lodclusters processing output (progress, statistics,
    // warnings) into the Remix log
    void nvproLogSink(int level, const char* message) {
      std::string text = str::format("[ClusterLOD/nvpro] ", message);

      // nvpro messages carry their own newlines
      while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
      }

      switch (level) {
      case 2: Logger::err(text); break;
      case 1: Logger::warn(text); break;
      default: Logger::info(text); break;
      }
    }

    // The compiled kernel variant matrix covers cluster sizes 64 and 128 (see the
    // //!variant annotations in shaders/rtx/pass/lodclusters); the SceneConfig has to
    // stay on one of those points or the P2+ GPU path could not consume the clusters.
    uint32_t validateClusterSize(int value, const char* what) {
      if (value != 64 && value != 128) {
        ONCE(Logger::warn(str::format("[ClusterLOD] ", what, " = ", value,
                                      " is not a compiled variant (64 or 128), clamping to 128")));
        return 128;
      }
      return uint32_t(value);
    }

    lodclusters_remix::ProcessorConfig buildProcessorConfig() {
      lodclusters_remix::ProcessorConfig config;

      config.clusterVertices = validateClusterSize(ClusterLodOptions::SceneConfig::clusterVertices(), "sceneConfig.clusterVertices");
      config.clusterTriangles = validateClusterSize(ClusterLodOptions::SceneConfig::clusterTriangles(), "sceneConfig.clusterTriangles");
      config.clusterGroupSize = uint32_t(std::max(1, ClusterLodOptions::SceneConfig::clusterGroupSize()));
      config.preferredNodeWidth = uint32_t(std::max(2, ClusterLodOptions::SceneConfig::preferredNodeWidth()));
      config.meshoptPreferRayTracing = ClusterLodOptions::SceneConfig::meshoptPreferRayTracing();
      config.useCompressedData = ClusterLodOptions::SceneConfig::useCompressedData();
      config.enabledAttributes = uint32_t(ClusterLodOptions::SceneConfig::enabledAttributes());
      config.meshoptFillWeight = ClusterLodOptions::SceneConfig::meshoptFillWeight();
      config.meshoptSplitFactor = ClusterLodOptions::SceneConfig::meshoptSplitFactor();
      config.lodLevelDecimationFactor = ClusterLodOptions::SceneConfig::lodLevelDecimationFactor();
      config.lodErrorMergePrevious = ClusterLodOptions::SceneConfig::lodErrorMergePrevious();
      config.lodErrorMergeAdditive = ClusterLodOptions::SceneConfig::lodErrorMergeAdditive();
      config.simplifyNormalWeight = ClusterLodOptions::SceneConfig::simplifyNormalWeight();
      config.simplifyTangentWeight = ClusterLodOptions::SceneConfig::simplifyTangentWeight();
      config.simplifyTangentSignWeight = ClusterLodOptions::SceneConfig::simplifyTangentSignWeight();
      config.simplifyTexCoordWeight = ClusterLodOptions::SceneConfig::simplifyTexCoordWeight();
      config.simplifyMaterialWeight = ClusterLodOptions::SceneConfig::simplifyMaterialWeight();
      config.compressionPosDropBits = uint32_t(ClusterLodOptions::SceneConfig::compressionPosDropBits());
      config.compressionTexDropBits = uint32_t(ClusterLodOptions::SceneConfig::compressionTexDropBits());

      config.processingThreadsPct = ClusterLodOptions::Processing::threadsPct();
      config.autoSaveCache = ClusterLodOptions::Processing::autoSaveCache();
      config.autoLoadCache = ClusterLodOptions::Processing::autoLoadCache();
      config.memoryMappedCache = ClusterLodOptions::Processing::memoryMappedCache();
      config.forcePreprocessMiB = uint64_t(std::max(1, ClusterLodOptions::Processing::forcePreprocessMiB()));

      // one .nvsngeo per geometry hash, per SceneConfig digest, next to mods/captures/logs
      config.cacheDirectoryUtf8 = "rtx-remix/cache/geometry";

      return config;
    }

    void writeMatrix(float out[16], const Matrix4& matrix) {
      static_assert(sizeof(Matrix4) == sizeof(float) * 16);
      std::memcpy(out, &matrix, sizeof(float) * 16);
    }

    // P4b: Path A/B discriminator inside the isClusterInstance ->
    // recordClusterInstance geometryId handoff (AccelManager passes the value
    // through opaquely)
    constexpr uint32_t kPathBTag = 0x80000000u;

    // P4b: topology-stable Path B identity - MUST match the provider's
    // makeTopologyKey (indices content hash + vertex count)
    uint64_t makeTopologyKey(const RasterGeometry& geometryData) {
      const XXH64_hash_t indicesHash = geometryData.hashes[HashComponents::Indices];
      return XXH64(&indicesHash, sizeof(indicesHash), geometryData.vertexCount);
    }

    lodclusters_remix::AnimatedConfig buildAnimatedConfig() {
      lodclusters_remix::AnimatedConfig config;

      config.clusterVertices = uint32_t(std::clamp(ClusterLodOptions::Animated::clusterVertices(), 8, 128));
      config.clusterTriangles = uint32_t(std::clamp(ClusterLodOptions::Animated::clusterTriangles(), 8, 128));
      config.useTemplates = ClusterLodOptions::Animated::useTemplates();
      config.useImplicitTemplates = ClusterLodOptions::Animated::useImplicitTemplates();
      config.templateBboxBloatPercentage = ClusterLodOptions::Animated::templateBboxBloatPercentage();
      config.positionTruncateBits = uint32_t(std::clamp(ClusterLodOptions::Animated::positionTruncateBits(), 0, 20));
      config.templateBuildFlags = ClusterLodOptions::Animated::templateBuildFastTrace()
        ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
      config.templateInstantiateFlags = ClusterLodOptions::Animated::instantiateFastTrace()
        ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
      config.clusterBuildFlags = config.templateInstantiateFlags;
      config.clusterBlasFlags = ClusterLodOptions::Animated::blasFastTrace()
        ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
      config.processingThreadsPct = ClusterLodOptions::Processing::threadsPct();
      config.maxPerFrameClusters = uint32_t(std::max(1024, ClusterLodOptions::Animated::maxPerFrameClusters()));

      return config;
    }

    uint32_t nextPowerOfTwo(uint32_t value) {
      uint32_t result = 1;
      while (result < value) {
        result <<= 1;
      }
      return result;
    }

  }  // namespace

  ClusterLodManager::ClusterLodManager(DxvkDevice* device)
    : m_device(device) {
    static std::once_flag s_logSinkOnce;
    std::call_once(s_logSinkOnce, [] {
      lodclusters_remix::installLogSink(&nvproLogSink);
    });

    const bool supported = checkIsSupported(device);

    Logger::info(str::format("[ClusterLOD] cluster LOD manager active. VK_NV_cluster_acceleration_structure ",
                             supported ? "supported - cluster rendering available"
                                       : "NOT supported - processing/caching only, rendering stays classic"));

    m_provider = std::make_unique<ClusterLodGeometryProvider>(
      &buildProcessorConfig,
      [] { return ClusterLodOptions::verifyCacheRoundTrip(); },
      [this](lodclusters_remix::GeometrySnapshot&& snapshot) {
        // provider worker thread (P4b Path B registration)
        return processAnimatedGeometry(std::move(snapshot));
      });
  }

  ClusterLodManager::~ClusterLodManager() {
    logStatistics();

    // join the provider worker BEFORE tearing down the systems its handler uses
    m_provider = nullptr;

    if (m_templateSystem != nullptr) {
      m_device->lockSubmission();
      m_templateSystem->deinit();
      m_device->unlockSubmission();
      m_templateSystem = nullptr;
    }

    if (m_renderSystem != nullptr) {
      m_device->lockSubmission();
      m_renderSystem->deinit();
      m_device->unlockSubmission();
      m_renderSystem = nullptr;
    }
  }

  bool ClusterLodManager::checkIsSupported(DxvkDevice* device) {
    if (device == nullptr) {
      return false;
    }

    const bool hasExtension = device->extensions().nvClusterAccelerationStructure;
    const bool hasFeature = device->features().nvClusterAccelerationStructureFeatures.clusterAccelerationStructure;

    // the ported kernels are compiled with SUBGROUP_SIZE 32 (NV hardware)
    const bool hasSubgroup32 = device->properties().coreSubgroup.subgroupSize == 32;

    return hasExtension && hasFeature && hasSubgroup32;
  }

  void ClusterLodManager::onDrawCallGeometry(const DrawCallState& drawCallState, bool vertexDataUpdated) {
    if (!ClusterLodOptions::enable()) {
      return;
    }

    // same stable identity that keys replacements and the runtime caches
    const XXH64_hash_t geometryHash = drawCallState.getHash(RtxOptions::geometryAssetHashRule());
    if (geometryHash == kEmptyHash) {
      return;
    }

    m_provider->onDrawCallGeometry(drawCallState, geometryHash, vertexDataUpdated);
  }

  // P4b: full Path B registration of one deforming geometry, on the provider's
  // worker thread: CPU clusterization (no Vulkan), then the GPU template build
  // with temporary submissions under the dxvk submission lock (RTXIO
  // precedent for off-thread queue use).
  bool ClusterLodManager::processAnimatedGeometry(lodclusters_remix::GeometrySnapshot&& snapshot) {
    if (!ClusterLodOptions::Animated::enable()) {
      return false;
    }

    if (!ensureTemplateSystem()) {
      return false;
    }

    const uint64_t token = m_templateSystem->clusterizeGeometry(snapshot);
    if (token == 0) {
      return false;
    }

    m_device->lockSubmission();
    const bool built = m_templateSystem->buildGeometryTemplates(token);
    m_device->unlockSubmission();

    return built;
  }

  bool ClusterLodManager::ensureTemplateSystem() {
    std::lock_guard<std::mutex> lock(m_templateSystemMutex);

    if (m_templateSystem != nullptr) {
      return true;
    }

    if (m_templateSystemFailed) {
      return false;
    }

    if (!checkIsSupported(m_device)) {
      ONCE(Logger::info("[ClusterLOD] cluster templates unavailable on this device - deforming geometry stays classic"));
      m_templateSystemFailed = true;
      return false;
    }

    lodclusters_remix::RenderDeviceInfo deviceInfo;
    deviceInfo.instance = m_device->instance()->handle();
    deviceInfo.physicalDevice = m_device->adapter()->handle();
    deviceInfo.device = m_device->handle();
    deviceInfo.graphicsQueueFamilyIndex = m_device->queues().graphics.queueFamily;
    deviceInfo.graphicsQueue = m_device->queues().graphics.queueHandle;
    deviceInfo.transferQueueFamilyIndex = m_device->queues().transfer.queueFamily;
    deviceInfo.transferQueue = m_device->queues().transfer.queueHandle;

    auto templateSystem = std::make_unique<lodclusters_remix::ClusterTemplateSystem>();

    m_device->lockSubmission();
    const bool initialized = templateSystem->init(deviceInfo, buildAnimatedConfig());
    m_device->unlockSubmission();

    if (!initialized) {
      Logger::err("[ClusterLOD] cluster template system initialization FAILED - deforming geometry stays classic");
      m_templateSystemFailed = true;
      return false;
    }

    m_templateSystem = std::move(templateSystem);

    Logger::info("[ClusterLOD] cluster template system initialized (Path B: deforming geometry)");
    return true;
  }

  void ClusterLodManager::logStatistics() const {
    if (m_provider == nullptr) {
      return;
    }

    const ClusterLodGeometryProvider::Stats stats = m_provider->getStats();

    Logger::info(str::format("[ClusterLOD] stats: submitted ", stats.submitted,
                             ", pending ", stats.pending,
                             " (", stats.pendingBytes / (1024 * 1024), " MiB)",
                             ", processed ", stats.processed,
                             " (cache hits ", stats.cacheHits, ")",
                             ", failed ", stats.failed,
                             ", deforming ", stats.deforming,
                             " (Path B ready ", stats.animatedReady,
                             ", failed ", stats.animatedFailed, ")",
                             ", ineligible ", stats.ineligible,
                             ", verified ", stats.verified,
                             ", verify failures ", stats.verifyFailed,
                             ", clusters ", stats.totalClusters,
                             ", cluster tris ", stats.totalTriangles));

    // P4b Path B (cluster templates)
    if (m_templateSystemMT != nullptr) {
      lodclusters_remix::AnimatedStats animatedStats;
      if (m_templateSystemMT->getStats(animatedStats)) {
        Logger::info(str::format("[ClusterLOD] animated: geometries ", animatedStats.registeredGeometries,
                                 ", poses ", animatedStats.activePoseSets,
                                 ", clusters ", animatedStats.totalClusters,
                                 ", templateMiB ", animatedStats.templateBytes / (1024 * 1024),
                                 ", topologyMiB ", animatedStats.geometryBytes / (1024 * 1024),
                                 ", clasMiB ", animatedStats.clasBytes / (1024 * 1024),
                                 " (actual ", animatedStats.clasActualBytes / (1024 * 1024), ")",
                                 ", blasReservedMiB ", animatedStats.blasReservedBytes / (1024 * 1024),
                                 " (actual ", animatedStats.blasActualBytes / (1024 * 1024), ")",
                                 ", opsMiB ", animatedStats.operationsBytes / (1024 * 1024)));
      }
    }

    if (m_renderSystem != nullptr && m_renderSystem->hasGeneration()) {
      lodclusters_remix::FrameStats frameStats;
      if (m_renderSystem->getFrameStats(frameStats)) {
        Logger::info(str::format("[ClusterLOD] render: generation ", m_generationCount,
                                 ", geometries ", m_geometryIdByHash.size(),
                                 ", renderClusters ", frameStats.numRenderClusters,
                                 ", blasBuilds ", frameStats.numBlasBuilds,
                                 ", blasBytes ", frameStats.blasActualSizeBytes,
                                 ", clasReservedBytes ", frameStats.reservedClasBytes,
                                 ", geoReservedBytes ", frameStats.reservedGeometryBytes));

        // P3: streaming residency/budget health. The couldNot* counters are
        // soft saturation (budget/table full) - persistent nonzero values mean
        // the rtx.clusterLod.streaming budgets are too small for the scene.
        if (frameStats.streaming) {
          Logger::info(str::format("[ClusterLOD] streaming: residentGroups ", frameStats.residentGroups,
                                   "/", frameStats.maxGroups,
                                   " (persistent ", frameStats.persistentGroups, ")",
                                   ", residentClusters ", frameStats.residentClusters,
                                   ", dataMiB ", frameStats.usedDataBytes / (1024 * 1024),
                                   "/", frameStats.maxDataBytes / (1024 * 1024),
                                   ", clasMiB ", frameStats.usedClasBytes / (1024 * 1024),
                                   " (wasted ", frameStats.wastedClasBytes / (1024 * 1024), ")",
                                   ", lastLoad ", frameStats.loadCount,
                                   ", lastUnload ", frameStats.unloadCount,
                                   ", pendingLoads ", frameStats.uncompletedLoadCount,
                                   ", transferKiB ", frameStats.transferBytes / 1024,
                                   ", couldNot g/c/t/s ", frameStats.couldNotAllocateGroup,
                                   "/", frameStats.couldNotAllocateClas,
                                   "/", frameStats.couldNotTransfer,
                                   "/", frameStats.couldNotStore));
        }
      }
    }
  }

  bool ClusterLodManager::ensureRenderSystem() {
    if (m_renderSystem != nullptr) {
      return true;
    }

    if (m_renderSystemFailed) {
      return false;
    }

    if (!checkIsSupported(m_device)) {
      ONCE(Logger::info("[ClusterLOD] cluster rendering unavailable on this device - geometry stays on the classic path"));
      m_renderSystemFailed = true;
      return false;
    }

    lodclusters_remix::RenderDeviceInfo deviceInfo;
    deviceInfo.instance = m_device->instance()->handle();
    deviceInfo.physicalDevice = m_device->adapter()->handle();
    deviceInfo.device = m_device->handle();
    deviceInfo.graphicsQueueFamilyIndex = m_device->queues().graphics.queueFamily;
    deviceInfo.graphicsQueue = m_device->queues().graphics.queueHandle;
    deviceInfo.transferQueueFamilyIndex = m_device->queues().transfer.queueFamily;
    deviceInfo.transferQueue = m_device->queues().transfer.queueHandle;

    lodclusters_remix::RenderConfig renderConfig;
    renderConfig.useSorting = ClusterLodOptions::Render::useSorting();
    renderConfig.useCulling = ClusterLodOptions::Render::useCulling();
    renderConfig.useBlasSharing = ClusterLodOptions::Render::useBlasSharing();
    renderConfig.useBlasMerging = ClusterLodOptions::Render::useBlasMerging();
    // P4: BLAS caching (streaming + sharing only; self-disables without)
    renderConfig.useBlasCaching = ClusterLodOptions::Render::useBlasCaching();
    renderConfig.maxBlasCachingMegaBytes = uint64_t(std::max(64, ClusterLodOptions::Streaming::maxBlasCachingMegaBytes()));
    renderConfig.useForcedInvisibleCulling = ClusterLodOptions::Render::useForcedInvisibleCulling();
    renderConfig.usePersistentTraversal = ClusterLodOptions::Render::usePersistentTraversal();

    // P3: streaming mode + budgets (lodclusters::StreamingConfig mirror)
    renderConfig.preferStreaming = ClusterLodOptions::Streaming::preferStreaming();
    renderConfig.useAsyncTransfer = ClusterLodOptions::Streaming::useAsyncTransfer();
    renderConfig.useDecoupledAsyncTransfer = ClusterLodOptions::Streaming::useDecoupledAsyncTransfer();
    renderConfig.usePersistentClasAllocator = ClusterLodOptions::Streaming::usePersistentClasAllocator();
    renderConfig.maxPerFrameLoadRequests = uint32_t(std::max(1, ClusterLodOptions::Streaming::maxPerFrameLoadRequests()));
    renderConfig.maxPerFrameUnloadRequests = uint32_t(std::max(1, ClusterLodOptions::Streaming::maxPerFrameUnloadRequests()));
    renderConfig.streamingMaxGroups = uint32_t(std::max(1024, ClusterLodOptions::Streaming::maxGroups()));
    renderConfig.streamingMaxClusters = uint32_t(std::max(0, ClusterLodOptions::Streaming::maxClusters()));
    renderConfig.maxTransferMegaBytes = uint64_t(std::max(1, ClusterLodOptions::Streaming::maxTransferMegaBytes()));
    renderConfig.maxGeometryMegaBytes = uint64_t(std::max(64, ClusterLodOptions::Streaming::maxGeometryMegaBytes()));
    renderConfig.maxClasMegaBytes = uint64_t(std::max(64, ClusterLodOptions::Streaming::maxClasMegaBytes()));
    renderConfig.clasAllocatorSectorSizeShift = uint32_t(std::clamp(ClusterLodOptions::Streaming::clasAllocatorSectorSizeShift(), 6, 20));
    renderConfig.clasAllocatorGranularityShift = uint32_t(std::clamp(ClusterLodOptions::Streaming::clasAllocatorGranularityShift(), 0, 8));
    renderConfig.numRenderClusterBits = uint32_t(std::clamp(ClusterLodOptions::Render::numRenderClusterBits(), 10, 24));
    renderConfig.numTraversalTaskBits = uint32_t(std::clamp(ClusterLodOptions::Render::numTraversalTaskBits(), 10, 24));
    renderConfig.clusterBlasFlags = ClusterLodOptions::Render::blasFastTrace()
      ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
      : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    renderConfig.clasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    renderConfig.clasPositionTruncateBits = uint32_t(std::clamp(ClusterLodOptions::Render::positionTruncateBits(), 0, 20));
    renderConfig.maxRenderInstances = uint32_t(std::max(64, ClusterLodOptions::Render::maxRenderInstances()));
    renderConfig.maxGeometries = uint32_t(std::max(64, ClusterLodOptions::Render::maxGeometries()));

    m_renderSystem = std::make_unique<lodclusters_remix::ClusterRenderSystem>();

    // the library submits (dummy HiZ transition) during init; Vulkan queues need
    // external synchronization against dxvk's submission thread
    m_device->lockSubmission();
    const bool initialized = m_renderSystem->init(deviceInfo, renderConfig);
    m_device->unlockSubmission();

    if (!initialized) {
      Logger::err("[ClusterLOD] cluster render system initialization FAILED - rendering stays classic");
      m_renderSystem = nullptr;
      m_renderSystemFailed = true;
      return false;
    }

    m_streamingActive = renderConfig.preferStreaming;
    m_asyncTransferActive = renderConfig.preferStreaming && renderConfig.useAsyncTransfer;
    // P4: kernel variants are picked at start; the HiZ feed follows this
    // captured state, not the live option (see dispatchBuild)
    m_cullingActive = renderConfig.useCulling;

    Logger::info(str::format("[ClusterLOD] cluster render system initialized (",
                             m_streamingActive ? "streaming" : "preloaded", ")"));
    return true;
  }

  void ClusterLodManager::buildGenerationIfDue(AccelManager& accelManager, InstanceManager& instanceManager) {
    // capacity growth request from last frame's overflow
    uint32_t requestedCapacity = m_renderSystem->getMaxRenderInstances();
    if (m_frameOverflowCount > 0) {
      requestedCapacity = nextPowerOfTwo(m_peakInstanceCount + m_frameOverflowCount + 1);
    }

    const bool needsCapacityGrowth = requestedCapacity > m_renderSystem->getMaxRenderInstances();

    if (m_pendingGeometryHashes.empty() && !(needsCapacityGrowth && !m_residentGeometryHashes.empty())) {
      return;
    }

    // batch generation updates
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const uint32_t cooldown = uint32_t(std::max(1, ClusterLodOptions::Render::generationCooldownFrames()));
    if (m_generationCount > 0 && currentFrame - m_lastGenerationFrame < cooldown) {
      return;
    }

    const lodclusters_remix::ProcessorConfig processorConfig = buildProcessorConfig();
    const std::string configDigest = lodclusters_remix::getConfigCacheDigestUtf8(processorConfig);

    // P2.5: while the running generation can absorb them, newly processed
    // geometries join incrementally - O(new) upload and CLAS/low-detail-BLAS
    // build, resident geometry untouched, no device-wait-idle swap. Existing
    // geometryIDs stay valid, the new ones extend the table.
    if (m_renderSystem->hasGeneration() && !needsCapacityGrowth
        && configDigest == m_generationConfigDigest && !m_pendingGeometryHashes.empty()) {
      std::vector<std::string> pendingFiles;
      pendingFiles.reserve(m_pendingGeometryHashes.size());
      for (const uint64_t hash : m_pendingGeometryHashes) {
        pendingFiles.push_back(lodclusters_remix::getGeometryCacheFileUtf8(hash, processorConfig));
      }

      m_device->lockSubmission();
      const lodclusters_remix::ClusterRenderSystem::AppendResult appendResult =
        m_renderSystem->appendToGeneration(pendingFiles, m_pendingGeometryHashes);
      m_device->unlockSubmission();

      if (appendResult == lodclusters_remix::ClusterRenderSystem::AppendResult::Ok) {
        const std::vector<lodclusters_remix::GeometryRenderInfo>& infos = m_renderSystem->getGeometryRenderInfos();
        for (uint32_t geometryId = uint32_t(m_residentGeometryHashes.size()); geometryId < infos.size(); geometryId++) {
          m_geometryIdByHash.emplace(infos[geometryId].geometryHash, geometryId);
        }

        m_residentGeometryHashes.insert(m_residentGeometryHashes.end(),
                                        m_pendingGeometryHashes.begin(), m_pendingGeometryHashes.end());

        Logger::info(str::format("[ClusterLOD] render generation ", m_generationCount, " grew by ",
                                 m_pendingGeometryHashes.size(), " geometries (", infos.size(), " total)"));

        m_pendingGeometryHashes.clear();
        m_lastGenerationFrame = currentFrame;

        // instances of the appended geometries flip classic -> cluster: their
        // cached BLAS buckets are stale and the full-skip fast path must not
        // reuse last frame's TLAS instance list (risk R8)
        accelManager.invalidateBucketCache();
        instanceManager.notifySceneChanged();
        return;
      }

      if (appendResult == lodclusters_remix::ClusterRenderSystem::AppendResult::Failed) {
        // unreadable/invalid cache entries; the generation itself is unchanged.
        // Drop the batch so a bad file cannot wedge the pipeline - those
        // geometries stay on the classic path.
        Logger::err(str::format("[ClusterLOD] appending ", m_pendingGeometryHashes.size(),
                                " geometries FAILED - they stay on the classic path"));
        m_pendingGeometryHashes.clear();
        m_lastGenerationFrame = currentFrame;
        return;
      }

      // AppendResult::NeedsRebuild: fall through to the full rebuild below.
      // (The combined Scene may already hold the appended views; the rebuild
      // re-assembles everything from scratch, so that is harmless.)
    }

    // full rebuild: bootstrap (first generation), instance-capacity growth,
    // SceneConfig change, or an append that exceeded the generation's sizing
    m_residentGeometryHashes.insert(m_residentGeometryHashes.end(),
                                    m_pendingGeometryHashes.begin(), m_pendingGeometryHashes.end());
    m_pendingGeometryHashes.clear();

    std::vector<std::string> cacheFiles;
    cacheFiles.reserve(m_residentGeometryHashes.size());
    for (const uint64_t hash : m_residentGeometryHashes) {
      cacheFiles.push_back(lodclusters_remix::getGeometryCacheFileUtf8(hash, processorConfig));
    }

    m_device->lockSubmission();
    const bool built = m_renderSystem->buildGeneration(cacheFiles, m_residentGeometryHashes, requestedCapacity);
    m_device->unlockSubmission();

    m_lastGenerationFrame = currentFrame;
    m_frameOverflowCount = 0;

    m_geometryIdByHash.clear();

    if (!built) {
      // isClusterInstance rejects everything while no generation exists; retried
      // when the next processed geometry arrives
      Logger::err(str::format("[ClusterLOD] render generation build FAILED (", m_residentGeometryHashes.size(),
                              " geometries) - instances stay on the classic path"));
      return;
    }

    m_generationConfigDigest = configDigest;

    const std::vector<lodclusters_remix::GeometryRenderInfo>& infos = m_renderSystem->getGeometryRenderInfos();
    for (uint32_t geometryId = 0; geometryId < infos.size(); geometryId++) {
      m_geometryIdByHash.emplace(infos[geometryId].geometryHash, geometryId);
    }

    m_generationCount++;

    // instances flip classic -> cluster: every cached BLAS bucket that contained
    // them is stale, and the full-skip fast path must not reuse last frame's TLAS
    // instance list (risk R8)
    accelManager.invalidateBucketCache();
    instanceManager.notifySceneChanged();

    Logger::info(str::format("[ClusterLOD] render generation ", m_generationCount, " active: ",
                             infos.size(), " geometries, instance capacity ",
                             m_renderSystem->getMaxRenderInstances()));
  }

  void ClusterLodManager::onFrameBegin(Rc<DxvkContext> ctx, AccelManager& accelManager, InstanceManager& instanceManager) {
    // per-frame slot state (rebuilt by mergeInstancesIntoBlas's full pass)
    m_peakInstanceCount = std::max(m_peakInstanceCount,
                                   uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size()));
    for (size_t tlasType = 0; tlasType < Tlas::Count; tlasType++) {
      m_slots[tlasType].clear();
      m_slotInstanceData[tlasType].clear();
      m_slotsB[tlasType].clear();
      m_slotInstanceDataB[tlasType].clear();
    }
    m_sssDuplicates.clear();
    m_sssDuplicatesB.clear();
    m_framePoses.clear();
    m_framePoseIndexByBlas.clear();
    m_frameClusterBudgetUsed = 0;

    if (!ClusterLodOptions::enable()) {
      return;
    }

    // ---- P4b Path B frame tick ----
    {
      // publish the worker-created template system to the main thread
      std::lock_guard<std::mutex> lock(m_templateSystemMutex);
      m_templateSystemMT = m_templateSystem.get();
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    if (m_templateSystemMT != nullptr) {
      m_templateSystemMT->beginFrame(currentFrame);

      // template sets whose worker-side registration completed
      bool anyReady = false;
      for (const lodclusters_remix::ClusterTemplateSystem::ReadyGeometry& ready : m_templateSystemMT->drainReadyGeometries()) {
        m_animatedGeometryByKey[ready.topologyKey] = ready.geometryIndex;
        anyReady = true;
      }

      if (anyReady) {
        // deforming instances flip classic -> cluster templates: cached BLAS
        // buckets containing them are stale and the full-skip fast path must
        // not reuse last frame's TLAS instance list (risk R8, Path A parity)
        accelManager.invalidateBucketCache();
        instanceManager.notifySceneChanged();
      }

      // age out pose sets of BlasEntries that stopped drawing (their CLAS
      // memory is the dominant per-pose cost)
      for (auto it = m_poseByBlas.begin(); it != m_poseByBlas.end();) {
        if (currentFrame - it->second.lastSeenFrame > kPoseSetKeepFrames) {
          m_templateSystemMT->releasePoseSet(it->second.poseSetId);
          it = m_poseByBlas.erase(it);
        } else {
          ++it;
        }
      }
    }

    if (!ensureRenderSystem()) {
      return;
    }

    // geometries whose background processing completed; they join the
    // generation (append or rebuild) once the cooldown batches them
    for (const uint64_t hash : m_provider->drainReadyGeometries()) {
      m_pendingGeometryHashes.push_back(hash);
    }

    buildGenerationIfDue(accelManager, instanceManager);
  }

  bool ClusterLodManager::needsFullMergePass() const {
    // Path A: the LOD traversal outcome changes per frame while a generation
    // renders. Path B: the per-frame slot/pose lists are rebuilt here.
    return (m_renderSystem != nullptr && m_renderSystem->hasGeneration())
        || (m_templateSystemMT != nullptr && !m_animatedGeometryByKey.empty());
  }

  bool ClusterLodManager::isClusterInstance(const RtInstance* instance, uint32_t& outGeometryId) {
    if (!ClusterLodOptions::enable()) {
      return false;
    }

    const BlasEntry* blasEntry = instance->getBlas();
    if (blasEntry == nullptr) {
      return false;
    }

    // ---- P4b Path B routing (plan 7.1) ----
    // Deforming geometry must never take Path A (its static CLAS would render
    // the bind pose / stale positions): skinned meshes always, and meshes
    // whose vertex data updated in place this frame (CPU mutation; their
    // churning asset hash also misses the Path A tables by construction).
    const RasterGeometry& geometryData = blasEntry->input.getGeometryData();
    const bool skinned = blasEntry->input.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const bool updatedInPlace = blasEntry->frameLastUpdated == currentFrame
                             && blasEntry->frameLastUpdated != blasEntry->frameCreated;

    if (skinned || updatedInPlace) {
      return isClusterTemplateInstance(instance, blasEntry, outGeometryId);
    }

    if (m_renderSystem == nullptr || !m_renderSystem->hasGeneration()) {
      return false;
    }

    const XXH64_hash_t geometryHash = blasEntry->input.getHash(RtxOptions::geometryAssetHashRule());

    const auto found = m_geometryIdByHash.find(geometryHash);
    if (found == m_geometryIdByHash.end()) {
      return false;
    }

    // capacity guard: overflowing instances render classic this frame; the next
    // generation rebuild grows the renderer's buffers
    const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
    if (usedSlots >= m_renderSystem->getMaxRenderInstances()) {
      m_frameOverflowCount++;
      ONCE(Logger::warn(str::format("[ClusterLOD] render instance capacity (",
                                    m_renderSystem->getMaxRenderInstances(),
                                    ") exceeded - overflowing instances render classic until the capacity grows")));
      return false;
    }

    outGeometryId = found->second;
    return true;
  }

  bool ClusterLodManager::isClusterTemplateInstance(const RtInstance* instance, const BlasEntry* blasEntry, uint32_t& outGeometryId) {
    if (m_templateSystemMT == nullptr || m_animatedGeometryByKey.empty() || !ClusterLodOptions::Animated::enable()) {
      return false;
    }

    const RasterGeometry& geometryData = blasEntry->input.getGeometryData();

    const auto foundGeometry = m_animatedGeometryByKey.find(makeTopologyKey(geometryData));
    if (foundGeometry == m_animatedGeometryByKey.end()) {
      // registration still running (or failed) - classic until ready
      return false;
    }
    const uint32_t geometryIndex = foundGeometry->second;

    // the per-frame instantiation consumes the live skinned/updated positions
    const RaytraceBuffer& positions = blasEntry->modifiedGeometryData.positionBuffer;
    if (!positions.defined()) {
      return false;
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // frame-local dedup: instances sharing a BlasEntry share its pose (one
    // CLAS + BLAS; each instance still gets its own TLAS slot)
    const auto foundFramePose = m_framePoseIndexByBlas.find(blasEntry);
    if (foundFramePose != m_framePoseIndexByBlas.end()) {
      outGeometryId = kPathBTag | foundFramePose->second;
      return true;
    }

    // pose set per BlasEntry, validated against reuse (a recycled BlasEntry
    // address or re-pointed topology gets a fresh pose set)
    auto poseIt = m_poseByBlas.find(blasEntry);
    if (poseIt != m_poseByBlas.end()
        && (poseIt->second.blasFrameCreated != blasEntry->frameCreated || poseIt->second.geometryIndex != geometryIndex)) {
      m_templateSystemMT->releasePoseSet(poseIt->second.poseSetId);
      m_poseByBlas.erase(poseIt);
      poseIt = m_poseByBlas.end();
    }

    if (poseIt == m_poseByBlas.end()) {
      const uint32_t poseSetId = m_templateSystemMT->createPoseSet(geometryIndex);
      if (poseSetId == ~0u) {
        return false;
      }

      PoseEntry entry;
      entry.poseSetId = poseSetId;
      entry.geometryIndex = geometryIndex;
      entry.blasFrameCreated = blasEntry->frameCreated;
      entry.lastSeenFrame = currentFrame;
      poseIt = m_poseByBlas.emplace(blasEntry, entry).first;
    }

    PoseEntry& pose = poseIt->second;
    pose.lastSeenFrame = currentFrame;

    // per-frame cluster budget (risk R15: degrade to classic, never corrupt)
    const uint32_t poseClusters = m_templateSystemMT->getPoseSetClusterCount(pose.poseSetId);
    const uint32_t budget = uint32_t(std::max(1024, ClusterLodOptions::Animated::maxPerFrameClusters()));
    if (m_frameClusterBudgetUsed + poseClusters > budget) {
      ONCE(Logger::warn(str::format("[ClusterLOD] animated per-frame cluster budget (", budget,
                                    ") exceeded - overflowing deforming instances render classic")));
      return false;
    }
    m_frameClusterBudgetUsed += poseClusters;

    FramePose framePose;
    framePose.poseSetId = pose.poseSetId;
    framePose.positionsAddress = positions.getDeviceAddress() + positions.offsetFromSlice();
    framePose.positionsStrideBytes = positions.stride();
    framePose.positionsBuffer = positions.buffer();

    const uint32_t framePoseIndex = uint32_t(m_framePoses.size());
    m_framePoses.push_back(std::move(framePose));
    m_framePoseIndexByBlas.emplace(blasEntry, framePoseIndex);

    outGeometryId = kPathBTag | framePoseIndex;
    return true;
  }

  void ClusterLodManager::recordClusterInstance(RtInstance* instance,
                                                uint32_t geometryId,
                                                size_t tlasType,
                                                bool isSssDuplicate,
                                                const VkAccelerationStructureInstanceKHR& blasInstance) {
    // ---- P4b Path B (cluster templates) ----
    if (geometryId & kPathBTag) {
      const uint32_t framePoseIndex = geometryId & ~kPathBTag;

      m_slotsB[tlasType].push_back({ instance, framePoseIndex });

      // cluster_blas_instances patches every recorded slot each frame; until
      // then 0 = inactive instance (never a stale address)
      VkAccelerationStructureInstanceKHR instanceData = blasInstance;
      instanceData.accelerationStructureReference = 0;
      m_slotInstanceDataB[tlasType].push_back(instanceData);

      if (isSssDuplicate) {
        assert(tlasType == Tlas::Opaque);
        m_sssDuplicatesB.push_back({ uint32_t(m_slotsB[Tlas::Opaque].size() - 1) });
      }

      // hit-side routing: classic buffer fetch with the cluster-local
      // primitive remap (surface_interaction template branch)
      instance->surface.isClusterLod = false;
      instance->surface.isClusterTemplate = true;
      return;
    }

    m_slots[tlasType].push_back({ instance, geometryId });

    // pre-fill blasReference with the geometry's low-detail BLAS: the safe default
    // instance_assign_blas expects for skipped/culled builds
    VkAccelerationStructureInstanceKHR instanceData = blasInstance;
    instanceData.accelerationStructureReference = m_renderSystem->getGeometryRenderInfos()[geometryId].lowDetailBlasAddress;
    m_slotInstanceData[tlasType].push_back(instanceData);

    if (isSssDuplicate) {
      // SSS entries duplicate an Opaque-region instance; they receive a copy of
      // its patched TlasInstance rather than their own kernel slot
      assert(tlasType == Tlas::Opaque);
      m_sssDuplicates.push_back({ uint32_t(m_slots[Tlas::Opaque].size() - 1) });
    }

    // hit-side routing (surface_interaction cluster fetch)
    instance->surface.isClusterLod = true;
    instance->surface.isClusterTemplate = false;
    instance->surface.clusterGeometryId = geometryId;
  }

  uint32_t ClusterLodManager::getClusterSlotCount(size_t tlasType) const {
    // region layout per type: [Path A block][Path B block] (the copies in
    // dispatchBuild write the blocks; arrival order does not matter)
    if (tlasType == Tlas::SSS) {
      return uint32_t(m_sssDuplicates.size() + m_sssDuplicatesB.size());
    }
    return uint32_t(m_slots[tlasType].size() + m_slotsB[tlasType].size());
  }

  void ClusterLodManager::dispatchBuild(Rc<DxvkContext> ctx, const CameraManager& cameraManager, AccelManager& accelManager) {
    const uint32_t numOpaque = uint32_t(m_slots[Tlas::Opaque].size());
    const uint32_t numUnordered = uint32_t(m_slots[Tlas::Unordered].size());
    const uint32_t count = numOpaque + numUnordered;

    const uint32_t countB = uint32_t(m_slotsB[Tlas::Opaque].size() + m_slotsB[Tlas::Unordered].size());

    const bool pathAActive = m_renderSystem != nullptr && m_renderSystem->hasGeneration() && count > 0;
    const bool pathBActive = m_templateSystemMT != nullptr && countB > 0;

    if (!pathAActive && !pathBActive) {
      return;
    }

    // ---- P4b Path B (independent of a Path A generation) ----
    if (pathBActive) {
      dispatchAnimated(ctx, accelManager, ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer));
    }

    if (!pathAActive) {
      return;
    }

    // flat kernel-array order: [Opaque region][Unordered region]
    std::vector<lodclusters_remix::InstanceInput> instanceInputs(count);
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(count);

    uint32_t flatIndex = 0;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (size_t i = 0; i < m_slots[tlasType].size(); i++, flatIndex++) {
        const ClusterSlot& slot = m_slots[tlasType][i];

        lodclusters_remix::InstanceInput& input = instanceInputs[flatIndex];
        writeMatrix(input.worldMatrix, slot.instance->getTransform());
        input.geometryID = slot.geometryId;
        // actual two-sidedness is carried per cluster in the baked state bits
        input.twoSided = false;
        input.opaqueStatus = slot.instance->surface.alphaState.isFullyOpaque ? 1u : 2u;

        tlasInstances[flatIndex] = m_slotInstanceData[tlasType][i];
      }
    }

    // per-frame traversal parameters from the main camera
    const RtCamera& camera = cameraManager.getMainCamera();

    lodclusters_remix::FrameParams frameParams;

    const Matrix4 worldToView = camera.getWorldToViewf();
    const Matrix4 viewToProjection = camera.getViewToProjectionf();
    const Matrix4 viewProjection = viewToProjection * worldToView;
    const Matrix4 previousViewProjection =
      Matrix4(camera.getPreviousViewToProjection()) * Matrix4(camera.getPreviousWorldToView());

    writeMatrix(frameParams.viewMatrix, worldToView);
    writeMatrix(frameParams.projMatrix, viewToProjection);
    writeMatrix(frameParams.viewProjMatrix, viewProjection);
    writeMatrix(frameParams.prevViewProjMatrix, previousViewProjection);

    const Vector3 cameraPosition = camera.getPosition();
    frameParams.viewPos[0] = cameraPosition.x;
    frameParams.viewPos[1] = cameraPosition.y;
    frameParams.viewPos[2] = cameraPosition.z;

    frameParams.fovRadians = camera.getFov();
    frameParams.viewportWidth = std::max(1u, camera.m_renderResolution[0]);
    frameParams.viewportHeight = std::max(1u, camera.m_renderResolution[1]);
    frameParams.nearPlane = camera.getNearPlane();
    frameParams.farPlane = camera.getFarPlane();

    frameParams.lodPixelError = std::max(0.01f, ClusterLodOptions::Render::lodPixelError());
    frameParams.culledErrorScale = ClusterLodOptions::Render::culledErrorScale();
    frameParams.traversalPersistentThreads = uint32_t(std::max(64, ClusterLodOptions::Render::traversalPersistentThreads()));
    // P3: per-frame streaming tunable (unlike the init-time budget options)
    frameParams.streamingAgeThreshold = uint32_t(std::clamp(ClusterLodOptions::Streaming::ageThreshold(), 1, 4096));

    // P4: per-frame culling / sharing / caching tunables
    frameParams.freezeCulling = ClusterLodOptions::Render::freezeCulling();
    frameParams.freezeLoD = ClusterLodOptions::Render::freezeLoD();
    frameParams.sharingPushCulled = ClusterLodOptions::Render::sharingPushCulled();
    frameParams.sharingTolerantLevels = uint32_t(std::clamp(ClusterLodOptions::Render::sharingTolerantLevels(), 0, 32));
    frameParams.sharingEnabledLevels = uint32_t(std::clamp(ClusterLodOptions::Render::sharingEnabledLevels(), 0, 32));
    frameParams.cachingAgeThreshold = uint32_t(std::clamp(ClusterLodOptions::Streaming::cachingAgeThreshold(), 1, 4096));
    frameParams.cachingEnabledLevels = uint32_t(std::clamp(ClusterLodOptions::Streaming::cachingEnabledLevels(), 0, 32));

    // P4: HiZ occlusion feed. This runs from SceneManager::prepareSceneData,
    // BEFORE injectRTX re-points m_primaryDepth at this frame's target - so
    // the resource still references the image the PREVIOUS frame's gbuffer
    // pass wrote: exactly the depth that matches prevViewProjMatrix and the
    // sample's end-of-frame HiZ build. The library converts it into its own
    // reversed-Z HiZ source (remix_depth_flip) and builds the far pyramid at
    // the start of recordFrame; freezeCulling skips the rebuild.
    // gated on the state captured at render-system start: the culling kernel
    // variants are baked then, and kernels compiled WITH culling must keep
    // receiving fresh HiZ regardless of later live edits to the option
    if (m_cullingActive) {
      const Resources::Resource& primaryDepth =
        m_device->getCommon()->getResources().getRaytracingOutput().m_primaryDepth;

      if (primaryDepth.isValid()) {
        const VkExtent3D depthExtent = primaryDepth.image->info().extent;

        // resize takes a device-idle wait inside the library; Vulkan queues
        // need external synchronization against dxvk's submission thread
        if (m_renderSystem->hizResolutionDiffers(depthExtent.width, depthExtent.height)) {
          m_device->lockSubmission();
          m_renderSystem->updateHizResolution(depthExtent.width, depthExtent.height);
          m_device->unlockSubmission();
          // recreated resources invalidate the seen-set (handles may recycle)
          m_hizDepthImagesSeen.clear();
        }

        // only feed a depth image that survived at least one full frame cycle:
        // a freshly created target is cleared to 0, which in the game's
        // standard-Z convention reads as "everything at the near plane" and
        // would falsely occlude the whole scene for a frame. First sight
        // (bootstrap, resolution change, each new DLFG queue entry) leaves the
        // far pyramid at its cleared everything-visible state instead.
        const uint64_t depthImageHandle = uint64_t(reinterpret_cast<uintptr_t>(primaryDepth.image->handle()));
        if (!m_hizDepthImagesSeen.insert(depthImageHandle).second) {
          frameParams.depthView = primaryDepth.view->handle();
          frameParams.depthWidth = depthExtent.width;
          frameParams.depthHeight = depthExtent.height;

          // the cluster build reads the depth image this frame - keep it alive
          // for dxvk's lifetime tracking
          ctx->getCommandList()->trackResource<DxvkAccess::Read>(primaryDepth.image);
          ctx->getCommandList()->trackResource<DxvkAccess::None>(primaryDepth.view);
        }
      }
    }

    VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    // P3: with async transfers, completed streaming requests submit upload
    // command buffers directly onto the transfer queue inside recordFrame.
    // dxvk's SDMA path submits to that same queue from the submission thread,
    // so park it for the duration (RTXIO's non-dedicated-queue precedent).
    const bool lockForAsyncTransfer = m_streamingActive && m_asyncTransferActive;
    if (lockForAsyncTransfer) {
      m_device->lockSubmission();
    }

    // traversal -> CLAS/BLAS builds -> instance_assign_blas (patched TlasInstances
    // land in the renderer's staging buffer, blasReference resolved); streaming
    // mode also records request handling, uploads and scene patches
    lodclusters_remix::FrameSubmitSync submitSync;
    m_renderSystem->recordFrame(cmd, frameParams, instanceInputs.data(), tlasInstances.data(), count, &submitSync);

    if (lockForAsyncTransfer) {
      m_device->unlockSubmission();
    }

    // P3: the streaming task queues track this frame through the primary
    // timeline semaphore - it must be signaled by the submission that executes
    // this command list, and any async-transfer waits must gate it
    if (submitSync.signal.semaphore != VK_NULL_HANDLE) {
      ctx->getCommandList()->addSignalSemaphore(submitSync.signal.semaphore, submitSync.signal.value);
    }
    for (const lodclusters_remix::FrameSubmitSync::Entry& wait : submitSync.waits) {
      ctx->getCommandList()->addWaitSemaphore(wait.semaphore, wait.value);
    }

    // copy the patched TlasInstances into AccelManager's instance buffer regions.
    // The source barrier (compute -> transfer) is recorded by the renderer after
    // instance_assign_blas; buildTlas's own barrier covers transfer -> TLAS build.
    const Rc<DxvkBuffer>& instanceBuffer = accelManager.getVkInstanceBuffer();
    const VkBuffer sourceBuffer = m_renderSystem->getTlasInstancesBuffer();

    if (instanceBuffer == nullptr || sourceBuffer == VK_NULL_HANDLE) {
      return;
    }

    const DxvkBufferSliceHandle instanceBufferSlice = instanceBuffer->getSliceHandle();
    constexpr VkDeviceSize kInstanceSize = sizeof(VkAccelerationStructureInstanceKHR);

    std::vector<VkBufferCopy> regions;
    regions.reserve(2 + m_sssDuplicates.size());

    if (numOpaque > 0) {
      VkBufferCopy region;
      region.srcOffset = 0;
      region.dstOffset = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::Opaque);
      region.size = kInstanceSize * numOpaque;
      regions.push_back(region);
    }

    if (numUnordered > 0) {
      VkBufferCopy region;
      region.srcOffset = kInstanceSize * numOpaque;
      region.dstOffset = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::Unordered);
      region.size = kInstanceSize * numUnordered;
      regions.push_back(region);
    }

    const VkDeviceSize sssRegionBase = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::SSS);
    for (size_t i = 0; i < m_sssDuplicates.size(); i++) {
      VkBufferCopy region;
      region.srcOffset = kInstanceSize * m_sssDuplicates[i].sourceFlatIndex;
      region.dstOffset = sssRegionBase + kInstanceSize * i;
      region.size = kInstanceSize;
      regions.push_back(region);
    }

    m_device->vkd()->vkCmdCopyBuffer(cmd, sourceBuffer, instanceBufferSlice.handle, uint32_t(regions.size()), regions.data());

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(instanceBuffer);
  }

  // P4b Path B: records the per-frame cluster-template build - CLAS
  // instantiation from the live (skinned/updated) vertex buffers, one cluster
  // BLAS per pose (BlasEntry), cluster_blas_instances TLAS-slot patch - and
  // copies the patched TlasInstances into the [Path B] blocks of the cluster
  // regions (after the Path A blocks).
  void ClusterLodManager::dispatchAnimated(Rc<DxvkContext> ctx, AccelManager& accelManager, VkCommandBuffer cmd) {
    const uint32_t numOpaqueB = uint32_t(m_slotsB[Tlas::Opaque].size());
    const uint32_t numUnorderedB = uint32_t(m_slotsB[Tlas::Unordered].size());
    const uint32_t countB = numOpaqueB + numUnorderedB;
    const uint32_t poseCount = uint32_t(m_framePoses.size());

    if (countB == 0 || poseCount == 0) {
      return;
    }

    // flat kernel-array order: [Opaque B block][Unordered B block]
    std::vector<lodclusters_remix::ClusterTemplateSystem::PoseInput> poses(poseCount);
    for (uint32_t p = 0; p < poseCount; p++) {
      poses[p].poseSetId = m_framePoses[p].poseSetId;
      poses[p].positionsAddress = m_framePoses[p].positionsAddress;
      poses[p].positionsStrideBytes = m_framePoses[p].positionsStrideBytes;

      // the instantiation reads the skinned/live positions this frame - keep
      // the buffer alive for dxvk's lifetime tracking
      if (m_framePoses[p].positionsBuffer != nullptr) {
        ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_framePoses[p].positionsBuffer);
      }
    }

    std::vector<uint32_t> slotPoseIndex(countB);
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(countB);

    uint32_t flatIndex = 0;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (size_t i = 0; i < m_slotsB[tlasType].size(); i++, flatIndex++) {
        slotPoseIndex[flatIndex] = m_slotsB[tlasType][i].geometryId;  // frame pose index
        tlasInstances[flatIndex] = m_slotInstanceDataB[tlasType][i];
      }
    }

    const bool recorded = m_templateSystemMT->recordFrame(cmd, poses.data(), poseCount, slotPoseIndex.data(),
                                                          tlasInstances.data(), countB);

    const Rc<DxvkBuffer>& instanceBuffer = accelManager.getVkInstanceBuffer();
    if (instanceBuffer == nullptr) {
      return;
    }

    const DxvkBufferSliceHandle instanceBufferSlice = instanceBuffer->getSliceHandle();
    constexpr VkDeviceSize kInstanceSize = sizeof(VkAccelerationStructureInstanceKHR);

    // Path B blocks start after the Path A blocks within each cluster region
    const VkDeviceSize opaqueBase = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::Opaque)
      + kInstanceSize * m_slots[Tlas::Opaque].size();
    const VkDeviceSize unorderedBase = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::Unordered)
      + kInstanceSize * m_slots[Tlas::Unordered].size();
    const VkDeviceSize sssBase = instanceBufferSlice.offset + accelManager.getClusterRegionByteOffset(Tlas::SSS)
      + kInstanceSize * m_sssDuplicates.size();

    if (!recorded) {
      // never leave the reserved slots with garbage: the CPU-known fields with
      // blasReference 0 are valid inactive TLAS instances (degrade, don't
      // corrupt - risk R15)
      ONCE(Logger::warn("[ClusterLOD] animated per-frame build failed - deforming instances inactive this frame"));

      if (numOpaqueB > 0) {
        ctx->writeToBuffer(instanceBuffer,
                           accelManager.getClusterRegionByteOffset(Tlas::Opaque) + kInstanceSize * m_slots[Tlas::Opaque].size(),
                           kInstanceSize * numOpaqueB, tlasInstances.data());
      }
      if (numUnorderedB > 0) {
        ctx->writeToBuffer(instanceBuffer, accelManager.getClusterRegionByteOffset(Tlas::Unordered) + kInstanceSize * m_slots[Tlas::Unordered].size(),
                           kInstanceSize * numUnorderedB, tlasInstances.data() + numOpaqueB);
      }
      for (size_t i = 0; i < m_sssDuplicatesB.size(); i++) {
        ctx->writeToBuffer(instanceBuffer,
                           accelManager.getClusterRegionByteOffset(Tlas::SSS) + kInstanceSize * (m_sssDuplicates.size() + i),
                           kInstanceSize, tlasInstances.data() + m_sssDuplicatesB[i].sourceFlatIndex);
      }
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(instanceBuffer);
      return;
    }

    // copy the patched TlasInstances into the reserved region blocks. The
    // source barrier (compute -> transfer) is recorded by recordFrame after
    // the patch kernel; buildTlas's own barrier covers transfer -> TLAS build.
    const VkBuffer sourceBuffer = m_templateSystemMT->getTlasInstancesBuffer();

    std::vector<VkBufferCopy> regions;
    regions.reserve(2 + m_sssDuplicatesB.size());

    if (numOpaqueB > 0) {
      VkBufferCopy region;
      region.srcOffset = 0;
      region.dstOffset = opaqueBase;
      region.size = kInstanceSize * numOpaqueB;
      regions.push_back(region);
    }

    if (numUnorderedB > 0) {
      VkBufferCopy region;
      region.srcOffset = kInstanceSize * numOpaqueB;
      region.dstOffset = unorderedBase;
      region.size = kInstanceSize * numUnorderedB;
      regions.push_back(region);
    }

    for (size_t i = 0; i < m_sssDuplicatesB.size(); i++) {
      VkBufferCopy region;
      region.srcOffset = kInstanceSize * m_sssDuplicatesB[i].sourceFlatIndex;
      region.dstOffset = sssBase + kInstanceSize * i;
      region.size = kInstanceSize;
      regions.push_back(region);
    }

    m_device->vkd()->vkCmdCopyBuffer(cmd, sourceBuffer, instanceBufferSlice.handle, uint32_t(regions.size()), regions.data());

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(instanceBuffer);
  }

  uint64_t ClusterLodManager::getGeometriesTableAddress() const {
    if (m_renderSystem == nullptr) {
      return 0;
    }
    return m_renderSystem->getGeometriesTableAddress();
  }

  uint64_t ClusterLodManager::getAnimatedClusterTableAddress() const {
    if (m_templateSystemMT == nullptr) {
      return 0;
    }
    return m_templateSystemMT->getClusterTableAddress();
  }

}  // namespace dxvk
