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

// Implementation of the C++17-clean lodclusters_remix interface. This file is the
// only place that translates between the plain boundary types and NVIDIA's
// lodclusters::Scene types; it compiles as part of the C++20 lodclusters library.

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <thread>

#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

#include "lodclusters_remix.h"
#include "scene.hpp"

namespace lodclusters_remix {

namespace {

// FNV-1a 64: deterministic digest of the SceneConfig, used to build a per-config
// cache subdirectory so that config changes can never load stale cluster data.
uint64_t fnv1a64(const void* data, size_t size, uint64_t hash = 0xcbf29ce484222325ull)
{
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for(size_t i = 0; i < size; i++)
  {
    hash ^= bytes[i];
    hash *= 0x100000001b3ull;
  }
  return hash;
}

lodclusters::SceneConfig toSceneConfig(const ProcessorConfig& config)
{
  // value-init zeroes padding + reservedData, keeping the config digest deterministic
  lodclusters::SceneConfig sceneConfig = {};

  sceneConfig.clusterVertices    = config.clusterVertices;
  sceneConfig.clusterTriangles   = config.clusterTriangles;
  sceneConfig.clusterGroupSize   = config.clusterGroupSize;
  sceneConfig.preferredNodeWidth = config.preferredNodeWidth;

  sceneConfig.meshoptPreferRayTracing = config.meshoptPreferRayTracing;
  sceneConfig.useCompressedData       = config.useCompressedData;

  // one Scene material per Remix geometry, so multi-material bookkeeping stays available
  sceneConfig.enableMultiMaterials = true;

  sceneConfig.enabledAttributes = config.enabledAttributes;

  sceneConfig.meshoptFillWeight  = config.meshoptFillWeight;
  sceneConfig.meshoptSplitFactor = config.meshoptSplitFactor;

  sceneConfig.lodLevelDecimationFactor = config.lodLevelDecimationFactor;
  sceneConfig.lodErrorMergePrevious    = config.lodErrorMergePrevious;
  sceneConfig.lodErrorMergeAdditive    = config.lodErrorMergeAdditive;

  sceneConfig.simplifyNormalWeight      = config.simplifyNormalWeight;
  sceneConfig.simplifyTangentWeight     = config.simplifyTangentWeight;
  sceneConfig.simplifyTangentSignWeight = config.simplifyTangentSignWeight;
  sceneConfig.simplifyTexCoordWeight    = config.simplifyTexCoordWeight;
  sceneConfig.simplifyMaterialWeight    = config.simplifyMaterialWeight;

  sceneConfig.compressionPosDropBits = config.compressionPosDropBits;
  sceneConfig.compressionTexDropBits = config.compressionTexDropBits;

  return sceneConfig;
}

lodclusters::SceneLoaderConfig toLoaderConfig(const ProcessorConfig& config)
{
  lodclusters::SceneLoaderConfig loaderConfig = {};

  // pool sized once from the pct; 0 = "use the pool as-is" so NVIDIA's
  // per-geometry ProcessingInfo::init/deinit pool resets never run (they cost
  // ~20 ms per geometry - see configureProcessingThreadPool)
  configureProcessingThreadPool(config.processingThreadsPct);
  loaderConfig.processingThreadsPct = 0.0f;
  loaderConfig.autoSaveCache        = config.autoSaveCache;
  loaderConfig.autoLoadCache        = config.autoLoadCache;
  loaderConfig.memoryMappedCache    = config.memoryMappedCache;
  loaderConfig.forcePreprocessMiB   = size_t(config.forcePreprocessMiB);

  return loaderConfig;
}

// deterministic digest (cache subdirectory name) of the SceneConfig a
// ProcessorConfig maps to
std::string configCacheDigestName(const ProcessorConfig& config)
{
  const lodclusters::SceneConfig sceneConfig = toSceneConfig(config);

  const uint32_t version = lodclusters::SceneConfig::version;

  uint64_t digest = fnv1a64(&version, sizeof(version));
  digest          = fnv1a64(&sceneConfig, sizeof(sceneConfig), digest);

  char digestName[32];
  snprintf(digestName, sizeof(digestName), "%016" PRIx64, digest);

  return digestName;
}

// cache layout: <cacheDirectory>/<config digest as 16 hex>/<geometry hash as 16 hex>(.nvsngeo)
// The returned path is extensionless; Scene::initFromMeshInputs appends the suffix.
std::filesystem::path cacheBasePath(uint64_t geometryHash, const ProcessorConfig& config)
{
  char hashName[32];
  snprintf(hashName, sizeof(hashName), "%016" PRIx64, geometryHash);

  std::filesystem::path dir = nvutils::pathFromUtf8(config.cacheDirectoryUtf8);
  dir /= configCacheDigestName(config);

  return dir / hashName;
}

std::filesystem::path cacheBasePath(const GeometrySnapshot& snapshot, const ProcessorConfig& config)
{
  return cacheBasePath(snapshot.geometryHash, config);
}

lodclusters::Scene::RemixMeshInput toMeshInput(const GeometrySnapshot& snapshot)
{
  lodclusters::Scene::RemixMeshInput input;

  input.name        = snapshot.name;
  input.indices     = std::span<const uint32_t>(snapshot.indices.data(), snapshot.indices.size());
  input.vertexCount = snapshot.vertexCount;

  input.positions      = snapshot.positions.empty() ? nullptr : snapshot.positions.data();
  input.positionStride = 3;
  input.normals        = snapshot.normals.empty() ? nullptr : snapshot.normals.data();
  input.normalStride   = 3;
  input.texcoords0     = snapshot.texcoords0.empty() ? nullptr : snapshot.texcoords0.data();
  input.texcoord0Stride = 2;
  input.tangents       = snapshot.tangents.empty() ? nullptr : snapshot.tangents.data();
  input.tangentStride  = 4;

  input.twoSided    = snapshot.twoSided;
  input.alphaMasked = snapshot.alphaMasked;
  input.alphaCutOff = snapshot.alphaCutOff;

  input.indicesHash  = snapshot.indicesHash;
  input.verticesHash = snapshot.verticesHash;

  return input;
}

void fillStats(const lodclusters::Scene& scene, ProcessStats& stats)
{
  stats.loadedFromCache = scene.m_loadedFromCache;
  stats.memoryMapped    = scene.isMemoryMappedCache();

  stats.lodLevelsCount = scene.m_maxLodLevelsCount;
  stats.totalClusters  = scene.m_totalClustersCount;
  stats.totalTriangles = scene.m_totalTrianglesCount;
  stats.totalVertices  = scene.m_totalVerticesCount;
  stats.hiClusters     = scene.m_hiClustersCount;
  stats.hiTriangles    = scene.m_hiTrianglesCount;

  stats.clusterTrianglesMax = scene.m_histograms.clusterTrianglesMax;
  stats.clusterVerticesMax  = scene.m_histograms.clusterVerticesMax;
  stats.groupClustersMax    = scene.m_histograms.groupClustersMax;
  stats.lodLevelsMax        = scene.m_histograms.lodLevelsMax;

  std::error_code ec;
  uint64_t fileSize = uint64_t(std::filesystem::file_size(scene.getCacheFilePath(), ec));
  stats.cacheFileSizeBytes = ec ? 0 : fileSize;
}

bool statsMatch(const ProcessStats& a, const ProcessStats& b, std::string& outMismatch)
{
  auto check = [&](const char* what, uint64_t va, uint64_t vb) {
    if(va != vb)
    {
      char buffer[256];
      snprintf(buffer, sizeof(buffer), "%s mismatch (%" PRIu64 " vs %" PRIu64 ") ", what, va, vb);
      outMismatch += buffer;
      return false;
    }
    return true;
  };

  bool match = true;
  match &= check("lodLevelsCount", a.lodLevelsCount, b.lodLevelsCount);
  match &= check("totalClusters", a.totalClusters, b.totalClusters);
  match &= check("totalTriangles", a.totalTriangles, b.totalTriangles);
  match &= check("totalVertices", a.totalVertices, b.totalVertices);
  match &= check("hiClusters", a.hiClusters, b.hiClusters);
  match &= check("hiTriangles", a.hiTriangles, b.hiTriangles);
  match &= check("clusterTrianglesMax", a.clusterTrianglesMax, b.clusterTrianglesMax);
  match &= check("clusterVerticesMax", a.clusterVerticesMax, b.clusterVerticesMax);
  match &= check("groupClustersMax", a.groupClustersMax, b.groupClustersMax);
  match &= check("cacheFileSizeBytes", a.cacheFileSizeBytes, b.cacheFileSizeBytes);
  return match;
}

}  // namespace

std::string getGeometryCacheFileUtf8(uint64_t geometryHash, const ProcessorConfig& config)
{
  std::filesystem::path path = cacheBasePath(geometryHash, config);
  path += ".nvsngeo";
  return nvutils::utf8FromPath(path);
}

bool geometryCacheFileExists(uint64_t geometryHash, const ProcessorConfig& config)
{
  std::filesystem::path path = cacheBasePath(geometryHash, config);
  path += ".nvsngeo";

  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

std::string getConfigCacheDigestUtf8(const ProcessorConfig& config)
{
  return configCacheDigestName(config);
}

void installLogSink(LogSink sink)
{
  nvutils::Logger& logger = nvutils::Logger::getInstance();

  // never DebugBreak the game process on a bad mesh, and never write nvpro's own
  // log file - everything is routed into the Remix logger through the sink
  logger.breakOnError(false);
  logger.enableFileOutput(false);

  logger.setLogCallback([sink](nvutils::Logger::LogLevel level, const std::string& message) {
    int sinkLevel = 0;
    if(level == nvutils::Logger::eWARNING)
    {
      sinkLevel = 1;
    }
    else if(level == nvutils::Logger::eERROR)
    {
      sinkLevel = 2;
    }
    sink(sinkLevel, message.c_str());
  });
}

void configureProcessingThreadPool(float threadsPct)
{
  static std::mutex s_mutex;
  std::lock_guard<std::mutex> lock(s_mutex);

  const uint32_t hwThreads = std::max(1u, std::thread::hardware_concurrency());

  uint32_t target = hwThreads;
  if(threadsPct > 0.0f && threadsPct < 1.0f)
  {
    target = std::min(hwThreads, std::max(1u, uint32_t(ceilf(float(hwThreads) * threadsPct))));
  }

  auto& pool = nvutils::get_thread_pool();
  if(pool.get_thread_count() != target)
  {
    pool.reset(target);
  }
}

struct GeometryProcessor::Impl
{
  // reserved for render-scene state in P2; processing itself is stateless
};

GeometryProcessor::GeometryProcessor()
    : m_impl(std::make_unique<Impl>())
{
}

GeometryProcessor::~GeometryProcessor() = default;

bool GeometryProcessor::processGeometry(const GeometrySnapshot& snapshot, const ProcessorConfig& config, ProcessStats& outStats)
{
  outStats = {};

  if(snapshot.isDeforming)
  {
    // Path B (cluster templates, vk_animated_clusters) arrives with P4b; the provider
    // routes deforming snapshots away from the LOD pipeline before getting here.
    LOGE("GeometryProcessor: deforming geometry %s reached the LOD processor\n", snapshot.name.c_str());
    return false;
  }

  if(snapshot.positions.empty() || snapshot.indices.size() < 3 || snapshot.vertexCount == 0)
  {
    LOGW("GeometryProcessor: empty geometry snapshot %s\n", snapshot.name.c_str());
    return false;
  }

  const std::filesystem::path basePath = cacheBasePath(snapshot, config);

  std::error_code ec;
  std::filesystem::create_directories(basePath.parent_path(), ec);
  if(ec)
  {
    LOGE("GeometryProcessor: failed to create cache directory %s\n",
         nvutils::utf8FromPath(basePath.parent_path()).c_str());
    return false;
  }

  const lodclusters::Scene::RemixMeshInput input = toMeshInput(snapshot);

  const auto timeBegin = std::chrono::steady_clock::now();

  lodclusters::Scene scene;
  lodclusters::Scene::Result result = scene.initFromMeshInputs(basePath, std::span(&input, 1), toSceneConfig(config),
                                                               toLoaderConfig(config), ".nvsngeo", false);

  const auto timeEnd = std::chrono::steady_clock::now();

  if(result != lodclusters::Scene::SCENE_RESULT_SUCCESS)
  {
    LOGW("GeometryProcessor: processing failed (%d) for %s\n", int(result), snapshot.name.c_str());
    scene.deinit();
    return false;
  }

  fillStats(scene, outStats);
  outStats.success      = true;
  outStats.processingMs = std::chrono::duration<double, std::milli>(timeEnd - timeBegin).count();

  scene.deinit();

  return true;
}

bool GeometryProcessor::verifyCacheRoundTrip(const GeometrySnapshot& snapshot,
                                             const ProcessorConfig& config,
                                             const ProcessStats& referenceStats,
                                             std::string& outMessage)
{
  outMessage.clear();

  const std::filesystem::path basePath = cacheBasePath(snapshot, config);

  const lodclusters::Scene::RemixMeshInput input = toMeshInput(snapshot);

  const lodclusters::SceneConfig sceneConfig = toSceneConfig(config);

  // pass 1: cache load into system RAM; pass 2: memory-mapped cache load
  for(int pass = 0; pass < 2; pass++)
  {
    lodclusters::SceneLoaderConfig loaderConfig = toLoaderConfig(config);
    loaderConfig.autoLoadCache                  = true;
    loaderConfig.autoSaveCache                  = false;
    loaderConfig.memoryMappedCache              = (pass == 1);

    lodclusters::Scene scene;
    lodclusters::Scene::Result result =
        scene.initFromMeshInputs(basePath, std::span(&input, 1), sceneConfig, loaderConfig, ".nvsngeo", false);

    if(result != lodclusters::Scene::SCENE_RESULT_SUCCESS)
    {
      outMessage = pass == 0 ? "RAM cache load failed" : "mapped cache load failed";
      scene.deinit();
      return false;
    }

    ProcessStats stats = {};
    fillStats(scene, stats);

    if(!scene.m_loadedFromCache)
    {
      outMessage = pass == 0 ? "RAM pass missed the cache" : "mapped pass missed the cache";
      scene.deinit();
      return false;
    }

    if(pass == 1 && !stats.memoryMapped)
    {
      outMessage = "mapped pass did not memory-map the cache";
      scene.deinit();
      return false;
    }

    std::string mismatch;
    if(!statsMatch(referenceStats, stats, mismatch))
    {
      outMessage = (pass == 0 ? "RAM pass: " : "mapped pass: ") + mismatch;
      scene.deinit();
      return false;
    }

    scene.deinit();
  }

  outMessage = "cache round-trip verified (RAM + mapped)";
  return true;
}

}  // namespace lodclusters_remix
