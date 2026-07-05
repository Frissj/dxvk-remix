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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>

#include "rtx_cluster_lod_manager.h"
#include "rtx_cluster_lod_geometry_provider.h"
#include "../../util/util_math.h"
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
      config.processingWorkerCount = uint32_t(std::max(0, ClusterLodOptions::Processing::workerCount()));
      config.autoSaveCache = ClusterLodOptions::Processing::autoSaveCache();
      config.autoLoadCache = ClusterLodOptions::Processing::autoLoadCache();
      config.memoryMappedCache = ClusterLodOptions::Processing::memoryMappedCache();
      config.forcePreprocessMiB = uint64_t(std::max(1, ClusterLodOptions::Processing::forcePreprocessMiB()));

      // P4c promotion foundation (plan 7.7)
      config.processCapturedGeometry = ClusterLodOptions::Promotion::processAtFirstSight();

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

    // P4c: promoted rigid-captured instance (plan 7.7) - Path A slot whose
    // worldMatrix/TLAS transform the promotion kernel patches per frame.
    // Layout: bit 30 tag, bits [29:17] promo state slot, bits [16:0] geometryId.
    constexpr uint32_t kPromotedTag = 0x40000000u;
    constexpr uint32_t kPromotedSlotShift = 17;
    constexpr uint32_t kPromotedGeometryMask = (1u << kPromotedSlotShift) - 1u;

    // 4x4 symmetric pseudoinverse via cyclic Jacobi (doubles; plan 7.7 probe
    // precompute). Rank-deficient Gram matrices (coplanar meshes) invert on
    // the non-null eigenspace only.
    void pseudoInverse4x4(const double g[16], double out[16]) {
      double a[16];
      double v[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
      for (int i = 0; i < 16; i++) {
        a[i] = g[i];
      }

      for (int sweep = 0; sweep < 32; sweep++) {
        double off = 0.0;
        for (int p = 0; p < 4; p++) {
          for (int q = p + 1; q < 4; q++) {
            off += a[p * 4 + q] * a[p * 4 + q];
          }
        }
        if (off < 1e-24) {
          break;
        }

        for (int p = 0; p < 4; p++) {
          for (int q = p + 1; q < 4; q++) {
            const double apq = a[p * 4 + q];
            if (std::abs(apq) < 1e-30) {
              continue;
            }
            const double theta = (a[q * 4 + q] - a[p * 4 + p]) / (2.0 * apq);
            const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;

            for (int k = 0; k < 4; k++) {
              const double akp = a[k * 4 + p];
              const double akq = a[k * 4 + q];
              a[k * 4 + p] = c * akp - s * akq;
              a[k * 4 + q] = s * akp + c * akq;
            }
            for (int k = 0; k < 4; k++) {
              const double apk = a[p * 4 + k];
              const double aqk = a[q * 4 + k];
              a[p * 4 + k] = c * apk - s * aqk;
              a[q * 4 + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 4; k++) {
              const double vkp = v[k * 4 + p];
              const double vkq = v[k * 4 + q];
              v[k * 4 + p] = c * vkp - s * vkq;
              v[k * 4 + q] = s * vkp + c * vkq;
            }
          }
        }
      }

      double eig[4];
      double maxEig = 0.0;
      for (int i = 0; i < 4; i++) {
        eig[i] = a[i * 4 + i];
        maxEig = std::max(maxEig, std::abs(eig[i]));
      }
      const double tol = maxEig * 1e-10;

      // out = V * diag(1/eig where |eig| > tol) * V^T
      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
          double acc = 0.0;
          for (int k = 0; k < 4; k++) {
            if (std::abs(eig[k]) > tol) {
              acc += v[r * 4 + k] * v[c * 4 + k] / eig[k];
            }
          }
          out[r * 4 + c] = acc;
        }
      }
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

    // chrono helper: milliseconds since a steady_clock start point
    double elapsedMs(const std::chrono::steady_clock::time_point& since) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
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
      [this](const lodclusters_remix::GeometrySnapshot& snapshot) {
        // provider worker thread (P4b Path B registration / P4c interim templates)
        return processAnimatedGeometry(snapshot);
      },
      [this](const lodclusters_remix::GeometrySnapshot& snapshot) {
        // provider worker thread (P4c promotion probe precompute + upload)
        buildAndUploadPromotionProbe(snapshot);
      });
  }

  ClusterLodManager::~ClusterLodManager() {
    logStatistics();
    logFrameTimes();

    // join the provider worker BEFORE tearing down the systems its handler uses
    m_provider = nullptr;

    if (m_templateSystem != nullptr) {
      // deinit runs under the external lock - drop the internal submit-lock
      // callbacks first so a stray temp submit inside cannot re-take the
      // (non-recursive) submission lock
      m_templateSystem->setSubmitLockCallbacks(nullptr, nullptr);
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

    // kernel requirements: 64-bit integer arithmetic + buffer atomics; the HiZ
    // builder additionally samples depth through a MIN-reduction sampler
    const bool hasShaderInt64 = device->features().core.features.shaderInt64
                             && device->features().vulkan12Features.shaderBufferInt64Atomics;
    const bool hasMinmaxSampler = device->features().vulkan12Features.samplerFilterMinmax;

    // the ported pipeline creation chains VkShaderModuleCreateInfo into the stage
    // info with module = VK_NULL_HANDLE, which maintenance5 legalizes
    const bool hasMaintenance5 = device->features().khrMaintenance5Features.maintenance5;

    return hasExtension && hasFeature && hasSubgroup32 && hasShaderInt64 && hasMinmaxSampler && hasMaintenance5;
  }

  void ClusterLodManager::onDrawCallGeometry(const DrawCallState& drawCallState, bool vertexDataUpdated) {
    // Whatever happens below, drop the pre-capture staging hold on the way out:
    // the snapshot (if one is taken) copies the data, and holding longer would
    // pin the staging memory for the lifetime of the BlasEntry's DrawCallState
    // copy, which shares this hold.
    struct HoldRelease {
      const std::shared_ptr<PreCaptureVertexData>& hold;
      ~HoldRelease() {
        if (hold != nullptr) {
          hold->release();
        }
      }
    } holdRelease { drawCallState.preCaptureVertexData };

    if (!ClusterLodOptions::enable()) {
      return;
    }

    // PURE geometry identity (P4c): DrawCallState::getHash XORs the draw's
    // legacy MATERIAL hash into the key - cluster data is pure geometry
    // (material rides the instance; opaqueStatus overrides per instance), so
    // keying on it would process + cache identical geometry once per material
    // variant AND make load-time keying (7.1a - no draw material exists yet)
    // impossible. Same rule, geometry hashes only.
    const XXH64_hash_t geometryHash = drawCallState.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());
    if (geometryHash == kEmptyHash) {
      return;
    }

    m_provider->onDrawCallGeometry(drawCallState, geometryHash, vertexDataUpdated);
  }

  void ClusterLodManager::onReplacementGeometryLoaded(const RasterGeometry& geometryData) {
    if (!ClusterLodOptions::enable()) {
      return;
    }

    // pure geometry hash - the exact key onDrawCallGeometry derives at draw
    // time, so the draw-time lookup finds the load-time entry
    const XXH64_hash_t geometryHash = geometryData.getHashForRule(RtxOptions::geometryAssetHashRule());
    if (geometryHash == kEmptyHash) {
      return;
    }

    m_provider->onReplacementGeometry(geometryData, geometryHash);
  }

  // P4b: full Path B registration of one deforming geometry, on the provider's
  // worker thread: CPU clusterization (no Vulkan), then the GPU template build
  // with temporary submissions under the dxvk submission lock (RTXIO
  // precedent for off-thread queue use).
  bool ClusterLodManager::processAnimatedGeometry(const lodclusters_remix::GeometrySnapshot& snapshot) {
    if (!ClusterLodOptions::Animated::enable()) {
      return false;
    }

    // P4c: interim templates for static geometry have their own opt-out.
    // Captured snapshots pass regardless of that option - Path B templates
    // are their PRIMARY render path until promotion, not an interim nicety
    if (!snapshot.isDeforming && !snapshot.isMutating && !snapshot.isCaptured
        && !ClusterLodOptions::Animated::interimTemplates()) {
      return false;
    }

    if (!ensureTemplateSystem()) {
      return false;
    }

    // chrono: split the registration into its three costs - the CPU
    // clusterization (meshopt, lock-free), the wait for Remix's submission
    // lock (contends with the render threads) and the GPU template build
    // (temp submissions + fence waits inside)
    const std::chrono::steady_clock::time_point clusterizeStart = std::chrono::steady_clock::now();
    const uint64_t token = m_templateSystem->clusterizeGeometry(snapshot);
    const double clusterizeMs = elapsedMs(clusterizeStart);
    if (token == 0) {
      return false;
    }

    // P4c: no external submission lock here anymore - the template system
    // locks only around its raw queue submits (setSubmitLockCallbacks); the
    // fence waits that dominate this call run without blocking render-thread
    // submissions
    const std::chrono::steady_clock::time_point buildStart = std::chrono::steady_clock::now();
    const bool built = m_templateSystem->buildGeometryTemplates(token);
    const double buildMs = elapsedMs(buildStart);

    Logger::info(str::format("[ClusterLOD] ", snapshot.name, ": Path B chrono: clusterize ", clusterizeMs,
                             " ms, template build ", buildMs, " ms"));

    return built;
  }

  // ---- P4c rigid-capture promotion (plan 7.7 spec) --------------------------

  namespace {
    // byte-layout mirrors of promotion_solve.comp's ProbeBlob (scalar layout)
    struct ProbeHeader {
      uint32_t sampleCount;
      uint32_t validationCount;
      uint32_t vertexCount;
      uint32_t pad;
      float centroid[3];
      float radius;
      float gInv[16];
    };
    static_assert(sizeof(ProbeHeader) == 96, "kernel mirrors this layout");

    struct ProbeSample {
      uint32_t index;
      float x, y, z;
    };
    static_assert(sizeof(ProbeSample) == 16, "kernel mirrors this layout");
  }

  // worker thread: probe precompute + upload (plan 7.7). Runs after the
  // captured snapshot's Path A processing; the snapshot's CPU data is alive.
  void ClusterLodManager::buildAndUploadPromotionProbe(const lodclusters_remix::GeometrySnapshot& snapshot) {
    if (!ClusterLodOptions::Promotion::enable()) {
      return;
    }
    if (!ensureTemplateSystem()) {
      return;
    }

    const uint32_t vertexCount = snapshot.vertexCount;
    if (vertexCount < 4 || snapshot.positions.size() < size_t(vertexCount) * 3) {
      return;
    }

    const float* positions = snapshot.positions.data();

    // centroid + bounding radius (residuals are relative to it)
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (uint32_t v = 0; v < vertexCount; v++) {
      cx += positions[v * 3 + 0];
      cy += positions[v * 3 + 1];
      cz += positions[v * 3 + 2];
    }
    cx /= vertexCount;
    cy /= vertexCount;
    cz /= vertexCount;

    double radiusSq = 0.0;
    for (uint32_t v = 0; v < vertexCount; v++) {
      const double dx = positions[v * 3 + 0] - cx;
      const double dy = positions[v * 3 + 1] - cy;
      const double dz = positions[v * 3 + 2] - cz;
      radiusSq = std::max(radiusSq, dx * dx + dy * dy + dz * dz);
    }
    const float radius = float(std::sqrt(radiusSq));
    if (!(radius > 0.0f)) {
      return;  // degenerate point cloud - unpromotable, stays Path B
    }

    // farthest-point sampling over a strided candidate subset: 64 spread solve
    // samples, then 32 validation samples continuing the same chain (spread
    // AND disjoint from the solve set)
    const uint32_t stride = std::max(1u, vertexCount / 4096u);
    std::vector<uint32_t> candidates;
    candidates.reserve(vertexCount / stride + 1);
    for (uint32_t v = 0; v < vertexCount; v += stride) {
      candidates.push_back(v);
    }

    const uint32_t solveCount = std::min<uint32_t>(64, uint32_t(candidates.size()));
    const uint32_t validationCount = std::min<uint32_t>(32, uint32_t(candidates.size()) > solveCount
                                                                ? uint32_t(candidates.size()) - solveCount : 0);

    std::vector<uint32_t> picked;
    picked.reserve(solveCount + validationCount);
    std::vector<float> minDistSq(candidates.size(), std::numeric_limits<float>::max());

    // seed: candidate farthest from the centroid
    {
      uint32_t seed = 0;
      float best = -1.0f;
      for (uint32_t i = 0; i < candidates.size(); i++) {
        const uint32_t v = candidates[i];
        const float dx = positions[v * 3 + 0] - float(cx);
        const float dy = positions[v * 3 + 1] - float(cy);
        const float dz = positions[v * 3 + 2] - float(cz);
        const float d = dx * dx + dy * dy + dz * dz;
        if (d > best) {
          best = d;
          seed = i;
        }
      }
      picked.push_back(seed);
    }

    while (picked.size() < size_t(solveCount) + validationCount) {
      const uint32_t lastVertex = candidates[picked.back()];
      uint32_t next = 0;
      float best = -1.0f;
      for (uint32_t i = 0; i < candidates.size(); i++) {
        const uint32_t v = candidates[i];
        const float dx = positions[v * 3 + 0] - positions[lastVertex * 3 + 0];
        const float dy = positions[v * 3 + 1] - positions[lastVertex * 3 + 1];
        const float dz = positions[v * 3 + 2] - positions[lastVertex * 3 + 2];
        minDistSq[i] = std::min(minDistSq[i], dx * dx + dy * dy + dz * dz);
        if (minDistSq[i] > best) {
          best = minDistSq[i];
          next = i;
        }
      }
      if (best <= 0.0f) {
        break;  // all remaining candidates coincide with picked ones
      }
      picked.push_back(next);
    }

    const uint32_t pickedSolve = std::min<uint32_t>(solveCount, uint32_t(picked.size()));
    uint32_t pickedValidation = uint32_t(picked.size()) - pickedSolve;

    // Gram matrix over the solve samples (centered homogeneous, doubles)
    double g[16] = {};
    for (uint32_t i = 0; i < pickedSolve; i++) {
      const uint32_t v = candidates[picked[i]];
      const double h[4] = { positions[v * 3 + 0] - cx, positions[v * 3 + 1] - cy, positions[v * 3 + 2] - cz, 1.0 };
      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
          g[r * 4 + c] += h[r] * h[c];
        }
      }
    }
    double gInv[16];
    pseudoInverse4x4(g, gInv);

    // blob assembly: header + solve + validation (falls back to the solve set
    // when no disjoint candidates exist - the 64-vs-12-DOF overdetermination
    // still exposes non-affine output) + full centered ref positions (gate)
    const uint32_t effectiveValidation = pickedValidation > 0 ? pickedValidation : pickedSolve;
    std::vector<uint8_t> blob(sizeof(ProbeHeader)
                              + sizeof(ProbeSample) * (size_t(pickedSolve) + effectiveValidation + vertexCount));

    ProbeHeader header = {};
    header.sampleCount = pickedSolve;
    header.validationCount = effectiveValidation;
    header.vertexCount = vertexCount;
    header.centroid[0] = float(cx);
    header.centroid[1] = float(cy);
    header.centroid[2] = float(cz);
    header.radius = radius;
    for (int i = 0; i < 16; i++) {
      header.gInv[i] = float(gInv[i]);
    }
    std::memcpy(blob.data(), &header, sizeof(header));

    ProbeSample* samples = reinterpret_cast<ProbeSample*>(blob.data() + sizeof(ProbeHeader));
    auto writeSample = [&](ProbeSample& out, uint32_t v) {
      out.index = v;
      out.x = positions[v * 3 + 0] - float(cx);
      out.y = positions[v * 3 + 1] - float(cy);
      out.z = positions[v * 3 + 2] - float(cz);
    };
    for (uint32_t i = 0; i < pickedSolve; i++) {
      writeSample(samples[i], candidates[picked[i]]);
    }
    for (uint32_t i = 0; i < effectiveValidation; i++) {
      const uint32_t pickIndex = pickedValidation > 0 ? pickedSolve + i : i;
      writeSample(samples[pickedSolve + i], candidates[picked[pickIndex]]);
    }
    for (uint32_t v = 0; v < vertexCount; v++) {
      writeSample(samples[size_t(pickedSolve) + effectiveValidation + v], v);
    }

    const uint64_t probeVa = m_templateSystem->uploadPromotionProbe(blob.data(), blob.size());
    if (probeVa == 0) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(m_promoPendingMutex);
      m_promoPendingProbes.push_back(PendingProbe { snapshot.geometryHash, snapshot.topologyKey, probeVa, vertexCount });
    }

    Logger::info(str::format("[ClusterLOD] ", snapshot.name, ": promotion probe uploaded (verts ", vertexCount,
                             ", blob ", blob.size() / 1024, " KiB)"));
  }

  void ClusterLodManager::updatePromotionStates() {
    if (m_renderSystem == nullptr || !ClusterLodOptions::Promotion::enable()) {
      return;
    }

    // adopt worker-uploaded probes
    {
      std::lock_guard<std::mutex> lock(m_promoPendingMutex);
      for (const PendingProbe& pending : m_promoPendingProbes) {
        if (m_promoNextStateSlot >= lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
          ONCE(Logger::warn("[ClusterLOD] promotion state slots exhausted - further candidates stay Path B"));
          break;
        }
        PromotionCandidate candidate;
        candidate.probeVa = pending.probeVa;
        candidate.vertexCount = pending.vertexCount;
        candidate.stateSlot = m_promoNextStateSlot++;
        m_promoCandidates.emplace(pending.geometryHash, candidate);
        // stable-key translation: a moving captured object cannot recompute its
        // intake asset hash at render (positions churn), so map its position-
        // independent topologyKey -> this intake hash for the render lookups
        if (pending.topologyKey != 0) {
          m_capturedStableHashByTopologyKey[pending.topologyKey] = pending.geometryHash;
        }
      }
      m_promoPendingProbes.clear();
    }

    if (m_promoCandidates.empty()) {
      return;
    }

    if (m_promoStates.size() != lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
      m_promoStates.resize(lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity);
    }
    m_promoStatesValid = m_renderSystem->readPromotionStates(m_promoStates.data());
    if (!m_promoStatesValid) {
      return;
    }

    const uint32_t rigidFrames = uint32_t(std::max(1, ClusterLodOptions::Promotion::rigidFrames()));
    const float epsilon = std::max(1e-5f, ClusterLodOptions::Promotion::residualEpsilon());
    const uint32_t gateLag = uint32_t(std::max(2, ClusterLodOptions::Promotion::gateLagFrames()));

    for (auto& entry : m_promoCandidates) {
      PromotionCandidate& candidate = entry.second;
      const lodclusters_remix::PromotionStateView& state = m_promoStates[candidate.stateSlot];

      switch (candidate.phase) {
      case PromotionCandidate::Phase::Probing:
        if (state.rigidStreak >= rigidFrames) {
          candidate.phase = PromotionCandidate::Phase::GateScheduled;
        }
        break;

      case PromotionCandidate::Phase::GateRunning:
        if (++candidate.gateFrames >= gateLag) {
          if (state.gateResidualRel > 0.0f && state.gateResidualRel <= epsilon) {
            candidate.phase = PromotionCandidate::Phase::Promoted;
            m_statsPromoted++;
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                     " PROMOTED to Path A (full-mesh residual ", state.gateResidualRel, ")"));
          } else if (state.gateResidualRel > epsilon) {
            candidate.phase = PromotionCandidate::Phase::Rejected;
            m_statsPromoRejected++;
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                     " gate REJECTED - partial deformation (full-mesh residual ",
                                     state.gateResidualRel, "), stays Path B"));
            // rejection is terminal - nothing references the probe blob (incl.
            // its full-mesh ref positions) anymore; deferred-free it (former
            // V1 limitation: blobs lived until shutdown)
            if (m_templateSystemMT != nullptr && candidate.probeVa != 0) {
              m_templateSystemMT->freePromotionProbe(candidate.probeVa);
              candidate.probeVa = 0;
            }
          } else {
            // gate never accumulated (instance off-screen that frame) - retry
            candidate.phase = PromotionCandidate::Phase::GateScheduled;
            candidate.gateFrames = 0;
          }
        }
        break;

      case PromotionCandidate::Phase::Promoted:
        break;  // demotion is detected on the per-instance slots below

      default:
        break;
      }
    }

    // demotion: PER-INSTANCE (former V1 limitation was geometry-level). An
    // instance whose last solve went non-rigid, or whose periodic full-mesh
    // sweep (risk R20) failed, re-routes to Path B by itself - its siblings
    // keep rendering Path A. A fresh rigid streak re-promotes it.
    m_statsPromoSolveSkipped = 0;
    for (auto& slotEntry : m_promoSlotByBlas) {
      PromoInstance& promoInstance = slotEntry.second;
      const lodclusters_remix::PromotionStateView& state = m_promoStates[promoInstance.stateSlot];

      // diagnostic: did this instance's last (readback-lagged) solve take the
      // GPU re-solve skip? (kernel PROMO_FLAG_SKIPPED = 8u)
      if ((state.flags & 8u) != 0) {
        m_statsPromoSolveSkipped++;
      }

      // periodic full-mesh sweep verdict (same lag handling as the gate)
      if (promoInstance.sweepPending && ++promoInstance.sweepLagFrames >= gateLag) {
        promoInstance.sweepPending = false;
        if (state.gateResidualRel > epsilon && !promoInstance.demoted) {
          promoInstance.demoted = true;
          Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                   ") DEMOTED to Path B - full-mesh sweep residual ", state.gateResidualRel,
                                   " (sparse-blind partial deformation, risk R20)"));
        }
      }

      if (!promoInstance.demoted && (state.flags & 4u) != 0) {
        promoInstance.demoted = true;
        Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                 ") DEMOTED to Path B (solve went non-rigid; last-good transform covered the lag)"));
      } else if (promoInstance.demoted && state.rigidStreak >= rigidFrames) {
        promoInstance.demoted = false;
        Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                 ") RE-PROMOTED to Path A (rigid streak rebuilt)"));
      }
    }
  }

  void ClusterLodManager::buildPromotionEntries() {
    m_framePromoEntries.clear();

    if (m_renderSystem == nullptr || !m_renderSystem->hasGeneration()
        || !ClusterLodOptions::Promotion::enable() || m_promoCandidates.empty()) {
      return;
    }

    // Path B instances: geometry-level probe / gate entries (one per candidate
    // per frame - rigidity is a property of the shared capture content class,
    // every instance solves the same way for the verdict's purposes)
    std::unordered_set<uint64_t> emitted;
    std::unordered_set<uint32_t> emittedInstanceSlots;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (size_t i = 0; i < m_slotsB[tlasType].size(); i++) {
        const ClusterSlot& slot = m_slotsB[tlasType][i];
        const uint32_t framePoseIndex = slot.geometryId & ~kPathBTag;
        if (framePoseIndex >= m_framePoses.size()) {
          continue;
        }
        const BlasEntry* blasEntry = slot.instance->getBlas();
        if (blasEntry == nullptr) {
          continue;
        }
        // stable key: a moving captured object's live hash churns; translate via
        // topologyKey so the SOLVE runs on the mover (a Path B slot here may be
        // skinned/mutating too - those simply won't map to a captured candidate)
        const XXH64_hash_t hash = stableClusterHash(blasEntry, blasEntry->input.preCaptureVertexData != nullptr);
        const auto found = m_promoCandidates.find(hash);
        if (found == m_promoCandidates.end()) {
          continue;
        }
        PromotionCandidate& candidate = found->second;

        // DEMOTED promoted-instance rendering Path B: keep solving ITS OWN
        // slot so a rebuilt rigid streak re-promotes it (per-instance
        // demotion; dedup by state slot - instances sharing a BlasEntry share
        // capture content and therefore a slot)
        if (candidate.phase == PromotionCandidate::Phase::Promoted && candidate.probeVa != 0) {
          const auto instanceIt = m_promoSlotByBlas.find(blasEntry);
          if (instanceIt != m_promoSlotByBlas.end() && instanceIt->second.demoted
              && emittedInstanceSlots.insert(instanceIt->second.stateSlot).second) {
            lodclusters_remix::PromotionEntry probeEntry;
            probeEntry.probeVa = candidate.probeVa;
            probeEntry.captureVa = m_framePoses[framePoseIndex].positionsAddress;
            probeEntry.captureStrideBytes = m_framePoses[framePoseIndex].positionsStrideBytes;
            probeEntry.stateSlot = instanceIt->second.stateSlot;
            probeEntry.patchSlot = 0xFFFFFFFFu;
            m_framePromoEntries.push_back(probeEntry);
          }
          continue;
        }

        if (candidate.phase == PromotionCandidate::Phase::Rejected
            || candidate.phase == PromotionCandidate::Phase::Promoted) {
          continue;
        }
        if (!emitted.insert(hash).second) {
          continue;
        }

        lodclusters_remix::PromotionEntry promoEntry;
        promoEntry.probeVa = candidate.probeVa;
        promoEntry.captureVa = m_framePoses[framePoseIndex].positionsAddress;
        promoEntry.captureStrideBytes = m_framePoses[framePoseIndex].positionsStrideBytes;
        promoEntry.stateSlot = candidate.stateSlot;
        promoEntry.patchSlot = 0xFFFFFFFFu;
        if (candidate.phase == PromotionCandidate::Phase::GateScheduled) {
          promoEntry.mode = 1;
          promoEntry.vertexCount = candidate.vertexCount;
          candidate.phase = PromotionCandidate::Phase::GateRunning;
          candidate.gateFrames = 0;
        }
        m_framePromoEntries.push_back(promoEntry);
      }
    }

    // Path A slots: per-INSTANCE solve+patch entries for promoted instances
    // (plan R21: each instance's capture carries its own transform, so each
    // gets its own state slot for M/prevM continuity and hit-side fetch)
    uint32_t flatIndex = 0;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (size_t i = 0; i < m_slots[tlasType].size(); i++, flatIndex++) {
        const ClusterSlot& slot = m_slots[tlasType][i];
        if ((slot.geometryId & kPromotedTag) == 0) {
          continue;
        }
        const BlasEntry* blasEntry = slot.instance->getBlas();
        if (blasEntry == nullptr) {
          continue;
        }
        // stable key: this is a tagged PROMOTED captured slot - its live hash
        // churns with motion, so translate via topologyKey (see stableClusterHash)
        const XXH64_hash_t hash = stableClusterHash(blasEntry, /*captured*/ true);
        const auto found = m_promoCandidates.find(hash);
        if (found == m_promoCandidates.end()) {
          continue;
        }

        const RaytraceBuffer& positions = blasEntry->modifiedGeometryData.positionBuffer;

        lodclusters_remix::PromotionEntry promoEntry;
        promoEntry.probeVa = found->second.probeVa;
        promoEntry.captureVa = positions.getDeviceAddress() + positions.offsetFromSlice();
        promoEntry.captureStrideBytes = positions.stride();
        promoEntry.stateSlot = (slot.geometryId >> kPromotedSlotShift) & 0x1FFFu;
        promoEntry.patchSlot = flatIndex;
        m_framePromoEntries.push_back(promoEntry);

        // periodic full-mesh residual sweep (risk R20): the sparse solve can
        // miss a VS animating a small vertex subset, so every promoted
        // instance re-runs the every-vertex gate on a stagger. Reads the M the
        // solve entry above writes this frame (recordPromotion orders gates
        // after solves); the verdict demotes just this instance.
        const uint32_t sweepInterval = uint32_t(std::max(0, ClusterLodOptions::Promotion::fullSweepIntervalFrames()));
        if (sweepInterval > 0 && found->second.probeVa != 0) {
          const auto instanceIt = m_promoSlotByBlas.find(blasEntry);
          if (instanceIt != m_promoSlotByBlas.end() && !instanceIt->second.sweepPending
              && ((m_device->getCurrentFrameId() + instanceIt->second.stateSlot) % sweepInterval) == 0) {
            lodclusters_remix::PromotionEntry sweepEntry = promoEntry;
            sweepEntry.patchSlot = 0xFFFFFFFFu;
            sweepEntry.mode = 1;
            sweepEntry.vertexCount = found->second.vertexCount;
            m_framePromoEntries.push_back(sweepEntry);
            instanceIt->second.sweepPending = true;
            instanceIt->second.sweepLagFrames = 0;
          }
        }
      }
    }

    // CRASH BREADCRUMB (survives a device-lost - emitted on the CPU BEFORE the
    // promotion GPU dispatch this frame). On a GPU hang the render thread blocks
    // on a fence and logging stops, so the LAST of these lines before the gap
    // names the frame whose dispatch hung, plus the exact entry/slot extents the
    // kernel was about to touch. Verbose by design while we chase the crash;
    // gate/remove once stable. Also flags any out-of-range slot (should be none).
    uint32_t maxStateSlot = 0, maxPatchSlot = 0, gateEntries = 0;
    for (const lodclusters_remix::PromotionEntry& e : m_framePromoEntries) {
      maxStateSlot = std::max(maxStateSlot, e.stateSlot);
      if (e.patchSlot != 0xFFFFFFFFu) {
        maxPatchSlot = std::max(maxPatchSlot, e.patchSlot);
      }
      if (e.mode == 1) {
        gateEntries++;
      }
    }
    Logger::info(str::format("[ClusterLOD] promo dispatch: frame ", m_device->getCurrentFrameId(),
                             ", entries ", m_framePromoEntries.size(), " (gate ", gateEntries, ")",
                             ", promoInstances ", m_promoSlotByBlas.size(),
                             ", maxStateSlot ", maxStateSlot, ", maxPatchSlot ", maxPatchSlot,
                             (maxStateSlot >= lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity
                                ? " *** STATESLOT OUT OF RANGE ***" : "")));
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

    const std::chrono::steady_clock::time_point initStart = std::chrono::steady_clock::now();
    m_device->lockSubmission();
    const bool initialized = templateSystem->init(deviceInfo, buildAnimatedConfig());
    m_device->unlockSubmission();
    const double initMs = elapsedMs(initStart);

    if (!initialized) {
      Logger::err("[ClusterLOD] cluster template system initialization FAILED - deforming geometry stays classic");
      m_templateSystemFailed = true;
      return false;
    }

    // P4c: from here on the template system locks around its raw vkQueueSubmit
    // calls itself - fence waits run unlocked, so a registration flood cannot
    // block the render threads' submissions for the GPU duration of each
    // template build (init above ran under the external lock, callbacks
    // deliberately installed after)
    DxvkDevice* device = m_device;
    templateSystem->setSubmitLockCallbacks(
      [device] { device->lockSubmission(); },
      [device] { device->unlockSubmission(); });

    m_templateSystem = std::move(templateSystem);

    Logger::info(str::format("[ClusterLOD] cluster template system initialized (Path B: deforming geometry) in ", initMs, " ms"));
    return true;
  }

  void ClusterLodManager::logStatistics() const {
    if (m_provider == nullptr) {
      return;
    }

    const ClusterLodGeometryProvider::Stats stats = m_provider->getStats();

    // chrono: CS-thread intake tax (every draw pays intake; only first-sight
    // draws pay a snapshot copy). Lifetime avg/max.
    if (stats.intakeCalls > 0) {
      Logger::info(str::format("[ClusterLOD] intake chrono: ", stats.intakeCalls,
                               " calls, avg ", stats.intakeUsTotal / stats.intakeCalls,
                               " us, max ", stats.intakeUsMax, " us; snapshots ", stats.snapshotCount,
                               ", avg ", stats.snapshotCount > 0 ? double(stats.snapshotUsTotal) * 1e-3 / double(stats.snapshotCount) : 0.0,
                               " ms, max ", double(stats.snapshotUsMax) * 1e-3, " ms"));
    }

    Logger::info(str::format("[ClusterLOD] stats: submitted ", stats.submitted,
                             " (topology-converted ", stats.convertedTopology, ")",
                             ", pending ", stats.pending,
                             " (", stats.pendingBytes / (1024 * 1024), " MiB)",
                             ", processed ", stats.processed,
                             " (cache hits ", stats.cacheHits, ")",
                             ", failed ", stats.failed,
                             ", deforming ", stats.deforming,
                             " (Path B ready ", stats.animatedReady,
                             ", failed ", stats.animatedFailed, ")",
                             ", ineligible ", stats.ineligible,
                             " (topology ", stats.skippedTopology,
                             ", tooSmall ", stats.skippedTooSmall,
                             ", format ", stats.skippedFormat,
                             ", noCpuData ", stats.skippedNoCpuData, ")",
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
                                 ", slots opaque ", m_statsSlotsOpaque,
                                 " unordered ", m_statsSlotsUnordered,
                                 " pathB ", m_statsSlotsPathB,
                                 ", renderClusters ", frameStats.numRenderClusters,
                                 ", blasBuilds ", frameStats.numBlasBuilds,
                                 ", blasBytes ", frameStats.blasActualSizeBytes,
                                 ", clasReservedBytes ", frameStats.reservedClasBytes,
                                 ", geoReservedBytes ", frameStats.reservedGeometryBytes));

        // P4c promotion routing: raw per-frame accounting of promoted candidates
        // seen at classify. routedA = actually rendering Path A this frame;
        // droppedTrivial/KeyMiss/Capacity = promoted verdict but held on Path B
        // for that reason. candidates/promoted are cumulative geometry-level.
        Logger::info(str::format("[ClusterLOD] promotion routing: candidates ", m_promoCandidates.size(),
                                 ", promoted ", m_statsPromoted, ", rejected ", m_statsPromoRejected,
                                 " | this frame routedA ", m_statsPromoRoutedALatched,
                                 ", droppedTrivial ", m_statsPromoDroppedTrivialLatched,
                                 ", droppedKeyMiss ", m_statsPromoDroppedKeyMissLatched,
                                 ", droppedCapacity ", m_statsPromoDroppedCapacityLatched,
                                 " | promoInstances ", m_promoSlotByBlas.size(),
                                 ", solveSkipped ", m_statsPromoSolveSkipped,
                                 ", stableKeys ", m_capturedStableHashByTopologyKey.size()));

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

  void ClusterLodManager::logFrameTimes() {
    const FrameTimes& times = m_frameTimes;

    // nothing dispatched since the last report: menus/loading frames record no
    // cluster work - stay silent instead of logging zero lines
    if (times.dispatchA.samples == 0 && times.dispatchB.samples == 0) {
      m_frameTimes = FrameTimes();
      return;
    }

    // avg = steady per-frame cost, max = the hitch a frame paid at least once
    auto avgMax = [](const SectionTimes& section) {
      char buffer[64];
      snprintf(buffer, sizeof(buffer), "%.3f/%.3f ms (%u)", section.avgMs(), section.maxMs, section.samples);
      return std::string(buffer);
    };

    Logger::info(str::format("[ClusterLOD] frame cpu chrono avg/max (frames) since last report:",
                             " onFrameBegin ", avgMax(times.frameBegin),
                             ", classify ", avgMax(times.classify),
                             ", dispatchA ", avgMax(times.dispatchA),
                             " [hizFeed ", avgMax(times.hizFeed),
                             ", asyncLockWait ", avgMax(times.lockWaitA),
                             ", record ", avgMax(times.recordA),
                             "], dispatchB ", avgMax(times.dispatchB),
                             " [record ", avgMax(times.recordB), "]"));

    // GPU section timers NVIDIA's kernels are bracketed with (Path A: traversal,
    // BLAS build, streaming, HiZ; Path B: instantiate, BLAS build, slot patch) -
    // the per-section GPU cost is THE optimization signal
    auto logReport = [](const char* label, const std::string& report) {
      size_t pos = 0;
      while (pos < report.size()) {
        size_t end = report.find('\n', pos);
        if (end == std::string::npos) {
          end = report.size();
        }
        Logger::info(str::format("[ClusterLOD] gpu chrono ", label, "| ", report.substr(pos, end - pos)));
        pos = end + 1;
      }
    };

    std::string report;
    if (m_renderSystem != nullptr && m_renderSystem->getProfilerReportUtf8(report)) {
      logReport("A", report);
    }
    if (m_templateSystemMT != nullptr && m_templateSystemMT->getProfilerReportUtf8(report)) {
      logReport("B", report);
    }

    m_frameTimes = FrameTimes();
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
    const std::chrono::steady_clock::time_point initStart = std::chrono::steady_clock::now();
    m_device->lockSubmission();
    const bool initialized = m_renderSystem->init(deviceInfo, renderConfig);
    m_device->unlockSubmission();
    const double initMs = elapsedMs(initStart);

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
                             m_streamingActive ? "streaming" : "preloaded", ") in ", initMs, " ms"));
    return true;
  }

  double ClusterLodManager::buildGenerationIfDue(AccelManager& accelManager, InstanceManager& instanceManager) {
    // capacity growth request from last frame's overflow
    uint32_t requestedCapacity = m_renderSystem->getMaxRenderInstances();
    if (m_frameOverflowCount > 0) {
      requestedCapacity = nextPowerOfTwo(m_peakInstanceCount + m_frameOverflowCount + 1);
    }

    const bool needsCapacityGrowth = requestedCapacity > m_renderSystem->getMaxRenderInstances();

    if (m_pendingGeometryHashes.empty() && !(needsCapacityGrowth && !m_residentGeometryHashes.empty())) {
      return 0.0;
    }

    // batch generation updates. Cache-hit batches take the fast lane (P4c,
    // plan 7.7): a .nvsngeo load costs milliseconds, so the full cooldown
    // would delay the classic->cluster flip with nothing to amortize.
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    uint32_t cooldown = uint32_t(std::max(1, ClusterLodOptions::Render::generationCooldownFrames()));
    if (m_pendingHasCacheHit) {
      cooldown = std::min(cooldown, uint32_t(std::max(1, ClusterLodOptions::Render::cacheHitCooldownFrames())));
    }
    if (m_generationCount > 0 && currentFrame - m_lastGenerationFrame < cooldown) {
      return 0.0;
    }

    // chrono: generation events are the frame hitches of the cluster pipeline
    // (appends should stay O(new); rebuilds device-idle) - every event logs
    // its wall time + how long acquiring the submission lock took
    const std::chrono::steady_clock::time_point generationStart = std::chrono::steady_clock::now();

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

      const std::chrono::steady_clock::time_point appendLockStart = std::chrono::steady_clock::now();
      m_device->lockSubmission();
      const double appendLockWaitMs = elapsedMs(appendLockStart);
      const lodclusters_remix::ClusterRenderSystem::AppendResult appendResult =
        m_renderSystem->appendToGeneration(pendingFiles, m_pendingGeometryHashes);
      m_device->unlockSubmission();

      if (appendResult == lodclusters_remix::ClusterRenderSystem::AppendResult::Ok) {
        const std::vector<lodclusters_remix::GeometryRenderInfo>& infos = m_renderSystem->getGeometryRenderInfos();
        for (uint32_t geometryId = uint32_t(m_residentGeometryHashes.size()); geometryId < infos.size(); geometryId++) {
          m_geometryIdByHash.emplace(infos[geometryId].geometryHash, geometryId);
          if (infos[geometryId].lodLevelsCount <= 1) {
            m_trivialGeometryIds.insert(geometryId);  // F7
          }
        }

        m_residentGeometryHashes.insert(m_residentGeometryHashes.end(),
                                        m_pendingGeometryHashes.begin(), m_pendingGeometryHashes.end());

        Logger::info(str::format("[ClusterLOD] render generation ", m_generationCount, " grew by ",
                                 m_pendingGeometryHashes.size(), " geometries (", infos.size(), " total)",
                                 " in ", elapsedMs(generationStart), " ms (lock wait ", appendLockWaitMs, " ms)"));

        m_pendingGeometryHashes.clear();
        m_pendingHasCacheHit = false;
        m_lastGenerationFrame = currentFrame;

        // instances of the appended geometries flip classic -> cluster: their
        // cached BLAS buckets are stale and the full-skip fast path must not
        // reuse last frame's TLAS instance list (risk R8)
        accelManager.invalidateBucketCache();
        instanceManager.notifySceneChanged();
        return elapsedMs(generationStart);
      }

      if (appendResult == lodclusters_remix::ClusterRenderSystem::AppendResult::Failed) {
        // unreadable/invalid cache entries; the generation itself is unchanged.
        // Drop the batch so a bad file cannot wedge the pipeline - those
        // geometries stay on the classic path.
        Logger::err(str::format("[ClusterLOD] appending ", m_pendingGeometryHashes.size(),
                                " geometries FAILED - they stay on the classic path (",
                                elapsedMs(generationStart), " ms)"));
        m_pendingGeometryHashes.clear();
        m_pendingHasCacheHit = false;
        m_lastGenerationFrame = currentFrame;
        return elapsedMs(generationStart);
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
    m_pendingHasCacheHit = false;

    std::vector<std::string> cacheFiles;
    cacheFiles.reserve(m_residentGeometryHashes.size());
    for (const uint64_t hash : m_residentGeometryHashes) {
      cacheFiles.push_back(lodclusters_remix::getGeometryCacheFileUtf8(hash, processorConfig));
    }

    const std::chrono::steady_clock::time_point rebuildLockStart = std::chrono::steady_clock::now();
    m_device->lockSubmission();
    const double rebuildLockWaitMs = elapsedMs(rebuildLockStart);
    const bool built = m_renderSystem->buildGeneration(cacheFiles, m_residentGeometryHashes, requestedCapacity);
    m_device->unlockSubmission();

    m_lastGenerationFrame = currentFrame;
    m_frameOverflowCount = 0;

    m_geometryIdByHash.clear();
    m_trivialGeometryIds.clear();

    if (!built) {
      // isClusterInstance rejects everything while no generation exists; retried
      // when the next processed geometry arrives
      Logger::err(str::format("[ClusterLOD] render generation build FAILED (", m_residentGeometryHashes.size(),
                              " geometries) - instances stay on the classic path (",
                              elapsedMs(generationStart), " ms)"));
      return elapsedMs(generationStart);
    }

    m_generationConfigDigest = configDigest;

    const std::vector<lodclusters_remix::GeometryRenderInfo>& infos = m_renderSystem->getGeometryRenderInfos();
    for (uint32_t geometryId = 0; geometryId < infos.size(); geometryId++) {
      m_geometryIdByHash.emplace(infos[geometryId].geometryHash, geometryId);
      if (infos[geometryId].lodLevelsCount <= 1) {
        m_trivialGeometryIds.insert(geometryId);  // F7
      }
    }

    m_generationCount++;

    // instances flip classic -> cluster: every cached BLAS bucket that contained
    // them is stale, and the full-skip fast path must not reuse last frame's TLAS
    // instance list (risk R8)
    accelManager.invalidateBucketCache();
    instanceManager.notifySceneChanged();

    Logger::info(str::format("[ClusterLOD] render generation ", m_generationCount, " active: ",
                             infos.size(), " geometries, instance capacity ",
                             m_renderSystem->getMaxRenderInstances(),
                             ", full rebuild in ", elapsedMs(generationStart),
                             " ms (lock wait ", rebuildLockWaitMs, " ms, device-idle swap)"));
    return elapsedMs(generationStart);
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
    // P4c routing diagnostics: reset before the classify pass repopulates them
    m_promoRoutedA = 0;
    m_promoDroppedTrivial = 0;
    m_promoDroppedKeyMiss = 0;
    m_promoDroppedCapacity = 0;
    m_framePoseIndexByBlas.clear();
    m_frameClusterBudgetUsed = 0;

    if (!ClusterLodOptions::enable()) {
      return;
    }

    // chrono: previous frame's isClusterInstance total (accumulated across the
    // merge pass, one sample per frame) + this function's own cost (generation
    // events excluded - they log their own wall time above)
    m_frameTimes.classify.add(m_frameClassifyMs);
    m_frameClassifyMs = 0.0;
    const std::chrono::steady_clock::time_point frameBeginStart = std::chrono::steady_clock::now();

    // ---- P4b Path B frame tick ----
    {
      // publish the worker-created template system to the main thread
      std::lock_guard<std::mutex> lock(m_templateSystemMutex);
      m_templateSystemMT = m_templateSystem.get();
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // ---- per-frame spike detector ----
    // Real frame time = wall-clock delta between consecutive onFrameBegin calls
    // (captures the WHOLE frame, incl. the path tracer + present, not just the
    // cluster sections). classify was just folded above, and frameBegin/
    // dispatchA/dispatchB hold last frame's lastMs - so on a spike we attribute
    // the just-completed frame. Fires immediately (the interval avg/max chrono
    // averages spikes away).
    {
      const std::chrono::steady_clock::time_point nowTp = std::chrono::steady_clock::now();
      const int thresholdMs = ClusterLodOptions::spikeLogThresholdMs();
      if (m_havePrevFrameTp && thresholdMs > 0) {
        const double frameMs = std::chrono::duration<double, std::milli>(nowTp - m_prevFrameTp).count();
        if (frameMs > double(thresholdMs)) {
          m_spikeCountInterval++;
          m_worstFrameMsInterval = std::max(m_worstFrameMsInterval, frameMs);
          const double clusterMs = m_frameTimes.frameBegin.lastMs + m_frameTimes.classify.lastMs
                                 + m_frameTimes.dispatchA.lastMs + m_frameTimes.dispatchB.lastMs;
          Logger::warn(str::format(
              "[Spike] frame ", currentFrame, ": ", frameMs, " ms (>", thresholdMs,
              ") | cluster CPU ", clusterMs, " ms = onFrameBegin ", m_frameTimes.frameBegin.lastMs,
              " + classify ", m_frameTimes.classify.lastMs, " + dispatchA ", m_frameTimes.dispatchA.lastMs,
              " (record ", m_frameTimes.recordA.lastMs, ", hizFeed ", m_frameTimes.hizFeed.lastMs,
              ", asyncLockWait ", m_frameTimes.lockWaitA.lastMs, ") + dispatchB ", m_frameTimes.dispatchB.lastMs,
              " (record ", m_frameTimes.recordB.lastMs, ") | slots A ", m_statsSlotsOpaque + m_statsSlotsUnordered,
              ", B ", m_statsSlotsPathB,
              " | ", (clusterMs > frameMs * 0.5 ? "CLUSTER-BOUND" : "outside cluster work (GPU/path-tracer/present)")));
        }
      }
      m_prevFrameTp = nowTp;
      m_havePrevFrameTp = true;
    }

    // null-record probe: direct read of the host-visible flag; logs if the
    // shader ever read a null record (must never happen with the visibility defer)
    updateNullRecordProbe();

    // Periodic stats so the log always carries the intake/skip counts - the
    // per-geometry skip messages are ONCE-per-reason and cannot show totals.
    // Only prints when the counters actually changed since the last line.
    if (m_provider != nullptr
        && ClusterLodOptions::logStatsIntervalFrames() > 0
        && currentFrame - m_lastStatsLogFrame >= uint32_t(ClusterLodOptions::logStatsIntervalFrames())) {
      // chrono fields change on every draw and must not force a log while the
      // actual counts are idle - digest the counters only
      ClusterLodGeometryProvider::Stats digestStats = m_provider->getStats();
      digestStats.intakeCalls = 0;
      digestStats.intakeUsTotal = 0;
      digestStats.intakeUsMax = 0;
      digestStats.snapshotCount = 0;
      digestStats.snapshotUsTotal = 0;
      digestStats.snapshotUsMax = 0;
      const uint64_t digest = XXH64(&digestStats, sizeof(digestStats), 0);
      if (digest != m_lastLoggedStatsDigest) {
        logStatistics();
        m_lastLoggedStatsDigest = digest;
      }

      // chrono: per-frame CPU/GPU section report; silent while nothing
      // dispatched (menus/loading), so this only speaks during gameplay
      logFrameTimes();

      // visibility-defer verification: holds = instances kept classic by the
      // gate this interval, flips = geometries that first crossed to Path B,
      // heldFrames = how long they waited (should equal kTemplateVisibilityDelay
      // Frames). Only speaks when the defer actually did something.
      if (m_deferHoldsInterval != 0 || m_deferFlipsInterval != 0) {
        Logger::info(str::format("[TemplateVis] deferGate: holds ", m_deferHoldsInterval,
                                 ", flips ", m_deferFlipsInterval,
                                 ", heldFrames ", (m_deferHeldMinInterval == ~0u ? 0u : m_deferHeldMinInterval),
                                 "/", m_deferHeldMaxInterval,
                                 " (delay ", kTemplateVisibilityDelayFrames, ")"));
      }
      m_deferHoldsInterval = 0;
      m_deferFlipsInterval = 0;
      m_deferHeldMinInterval = ~0u;
      m_deferHeldMaxInterval = 0;

      // spike summary for the interval (individual spikes log immediately above)
      if (m_spikeCountInterval != 0) {
        Logger::info(str::format("[Spike] ", m_spikeCountInterval, " frame(s) over ",
                                 ClusterLodOptions::spikeLogThresholdMs(), " ms this interval, worst ",
                                 m_worstFrameMsInterval, " ms"));
      }
      m_spikeCountInterval = 0;
      m_worstFrameMsInterval = 0.0;

      m_lastStatsLogFrame = currentFrame;
    }

    if (m_templateSystemMT != nullptr) {
      m_templateSystemMT->beginFrame(currentFrame);

      // template sets whose worker-side registration completed
      bool anyReady = false;
      for (const lodclusters_remix::ClusterTemplateSystem::ReadyGeometry& ready : m_templateSystemMT->drainReadyGeometries()) {
        // adoptedFrame stamps the visibility defer (see AnimatedGeometryEntry).
        // A topology registers once (provider dedups), so first-seen wins; guard
        // against a re-adoption resetting the clock backwards regardless.
        auto& entry = m_animatedGeometryByKey[ready.topologyKey];
        if (entry.geometryIndex == ~0u) {
          entry.adoptedFrame = currentFrame;
        }
        entry.geometryIndex = ready.geometryIndex;
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
    // P4c: adopt worker-uploaded promotion probes, read GPU verdicts, run the
    // promote/demote state machine (plan 7.7)
    updatePromotionStates();

    for (const ClusterLodGeometryProvider::ReadyGeometry& ready : m_provider->drainReadyGeometries()) {
      m_pendingGeometryHashes.push_back(ready.hash);
      m_pendingHasCacheHit |= ready.fromCache;
    }

    const double generationMs = buildGenerationIfDue(accelManager, instanceManager);

    m_frameTimes.frameBegin.add(elapsedMs(frameBeginStart) - generationMs);
  }

  bool ClusterLodManager::needsFullMergePass() const {
    // Path A: the LOD traversal outcome changes per frame while a generation
    // renders. Path B: the per-frame slot/pose lists are rebuilt here.
    return (m_renderSystem != nullptr && m_renderSystem->hasGeneration())
        || (m_templateSystemMT != nullptr && !m_animatedGeometryByKey.empty());
  }

  uint64_t ClusterLodManager::stableClusterHash(const BlasEntry* blasEntry, bool captured) const {
    const RasterGeometry& geometryData = blasEntry->input.getGeometryData();
    const XXH64_hash_t live = geometryData.getHashForRule(RtxOptions::geometryAssetHashRule());
    if (!captured) {
      // non-captured static geometry never moves in place - the live asset hash
      // equals the intake hash it was registered under
      return live;
    }
    // captured: the live hash churns with motion. Recover the intake hash via
    // the position-independent topologyKey (stable across frames). Fall back to
    // the live hash if this geometry has no probe yet (not a candidate) - it
    // then simply won't match a candidate, which is correct.
    const uint64_t topologyKey = ClusterLodGeometryProvider::makeTopologyKey(geometryData);
    const auto it = m_capturedStableHashByTopologyKey.find(topologyKey);
    return it != m_capturedStableHashByTopologyKey.end() ? it->second : live;
  }

  bool ClusterLodManager::isClusterInstance(const RtInstance* instance, uint32_t& outGeometryId) {
    if (!ClusterLodOptions::enable()) {
      return false;
    }

    // chrono: called once per instance from mergeInstancesIntoBlas' full pass -
    // the sum lands in m_frameTimes.classify at the next onFrameBegin. RAII so
    // every return path (incl. the Path B branch) is counted.
    struct ClassifyChrono {
      double& accumMs;
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      ~ClassifyChrono() { accumMs += elapsedMs(start); }
    } classifyChrono { m_frameClassifyMs };

    const BlasEntry* blasEntry = instance->getBlas();
    if (blasEntry == nullptr) {
      return false;
    }

    // ---- P4b Path B routing (plan 7.1) ----
    // Deforming geometry must never take Path A (its static CLAS would render
    // the bind pose / stale positions): skinned meshes always, meshes whose
    // vertex data updated in place this frame (CPU mutation; their churning
    // asset hash also misses the Path A tables by construction), and
    // vertex-captured draws - their rendered mesh is the GPU capture buffer
    // content whose model->world transform exists only in the game's shader
    // constants, so the input-space Path A clusters would render untransformed
    // (user decision 2026-07-03: captured -> Path B; the pose reads the
    // capture-derived modifiedGeometryData, matching classic by construction).
    const RasterGeometry& geometryData = blasEntry->input.getGeometryData();
    const bool skinned = blasEntry->input.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const bool captured = blasEntry->input.preCaptureVertexData != nullptr;
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const bool updatedInPlace = blasEntry->frameLastUpdated == currentFrame
                             && blasEntry->frameLastUpdated != blasEntry->frameCreated;

    if (skinned || captured || updatedInPlace) {
      // ---- P4c rigid-capture promotion (plan 7.7): PROMOTED captured
      // instances render Path A LOD clusters; the promotion kernel patches
      // their worldMatrix/TLAS transform from the per-frame solve ----
      // NOTE: no !updatedInPlace guard. A rigidly-MOVING captured object is
      // updatedInPlace every frame (its positions churn) yet is the very thing
      // promotion targets - the gate already proved it rigid, so a Promoted
      // verdict is authoritative regardless of this frame's in-place update.
      if (captured && !skinned
          && m_renderSystem != nullptr && m_renderSystem->hasGeneration()
          && ClusterLodOptions::Promotion::enable() && !m_promoCandidates.empty()) {
        // stable key: a moved captured object's live asset hash no longer
        // matches its intake hash - translate via topologyKey (see stableClusterHash)
        const XXH64_hash_t geometryHash = stableClusterHash(blasEntry, /*captured*/ true);
        const auto candidate = m_promoCandidates.find(geometryHash);
        if (candidate != m_promoCandidates.end()
            && candidate->second.phase == PromotionCandidate::Phase::Promoted) {
          const auto found = m_geometryIdByHash.find(geometryHash);
          if (found == m_geometryIdByHash.end() || found->second > kPromotedGeometryMask) {
            m_promoDroppedKeyMiss++;
          } else if (ClusterLodOptions::Render::routeTrivialToClassic() && m_trivialGeometryIds.count(found->second) != 0) {
            m_promoDroppedTrivial++;
          } else {
            const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
            if (usedSlots >= m_renderSystem->getMaxRenderInstances()) {
              m_promoDroppedCapacity++;
            } else {
              // per-INSTANCE promotion state slot (plan R21: every captured
              // instance's buffer carries its own transform)
              auto slotIt = m_promoSlotByBlas.find(blasEntry);
              if (slotIt == m_promoSlotByBlas.end()
                  && m_promoNextStateSlot < lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
                PromoInstance promoInstance;
                promoInstance.stateSlot = m_promoNextStateSlot++;
                slotIt = m_promoSlotByBlas.emplace(blasEntry, promoInstance).first;
              }
              // per-instance demotion: a demoted instance falls through to
              // Path B below while its siblings stay promoted; its slot keeps
              // solving (buildPromotionEntries) so it can re-promote
              if (slotIt != m_promoSlotByBlas.end() && !slotIt->second.demoted) {
                outGeometryId = kPromotedTag | (slotIt->second.stateSlot << kPromotedSlotShift) | found->second;
                m_promoRoutedA++;
                return true;
              }
            }
          }
        }
      }

      return isClusterTemplateInstance(instance, blasEntry, outGeometryId);
    }

    // ---- P4c ladder: Path A when resident, interim templates while it loads ----
    if (m_renderSystem != nullptr && m_renderSystem->hasGeneration()) {
      // pure geometry hash - MUST match the intake's keying (see onDrawCallGeometry)
      const XXH64_hash_t geometryHash = blasEntry->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());

      const auto found = m_geometryIdByHash.find(geometryHash);
      if (found != m_geometryIdByHash.end()) {
        // F7: single-LOD-level geometry gains nothing from the cluster path -
        // NVIDIA's own guidance keeps purely static no-LOD data on the classic
        // triangle BLAS (which also bucket-merges it)
        if (ClusterLodOptions::Render::routeTrivialToClassic() && m_trivialGeometryIds.count(found->second) != 0) {
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
    }

    // not (yet) resident in the LOD generation: render through the interim
    // template set the worker registered at first sight. A lookup miss falls
    // through to classic - that covers interim disabled, the cache-hit skip,
    // and the first frames before the registration lands. Once the geometry
    // joins the generation the branch above wins and the interim pose sets
    // age out via the normal 60-frame pose GC.
    return isClusterTemplateInstance(instance, blasEntry, outGeometryId);
  }

  bool ClusterLodManager::isClusterTemplateInstance(const RtInstance* instance, const BlasEntry* blasEntry, uint32_t& outGeometryId) {
    if (m_templateSystemMT == nullptr || m_animatedGeometryByKey.empty() || !ClusterLodOptions::Animated::enable()) {
      return false;
    }

    const RasterGeometry& geometryData = blasEntry->input.getGeometryData();

    const auto foundGeometry = m_animatedGeometryByKey.find(ClusterLodGeometryProvider::makeTopologyKey(geometryData));
    if (foundGeometry == m_animatedGeometryByKey.end()) {
      // registration still running (or failed) - classic until ready
      return false;
    }
    const uint32_t geometryIndex = foundGeometry->second.geometryIndex;

    // the per-frame instantiation consumes the live skinned/updated positions
    const RaytraceBuffer& positions = blasEntry->modifiedGeometryData.positionBuffer;
    if (!positions.defined()) {
      return false;
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // visibility defer (see AnimatedGeometryEntry): keep the instance on the
    // classic path until a frame boundary has passed since adoption, so this
    // frame's path tracer can never read a not-yet-visible (zero) cluster-table
    // record. Classic renders the same mesh identically, so this is seamless.
    AnimatedGeometryEntry& animEntry = foundGeometry->second;
    if (currentFrame < animEntry.adoptedFrame + kTemplateVisibilityDelayFrames) {
      m_deferHoldsInterval++;  // instrumentation: the gate is actively holding
      return false;
    }
    // first frame this geometry flips to the cluster-template path (verifies the
    // defer executed and how many classic frames it held)
    if (animEntry.firstFlipFrame == 0) {
      animEntry.firstFlipFrame = currentFrame;
      const uint32_t held = currentFrame - animEntry.adoptedFrame;
      m_deferFlipsInterval++;
      m_deferHeldMinInterval = std::min(m_deferHeldMinInterval, held);
      m_deferHeldMaxInterval = std::max(m_deferHeldMaxInterval, held);
    }

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

    // [ClusterLOD] AnimVtx probe (2026-07-05): the per-frame cluster build reads
    // this pose's positions from the live buffer. Verify the byte range the build
    // will touch is in-bounds, the format/stride are sane, and the data was
    // actually written this frame - the flicker-razor suspects. Each offending
    // mesh logs once; a periodic summary shows prevalence. Gameplay-gated.
    if (currentFrame > 200) {
      static std::unordered_set<uint64_t> s_flagged;
      static uint32_t s_poses = 0, s_oob = 0, s_stale = 0, s_badFmt = 0, s_badStride = 0, s_lastSummary = 0;

      const RasterGeometry& g = blasEntry->input.getGeometryData();
      const uint32_t vtx = g.vertexCount;
      const uint32_t stride = positions.stride();
      const uint64_t offset = positions.offsetFromSlice();
      const uint64_t bufSize = positions.buffer() != nullptr ? positions.buffer()->info().size : 0;
      const uint64_t requiredEnd = vtx > 0 ? offset + uint64_t(stride) * (vtx - 1u) + 3u * sizeof(float) : 0;
      const VkFormat fmt = positions.vertexFormat();
      const bool writtenThisFrame = blasEntry->frameLastUpdated == currentFrame;
      const bool oob = requiredEnd > bufSize;
      const bool badStride = stride < 3u * sizeof(float);
      const bool badFmt = fmt != VK_FORMAT_R32G32B32_SFLOAT && fmt != VK_FORMAT_R32G32B32A32_SFLOAT;

      s_poses++;
      if (oob) { s_oob++; }
      if (!writtenThisFrame) { s_stale++; }
      if (badFmt) { s_badFmt++; }
      if (badStride) { s_badStride++; }

      if (oob || badStride || badFmt || !writtenThisFrame) {
        const uint64_t key = ClusterLodGeometryProvider::makeTopologyKey(g);
        if (s_flagged.insert(key).second) {
          Logger::warn(str::format(
            "[ClusterLOD] AnimVtx FLAG key=0x", std::hex, key, std::dec,
            " verts=", vtx, " stride=", stride, " offset=", offset,
            " bufSize=", bufSize, " requiredEnd=", requiredEnd, " fmt=", uint32_t(fmt),
            " frameLastUpdated=", blasEntry->frameLastUpdated, " cur=", currentFrame,
            " frameCreated=", blasEntry->frameCreated,
            oob ? " *OOB*" : "", badStride ? " *STRIDE*" : "",
            badFmt ? " *FMT*" : "", !writtenThisFrame ? " *STALE*" : ""));
        }
      }

      if (currentFrame - s_lastSummary >= 300) {
        s_lastSummary = currentFrame;
        Logger::info(str::format(
          "[ClusterLOD] AnimVtx summary @frame ", currentFrame, ": posesThisInterval=", s_poses,
          " oob=", s_oob, " stale=", s_stale, " badFmt=", s_badFmt, " badStride=", s_badStride,
          " distinctFlagged=", s_flagged.size()));
        s_poses = s_oob = s_stale = s_badFmt = s_badStride = 0;
      }
    }

    FramePose framePose;
    framePose.poseSetId = pose.poseSetId;
    framePose.positionsAddress = positions.getDeviceAddress() + positions.offsetFromSlice();
    framePose.positionsStrideBytes = positions.stride();
    framePose.positionsBuffer = positions.buffer();
    if (framePose.positionsBuffer != nullptr) {
      // tracked-staging inputs (flicker fix, see kPoseStagingRing): byte range
      // of the position data within the live DxvkBuffer
      const uint32_t vertexCount = blasEntry->input.getGeometryData().vertexCount;
      framePose.positionsByteOffset = framePose.positionsAddress - framePose.positionsBuffer->getDeviceAddress();
      framePose.positionsLengthBytes = vertexCount > 0
        ? VkDeviceSize(framePose.positionsStrideBytes) * (vertexCount - 1u) + 3u * sizeof(float)
        : 0;
    }

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

    // P4c: promoted rigid-captured instances arrive TAGGED (bit 30 + state
    // slot); the slot list keeps the tagged id (buildPromotionEntries decodes
    // it), table lookups and the surface use the plain geometryId
    const bool promoted = (geometryId & kPromotedTag) != 0;
    const uint32_t plainGeometryId = promoted ? (geometryId & kPromotedGeometryMask) : geometryId;
    const uint32_t promoStateSlotPlusOne = promoted ? (((geometryId >> kPromotedSlotShift) & 0x1FFFu) + 1u) : 0u;

    m_slots[tlasType].push_back({ instance, geometryId });

    // pre-fill blasReference with the geometry's low-detail BLAS: the safe default
    // instance_assign_blas expects for skipped/culled builds
    VkAccelerationStructureInstanceKHR instanceData = blasInstance;
    instanceData.accelerationStructureReference = m_renderSystem->getGeometryRenderInfos()[plainGeometryId].lowDetailBlasAddress;
    m_slotInstanceData[tlasType].push_back(instanceData);

    if (isSssDuplicate) {
      // SSS entries duplicate an Opaque-region instance; they receive a copy of
      // its patched TlasInstance rather than their own kernel slot
      assert(tlasType == Tlas::Opaque);
      m_sssDuplicates.push_back({ uint32_t(m_slots[Tlas::Opaque].size() - 1) });
    }

    // hit-side routing (surface_interaction cluster fetch). Promoted instances
    // pack (stateSlot+1) into bits [30:18] - their object->world transforms
    // are GPU-solved and fetched from the promotion state buffer at hit time;
    // plain cluster surfaces keep 0 there and use the CPU surface matrices.
    instance->surface.isClusterLod = true;
    instance->surface.isClusterTemplate = false;
    instance->surface.clusterGeometryId = plainGeometryId | (promoStateSlotPlusOne << 18);
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

    // stats latch: the periodic digest runs in onFrameBegin AFTER the per-frame
    // slot reset, so reading m_slots there always shows 0 (verified in the
    // 2026-07-04 log against a live first-cluster-dispatch of 18) - report the
    // counts captured here, while they are live
    m_statsSlotsOpaque = numOpaque;
    m_statsSlotsUnordered = numUnordered;

    // P4c routing diagnostics: latch the classify-pass accumulators the same way
    m_statsPromoRoutedALatched = m_promoRoutedA;
    m_statsPromoDroppedTrivialLatched = m_promoDroppedTrivial;
    m_statsPromoDroppedKeyMissLatched = m_promoDroppedKeyMiss;
    m_statsPromoDroppedCapacityLatched = m_promoDroppedCapacity;

    const uint32_t countB = uint32_t(m_slotsB[Tlas::Opaque].size() + m_slotsB[Tlas::Unordered].size());

    // P4c: this frame's promotion solve/gate/patch work items (needs the slot
    // lists, which are final here). Probing runs even when no Path A instance
    // rendered this frame - recordFrame handles the promotion-only case.
    buildPromotionEntries();

    const bool pathAActive = m_renderSystem != nullptr && m_renderSystem->hasGeneration()
                          && (count > 0 || !m_framePromoEntries.empty());
    const bool pathBActive = m_templateSystemMT != nullptr && countB > 0;

    if (!pathAActive && !pathBActive) {
      return;
    }

    // ---- P4b Path B (independent of a Path A generation) ----
    if (pathBActive) {
      const std::chrono::steady_clock::time_point dispatchBStart = std::chrono::steady_clock::now();
      dispatchAnimated(ctx, accelManager, ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer));
      m_frameTimes.dispatchB.add(elapsedMs(dispatchBStart));
    }

    if (!pathAActive) {
      return;
    }

    // chrono: Path A's CPU record cost (input fill, HiZ feed, recordFrame,
    // region copies), counted on every exit path below
    struct DispatchChrono {
      SectionTimes& bucket;
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      ~DispatchChrono() { bucket.add(elapsedMs(start)); }
    } dispatchChrono { m_frameTimes.dispatchA };

    // flat kernel-array order: [Opaque region][Unordered region]
    std::vector<lodclusters_remix::InstanceInput> instanceInputs(count);
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(count);

    uint32_t flatIndex = 0;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (size_t i = 0; i < m_slots[tlasType].size(); i++, flatIndex++) {
        const ClusterSlot& slot = m_slots[tlasType][i];

        lodclusters_remix::InstanceInput& input = instanceInputs[flatIndex];
        writeMatrix(input.worldMatrix, slot.instance->getTransform());
        // P4c: promoted slots carry the tag + state slot; the renderer wants
        // the plain table index (its worldMatrix gets kernel-patched anyway)
        input.geometryID = (slot.geometryId & kPromotedTag) ? (slot.geometryId & kPromotedGeometryMask)
                                                            : slot.geometryId;
        // actual two-sidedness is carried per cluster in the baked state bits
        input.twoSided = false;
        input.opaqueStatus = slot.instance->surface.alphaState.isFullyOpaque ? 1u : 2u;

        tlasInstances[flatIndex] = m_slotInstanceData[tlasType][i];
      }
    }

    // one-time proof of what the reserved TLAS slots carry before the GPU patch:
    // mask 0 = invisible to every ray, blas 0 = rays pass through until
    // instance_assign_blas patches the slot (skipped on promotion-only frames)
    if (count > 0)
    ONCE(Logger::info(str::format("[ClusterLOD] first cluster dispatch: slots opaque ", numOpaque,
                                  ", unordered ", numUnordered,
                                  "; slot0 mask 0x", std::hex, uint32_t(tlasInstances[0].mask),
                                  ", customIndex 0x", uint32_t(tlasInstances[0].instanceCustomIndex),
                                  ", flags 0x", uint32_t(tlasInstances[0].flags),
                                  ", lowDetailBlas 0x", tlasInstances[0].accelerationStructureReference,
                                  std::dec, ", geometryId ", m_slots[numOpaque > 0 ? Tlas::Opaque : Tlas::Unordered][0].geometryId)));

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

    // P4c rigid-capture promotion: this frame's solve/gate/patch work items
    frameParams.promotionEntries = m_framePromoEntries.empty() ? nullptr : m_framePromoEntries.data();
    frameParams.promotionEntryCount = uint32_t(m_framePromoEntries.size());
    frameParams.promotionResidualEpsilon = std::max(1e-5f, ClusterLodOptions::Promotion::residualEpsilon());

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
      const std::chrono::steady_clock::time_point hizStart = std::chrono::steady_clock::now();

      const Resources::Resource& primaryDepth =
        m_device->getCommon()->getResources().getRaytracingOutput().m_primaryDepth;

      if (primaryDepth.isValid()) {
        const VkExtent3D depthExtent = primaryDepth.image->info().extent;

        // resize takes a device-idle wait inside the library; Vulkan queues
        // need external synchronization against dxvk's submission thread
        if (m_renderSystem->hizResolutionDiffers(depthExtent.width, depthExtent.height)) {
          const std::chrono::steady_clock::time_point resizeStart = std::chrono::steady_clock::now();
          m_device->lockSubmission();
          m_renderSystem->updateHizResolution(depthExtent.width, depthExtent.height);
          m_device->unlockSubmission();
          // chrono: device-idle event - shows up as a one-frame hitch
          Logger::info(str::format("[ClusterLOD] HiZ resize to ", depthExtent.width, "x", depthExtent.height,
                                   " took ", elapsedMs(resizeStart), " ms (device idle)"));
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

      m_frameTimes.hizFeed.add(elapsedMs(hizStart));
    }

    VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    // P3: with async transfers, completed streaming requests submit upload
    // command buffers directly onto the transfer queue inside recordFrame.
    // dxvk's SDMA path submits to that same queue from the submission thread,
    // so park it for the duration (RTXIO's non-dedicated-queue precedent).
    const bool lockForAsyncTransfer = m_streamingActive && m_asyncTransferActive;
    if (lockForAsyncTransfer) {
      const std::chrono::steady_clock::time_point lockStart = std::chrono::steady_clock::now();
      m_device->lockSubmission();
      m_frameTimes.lockWaitA.add(elapsedMs(lockStart));
    }

    // traversal -> CLAS/BLAS builds -> instance_assign_blas (patched TlasInstances
    // land in the renderer's staging buffer, blasReference resolved); streaming
    // mode also records request handling, uploads and scene patches
    const std::chrono::steady_clock::time_point recordStart = std::chrono::steady_clock::now();
    lodclusters_remix::FrameSubmitSync submitSync;
    m_renderSystem->recordFrame(cmd, frameParams, instanceInputs.data(), tlasInstances.data(), count, &submitSync);
    m_frameTimes.recordA.add(elapsedMs(recordStart));

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

    // stats latch (see dispatchBuild)
    m_statsSlotsPathB = countB;

    if (countB == 0 || poseCount == 0) {
      return;
    }

    // flat kernel-array order: [Opaque B block][Unordered B block]
    // DIAG (2026-07-05): the Aftermath page fault (0x54975000) is disjoint from
    // every logged cluster buffer (all at 0x116x...), so the AS build's remaining
    // input - pose.positionsAddress (the live game vertex buffer) - is the
    // culprit. Log its ranges (deduped) + LOUDLY flag any pose whose buffer Rc is
    // null: those skip trackResource below yet still feed the CLAS build an
    // untracked address that can be freed -> exactly this fault.
    // ---- tracked live-position staging (flicker fix, see kPoseStagingRing) ----
    // The raw gather/CLAS reads below are invisible to dxvk's barrier tracker,
    // so they must never touch the game's live buffers directly (the kUpdateBVH
    // ping-pong rewrites them every 2 frames while GPU frames overlap 4 deep -
    // torn positions = flickering garbage clusters). Stage each pose's position
    // range through TRACKED ctx->copyBuffer first: dxvk orders those copies
    // against the game's vertex uploads in both directions, and everything raw
    // downstream reads only Path-B-owned staging.
    VkDeviceSize stagingTotal = 0;
    std::vector<VkDeviceSize> stagingOffsets(poseCount);
    for (uint32_t p = 0; p < poseCount; p++) {
      stagingOffsets[p] = stagingTotal;
      stagingTotal += align(m_framePoses[p].positionsLengthBytes, 16);
    }

    const uint32_t stagingIndex = m_device->getCurrentFrameId() % kPoseStagingRing;
    Rc<DxvkBuffer>& staging = m_poseStagingBuffers[stagingIndex];
    if (stagingTotal > 0 && (staging == nullptr || staging->info().size < stagingTotal)) {
      DxvkBufferCreateInfo info;
      info.size = align(std::max<VkDeviceSize>(stagingTotal, 1 << 20), 1 << 20);  // MiB-align growth, no churn
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                 | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                  | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                  | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      // old buffer's Rc drops here; dxvk keeps it alive while any tracked
      // command list still references it
      staging = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       DxvkMemoryStats::Category::RTXAccelerationStructure, "PathB Pose Staging");
    }

    uint32_t nullBufferPoses = 0;
    std::vector<lodclusters_remix::ClusterTemplateSystem::PoseInput> poses(poseCount);
    for (uint32_t p = 0; p < poseCount; p++) {
      poses[p].poseSetId = m_framePoses[p].poseSetId;
      poses[p].positionsStrideBytes = m_framePoses[p].positionsStrideBytes;

      if (m_framePoses[p].positionsBuffer != nullptr && staging != nullptr
          && m_framePoses[p].positionsLengthBytes > 0) {
        // tracked copy: live buffer -> this frame's staging slice (dxvk
        // barriers this against the game's writes to the live buffer)
        ctx->copyBuffer(staging, stagingOffsets[p],
                        m_framePoses[p].positionsBuffer, m_framePoses[p].positionsByteOffset,
                        m_framePoses[p].positionsLengthBytes);
        poses[p].positionsAddress = staging->getDeviceAddress() + stagingOffsets[p];
      } else {
        // no Rc (untracked capture path) - fall back to the raw live address;
        // rare, flagged loudly below
        poses[p].positionsAddress = m_framePoses[p].positionsAddress;
        nullBufferPoses++;
        Logger::warn(str::format("[ClusterLOD] *** UNSTAGED POSITIONS *** pose ", p,
                                 " addr 0x", std::hex, m_framePoses[p].positionsAddress, std::dec,
                                 " (null buffer Rc -> raw unordered read; tearing/lifetime hazards possible)"));
      }
    }
    if (nullBufferPoses > 0) {
      Logger::warn(str::format("[ClusterLOD] frame ", m_device->getCurrentFrameId(),
                               ": ", nullBufferPoses, "/", poseCount, " Path B poses have UNSTAGED positions"));
    }
    if (staging != nullptr) {
      // the raw gather reads the staging after the tracked copies; keep it on
      // this command list for lifetime (ordering is the ring + same-cmdbuf
      // barriers inside recordFrame)
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(staging);
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

    const std::chrono::steady_clock::time_point recordStart = std::chrono::steady_clock::now();
    const bool recorded = m_templateSystemMT->recordFrame(cmd, poses.data(), poseCount, slotPoseIndex.data(),
                                                          tlasInstances.data(), countB);
    m_frameTimes.recordB.add(elapsedMs(recordStart));

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

  uint64_t ClusterLodManager::getResidentClustersTableAddress() const {
    if (m_renderSystem == nullptr) {
      return 0;
    }
    return m_renderSystem->getResidentClustersTableAddress();
  }

  uint64_t ClusterLodManager::getPromotionStateAddress() const {
    if (m_renderSystem == nullptr || !ClusterLodOptions::Promotion::enable()) {
      return 0;
    }
    return m_renderSystem->getPromotionStateAddress();
  }

  uint64_t ClusterLodManager::getAnimatedClusterTableAddress() const {
    if (m_templateSystemMT == nullptr) {
      return 0;
    }
    return m_templateSystemMT->getClusterTableAddress();
  }

  // ---- Path B null-record probe (2026-07-04) ----
  // Verifies the visibility defer: the template hit path stores {lastNullFrame,
  // lastNullClusterId} into m_templateDiag via BDA whenever it reads a
  // not-yet-resident (zero) cluster-table record - the device-loss root cause.
  // ONE host-visible buffer, read directly (no ring, no GPU->host copy): a
  // readback pipeline cannot catch the crashing frame anyway (device-loss kills
  // the copy), so its only job is to detect a null while the guard keeps the
  // shader alive - and the flag is persistent, so a direct read never misses it.
  // With the defer in place lastNullFrame must never advance.
  void ClusterLodManager::ensureTemplateDiagBuffers() {
    if (m_templateDiagReady || !ClusterLodOptions::Animated::diagnoseNullRecords()) {
      return;
    }

    // 8 bytes: [+0] lastNullFrame, [+4] lastNullClusterId. Host-visible so the
    // shader writes it via BDA and the CPU reads it directly (host-coherent).
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
                | VK_PIPELINE_STAGE_HOST_BIT;
    info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    info.size = 8;
    m_templateDiag = m_device->createBuffer(info,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            DxvkMemoryStats::Category::RTXBuffer, "ClusterLOD NullRecordProbe");

    // zero-init on the CPU (host-visible - no GPU clear needed)
    m_templateDiagMapped = reinterpret_cast<uint32_t*>(m_templateDiag->mapPtr(0));
    m_templateDiagMapped[0] = 0;
    m_templateDiagMapped[1] = 0;

    m_templateDiagAddress = m_templateDiag->getDeviceAddress();
    m_templateDiagReady = true;
  }

  uint64_t ClusterLodManager::getTemplateDiagAddress() const {
    return m_templateDiagAddress;  // 0 until ensureTemplateDiagBuffers runs / when disabled
  }

  void ClusterLodManager::updateNullRecordProbe() {
    if (!ClusterLodOptions::Animated::diagnoseNullRecords()) {
      return;
    }
    ensureTemplateDiagBuffers();
    if (!m_templateDiagReady) {
      return;
    }

    // direct read of the host-coherent buffer - reflects the GPU's completed
    // writes from prior frames; the flag persists (last-writer), so any null
    // the shader ever hit is caught here within a frame
    const uint32_t nullFrame = m_templateDiagMapped[0];
    const uint32_t nullClusterId = m_templateDiagMapped[1];
    if (nullFrame != 0 && nullFrame != m_lastLoggedNullFrame) {
      m_lastLoggedNullFrame = nullFrame;
      Logger::warn(str::format("[TemplateVis] NULL RECORD read by the path tracer at frame ", nullFrame,
                               ", clusterId ", nullClusterId,
                               " - the visibility defer (", kTemplateVisibilityDelayFrames,
                               " frames) did NOT cover this adopt->hit gap. Raise kTemplateVisibilityDelayFrames"
                               " or move to in-frame table upload."));
    }
  }

}  // namespace dxvk
