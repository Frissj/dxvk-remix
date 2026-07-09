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

#include "rtx_cluster_lod_manager.h"
#include "rtx_cluster_lod_geometry_provider.h"
// NV-DXVK: [GhostSurface] SURFACE_INDEX_INVALID for the transition ghost requests
#include "rtx/pass/common_binding_indices.h"
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
    // 4a: captured meshes are exempt - they MUST build templates (Path B is the
    // only correct render for them until promotion recovers their transform), so
    // only pure-static non-captured geometry honors the interimTemplates opt-out.
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
      m_promoPendingProbes.push_back(PendingProbe { snapshot.geometryHash, probeVa, vertexCount });
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
      candidate.anyInstanceDemoted = false;  // recomputed by the demotion loop below
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
    for (auto& slotEntry : m_promoSlotByBlas) {
      PromoInstance& promoInstance = slotEntry.second;
      const lodclusters_remix::PromotionStateView& state = m_promoStates[promoInstance.stateSlot];

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

      // NV-DXVK: atomicDemotion fix - aggregate the CURRENT demoted state up to the
      // geometry so isClusterInstance can keep the whole mesh single-path.
      if (promoInstance.demoted && slotEntry.first != nullptr) {
        const XXH64_hash_t hash =
          slotEntry.first->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());
        const auto candIt = m_promoCandidates.find(hash);
        if (candIt != m_promoCandidates.end()) {
          candIt->second.anyInstanceDemoted = true;
        }
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
        const XXH64_hash_t hash = blasEntry->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());
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
        const XXH64_hash_t hash = blasEntry->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());
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

      m_lastStatsLogFrame = currentFrame;
    }

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
    // [DualRoute] topology identity shared by this mesh's Path A and Path B forms.
    const uint64_t topologyKey = ClusterLodGeometryProvider::makeTopologyKey(geometryData);
    const bool skinned = blasEntry->input.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const bool captured = blasEntry->input.preCaptureVertexData != nullptr;
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const bool updatedInPlace = blasEntry->frameLastUpdated == currentFrame
                             && blasEntry->frameLastUpdated != blasEntry->frameCreated;

    if (skinned || captured || updatedInPlace) {
      // ---- pinned Path A fast-path ----
      // Once an instance has PROMOTED, it is identified by its stable BlasEntry*,
      // NOT the asset hash. The draw-call cache keeps the same BlasEntry across
      // camera moves (topological bucket + material-match reuse, rtx_draw_call_cache.cpp),
      // but this game's captured-draw asset hash is unstable frame-to-frame, so the
      // m_geometryIdByHash lookup below MISSES on every camera-move frame and dropped
      // the mesh to Path B. Here we route straight off the cached residentGeometryId,
      // deliberately IGNORING updatedInPlace: a changed asset hash on an already-rigid
      // promoted instance is the transform moving, not deformation. Genuine deformation
      // is still caught downstream by the async promotion solve (updatePromotionStates
      // sets slot.demoted on a non-rigid solve, state.flags & 4u) - the position hash
      // was never the right deformation signal because it also fires on rigid motion.
      if (!skinned && ClusterLodOptions::Promotion::enable()
          && m_renderSystem != nullptr && m_renderSystem->hasGeneration()) {
        auto pinIt = m_promoSlotByBlas.find(blasEntry);
        if (pinIt != m_promoSlotByBlas.end()
            && !pinIt->second.demoted
            && pinIt->second.residentGeometryId != ~0u
            && pinIt->second.blasFrameCreated == blasEntry->frameCreated) {
          // atomicDemotion honored via the CACHED ingest-time key (stable), not the
          // churning current hash - so sibling-deform suppression still works mid-motion.
          const auto pinCandidate = m_promoCandidates.find(pinIt->second.geometryHash);
          const bool atomicSuppress = ClusterLodOptions::Promotion::atomicDemotion()
            && pinCandidate != m_promoCandidates.end() && pinCandidate->second.anyInstanceDemoted;
          const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
          if (!atomicSuppress && usedSlots < m_renderSystem->getMaxRenderInstances()
              && (!ClusterLodOptions::Render::routeTrivialToClassic()
                  || m_trivialGeometryIds.count(pinIt->second.residentGeometryId) == 0)) {
            outGeometryId = kPromotedTag | (pinIt->second.stateSlot << kPromotedSlotShift) | pinIt->second.residentGeometryId;
            recordTopoRoute(topologyKey, 1u, instance, pinIt->second.residentGeometryId, 2u);  // Path A (promoted, pinned)
            return true;
          }
        }
      }

      // ---- P4c rigid-capture promotion (plan 7.7): PROMOTED captured
      // instances render Path A LOD clusters; the promotion kernel patches
      // their worldMatrix/TLAS transform from the per-frame solve ----
      // (ESTABLISH path: still gated on !updatedInPlace + a live hash match so a
      // mesh only enters Path A on a frame it has proven rigid; the pin above then
      // holds it there across subsequent camera-driven hash churn.)
      if (captured && !skinned && !updatedInPlace
          && m_renderSystem != nullptr && m_renderSystem->hasGeneration()
          && ClusterLodOptions::Promotion::enable() && !m_promoCandidates.empty()) {
        const XXH64_hash_t geometryHash = blasEntry->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule());
        const auto candidate = m_promoCandidates.find(geometryHash);
        // atomicDemotion fix: if ANY instance of this geometry is demoted, keep the WHOLE
        // mesh on Path B (single-path) so the resident Path A CLAS and the Path B surface
        // never coexist for one topology. Proven root: [DualRoute].
        const bool atomicSuppress = ClusterLodOptions::Promotion::atomicDemotion()
          && candidate != m_promoCandidates.end() && candidate->second.anyInstanceDemoted;
        if (atomicSuppress) {
          static uint32_t s_atomicLogs = 0;
          if (s_atomicLogs < 2000) {
            s_atomicLogs++;
            Logger::info(str::format("[PromoAtomic] geometry 0x", std::hex, geometryHash, std::dec,
                                     " promotion SUPPRESSED (sibling instance demoted) -> whole mesh Path B, frame ",
                                     m_device->getCurrentFrameId()));
          }
        }
        if (!atomicSuppress
            && candidate != m_promoCandidates.end()
            && candidate->second.phase == PromotionCandidate::Phase::Promoted) {
          const auto found = m_geometryIdByHash.find(geometryHash);
          if (found != m_geometryIdByHash.end()
              && found->second <= kPromotedGeometryMask
              && (!ClusterLodOptions::Render::routeTrivialToClassic() || m_trivialGeometryIds.count(found->second) == 0)) {
            const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
            if (usedSlots < m_renderSystem->getMaxRenderInstances()) {
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
                // Cache the stable identity so the pinned fast-path above can route
                // this instance every subsequent frame WITHOUT the churning-hash lookup.
                slotIt->second.residentGeometryId = found->second;
                slotIt->second.geometryHash = geometryHash;
                slotIt->second.blasFrameCreated = blasEntry->frameCreated;
                outGeometryId = kPromotedTag | (slotIt->second.stateSlot << kPromotedSlotShift) | found->second;
                recordTopoRoute(topologyKey, 1u, instance, found->second, 2u);  // Path A (promoted)
                return true;
              }
            }
          }
        }
      }

      // FIX (rtx.clusterLod.promotion.deformingPromotedToClassic): if this TOPOLOGY has a
      // Path A (promoted/resident) instance this-or-last frame, a sibling renders its Path A
      // resident CLAS (id 4096+). Rendering THIS deforming instance on Path B would let a ray
      // commit that resident id under the Path B surface. Topology-keyed (not the instance's
      // own hash) so it also catches DIFFERENT meshes conflated by the deformation-invariant
      // topology key ([DualRoute] proved both). Route it CLASSIC - rigid instances KEEP promotion.
      if (ClusterLodOptions::Promotion::deformingPromotedToClassic()) {
        const auto paIt = m_topoPathAFrame.find(topologyKey);
        if (paIt != m_topoPathAFrame.end() && (currentFrame - paIt->second) <= 2u) {
          static uint32_t s_defClassicLogs = 0;
          if (s_defClassicLogs < 2000) {
            s_defClassicLogs++;
            Logger::info(str::format("[DeformClassic] topo=0x", std::hex, topologyKey, std::dec,
                                     " deforming instance -> CLASSIC (topology is on Path A; avoids resident CLAS +"
                                     " Path B surface coexistence), frame ", currentFrame));
          }
          return false;  // classic
        }
      }

      // skinned/captured/updatedInPlace is genuinely deforming -> always Path B, never
      // subject to routing hysteresis (holding it classic would render the bind pose).
      {
        const bool routedB = isClusterTemplateInstance(instance, blasEntry, outGeometryId);
        if (routedB) {
          recordTopoRoute(topologyKey, 2u, instance, outGeometryId, 3u);  // Path B (deforming)
        }
        return routedB;
      }
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

        // FIX (rtx.clusterLod.render.animatedTopologyExcludesPathA): if this topology is
        // ALSO animated-registered, a sibling instance renders it Path B (template surface).
        // Rendering THIS static instance Path A puts the resident CLAS (id 4096+) in the same
        // cluster TLAS region as that Path B surface -> a ray commits the Path A id under the
        // Path B surface (foreign clusterId). Route it classic so the topology is single-path.
        // Root proven by [DualRoute].
        if (ClusterLodOptions::Render::animatedTopologyExcludesPathA()
            && m_animatedGeometryByKey.count(topologyKey) != 0) {
          return false;
        }

        outGeometryId = found->second;
        recordTopoRoute(topologyKey, 1u, instance, found->second, 1u);  // Path A (resident-static)
        return true;
      }
    }

    // NV-DXVK: routing hysteresis (rtx.clusterLod.render.pathHysteresisFrames). A static
    // instance that was Path A (resident) within N frames must NOT drop to the interim
    // Path B template on a transient residency-lookup miss (generation swap / streaming
    // churn) - that A->B flip is what lets the fresh Path B surface commit the lingering
    // Path A resident CLAS (foreign clusterId 4096+). Hold it on classic until residency
    // restabilizes. 0 disables (default, so [DualRoute] can still reproduce the symptom).
    {
      const int holdFrames = ClusterLodOptions::Render::pathHysteresisFrames();
      if (holdFrames > 0) {
        const auto affIt = m_pathAffiliation.find(instance);
        if (affIt != m_pathAffiliation.end() && affIt->second.path == 1u
            && (currentFrame - affIt->second.frame) <= uint32_t(holdFrames)) {
          return false;
        }
      }
    }

    // not (yet) resident in the LOD generation: render through the interim
    // template set the worker registered at first sight. A lookup miss falls
    // through to classic - that covers interim disabled, the cache-hit skip,
    // and the first frames before the registration lands. Once the geometry
    // joins the generation the branch above wins and the interim pose sets
    // age out via the normal 60-frame pose GC.
    {
      const bool routedB = isClusterTemplateInstance(instance, blasEntry, outGeometryId);
      if (routedB) {
        recordTopoRoute(topologyKey, 2u, instance, outGeometryId, 4u);  // Path B (interim)
      }
      return routedB;
    }
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

  void ClusterLodManager::beginInstanceRecording() {
    m_ghostRequests.clear();
    m_posBufPathThisFrame.clear();
    m_topoRouteThisFrame.clear();

    // prune affiliation entries for instances that stopped being cluster-recorded
    // (destroyed or gone classic) so the map does not grow unboundedly
    const uint32_t frameId = m_device->getCurrentFrameId();

    // NV-DXVK: [DualRoute] periodic rate heartbeat so an overnight run shows the ongoing
    // dual-route rate and distinct-offender count even after every key has been logged.
    if ((frameId % 300u) == 0u && m_dualRouteEvents > 0) {
      Logger::info(str::format("[DualRoute] heartbeat frame ", frameId, ": ", m_dualRouteEvents,
                               " total dual-route events across ", m_dualRouteSeenKeys.size(),
                               " distinct topologies so far"));
    }

    if ((frameId % 512u) == 0u) {
      for (auto it = m_pathAffiliation.begin(); it != m_pathAffiliation.end();) {
        if (frameId - it->second.frame > 512u) {
          it = m_pathAffiliation.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  // NV-DXVK: [DualRoute] record this frame's path decision for a topology and log the
  // first frame it is seen on BOTH paths (the resident-A + template-B coexistence that
  // produces the foreign clusterId 4096+). path: 1 = Path A, 2 = Path B.
  void ClusterLodManager::recordTopoRoute(uint64_t topologyKey, uint8_t path, const RtInstance* instance, uint32_t outGeometryId, uint8_t source) {
    TopoRoute& tr = m_topoRouteThisFrame[topologyKey];
    const uint32_t posBuf = instance != nullptr ? uint32_t(instance->surface.positionBufferIndex) : 0u;
    // asset hash (unique per mesh) to distinguish "same mesh split" from "two different
    // meshes conflated by the deformation-invariant topology key".
    const BlasEntry* be = instance != nullptr ? instance->getBlas() : nullptr;
    const uint64_t geomHash = be != nullptr
      ? uint64_t(be->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule())) : 0u;
    if (path == 1u) {
      tr.aInstance = instance;
      tr.residentGeometryId = outGeometryId;
      tr.aPosBuf = posBuf;
      tr.aSource = source;
      tr.aGeomHash = geomHash;
      // deformingPromotedToClassic signal: this topology has a Path A instance this frame.
      m_topoPathAFrame[topologyKey] = m_device->getCurrentFrameId();
    } else {
      tr.bInstance = instance;
      tr.bOutGeometryId = outGeometryId;
      tr.bPosBuf = posBuf;
      tr.bSource = source;
      tr.bGeomHash = geomHash;
      // atomicDemotion fix signal: this topology is deforming this frame (Path B).
      if (source == 3u) {
        m_topoDeformingFrame[topologyKey] = m_device->getCurrentFrameId();
      }
    }

    if (tr.aInstance != nullptr && tr.bInstance != nullptr && !tr.loggedDual) {
      tr.loggedDual = true;
      m_dualRouteEvents++;
      // one full line per NEW offending topology (the periodic heartbeat carries the rate).
      if (m_dualRouteSeenKeys.insert(topologyKey).second) {
        const char* aSrc = tr.aSource == 2u ? "PROMOTED" : (tr.aSource == 1u ? "resident-static" : "?");
        const char* bSrc = tr.bSource == 3u ? "deforming(skinned/captured)" : (tr.bSource == 4u ? "interim" : "?");
        Logger::err(str::format(
          "[DualRoute] NEW topo=0x", std::hex, topologyKey, std::dec,
          " on BOTH paths frame ", m_device->getCurrentFrameId(),
          " | A(", aSrc, ") inst=", tr.aInstance, " geomHash=0x", std::hex, tr.aGeomHash, std::dec,
          " residentGeomId=", tr.residentGeometryId, " posBuf=", tr.aPosBuf,
          " | B(", bSrc, ") inst=", tr.bInstance, " geomHash=0x", std::hex, tr.bGeomHash,
          " outId=0x", tr.bOutGeometryId, std::dec, " posBuf=", tr.bPosBuf,
          (tr.aGeomHash == tr.bGeomHash ? "  *** SAME mesh split across paths ***"
                                        : "  *** DIFFERENT meshes CONFLATED by weak topology key ***")));
      }
    }
  }

  void ClusterLodManager::recordClusterInstance(RtInstance* instance,
                                                uint32_t geometryId,
                                                size_t tlasType,
                                                bool isSssDuplicate,
                                                const VkAccelerationStructureInstanceKHR& blasInstance) {
    // NV-DXVK: [GhostSurface] detect a cluster-path transition (A <-> B) on this
    // instance. The previous-frame TLAS still holds its OLD path's BLAS, so a ghost
    // surface with the OLD routing is requested for prev-TLAS hit decoding (see header).
    // Only if the instance was recorded LAST frame (a gap means it is not in the
    // previous TLAS) and has a previous surface index for the mapping to redirect.
    {
      const uint8_t currentPath = (geometryId & kPathBTag) ? 2u : 1u;
      const uint32_t frameId = m_device->getCurrentFrameId();
      PathAffiliation& aff = m_pathAffiliation[instance];
      const uint32_t prevSurfaceIndex = instance->getPreviousSurfaceIndex();
      if (aff.path != 0 && aff.path != currentPath && aff.frame + 1 == frameId
          && prevSurfaceIndex != SURFACE_INDEX_INVALID) {
        GhostSurfaceRequest req;
        req.instance = instance;
        req.prevSurfaceIndex = prevSurfaceIndex;
        req.prevIsClusterLod = (aff.path == 1u);
        req.prevIsClusterTemplate = (aff.path == 2u);
        req.prevClusterGeometryId = aff.clusterGeometryId;
        m_ghostRequests.push_back(req);

        // NV-DXVK: [PathFlap] per-transition identity + direction. Repeated lines for
        // the SAME inst with alternating direction = the routing ladder is FLAPPING
        // (mutation classification / hash alternating between the deforming-B branch
        // and the resident-A ladder) - the upstream disease behind the persistent
        // foreign-clusterId misroutes ([ClusterDecodeProbe] posBufferIndex cross-links
        // to posBuf here). Throttled; diagnostic - revert.
        static uint32_t s_flapLogs = 0;
        if (s_flapLogs < 20000) {  // raised from 200 for overnight coverage
          s_flapLogs++;
          Logger::info(str::format("[PathFlap] inst=", instance,
                                   " dir=", (aff.path == 1u ? "A->B" : "B->A"),
                                   " posBuf=", uint32_t(instance->surface.positionBufferIndex),
                                   " prevIdx=", prevSurfaceIndex, " frame=", frameId));
        }
      }
      aff.path = currentPath;
      aff.frame = frameId;
      // clusterGeometryId refreshed below by the per-path branches (0 for Path B)
      aff.clusterGeometryId = 0;

      // NV-DXVK: [PathCollision] same geometry (posBufferIndex) on BOTH paths this frame?
      const uint32_t posBuf = uint32_t(instance->surface.positionBufferIndex);
      auto pcIt = m_posBufPathThisFrame.find(posBuf);
      if (pcIt != m_posBufPathThisFrame.end() && pcIt->second != currentPath) {
        static uint32_t s_collisionLogs = 0;
        if (s_collisionLogs < 100) {
          s_collisionLogs++;
          Logger::err(str::format("[PathCollision] posBuf=", posBuf,
                                  " present on BOTH Path A AND Path B this frame (frame ", frameId,
                                  ") - Path B surface can commit the resident Path A ClusterID (4096+)"
                                  "  *** SAME GEOMETRY DUAL-ROUTED ***"));
        }
      } else {
        m_posBufPathThisFrame[posBuf] = currentPath;
      }
    }

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
    // NV-DXVK: [PromoBlasNull] a 0 BLAS reference on a VISIBLE instance (mask != 0)
    // is a null fed to the driver's TLAS build (compute_01, render queue) - the
    // deterministic Read@VA=0 device-lost, which survives useTemplates=False and
    // fires no cluster-build probe. Unlike Path B (which sets 0 deliberately as
    // "inactive, patched by cluster_blas_instances each frame"), this Path A /
    // promoted fallback is lowDetailBlasAddress, which is 0 before the low-detail
    // build lands or after deinitClas - and instance_assign_blas is SKIPPED on
    // promotion-only frames, so the null survives into the TLAS build. Log it
    // CPU-side, before the TLAS submit (crash-safe).
    // NOTE: NOT gated on mask - the TLAS BUILD dereferences a slot's BLAS address
    // to read its header regardless of mask (mask only gates traversal). The
    // sample's "blas 0 = pass-through" assumption holds for traversal, not the
    // build; a mask==0, blas==0 slot still faults the build on this driver.
    if (instanceData.accelerationStructureReference == 0) {
      Logger::err(str::format("[PromoBlasNull] geometryId 0x", std::hex, geometryId, std::dec, " plainGeom ", plainGeometryId,
                              " promoted ", promoted ? 1 : 0, " tlasType ", tlasType, " mask 0x", std::hex,
                              uint32_t(instanceData.mask), std::dec, " lowDetailBlas 0 (unbuilt/deinited)"
                              "  *** NULL TLAS BLAS (pre-patch) ***"));
    }
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

    // NV-DXVK: [GhostSurface] remember the Path A decode id for a potential A->B
    // transition ghost next frame
    m_pathAffiliation[instance].clusterGeometryId = instance->surface.clusterGeometryId;
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

    // NV-DXVK: [ClusterSlotReserve] DIAGNOSTIC (cpp-only). Confirm the foreign-ClusterID
    // root: the TLAS is built over m_clusterSlotsPerType[type] slots, but this frame only
    // Path A (m_slots) + Path B (m_slotsB) slots are written. If reserved > written, the
    // tail slots keep last frame's bytes -> recycled blasReference -> a Path B surface
    // commits a Path A resident CLAS (id 4096+). Fires only on the mismatching frames;
    // cross-reference the frame number against the [ClusterDecodeProbe] symptom frames.
    {
      const uint32_t reservedOpaque = accelManager.getClusterSlotCount(Tlas::Opaque);
      const uint32_t reservedUnordered = accelManager.getClusterSlotCount(Tlas::Unordered);
      const uint32_t writtenOpaque = uint32_t(m_slots[Tlas::Opaque].size()) + numOpaqueB;
      const uint32_t writtenUnordered = uint32_t(m_slots[Tlas::Unordered].size()) + numUnorderedB;
      if (reservedOpaque != writtenOpaque || reservedUnordered != writtenUnordered) {
        Logger::err(str::format(
          "[ClusterSlotReserve] frame ", m_device->getCurrentFrameId(),
          " STALE TAIL: opaque reserved=", reservedOpaque, " written=", writtenOpaque,
          " (A=", m_slots[Tlas::Opaque].size(), " B=", numOpaqueB, ")",
          " | unordered reserved=", reservedUnordered, " written=", writtenUnordered,
          " (A=", m_slots[Tlas::Unordered].size(), " B=", numUnorderedB, ")",
          " -> ", (reservedOpaque - writtenOpaque), " opaque + ", (reservedUnordered - writtenUnordered),
          " unordered stale slots traversed with recycled blasReference"));
      }
    }

    // stats latch (see dispatchBuild)
    m_statsSlotsPathB = countB;

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

    // NV-DXVK: [UvBindProbe] DIAGNOSTIC (revert). The visible "black" Path B geometry renders with
    // correct positions AND correct vertex colors but no/wrong texture -> position, UV and color all
    // key off the same clusterId-remapped idx[] in the hit shader, so a wrong idx would corrupt all
    // three. Correct geo+color => idx is fine => the defect is UV-SPECIFIC: texcoordBufferIndex,
    // texgenMode, or textureTransform on the surface record. Dump those for Path B (template)
    // surfaces vs a Path A (cluster) surface, and FLAG any template surface with a bound position
    // but an invalid texcoord binding (the "right geo, black texture" signature).
    {
      const uint32_t frameId = m_device->getCurrentFrameId();
      static uint32_t s_uvFlagLogs = 0;
      auto dumpSurf = [&](const char* tag, const RtInstance* inst) {
        if (inst == nullptr) { return; }
        const RtSurface& s = inst->surface;
        Logger::err(str::format(
          "[UvBindProbe] ", tag, " frame ", frameId,
          " isTemplate=", s.isClusterTemplate ? 1 : 0,
          " posIdx=", s.positionBufferIndex, " texIdx=", s.texcoordBufferIndex, " colIdx=", s.color0BufferIndex,
          (s.texcoordBufferIndex == kSurfaceInvalidBufferIndex ? " *TEXCOORD-INVALID*" : ""),
          " texgen=", uint32_t(s.texgenMode),
          " xformIdentity=", (s.textureTransform == Matrix4()) ? 1 : 0,
          " xform00=", s.textureTransform[0][0], " xform11=", s.textureTransform[1][1],
          " xformTx=", s.textureTransform[3][0], " xformTy=", s.textureTransform[3][1]));
      };
      // unconditional flag: a template surface with a valid position but no texcoord binding
      for (const size_t tt : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
        for (size_t i = 0; i < m_slotsB[tt].size(); i++) {
          const RtInstance* inst = m_slotsB[tt][i].instance;
          if (inst != nullptr
              && inst->surface.positionBufferIndex != kSurfaceInvalidBufferIndex
              && (inst->surface.texcoordBufferIndex == kSurfaceInvalidBufferIndex
                  || inst->surface.texgenMode != TexGenMode::None)
              && s_uvFlagLogs < 40) {
            s_uvFlagLogs++;
            dumpSurf("FLAG-B", inst);
          }
        }
      }
      // periodic sample: first few Path B template surfaces + first Path A cluster surface for contrast
      if ((frameId % 120u) == 0u) {
        uint32_t shown = 0;
        for (const size_t tt : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
          for (size_t i = 0; i < m_slotsB[tt].size() && shown < 6; i++, shown++) {
            dumpSurf("sample-B", m_slotsB[tt][i].instance);
          }
        }
        for (const size_t tt : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
          if (!m_slots[tt].empty()) { dumpSurf("sample-A", m_slots[tt][0].instance); break; }
        }
      }
    }

    // NV-DXVK: DIAGNOSTIC (revert) - push the instantiate clusterIdOffset sentinel so the
    // committed-clusterId origin test is runtime-toggleable without a rebuild.
    m_templateSystemMT->setDbgClusterIdOffsetSentinel(uint32_t(ClusterLodOptions::Animated::dbgClusterIdOffsetSentinel()));

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

    // NV-DXVK: [SceneAnimInstScan] mirror ALL animated instances (OPAQUE block then
    // UNORDERED block, contiguous in sourceBuffer starting at srcOffset 0) exactly as
    // fed into the scene TLAS above - same source, same main cmd; device-lost hook
    // dumps it. The unordered block was a blind spot: the persistent foreign-clusterId
    // probe hits track translucent geometries ("surfaces missing sometimes").
    // Diagnostic - revert.
    if (countB > 0) {
      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t curSlot = frameId & 1u;

      // live scan of the OTHER slot (completed last frame) BEFORE overwriting meta
      scanSceneAnimInstMirror(curSlot ^ 1u);

      const VkDeviceSize needStride = kInstanceSize * VkDeviceSize(countB);
      if (m_dbgSceneAnimInstHost == nullptr || m_dbgSceneAnimInstStride < needStride) {
        // grow-only stride (already kInstanceSize-aligned); 2x headroom halves realloc churn
        m_dbgSceneAnimInstStride = needStride * 2;
        DxvkBufferCreateInfo bi;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        bi.access = VK_ACCESS_TRANSFER_WRITE_BIT;
        bi.size = m_dbgSceneAnimInstStride * 2;
        m_dbgSceneAnimInstHost = m_device->createBuffer(bi,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "SceneAnimInst Mirror");
        m_dbgSceneAnimInstCount[0] = m_dbgSceneAnimInstCount[1] = 0;
      }
      const DxvkBufferSliceHandle mirrorSlice = m_dbgSceneAnimInstHost->getSliceHandle();
      VkBufferCopy mr;
      mr.srcOffset = 0;
      mr.dstOffset = mirrorSlice.offset + m_dbgSceneAnimInstStride * curSlot;
      mr.size = kInstanceSize * VkDeviceSize(countB);
      m_device->vkd()->vkCmdCopyBuffer(cmd, sourceBuffer, mirrorSlice.handle, 1, &mr);
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dbgSceneAnimInstHost);
      m_dbgSceneAnimInstCount[curSlot] = countB;
      m_dbgSceneAnimInstFrame[curSlot] = frameId;
      // remember which ring pool THIS capture's patch targeted (recordFrame for this
      // frame just completed, so its ring slot = (counter - 1) % poolCount)
      {
        uint64_t tmpLo[8], tmpHi[8];
        uint32_t fc = 0;
        const uint32_t pc = m_templateSystemMT != nullptr ? m_templateSystemMT->getPoseBlasPools(tmpLo, tmpHi, 8, &fc) : 0;
        m_dbgSceneAnimInstExpectedPool[curSlot] = pc > 0 ? (fc + pc - 1u) % pc : 0u;
        // snapshot the FULL pool ranges live-at-capture so the scan is lag-immune
        m_dbgSceneAnimInstPoolCount[curSlot] = pc;
        for (uint32_t p = 0; p < pc && p < 8; p++) {
          m_dbgSceneAnimInstPoolLo[curSlot][p] = tmpLo[p];
          m_dbgSceneAnimInstPoolHi[curSlot][p] = tmpHi[p];
        }
      }
      m_dbgSceneAnimInstLastSlot = curSlot;
      if (!m_dbgSceneAnimInstArmed) {
        m_dbgSceneAnimInstArmed = true;
        rtxDeviceLostInstanceDumpFn() = [this]() { dumpSceneAnimInstOnDeviceLost(); };
      }
    }
  }

  // NV-DXVK: [SceneAnimInstScan] LIVE scan of the lagged mirror slot: validate every
  // animated opaque instance's BLAS ref against the animated system's CURRENT pose-BLAS
  // ring pools. A nonzero ref OUTSIDE all of them = the traced instance rides memory the
  // animated system does not own (recycled -> ghost Path A CLAS, the clusterId 4096+
  // mechanism). [HeadWatch] is blind to this case (such refs land inside OTHER systems'
  // registered pools and pass). Transient false positives for ~1 frame after a
  // "frame capacities" growth are expected (old pools freed) - noted in the message.
  void ClusterLodManager::scanSceneAnimInstMirror(uint32_t slot) {
    if (m_dbgSceneAnimInstHost == nullptr || m_dbgSceneAnimInstCount[slot] == 0 || m_templateSystemMT == nullptr) {
      return;
    }
    const auto* inst = reinterpret_cast<const VkAccelerationStructureInstanceKHR*>(
      m_dbgSceneAnimInstHost->mapPtr(m_dbgSceneAnimInstStride * slot));
    if (inst == nullptr) {
      return;
    }

    // use the pool ranges SNAPSHOTTED at capture time (lag-immune) instead of the live
    // pools, which may have grown/reallocated between capture and this scan.
    const uint64_t* poolLo = m_dbgSceneAnimInstPoolLo[slot];
    const uint64_t* poolHi = m_dbgSceneAnimInstPoolHi[slot];
    const uint32_t poolCount = m_dbgSceneAnimInstPoolCount[slot];
    if (poolCount == 0) {
      return;
    }
    // the capture's entries were patched into the ring pool recorded at CAPTURE time
    // (immune to skipped recordFrames). A ref inside a DIFFERENT ring pool passed the
    // old any-pool check while being STALE - a 1..3-frame-old patch whose pose set may
    // already be gone (CLAS memory recycled -> the foreign clusterId 4096+ hits on
    // TLAS=CURRENT).
    const uint32_t expectedPool = m_dbgSceneAnimInstExpectedPool[slot] % poolCount;

    // NV-DXVK: [ClasAlias] this frame's Path A resident/low-detail CLAS ranges. The
    // decisive test: if a Path B pose ref (or the pose pools themselves) fall inside
    // Path A CLAS memory, the allocator handed the same physical bytes to both paths
    // -> a Path B surface commits a Path A resident CLAS (foreign clusterId 4096+).
    // Path A ranges are few (resident + low-detail + append buffers); 64 covers them
    // with headroom. The accessor returns the TOTAL so truncation is still detected.
    constexpr uint32_t kPaCap = 64;
    uint64_t paLo[kPaCap], paHi[kPaCap];
    const uint32_t paTotal = m_renderSystem != nullptr
      ? m_renderSystem->getPathAClasRanges(paLo, paHi, kPaCap) : 0;
    const uint32_t paCount = std::min(paTotal, kPaCap);

    // [ClasAlias] the Path B CLAS-content buffers (where baked clusterIDs live and what
    // pose BLASes reference) - NOT the BLAS pools. Overlap with Path A CLAS memory is the
    // direct test for the foreign clusterId 4096+. These number in the thousands, so
    // size exactly via the two-call pattern (maxCount 0 -> writes nothing, returns total).
    const uint32_t poseClasTotal = m_templateSystemMT != nullptr
      ? m_templateSystemMT->getPoseClasRanges(nullptr, nullptr, 0) : 0;
    std::vector<uint64_t> poseClasLo(poseClasTotal), poseClasHi(poseClasTotal);
    if (poseClasTotal > 0) {
      m_templateSystemMT->getPoseClasRanges(poseClasLo.data(), poseClasHi.data(), poseClasTotal);
    }
    const uint32_t poseClasCount = poseClasTotal;  // full coverage, no cap
    const bool clasRangesTruncated = (paTotal > kPaCap);

    // [TplAlias] the geometry TEMPLATE buffers - the memory the instantiate reads
    // clusterTemplateAddress from (clusterIdOffset=0 -> committed clusterId == template's baked
    // id). NEVER cleared by [ClasAlias] (which only checked pose CLAS buffers/pools). If a
    // template buffer overlaps Path A resident CLAS memory, the instantiate copies a resident
    // id (4096+) into a genuine Path B pose CLAS - the last unchecked path.
    const uint32_t tplTotal = m_templateSystemMT != nullptr
      ? m_templateSystemMT->getTemplateBufferRanges(nullptr, nullptr, 0) : 0;
    std::vector<uint64_t> tplLo(tplTotal), tplHi(tplTotal);
    if (tplTotal > 0) {
      m_templateSystemMT->getTemplateBufferRanges(tplLo.data(), tplHi.data(), tplTotal);
    }

    uint32_t outside = 0, zeroVisible = 0, wrongSlot = 0;
    int firstBad = -1, firstWrong = -1;
    uint64_t firstBadRef = 0, firstWrongRef = 0;
    uint32_t firstWrongPool = 0, firstWrongCustom = 0;
    // [ClasAlias] refs whose blasReference lands inside a Path A CLAS range
    uint32_t refInPathA = 0;
    int firstPathAInst = -1;
    uint64_t firstPathARef = 0;
    uint32_t firstPathACustom = 0;
    for (uint32_t i = 0; i < m_dbgSceneAnimInstCount[slot]; i++) {
      const uint64_t ref = inst[i].accelerationStructureReference;
      if (ref == 0) {
        if (inst[i].mask != 0) {
          zeroVisible++;
          if (firstBad < 0) { firstBad = int(i); firstBadRef = 0; }
        }
        continue;
      }
      int poolIdx = -1;
      for (uint32_t p = 0; p < poolCount; p++) {
        if (ref >= poolLo[p] && ref < poolHi[p]) { poolIdx = int(p); break; }
      }
      if (poolIdx < 0) {
        outside++;
        if (firstBad < 0) { firstBad = int(i); firstBadRef = ref; }
      } else if (uint32_t(poolIdx) != expectedPool) {
        wrongSlot++;
        if (firstWrong < 0) {
          firstWrong = int(i);
          firstWrongRef = ref;
          firstWrongPool = uint32_t(poolIdx);
          firstWrongCustom = inst[i].instanceCustomIndex;
        }
      }

      // [ClasAlias] independent of pool bookkeeping: does this ref point into
      // Path A CLAS memory? (a BLAS ref should never live in a CLAS heap - if it
      // does, pose-BLAS and resident-CLAS allocations alias.)
      for (uint32_t r = 0; r < paCount; r++) {
        if (ref >= paLo[r] && ref < paHi[r]) {
          refInPathA++;
          if (firstPathAInst < 0) {
            firstPathAInst = int(i);
            firstPathARef = ref;
            firstPathACustom = inst[i].instanceCustomIndex;
          }
          break;
        }
      }
    }

    if (wrongSlot > 0) {
      static uint32_t s_wrongLogs = 0;
      if (s_wrongLogs < 40 || (m_dbgSceneAnimInstFrame[slot] % 64) == 0) {
        s_wrongLogs++;
        Logger::err(str::format(
          "[SceneAnimInstScan] frame ", m_dbgSceneAnimInstFrame[slot], ": ", wrongSlot, " ref(s) in the WRONG ring pool",
          " (expected pool ", expectedPool, ") of ", m_dbgSceneAnimInstCount[slot],
          " | first inst ", firstWrong, " custom=", firstWrongCustom,
          " ref=0x", std::hex, firstWrongRef, std::dec, " pool ", firstWrongPool,
          "  *** STALE PATCH - entry not re-patched this frame ***"));
      }
    }

    if (outside > 0 || zeroVisible > 0) {
      static uint32_t s_badLogs = 0;
      if (s_badLogs < 40 || (m_dbgSceneAnimInstFrame[slot] % 64) == 0) {
        s_badLogs++;
        Logger::err(str::format(
          "[SceneAnimInstScan] frame ", m_dbgSceneAnimInstFrame[slot], ": ", outside, " ref(s) OUTSIDE all ",
          poolCount, " pose-BLAS pools (SNAPSHOTTED live-at-capture -> lag-immune), ", zeroVisible,
          " visible NULL ref(s) of ", m_dbgSceneAnimInstCount[slot],
          " | first bad inst ", firstBad, " ref=0x", std::hex, firstBadRef, std::dec,
          "  *** DANGLING REF: scene TLAS instance references a pose-BLAS pool already dead at capture"
          " (kept-but-not-rebuilt instance whose pool was freed by a capacity grow) -> resident clusterId ***"));
      }
    } else {
      static uint32_t s_lastCleanFrame = 0;
      if (m_dbgSceneAnimInstFrame[slot] >= s_lastCleanFrame + 300) {
        s_lastCleanFrame = m_dbgSceneAnimInstFrame[slot];
        Logger::info(str::format("[SceneAnimInstScan] frame ", m_dbgSceneAnimInstFrame[slot], ": ",
                                 m_dbgSceneAnimInstCount[slot], " animated (opaque+unordered) refs all inside pose-BLAS pools"));
      }
    }

    // NV-DXVK: [ClasAlias] verdict - does Path B pose memory share bytes with Path A
    // CLAS memory? Overlap = poolLo < paHi && paLo < poolHi.
    {
      uint32_t poolOverlapsPathA = 0;
      int firstOverlapPool = -1, firstOverlapRange = -1;
      for (uint32_t p = 0; p < poolCount; p++) {
        for (uint32_t r = 0; r < paCount; r++) {
          if (poolLo[p] < paHi[r] && paLo[r] < poolHi[p]) {
            poolOverlapsPathA++;
            if (firstOverlapPool < 0) { firstOverlapPool = int(p); firstOverlapRange = int(r); }
          }
        }
      }
      // the decisive one: Path B CLAS-content buffers vs Path A CLAS memory
      uint32_t clasOverlapsPathA = 0;
      int firstClasBuf = -1, firstClasRange = -1;
      for (uint32_t c = 0; c < poseClasCount; c++) {
        for (uint32_t r = 0; r < paCount; r++) {
          if (poseClasLo[c] < paHi[r] && paLo[r] < poseClasHi[c]) {
            clasOverlapsPathA++;
            if (firstClasBuf < 0) { firstClasBuf = int(c); firstClasRange = int(r); }
          }
        }
      }
      // [TplAlias] the never-checked link: TEMPLATE buffers vs Path A CLAS memory. The
      // instantiate reads clusterTemplateAddress from these; an overlap means it copies a
      // resident cluster's baked id (4096+) into a genuine Path B pose CLAS.
      uint32_t tplOverlapsPathA = 0;
      int firstTplBuf = -1, firstTplRange = -1;
      for (uint32_t c = 0; c < tplTotal; c++) {
        for (uint32_t r = 0; r < paCount; r++) {
          if (tplLo[c] < paHi[r] && paLo[r] < tplHi[c]) {
            tplOverlapsPathA++;
            if (firstTplBuf < 0) { firstTplBuf = int(c); firstTplRange = int(r); }
          }
        }
      }
      if (tplOverlapsPathA > 0) {
        static uint32_t s_tplAliasLogs = 0;
        if (s_tplAliasLogs < 40 || (m_dbgSceneAnimInstFrame[slot] % 64) == 0) {
          s_tplAliasLogs++;
          Logger::err(str::format(
            "[TplAlias] frame ", m_dbgSceneAnimInstFrame[slot], ": ", tplOverlapsPathA,
            " TEMPLATE-buffer/Path-A CLAS OVERLAP(s) (first tplBuf ", firstTplBuf, " range ", firstTplRange,
            ") tplRanges=", tplTotal, " paRanges=", paCount, "/", paTotal,
            "  *** TEMPLATE MEMORY == Path A CLAS MEMORY -> instantiate copies resident id 4096+ into Path B pose CLAS ***"));
        }
      } else if (tplTotal > 0 && paCount > 0 && (m_dbgSceneAnimInstFrame[slot] % 300) == 0) {
        Logger::info(str::format(
          "[TplAlias] frame ", m_dbgSceneAnimInstFrame[slot], ": clean - no template buffer in Path A CLAS memory (",
          tplTotal, " template ranges, ", paCount, "/", paTotal, " Path A ranges checked)"));
      }
      if (refInPathA > 0 || poolOverlapsPathA > 0 || clasOverlapsPathA > 0) {
        static uint32_t s_aliasLogs = 0;
        if (s_aliasLogs < 40 || (m_dbgSceneAnimInstFrame[slot] % 64) == 0) {
          s_aliasLogs++;
          Logger::err(str::format(
            "[ClasAlias] frame ", m_dbgSceneAnimInstFrame[slot],
            ": ", refInPathA, " anim ref(s) INSIDE Path A CLAS memory (first inst ", firstPathAInst,
            " custom=", firstPathACustom, " ref=0x", std::hex, firstPathARef, std::dec, ")",
            " | ", poolOverlapsPathA, " pose-pool/Path-A OVERLAP(s) (pool ", firstOverlapPool,
            " range ", firstOverlapRange, ")",
            " | ", clasOverlapsPathA, " pose-CLASbuf/Path-A OVERLAP(s) (clasBuf ", firstClasBuf,
            " range ", firstClasRange, ") paRanges=", paCount, "/", paTotal,
            " poseClasRanges=", poseClasCount, "/", poseClasTotal,
            "  *** ALLOCATOR ALIASING: Path B CLAS memory == Path A CLAS memory -> foreign clusterId 4096+ ***"));
        }
      } else if (clasRangesTruncated) {
        // a "clean" result here would be UNSOUND - we didn't check every range
        static uint32_t s_truncLogs = 0;
        if (s_truncLogs < 40 || (m_dbgSceneAnimInstFrame[slot] % 64) == 0) {
          s_truncLogs++;
          Logger::err(str::format(
            "[ClasAlias] frame ", m_dbgSceneAnimInstFrame[slot], ": RANGE LIST TRUNCATED - checked ",
            paCount, "/", paTotal, " Path A and ", poseClasCount, "/", poseClasTotal,
            " pose-CLAS ranges; overlap verdict UNRELIABLE, raise the cap"));
        }
      } else if (paCount > 0) {
        static uint32_t s_aliasCleanFrame = 0;
        if (m_dbgSceneAnimInstFrame[slot] >= s_aliasCleanFrame + 300) {
          s_aliasCleanFrame = m_dbgSceneAnimInstFrame[slot];
          Logger::info(str::format(
            "[ClasAlias] frame ", m_dbgSceneAnimInstFrame[slot], ": clean - no pose ref/pool/CLASbuf in Path A memory (ALL ",
            paTotal, " Path A, ", poseClasTotal, " pose-CLAS ranges checked) -> aliasing REFUTED; foreign clusterId is a routing bug"));
        }
      }
    }
  }

  // NV-DXVK: [SceneAnimInstScan] device-lost dump - reads the last-captured animated
  // OPAQUE scene-TLAS instances and names any with a null accelerationStructureReference
  // (the reflection-PSR traversal VA=0). Non-null refs should be compared against the
  // [AnimTlasCapture] pool ranges printed alongside - a ref outside them = stale/freed
  // BLAS pool. Diagnostic - revert.
  void ClusterLodManager::dumpSceneAnimInstOnDeviceLost() {
    const uint32_t slot = m_dbgSceneAnimInstLastSlot;
    const uint32_t count = m_dbgSceneAnimInstCount[slot];
    if (m_dbgSceneAnimInstHost == nullptr || count == 0) {
      Logger::err("[SceneAnimInstScan] no mirror captured (device lost before first animated instance copy)");
      return;
    }
    const auto* inst = reinterpret_cast<const VkAccelerationStructureInstanceKHR*>(
      m_dbgSceneAnimInstHost->mapPtr(m_dbgSceneAnimInstStride * slot));
    if (inst == nullptr) {
      Logger::err("[SceneAnimInstScan] mirror mapPtr NULL");
      return;
    }
    Logger::err(str::format("[SceneAnimInstScan] ==== device lost: scene-TLAS animated OPAQUE instances (frame ",
      m_dbgSceneAnimInstFrame[slot], ", count ", count, ") - the refs the reflection-PSR ray traverses ===="));
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < count; ++i) {
      const uint64_t ref = inst[i].accelerationStructureReference;
      const uint32_t mask = inst[i].mask;
      const uint32_t custom = inst[i].instanceCustomIndex;
      const bool isNull = (ref == 0);
      if (isNull) {
        ++zeros;
      }
      if (isNull || i < 8u) {
        Logger::err(str::format("[SceneAnimInstScan]   inst ", i, " ref=0x", std::hex, ref, std::dec,
          " mask=", mask, " custom=", custom, isNull ? "  *** NULL BLAS REF (traced -> VA=0) ***" : ""));
      }
    }
    Logger::err(str::format("[SceneAnimInstScan] ", zeros, " of ", count,
      " animated (opaque+unordered) instances have NULL blasReference (compare non-null refs to [AnimTlasCapture] pool ranges) ==== dump end ===="));
  }

  uint64_t ClusterLodManager::getGeometriesTableAddress() const {
    if (m_renderSystem == nullptr) {
      return 0;
    }
    return m_renderSystem->getGeometriesTableAddress();
  }

  uint64_t ClusterLodManager::getResidentClustersAddress() const {
    if (m_renderSystem == nullptr) {
      return 0;
    }
    return m_renderSystem->getResidentClustersAddress();
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

  bool ClusterLodManager::getPathBExpectedClusterRange(size_t tlasType, uint32_t localIdxB,
                                                       uint32_t& outBase, uint32_t& outCount) const {
    outBase = 0;
    outCount = 0;
    if (m_templateSystemMT == nullptr || localIdxB >= m_slotsB[tlasType].size()) {
      return false;
    }
    const uint32_t framePoseIndex = m_slotsB[tlasType][localIdxB].geometryId;  // Path B slot stores the plain pose index
    if (framePoseIndex >= m_framePoses.size()) {
      return false;
    }
    return m_templateSystemMT->getPoseSetClusterIdRange(m_framePoses[framePoseIndex].poseSetId, outBase, outCount);
  }

  uint32_t ClusterLodManager::getAnimatedClusterTableTotal() const {
    return m_templateSystemMT != nullptr ? m_templateSystemMT->getAnimatedClusterTableCount() : 0;
  }

  bool ClusterLodManager::readPathBSlotPatchedBlasRef(size_t tlasType, uint32_t localIdxB,
                                                      uint64_t& outRef, bool& outInPosePool, uint32_t& outPoolCount) const {
    outRef = 0;
    outInPosePool = false;
    outPoolCount = 0;
    if (m_dbgSceneAnimInstHost == nullptr || m_templateSystemMT == nullptr) {
      return false;
    }
    // mirror flat order matches recordFrame: [m_slotsB[Opaque]...][m_slotsB[Unordered]...]. SSS
    // Path B is not recorded into the scene-anim mirror.
    uint32_t flatIndex;
    if (tlasType == size_t(Tlas::Opaque)) {
      flatIndex = localIdxB;
    } else if (tlasType == size_t(Tlas::Unordered)) {
      flatIndex = uint32_t(m_slotsB[size_t(Tlas::Opaque)].size()) + localIdxB;
    } else {
      return false;
    }
    const uint32_t slot = m_dbgSceneAnimInstLastSlot;
    if (flatIndex >= m_dbgSceneAnimInstCount[slot]) {
      return false;
    }
    const auto* inst = reinterpret_cast<const VkAccelerationStructureInstanceKHR*>(
      m_dbgSceneAnimInstHost->mapPtr(m_dbgSceneAnimInstStride * slot));
    if (inst == nullptr) {
      return false;
    }
    outRef = inst[flatIndex].accelerationStructureReference;
    // classify against pose-BLAS pool ranges (the correct home for a Path B ref)
    uint64_t poolLo[16], poolHi[16];
    outPoolCount = m_templateSystemMT->getPoseBlasPools(poolLo, poolHi, 16, nullptr);
    for (uint32_t p = 0; p < outPoolCount && p < 16; p++) {
      if (outRef != 0 && outRef >= poolLo[p] && outRef < poolHi[p]) { outInPosePool = true; break; }
    }
    return true;
  }

  void ClusterLodManager::countSlotsByPosBuf(uint32_t posBuf, uint32_t& outPathA, uint32_t& outPathB) const {
    outPathA = 0;
    outPathB = 0;
    for (uint32_t t = 0; t < uint32_t(Tlas::Count); t++) {
      for (const ClusterSlot& s : m_slots[t]) {
        if (s.instance != nullptr && uint32_t(s.instance->surface.positionBufferIndex) == posBuf) { outPathA++; }
      }
      for (const ClusterSlot& s : m_slotsB[t]) {
        if (s.instance != nullptr && uint32_t(s.instance->surface.positionBufferIndex) == posBuf) { outPathB++; }
      }
    }
  }

}  // namespace dxvk
