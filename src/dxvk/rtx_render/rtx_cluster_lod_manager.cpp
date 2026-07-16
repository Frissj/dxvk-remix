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

    // 3x3 symmetric eigendecomposition via cyclic Jacobi (doubles). Used by the
    // promotion probe to build the SAMPLE-CENTERED ref Gram pseudoinverse and, for
    // rank-2 (planar) meshes, the null-plane normal the kernel completes the affine
    // solve with. eigVec is column-major: column i = eigenvector of eig[i].
    void eigenSym3x3(const double g[9], double eig[3], double eigVec[9]) {
      double a[9];
      double v[9] = { 1,0,0, 0,1,0, 0,0,1 };
      for (int i = 0; i < 9; i++) {
        a[i] = g[i];
      }
      for (int sweep = 0; sweep < 32; sweep++) {
        double off = 0.0;
        for (int p = 0; p < 3; p++) {
          for (int q = p + 1; q < 3; q++) {
            off += a[p * 3 + q] * a[p * 3 + q];
          }
        }
        if (off < 1e-28) {
          break;
        }
        for (int p = 0; p < 3; p++) {
          for (int q = p + 1; q < 3; q++) {
            const double apq = a[p * 3 + q];
            if (std::abs(apq) < 1e-32) {
              continue;
            }
            const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq);
            const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            for (int k = 0; k < 3; k++) {
              const double akp = a[k * 3 + p];
              const double akq = a[k * 3 + q];
              a[k * 3 + p] = c * akp - s * akq;
              a[k * 3 + q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; k++) {
              const double apk = a[p * 3 + k];
              const double aqk = a[q * 3 + k];
              a[p * 3 + k] = c * apk - s * aqk;
              a[q * 3 + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; k++) {
              const double vkp = v[k * 3 + p];
              const double vkq = v[k * 3 + q];
              v[k * 3 + p] = c * vkp - s * vkq;
              v[k * 3 + q] = s * vkp + c * vkq;
            }
          }
        }
      }
      for (int i = 0; i < 3; i++) {
        eig[i] = a[i * 3 + i];
      }
      for (int i = 0; i < 9; i++) {
        eigVec[i] = v[i];
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

    // DIAG (DrawTrace/intake): this is the cluster manager's entry - a draw that
    // reaches here passed SceneManager but may still be dropped by the empty-hash
    // guard below (geometry with no hashable data never enters the cluster system).
    const uint64_t drawMat = drawCallState.getMaterialData().getHash();
    const bool traceThis = clusterLodPromoTraceMatchesMaterial(drawMat);
    if (traceThis) {
      const RasterGeometry& gd = drawCallState.getGeometryData();
      const bool captured = drawCallState.preCaptureVertexData != nullptr;
      const bool skinned = drawCallState.getSkinningState().numBones > 0 && gd.numBonesPerVertex > 0;
      static std::mutex s_mx;
      static std::unordered_map<uint64_t, uint32_t> s_last;
      const uint32_t fr = m_device->getCurrentFrameId();
      std::lock_guard<std::mutex> lk(s_mx);
      uint32_t& last = s_last[geometryHash];
      if (last == 0u || fr - last > 300u) {
        last = fr;
        Logger::info(str::format("[DrawTrace/intake] geom 0x", std::hex, geometryHash,
                                 " mat 0x", drawMat, std::dec, " captured ", captured,
                                 " skinned ", skinned, " vtxUpd ", vertexDataUpdated,
                                 " verts ", gd.vertexCount, " indices ", gd.indexCount,
                                 " emptyHash ", (geometryHash == kEmptyHash),
                                 " -> ", (geometryHash == kEmptyHash ? "DROPPED (no hash)" : "to provider"),
                                 " frame ", fr));
      }
    }

    if (geometryHash == kEmptyHash) {
      return;
    }

    m_provider->onDrawCallGeometry(drawCallState, geometryHash, vertexDataUpdated, traceThis);
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

    // Only vertices actually REFERENCED by the index buffer get a vertex-shader
    // capture write: the VS runs per index entry and writes capture slot
    // gl_VertexIndex - baseVertex, so a vertex in [0,vertexCount) that no triangle
    // references never receives a capture. Its slot holds stale VRAM (often another
    // mesh's positions - same spread, so it looks fine to a coarse check but
    // scrambles the per-vertex correspondence). Sampling/gating those garbage slots
    // is what rejected genuinely-rigid meshes that draw a SUBSET of a shared vertex
    // buffer (e.g. buildings): the fit ran against positions the capture never
    // wrote. Restrict the whole probe to the referenced set so every sampled and
    // gated vertex has a real capture behind it.
    std::vector<uint32_t> referenced;
    referenced.reserve(std::min<size_t>(snapshot.indices.size(), vertexCount));
    {
      std::vector<uint8_t> seen(vertexCount, 0);
      for (const uint32_t idx : snapshot.indices) {
        if (idx < vertexCount && !seen[idx]) {
          seen[idx] = 1;
          referenced.push_back(idx);
        }
      }
    }
    const uint32_t refCount = uint32_t(referenced.size());
    if (refCount < 4) {
      return;  // too few referenced vertices to fit a transform
    }

    // centroid + bounding radius over the REFERENCED vertices (residuals relative)
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (uint32_t r = 0; r < refCount; r++) {
      const uint32_t v = referenced[r];
      cx += positions[v * 3 + 0];
      cy += positions[v * 3 + 1];
      cz += positions[v * 3 + 2];
    }
    cx /= refCount;
    cy /= refCount;
    cz /= refCount;

    double radiusSq = 0.0;
    for (uint32_t r = 0; r < refCount; r++) {
      const uint32_t v = referenced[r];
      const double dx = positions[v * 3 + 0] - cx;
      const double dy = positions[v * 3 + 1] - cy;
      const double dz = positions[v * 3 + 2] - cz;
      radiusSq = std::max(radiusSq, dx * dx + dy * dy + dz * dz);
    }
    const float radius = float(std::sqrt(radiusSq));
    if (!(radius > 0.0f)) {
      return;  // degenerate point cloud - unpromotable, stays Path B
    }

    // farthest-point sampling over a strided subset of the REFERENCED vertices: 64
    // spread solve samples, then 32 validation samples continuing the same chain
    // (spread AND disjoint from the solve set)
    const uint32_t stride = std::max(1u, refCount / 4096u);
    std::vector<uint32_t> candidates;
    candidates.reserve(refCount / stride + 1);
    for (uint32_t r = 0; r < refCount; r += stride) {
      candidates.push_back(referenced[r]);
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

    // SAMPLE-CENTERED 3x3 ref Gram inverse (doubles) for the kernel's affine solve:
    //   A = Mcov * G3inv,  t = capBar - A*refBar   (translation decoupled)
    // This replaces the old 4x4 centered-homogeneous pseudoinverse: evaluating
    // gInv4x4 * b in the kernel multiplied huge inverse entries (~1/variance, 1e3+
    // for small-radius meshes) by huge UNCENTERED capture sums - fp32 catastrophic
    // cancellation that read a perfectly rigid 0.068-radius piece as residual 0.072
    // ([PromoDump] verified: double-precision affine on the same samples = 2e-4).
    // With both sides sample-centered every kernel intermediate is O(1)-sane.
    double sm[3] = { 0.0, 0.0, 0.0 };  // solve-sample mean of the CENTERED positions
    for (uint32_t i = 0; i < pickedSolve; i++) {
      const uint32_t v = candidates[picked[i]];
      sm[0] += positions[v * 3 + 0] - cx;
      sm[1] += positions[v * 3 + 1] - cy;
      sm[2] += positions[v * 3 + 2] - cz;
    }
    const double invN = pickedSolve > 0 ? 1.0 / double(pickedSolve) : 0.0;
    sm[0] *= invN; sm[1] *= invN; sm[2] *= invN;

    double g3[9] = {};
    for (uint32_t i = 0; i < pickedSolve; i++) {
      const uint32_t v = candidates[picked[i]];
      const double h[3] = { positions[v * 3 + 0] - cx - sm[0],
                            positions[v * 3 + 1] - cy - sm[1],
                            positions[v * 3 + 2] - cz - sm[2] };
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          g3[r * 3 + c] += h[r] * h[c];
        }
      }
    }
    // Eigendecomposed pseudoinverse + PLANE COMPLETION support:
    //   gInv[0..8]  = row-major 3x3 Gram PSEUDOinverse (null eigenspace truncated)
    //   gInv[9..11] = null-plane normal nHat (rank-2 only; kernel completes the
    //                 affine solve's out-of-plane column with s*R*nHat)
    //   gInv[12]    = 1.0 when rank-2 completion applies, else 0.0
    // Rank <= 1 (collinear/point sample sets) stores all zeros -> the kernel's
    // affine guards reject -> rigid fallback (unchanged behavior).
    double gInv[16] = {};
    {
      double eig[3];
      double eigVec[9];
      eigenSym3x3(g3, eig, eigVec);
      double maxEig = 0.0;
      for (int i = 0; i < 3; i++) {
        maxEig = std::max(maxEig, std::abs(eig[i]));
      }
      const double tol = maxEig * 1e-8;
      int rank = 0;
      int nullIdx = -1;
      for (int i = 0; i < 3; i++) {
        if (std::abs(eig[i]) > tol) {
          rank++;
        } else {
          nullIdx = i;
        }
      }
      if (rank >= 2 && maxEig > 0.0) {
        // pseudoinverse = V * diag(1/eig | non-null) * V^T
        for (int r = 0; r < 3; r++) {
          for (int c = 0; c < 3; c++) {
            double acc = 0.0;
            for (int k = 0; k < 3; k++) {
              if (std::abs(eig[k]) > tol) {
                acc += eigVec[r * 3 + k] * eigVec[c * 3 + k] / eig[k];
              }
            }
            gInv[r * 3 + c] = acc;
          }
        }
        if (rank == 2 && nullIdx >= 0) {
          gInv[9]  = eigVec[0 * 3 + nullIdx];
          gInv[10] = eigVec[1 * 3 + nullIdx];
          gInv[11] = eigVec[2 * 3 + nullIdx];
          gInv[12] = 1.0;
        }
      }
      // gInv[13..15] unused (kept zero; header layout unchanged)
    }

    // blob assembly: header + solve + validation (falls back to the solve set
    // when no disjoint candidates exist - the 64-vs-12-DOF overdetermination
    // still exposes non-affine output) + full centered ref positions (gate)
    // + (DIAG) validation-sample REF NORMALS appended at the very end when the
    // snapshot has them: the kernel measures how well the validation error
    // vectors align with the normals (normal-push displacement discriminator).
    const uint32_t effectiveValidation = pickedValidation > 0 ? pickedValidation : pickedSolve;
    const bool hasNormals = snapshot.normals.size() >= size_t(vertexCount) * 3;
    std::vector<uint8_t> blob(sizeof(ProbeHeader)
                              + sizeof(ProbeSample) * (size_t(pickedSolve) + effectiveValidation + refCount
                                                       + (hasNormals ? effectiveValidation : 0)));

    ProbeHeader header = {};
    header.sampleCount = pickedSolve;
    header.validationCount = effectiveValidation;
    header.vertexCount = refCount;  // gate sweeps only referenced vertices
    header.pad = hasNormals ? 1u : 0u;  // DIAG: 1 = validation normals appended
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
    for (uint32_t r = 0; r < refCount; r++) {
      writeSample(samples[size_t(pickedSolve) + effectiveValidation + r], referenced[r]);
    }
    if (hasNormals) {
      // DIAG: ref normals of the validation samples, same order as the validation
      // region (uncentered unit vectors; ProbeSample.index kept for sanity)
      const float* normals = snapshot.normals.data();
      for (uint32_t i = 0; i < effectiveValidation; i++) {
        const uint32_t pickIndex = pickedValidation > 0 ? pickedSolve + i : i;
        const uint32_t v = candidates[picked[pickIndex]];
        ProbeSample& out = samples[size_t(pickedSolve) + effectiveValidation + refCount + i];
        out.index = v;
        out.x = normals[v * 3 + 0];
        out.y = normals[v * 3 + 1];
        out.z = normals[v * 3 + 2];
      }
    }

    // DIAG raw dump: log the solve samples' REF positions once for the traced hash
    // (absolute object space = centered + centroid; order matches the kernel's
    // s_capNow / the [PromoDump] cap lines, so line i pairs with line i)
    {
      const std::string& dumpHashStr = ClusterLodOptions::Promotion::dumpGeometryHash();
      if (!dumpHashStr.empty()) {
        uint64_t dumpHash = 0;
        try { dumpHash = std::stoull(dumpHashStr, nullptr, 16); } catch (...) { dumpHash = 0; }
        if (dumpHash != 0 && dumpHash == snapshot.geometryHash) {
          Logger::info(str::format("[PromoDump] ref geometry 0x", std::hex, snapshot.geometryHash, std::dec,
                                   " solveSamples ", pickedSolve,
                                   " centroid (", float(cx), ", ", float(cy), ", ", float(cz),
                                   ") radius ", radius));
          for (uint32_t i = 0; i < pickedSolve; i++) {
            const uint32_t v = candidates[picked[i]];
            Logger::info(str::format("[PromoDump] ref[", i, "] idx ", v,
                                     " pos (", positions[v * 3 + 0], ", ", positions[v * 3 + 1],
                                     ", ", positions[v * 3 + 2], ")"));
          }
        }
      }
    }

    const uint64_t probeVa = m_templateSystem->uploadPromotionProbe(blob.data(), blob.size());
    if (probeVa == 0) {
      return;
    }

    // Retain the topology of captured candidates so a rest-capture snapshot can be
    // assembled later from a GPU readback alone (draw data may be long dead then).
    if (snapshot.isCaptured && !snapshot.isRestCapture
        && ClusterLodOptions::Promotion::restCaptureReference()) {
      std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
      RetainedTopology& topo = m_promoTopologyByHash[snapshot.geometryHash];
      topo.indices = snapshot.indices;
      topo.indicesHash = snapshot.indicesHash;
      topo.topologyKey = snapshot.topologyKey;
      topo.vertexCount = snapshot.vertexCount;
      topo.name = snapshot.name;
    }

    {
      std::lock_guard<std::mutex> lock(m_promoPendingMutex);
      // rest probes key the ORIGINAL candidate (promoKeyHash) and carry the
      // space-tagged rest hash for residency routing
      PendingProbe pending;
      pending.geometryHash = snapshot.isRestCapture ? snapshot.promoKeyHash : snapshot.geometryHash;
      pending.probeVa = probeVa;
      pending.vertexCount = refCount;
      pending.routeHash = snapshot.isRestCapture ? snapshot.geometryHash : 0;
      pending.topologyKey = snapshot.topologyKey;  // stable identity for the churning-hash draw side
      m_promoPendingProbes.push_back(pending);
    }

    Logger::info(str::format("[ClusterLOD] ", snapshot.name, ": promotion probe uploaded (referenced verts ", refCount,
                             " of ", vertexCount, ", blob ", blob.size() / 1024, " KiB)"));
  }

  // DIAG correspondence-scan offset table - MUST mirror kScanOffsets in
  // promotion_solve.comp exactly (index -> ref->capture vertex-index offset). The
  // shader reports the winning table index in diagGuard[2:8]; this maps it back.
  static constexpr int kPromoScanOffsets[64] = {
       0,
       1,   -1,    2,   -2,    3,   -3,    4,   -4,    5,   -5,    6,   -6,    7,   -7,    8,   -8,
      10,  -10,   12,  -12,   16,  -16,   20,  -20,   24,  -24,   32,  -32,   48,  -48,   64,  -64,
      96,  -96,  128, -128,  192, -192,  256, -256,  384, -384,  512, -512,  768, -768, 1024,-1024,
    1536,-1536, 2048,-2048, 3072,-3072, 4096,-4096, 8192,-8192,16384,-16384,32768,-32768,65536
  };

  void ClusterLodManager::updatePromotionStates() {
    if (m_renderSystem == nullptr || !ClusterLodOptions::Promotion::enable()) {
      return;
    }

    // adopt worker-uploaded probes
    {
      std::lock_guard<std::mutex> lock(m_promoPendingMutex);
      for (const PendingProbe& pending : m_promoPendingProbes) {
        // REST probe for an existing candidate: swap the reference in place - free
        // the old (object-space) blob, adopt the rest probe, restart Probing, and
        // route residency to the space-tagged rest hash. The state slot persists.
        const auto existing = m_promoCandidates.find(pending.geometryHash);
        if (existing != m_promoCandidates.end()) {
          if (pending.routeHash != 0) {
            if (m_templateSystemMT != nullptr && existing->second.probeVa != 0) {
              m_templateSystemMT->freePromotionProbe(existing->second.probeVa);
            }
            existing->second.probeVa = pending.probeVa;
            existing->second.vertexCount = pending.vertexCount;
            existing->second.routeHash = pending.routeHash;
            existing->second.phase = PromotionCandidate::Phase::Probing;
            existing->second.gateFrames = 0;
            existing->second.stuckFrames = 0;
            existing->second.loggedTemporalHold = false;
            existing->second.restState = PromotionCandidate::RestState::Referenced;
            // per-instance rest verdicts start fresh: instance state from the
            // object-space probe era (pins, demotions, GPU streaks) is void -
            // residency moved to the rest hash and the reference content changed
            for (auto& slotEntry : m_promoSlotByBlas) {
              if (slotEntry.second.geometryHash == pending.geometryHash) {
                slotEntry.second.restPhase = PromoInstance::RestPhase::Probing;
                slotEntry.second.restGateFrames = 0;
                slotEntry.second.restStuckFrames = 0;
                slotEntry.second.restLastSolveFrame = 0;
                slotEntry.second.demoted = false;
                slotEntry.second.sweepPending = false;
                slotEntry.second.residentGeometryId = ~0u;
              }
            }
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, pending.geometryHash,
                                     " probe re-referenced to REST CAPTURE 0x", pending.routeHash, std::dec));
          }
          else if (m_templateSystemMT != nullptr && pending.probeVa != 0) {
            // duplicate non-rest probe for a live candidate: free the fresh blob
            m_templateSystemMT->freePromotionProbe(pending.probeVa);
          }
          continue;
        }
        if (m_promoNextStateSlot >= lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
          ONCE(Logger::warn("[ClusterLOD] promotion state slots exhausted - further candidates stay Path B"));
          break;
        }
        PromotionCandidate candidate;
        candidate.probeVa = pending.probeVa;
        candidate.vertexCount = pending.vertexCount;
        candidate.stateSlot = m_promoNextStateSlot++;
        candidate.routeHash = pending.routeHash;
        candidate.topologyKey = pending.topologyKey;
        candidate.createdFrame = m_device->getCurrentFrameId();
        candidate.createdTime = std::chrono::steady_clock::now();
        m_promoCandidates.emplace(pending.geometryHash, candidate);
        // index by stable topology so a churned-hash draw resolves to this candidate
        // every frame (this game's captured hashes churn per frame; see topologyKey).
        if (pending.topologyKey != 0) {
          m_promoCandidateByTopology[pending.topologyKey] = pending.geometryHash;
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

    // DIAG raw dump: the traced geometry's solve-sample CAPTURE positions (ring-lagged;
    // line i pairs with the [PromoDump] ref[i] line from the probe build). Throttled.
    {
      static uint32_t s_lastDumpFrame = 0;
      const uint32_t frameNow = m_device->getCurrentFrameId();
      if (!ClusterLodOptions::Promotion::dumpGeometryHash().empty()
          && (s_lastDumpFrame == 0u || frameNow - s_lastDumpFrame > 300u)) {
        float cap[64 * 3];
        if (m_renderSystem->readPromotionSampleDump(cap)) {
          s_lastDumpFrame = frameNow;
          Logger::info(str::format("[PromoDump] cap geometry 0x", ClusterLodOptions::Promotion::dumpGeometryHash(),
                                   " frame ", frameNow));
          for (uint32_t i = 0; i < 64; i++) {
            Logger::info(str::format("[PromoDump] cap[", i, "] pos (", cap[i * 3 + 0], ", ",
                                     cap[i * 3 + 1], ", ", cap[i * 3 + 2], ")"));
          }
        }
      }
    }

    const uint32_t rigidFrames = uint32_t(std::max(1, ClusterLodOptions::Promotion::rigidFrames()));
    const float epsilon = std::max(1e-5f, ClusterLodOptions::Promotion::residualEpsilon());
    const uint32_t gateLag = uint32_t(std::max(2, ClusterLodOptions::Promotion::gateLagFrames()));

    // fresh per-pass snapshot of the solve diagnostics
    m_diagMaxAffineNonRigid = 0.0f;
    m_diagProbeZeroSlots = 0;
    m_diagDegenSlots = 0;
    m_diagWorstGeom = 0;
    m_diagWorstRefVar = 0.0f;
    m_diagWorstSampleN = 0;
    for (uint32_t& c : m_diagReasonHist) { c = 0; }
    float worstRefVarSeen = std::numeric_limits<float>::max();

    for (auto& entry : m_promoCandidates) {
      PromotionCandidate& candidate = entry.second;
      const lodclusters_remix::PromotionStateView& state = m_promoStates[candidate.stateSlot];
      m_diagMaxAffineNonRigid = std::max(m_diagMaxAffineNonRigid, state.affineNonRigid);

      // [PromoLat] tally an ACTUAL solve for this candidate: state.lastFrame is
      // the frame the GPU last solved this slot, so it stepping forward means a
      // solve ran (i.e. an instance was drawn and emitted a probe). A candidate
      // whose lastFrame stops advancing is off-screen - the decisive signal for
      // "waiting to be drawn" vs "drawn but grinding".
      if (state.lastFrame != 0u && state.lastFrame != candidate.lastCountedSolveFrame) {
        candidate.solveCount++;
        candidate.lastCountedSolveFrame = state.lastFrame;
        // rigid streak fell (a non-rigid spike reset the consecutive count)
        if (state.rigidStreak < candidate.prevRigidStreak && candidate.prevRigidStreak > 0u) {
          candidate.streakResets++;
        }
        candidate.prevRigidStreak = state.rigidStreak;
      }
      if ((state.diagGuard & 1u) != 0u) {
        m_diagProbeZeroSlots++;
      }
      if ((state.diagGuard & 2u) != 0u) {
        // a DEGENERATE fit (why it would have exploded). Name the worst one by the
        // smallest ref-sample spread. diagAux = refVar (float bits); diagGuard[8:16]
        // = sampleCount used. refVar ~ 0 with sampleCount full == coincident refs;
        // sampleCount ~ 0 == probe not populated (solved before ready).
        m_diagDegenSlots++;
        const uint32_t reason = (state.diagGuard >> 16) & 0xFFu;
        if (reason < 7u) { m_diagReasonHist[reason]++; }
        float rv = 0.0f;
        std::memcpy(&rv, &state.diagAux, sizeof(float));
        if (rv < worstRefVarSeen) {
          worstRefVarSeen = rv;
          m_diagWorstGeom = entry.first;
          m_diagWorstRefVar = rv;
          m_diagWorstSampleN = (state.diagGuard >> 8) & 0xFFu;
        }
      }

      switch (candidate.phase) {
      case PromotionCandidate::Phase::Probing:
        // REST-referenced: per-instance verdicts (instance loop below) drive this
        // candidate - its own state slot receives no solves anymore, so the state
        // here is stale. The phase flips to Promoted when the first instance
        // passes its gate (residency routing keys on the candidate phase).
        if (candidate.routeHash != 0
            && candidate.restState == PromotionCandidate::RestState::Referenced) {
          break;
        }
        if (state.rigidStreak >= rigidFrames) {
          candidate.phase = PromotionCandidate::Phase::GateScheduled;
          candidate.gateScheduledFrame = m_device->getCurrentFrameId();  // [PromoLat] Probing->Gate handoff
        } else if (ClusterLodOptions::Promotion::restCaptureReference()
                   && candidate.restState == PromotionCandidate::RestState::None
                   && state.residualRel > epsilon
                   && state.lastFrame != 0u) {
          // REST-CAPTURE trigger: temporally static (not animating) yet never fits
          // any single transform of the CPU snapshot -> the VS builds a genuinely
          // different shape. Re-reference the probe (and Path A clusters) to the
          // captured rest pose; the solve then fits identity(+motion) and promotes.
          if (state.temporalDeformRel <= ClusterLodOptions::Promotion::temporalEpsilon()) {
            candidate.stuckFrames++;
            if (candidate.stuckFrames >= uint32_t(std::max(10, ClusterLodOptions::Promotion::restCaptureStuckFrames()))) {
              candidate.restState = PromotionCandidate::RestState::Requested;
              Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                       " static but non-affine (residual ", state.residualRel,
                                       ") - requesting REST-CAPTURE reference"));
            }
          } else {
            candidate.stuckFrames = 0;  // it moves - genuinely animated, leave on Path B
          }
        } else if (!candidate.loggedTemporalHold
                   && (state.temporalDeformRel > ClusterLodOptions::Promotion::temporalEpsilon()
                       || state.residualRel > epsilon)) {
          // One-shot: WHY this candidate never builds a rigid streak, showing BOTH bars.
          //  - sparseResidual = per-frame rigid-fit error. >> epsilon => the mesh does
          //    not fit ANY rigid transform this frame => genuinely non-rigid geometry,
          //    independent of the temporal gate (won't promote even with it disabled).
          //  - tDeform = inter-frame distance drift. High with a LOW sparseResidual =
          //    fits rigidly each frame but drifts across frames (the temporal signal).
          // marginal (just over epsilon) => a tuning issue; large => real animation.
          candidate.loggedTemporalHold = true;
          // scan result too (needs rtx.clusterLod.promotion.correspondenceScan=True):
          // scanOff!=0 / scanScore>0 => a ref->cap index skew is the non-rigidity (the
          // old correspondence class, fixable); scanScore~0 with high sparseResidual =>
          // the point cloud is genuinely a non-rigid deform, no offset can save it.
          const uint32_t scanIdx = (state.diagGuard >> 2) & 0x3Fu;
          const uint32_t scanScoreQ = (state.diagGuard >> 27) & 0x1Fu;
          const uint32_t reflected = (state.diagGuard >> 26) & 0x1u;
          const int scanOff = kPromoScanOffsets[scanIdx];
          const std::string scanScore = scanScoreQ >= 31u ? std::string("n/a")
                                                          : str::format(float(scanScoreQ) / 20.0f);
          // aff: which solve ran - "used" = affine accepted; otherwise the first
          // guard that pushed it onto the rigid fallback (finite/norm/aniso/cond/var)
          const uint32_t affFail = state.solveInfo & 0xFFu;
          const char* affStr = (state.solveInfo & 0x100u) != 0u ? "used"
                             : affFail == 1u ? "FAIL-finite"
                             : affFail == 2u ? "FAIL-norm"
                             : affFail == 3u ? "FAIL-aniso"
                             : affFail == 4u ? "FAIL-cond"
                             : affFail == 5u ? "FAIL-refVar" : "none";
          Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                   " stuck in Probing (sparseResidual ", state.residualRel,
                                   " vs eps ", epsilon, ", aff ", affStr,
                                   ", meanDev ", state.meanDevRel,
                                   ", dirCoh ", state.dirCoherence,
                                   ", tDeform ", state.temporalDeformRel,
                                   ", scanOff ", scanOff, ", scanScore ", scanScore, ", refl ", reflected,
                                   ", verts ", candidate.vertexCount, ")"));
        }
        break;

      case PromotionCandidate::Phase::GateRunning:
        if (++candidate.gateFrames >= gateLag) {
          if (state.gateResidualRel > 0.0f && state.gateResidualRel <= epsilon) {
            candidate.phase = PromotionCandidate::Phase::Promoted;
            m_statsPromoted++;
            // [PromoLat] decompose the promotion latency. elapsedFrames = wall
            // frames since adoption; solveCount = frames actually solved (drawn).
            // coverage = solveCount/elapsedFrames: LOW => the mesh was off-screen
            // most of the time (inherent, not a pipeline bug); HIGH but slow =>
            // streak resets / gate round-trips are the cost. probingFrames vs
            // gateFrames splits the two phases.
            const uint32_t nowFrame = m_device->getCurrentFrameId();
            const uint32_t elapsedFrames = nowFrame - candidate.createdFrame;
            const uint32_t probingFrames = candidate.gateScheduledFrame != 0
              ? candidate.gateScheduledFrame - candidate.createdFrame : elapsedFrames;
            const double elapsedSec =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - candidate.createdTime).count() / 1000.0;
            const double coverage = elapsedFrames > 0 ? double(candidate.solveCount) / double(elapsedFrames) : 0.0;
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                     " PROMOTED to Path A (full-mesh residual ", state.gateResidualRel, ")",
                                     " | [PromoLat] ", elapsedSec, "s over ", elapsedFrames, " frames",
                                     " | solved ", candidate.solveCount, " (coverage ", coverage,
                                     ") | probing ", probingFrames, " frames, gate ", candidate.gateFrames,
                                     " frames | streakResets ", candidate.streakResets,
                                     " | verdict=", coverage < 0.5 ? "OFF-SCREEN-BOUND(inherent)"
                                       : candidate.streakResets > 3 ? "STREAK-THRASH(non-rigid spikes)"
                                       : "solve-bound"));
          } else if (state.gateResidualRel > epsilon) {
            candidate.phase = PromotionCandidate::Phase::Rejected;
            m_statsPromoRejected++;
            // DIAG: capVar (state.affineNonRigid, repurposed) vs refVar (diagAux).
            // sEff = sqrt(capVar/refVar). sEff~1 + high residual = ref/cap same
            // spread but scrambled correspondence; sEff>>1 = capture in a bigger
            // space (scale); residual>2 (impossible for a permutation) = the solve
            // blew up on a degenerate/coplanar mesh.
            const float capVar = state.affineNonRigid;
            float refVarDbg = 0.0f;
            std::memcpy(&refVarDbg, &state.diagAux, sizeof(float));
            const float sEff = refVarDbg > 1e-12f ? std::sqrt(capVar / refVarDbg) : -1.0f;
            // DIAG correspondence scan (probe only): diagGuard[2:8] = scan table index,
            // [24:26] = verdict. Decoded to the actual ref->cap index offset that best
            // matches the point cloud. scanV COLLAPSE at a nonzero scanOff == the
            // shared-vertex-buffer index skew that scrambled this mesh.
            const uint32_t scanIdx = (state.diagGuard >> 2) & 0x3Fu;
            const uint32_t scanVerdict = (state.diagGuard >> 24) & 0x3u;
            const uint32_t reflected = (state.diagGuard >> 26) & 0x1u;
            const uint32_t scanScoreQ = (state.diagGuard >> 27) & 0x1Fu;
            const int scanOff = kPromoScanOffsets[scanIdx];
            const char* scanV = scanVerdict == 2u ? "COLLAPSE" : (scanVerdict == 1u ? "impr" : "none");
            // scanScore = offset-0 pairwise mismatch (~0 => perfect correspondence, fit
            // is the problem; large => genuinely non-rigid). 31 => scan did NOT run.
            // refl = the fit's cross-covariance is improper (reflection the proper-only
            // solve cannot represent): the prime suspect for scanScore~0 + high residual.
            const std::string scanScore = scanScoreQ >= 31u ? std::string("n/a(off?)")
                                                            : str::format(float(scanScoreQ) / 20.0f);
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                     " gate REJECTED (full-mesh residual ", state.gateResidualRel,
                                     ", verts ", candidate.vertexCount,
                                     ", refVar ", refVarDbg, ", capVar ", capVar, ", sEff ", sEff,
                                     ", scanOff ", scanOff, ", scanV ", scanV, ", scanScore ", scanScore,
                                     ", refl ", reflected,
                                     ", gateOver ", state.gateOverCount, "/", candidate.vertexCount,
                                     ", gateStale ", state.gateStaleFrames,
                                     ", tDeform ", state.temporalDeformRel,
                                     ", meanDev ", state.meanDevRel, ", dirCoh ", state.dirCoherence,
                                     "), stays Path B"));
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

      // ---- per-instance REST verdicts (pre-promotion phases) ----
      // Mirrors the candidate state machine, per instance: streak -> gate ->
      // Promoted/Rejected. Pre-promotion phases skip the demote logic below
      // (their solves legitimately read non-rigid while they probe). Rejection
      // is terminal: a static instance that never fits the shared rest
      // reference has a divergent VS-built shape - buildPromotionEntries stops
      // emitting its solves, so it costs nothing steady-state.
      if (promoInstance.restPhase != PromoInstance::RestPhase::None
          && promoInstance.restPhase != PromoInstance::RestPhase::Promoted) {
        switch (promoInstance.restPhase) {
        case PromoInstance::RestPhase::Probing:
          if (state.rigidStreak >= rigidFrames) {
            promoInstance.restPhase = PromoInstance::RestPhase::GateScheduled;
          } else if (state.lastFrame != 0u && state.lastFrame != promoInstance.restLastSolveFrame) {
            // a fresh solve landed (stale passes change nothing - an absent
            // instance must not be judged on its last state)
            promoInstance.restLastSolveFrame = state.lastFrame;
            if (state.residualRel > epsilon
                && state.temporalDeformRel <= ClusterLodOptions::Promotion::temporalEpsilon()) {
              // static but not matching the shared rest reference; same patience
              // the rest trigger itself used before terminating
              if (++promoInstance.restStuckFrames
                  >= uint32_t(std::max(10, ClusterLodOptions::Promotion::restCaptureStuckFrames()))) {
                promoInstance.restPhase = PromoInstance::RestPhase::Rejected;
                Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                         ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                         ") REST verdict: static but does not match the shared rest reference (residual ",
                                         state.residualRel, ") - stays Path B (terminal)"));
              }
            } else {
              promoInstance.restStuckFrames = 0;
            }
          }
          break;

        case PromoInstance::RestPhase::GateRunning:
          if (++promoInstance.restGateFrames >= gateLag) {
            if (state.gateResidualRel > 0.0f && state.gateResidualRel <= epsilon) {
              promoInstance.restPhase = PromoInstance::RestPhase::Promoted;
              Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                       ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                       ") REST instance PROMOTED to Path A (full-mesh residual ",
                                       state.gateResidualRel, ")"));
              // first instance to pass promotes the candidate (routing gate)
              const auto candIt = m_promoCandidates.find(promoInstance.geometryHash);
              if (candIt != m_promoCandidates.end()
                  && candIt->second.phase != PromotionCandidate::Phase::Promoted) {
                candIt->second.phase = PromotionCandidate::Phase::Promoted;
                m_statsPromoted++;
                Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                         promoInstance.geometryHash, std::dec,
                                         " PROMOTED to Path A via per-instance REST verdict"));
              }
            } else if (state.gateResidualRel > epsilon) {
              promoInstance.restPhase = PromoInstance::RestPhase::Rejected;
              Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                       ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                       ") REST gate REJECTED (full-mesh residual ", state.gateResidualRel,
                                       ") - stays Path B (terminal)"));
            } else {
              // gate never accumulated (instance off-screen that frame) - retry
              promoInstance.restPhase = PromoInstance::RestPhase::GateScheduled;
              promoInstance.restGateFrames = 0;
            }
          }
          break;

        default:
          break;  // GateScheduled waits on buildPromotionEntries; Rejected is terminal
        }
        continue;
      }

      // periodic full-mesh sweep verdict (same lag handling as the gate). The sweep
      // judges an ALREADY-PROMOTED instance, so the demote hysteresis applies here
      // exactly like the per-frame solve verdict.
      const float sweepEpsilon = epsilon * std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis());
      if (promoInstance.sweepPending && ++promoInstance.sweepLagFrames >= gateLag) {
        promoInstance.sweepPending = false;
        if (state.gateResidualRel > sweepEpsilon && !promoInstance.demoted) {
          promoInstance.demoted = true;
          Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                   ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                   ") DEMOTED to Path B - full-mesh sweep residual ", state.gateResidualRel,
                                   " gateOver ", state.gateOverCount,
                                   " (sparse-blind partial deformation, risk R20)"));
        }
      }

      if (!promoInstance.demoted && (state.flags & 4u) != 0) {
        promoInstance.demoted = true;
        // DIAG: WHY it went non-rigid - the offending solve's residual + temporal drift.
        // A static building spiking residual >> its steady value on isolated frames is
        // a capture-data glitch (mid-upload read / wrong buffer), not motion; steady
        // moderate residual is the calibration mystery; high tDeform is real deform.
        Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                 ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                 ") DEMOTED to Path B (solve non-rigid: residual ", state.residualRel,
                                 ", tDeform ", state.temporalDeformRel,
                                 ", solveFrame ", state.lastFrame, ")"));
      } else if (promoInstance.demoted && state.rigidStreak >= rigidFrames) {
        promoInstance.demoted = false;
        Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                 ") RE-PROMOTED to Path A (rigid streak rebuilt)"));
      }
    }
  }

  uint64_t ClusterLodManager::resolvePromoCandidateKey(const RasterGeometry& geometryData) const {
    // direct hit: stable-hash geometry, or the exact frame the churned hash recurs
    const XXH64_hash_t hash = geometryData.getHashForRule(RtxOptions::geometryAssetHashRule());
    if (m_promoCandidates.count(hash) != 0) {
      return hash;
    }
    // churned hash: resolve through the stable topology index
    const uint64_t topo = ClusterLodGeometryProvider::makeTopologyKey(geometryData);
    const auto it = m_promoCandidateByTopology.find(topo);
    if (it != m_promoCandidateByTopology.end() && m_promoCandidates.count(it->second) != 0) {
      return it->second;
    }
    return 0;
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

    // ---- PIN the geometry-level temporal probe to one stable instance ----
    // (see PromotionCandidate::probeBlas). Pre-pass: for each plain (non-rest)
    // candidate still proving rigidity, choose the frame pose of the instance it
    // used last frame if that instance is present this frame; otherwise adopt the
    // first available. Without this the emit's "first slot in arrival order" swaps
    // instances between frames and the temporalDeform gate never lets a rigid mesh
    // build a streak (20+ s promotion latency observed on perfectly-rigid meshes).
    std::unordered_map<uint64_t, uint32_t> probePoseByHash;
    std::unordered_map<uint64_t, const BlasEntry*> probeBlasByHash;
    std::unordered_set<uint64_t> probePinLocked;
    for (const size_t tlasType : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
      for (const ClusterSlot& slot : m_slotsB[tlasType]) {
        const uint32_t fpi = slot.geometryId & ~kPathBTag;
        if (fpi >= m_framePoses.size()) {
          continue;
        }
        const BlasEntry* be = slot.instance->getBlas();
        if (be == nullptr) {
          continue;
        }
        const uint64_t hash = resolvePromoCandidateKey(be->input.getGeometryData());
        const auto cit = m_promoCandidates.find(hash);
        if (cit == m_promoCandidates.end()) {
          continue;
        }
        const PromotionCandidate& cand = cit->second;
        // rest-referenced candidates solve per-instance (handled below), not through
        // the shared geometry-level probe; promoted/rejected need no probe pin
        if (cand.routeHash != 0 && cand.restState == PromotionCandidate::RestState::Referenced) {
          continue;
        }
        if (cand.phase != PromotionCandidate::Phase::Probing
            && cand.phase != PromotionCandidate::Phase::GateScheduled
            && cand.phase != PromotionCandidate::Phase::GateRunning) {
          continue;
        }
        if (cand.probeBlas == be && cand.probeBlasFrameCreated == be->frameCreated) {
          // the pinned instance is present this frame - lock it (wins over any provisional)
          probePoseByHash[hash] = fpi;
          probeBlasByHash[hash] = be;
          probePinLocked.insert(hash);
        } else if (probePinLocked.count(hash) == 0 && probeBlasByHash.count(hash) == 0) {
          probePoseByHash[hash] = fpi;   // provisional fallback (pinned instance absent)
          probeBlasByHash[hash] = be;
        }
      }
    }
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
        // stable candidate key (the draw's own hash churns per frame on this game's
        // captured geometry; resolve through topology so the solve runs EVERY frame)
        const uint64_t hash = resolvePromoCandidateKey(blasEntry->input.getGeometryData());
        const auto found = m_promoCandidates.find(hash);
        if (found == m_promoCandidates.end()) {
          continue;
        }
        PromotionCandidate& candidate = found->second;

        // REST-referenced candidate: PER-INSTANCE solve/gate entries, deduped by
        // BlasEntry state slot instead of by hash. Every instance's VS can build
        // a different shape, so each solves ITS OWN capture buffer against the
        // shared rest reference and promotes (or terminally fails) individually -
        // a divergent sibling then never routes Path A instead of promoting
        // wrongly and demote-flapping (0x84974cdd instance 1181, residual 0.223).
        if (candidate.routeHash != 0
            && candidate.restState == PromotionCandidate::RestState::Referenced
            && candidate.probeVa != 0) {
          auto instanceIt = m_promoSlotByBlas.find(blasEntry);
          if (instanceIt == m_promoSlotByBlas.end()) {
            if (m_promoNextStateSlot >= lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
              ONCE(Logger::warn("[ClusterLOD] promotion state slots exhausted - rest instances stay Path B"));
              continue;
            }
            PromoInstance promoInstance;
            promoInstance.stateSlot = m_promoNextStateSlot++;
            promoInstance.geometryHash = hash;
            promoInstance.blasFrameCreated = blasEntry->frameCreated;
            promoInstance.restPhase = PromoInstance::RestPhase::Probing;
            instanceIt = m_promoSlotByBlas.emplace(blasEntry, promoInstance).first;
          }
          PromoInstance& promoInstance = instanceIt->second;
          if (promoInstance.blasFrameCreated != blasEntry->frameCreated) {
            // recycled BlasEntry address = fresh capture content on the same slot:
            // restart this instance's rest probing
            promoInstance.blasFrameCreated = blasEntry->frameCreated;
            promoInstance.geometryHash = hash;
            promoInstance.restPhase = PromoInstance::RestPhase::Probing;
            promoInstance.restGateFrames = 0;
            promoInstance.restStuckFrames = 0;
            promoInstance.restLastSolveFrame = 0;
            promoInstance.demoted = false;
            promoInstance.sweepPending = false;
            promoInstance.residentGeometryId = ~0u;
          }
          if (promoInstance.restPhase == PromoInstance::RestPhase::None) {
            // instance predates the rest swap (pre-rest establish) - enter probing
            promoInstance.restPhase = PromoInstance::RestPhase::Probing;
            promoInstance.geometryHash = hash;
            promoInstance.restStuckFrames = 0;
          }
          if (promoInstance.restPhase == PromoInstance::RestPhase::Rejected
              || (promoInstance.restPhase == PromoInstance::RestPhase::Promoted && !promoInstance.demoted)) {
            // terminal / routes Path A (a demoted rest instance falls through and
            // keeps solving so a rebuilt rigid streak re-promotes it)
            continue;
          }
          if (!emittedInstanceSlots.insert(promoInstance.stateSlot).second) {
            continue;  // instances sharing a BlasEntry share capture content + slot
          }
          lodclusters_remix::PromotionEntry instEntry;
          instEntry.probeVa = candidate.probeVa;
          instEntry.captureVa = m_framePoses[framePoseIndex].positionsAddress;
          instEntry.captureStrideBytes = m_framePoses[framePoseIndex].positionsStrideBytes;
          instEntry.captureVertexCount = m_framePoses[framePoseIndex].positionsCount;
          instEntry.stateSlot = promoInstance.stateSlot;
          instEntry.patchSlot = 0xFFFFFFFFu;
          m_framePromoEntries.push_back(instEntry);
          if (promoInstance.restPhase == PromoInstance::RestPhase::GateScheduled) {
            // same-frame gate pairing as the candidate flow (solves -> barrier ->
            // gates in recordPromotion, so the gate reads this frame's fresh M)
            lodclusters_remix::PromotionEntry gateEntry = instEntry;
            gateEntry.mode = 1;
            gateEntry.vertexCount = candidate.vertexCount;
            m_framePromoEntries.push_back(gateEntry);
            promoInstance.restPhase = PromoInstance::RestPhase::GateRunning;
            promoInstance.restGateFrames = 0;
          }
          continue;
        }

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
            probeEntry.captureVertexCount = m_framePoses[framePoseIndex].positionsCount;
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

        // Use the PINNED instance's frame pose (chosen in the pre-pass) instead of
        // this arrival-order slot, so the temporal gate compares like-for-like.
        const uint32_t probePoseIndex = probePoseByHash.count(hash) ? probePoseByHash[hash] : framePoseIndex;
        const BlasEntry* probeBe = probeBlasByHash.count(hash) ? probeBlasByHash[hash] : blasEntry;

        // Pin change (the previous probe instance left the scene, or first adoption):
        // the state slot's stored previous-frame samples belong to a DIFFERENT
        // placement, so a temporalDeform computed against them is meaningless. Skip
        // this frame's solve when switching between two real instances - the kernel
        // then sees a frame gap (lastFrame not contiguous) next solve and reports
        // tDeform 0, discarding the stale cross-instance sample. First adoption
        // (old probeBlas == nullptr) has no prior samples to discard, so it proceeds.
        const bool hadPrevPin = candidate.probeBlas != nullptr;
        const bool pinChanged = candidate.probeBlas != probeBe
                             || candidate.probeBlasFrameCreated != (probeBe ? probeBe->frameCreated : 0u);
        candidate.probeBlas = probeBe;
        candidate.probeBlasFrameCreated = probeBe ? probeBe->frameCreated : 0u;
        if (pinChanged && hadPrevPin) {
          static std::mutex s_pinMx;
          static std::unordered_map<uint64_t, uint32_t> s_pinLastLog;
          const uint32_t frameNow = m_device->getCurrentFrameId();
          std::lock_guard<std::mutex> lk(s_pinMx);
          uint32_t& last = s_pinLastLog[hash];
          if (last == 0u || frameNow - last > 120u) {
            last = frameNow;
            Logger::info(str::format("[PromoPin] geometry 0x", std::hex, hash, std::dec,
                                     " temporal-probe instance switched (pinned placement left view)"
                                     " - resetting temporal history this frame"));
          }
          continue;  // skip solve -> temporal reset on the next contiguous frame
        }

        // REST-CAPTURE staging: the frame pose IS the capture buffer, so stage the
        // one-time readback here (copy recorded in dispatchBuild where ctx lives).
        if (candidate.restState == PromotionCandidate::RestState::Requested) {
          uint32_t topoVertexCount = 0;
          {
            std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
            const auto topoIt = m_promoTopologyByHash.find(hash);
            if (topoIt != m_promoTopologyByHash.end()) {
              topoVertexCount = topoIt->second.vertexCount;
            }
          }
          const FramePose& pose = m_framePoses[probePoseIndex];
          if (topoVertexCount > 0 && pose.positionsCount >= topoVertexCount
              && pose.positionsStrideBytes >= 3 * sizeof(float)) {
            DxvkBufferCreateInfo stagingInfo;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
            stagingInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
            stagingInfo.size = VkDeviceSize(topoVertexCount) * pose.positionsStrideBytes;
            RestCaptureRequest request;
            request.geometryHash = hash;
            request.source = pose.positionsBuffer;
            request.sourceOffset = pose.positionsBufferOffset;
            request.strideBytes = pose.positionsStrideBytes;
            request.vertexCount = topoVertexCount;
            request.staging = m_device->createBuffer(stagingInfo,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              DxvkMemoryStats::Category::RTXBuffer, "promo rest-capture readback");
            m_restCaptureRequests.push_back(std::move(request));
            // Referenced = rest path initiated; the probe swap on adoption finishes it
            candidate.restState = PromotionCandidate::RestState::Referenced;
          } else {
            // topology missing or pose too small - cannot rest-reference this one
            candidate.restState = PromotionCandidate::RestState::Referenced;
            ONCE(Logger::warn(str::format("[ClusterLOD] rest-capture: geometry 0x", std::hex, hash, std::dec,
                                          " has no retained topology / undersized pose - stays Path B")));
          }
        }

        lodclusters_remix::PromotionEntry promoEntry;
        promoEntry.probeVa = candidate.probeVa;
        promoEntry.captureVa = m_framePoses[probePoseIndex].positionsAddress;
        promoEntry.captureStrideBytes = m_framePoses[probePoseIndex].positionsStrideBytes;
        promoEntry.captureVertexCount = m_framePoses[probePoseIndex].positionsCount;
        promoEntry.stateSlot = candidate.stateSlot;
        promoEntry.patchSlot = 0xFFFFFFFFu;
        // Always emit the per-frame mode-0 solve so matrices.m[slot] holds an M
        // solved for THIS frame's capture. The GateScheduled frame used to REPLACE
        // the solve with the gate, so the full-mesh gate scored a 1-frame-STALE M
        // against a capture whose (camera-relative) space had moved between frames:
        // every vertex then failed the tiny residual epsilon and no genuinely rigid
        // mesh could ever promote (diagnosed as gateStale 1 + gateOver N/N while the
        // data was a clean isometry - scanScore 0, sEff 1, refl 0). recordPromotion
        // runs all mode-0 solves, then a barrier, then the per-entry gates, so a
        // companion gate emitted this same frame reads the fresh M.
        m_framePromoEntries.push_back(promoEntry);
        if (candidate.phase == PromotionCandidate::Phase::GateScheduled) {
          lodclusters_remix::PromotionEntry gateEntry = promoEntry;
          gateEntry.mode = 1;
          gateEntry.vertexCount = candidate.vertexCount;
          m_framePromoEntries.push_back(gateEntry);
          candidate.phase = PromotionCandidate::Phase::GateRunning;
          candidate.gateFrames = 0;
        }
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
        const uint64_t hash = resolvePromoCandidateKey(blasEntry->input.getGeometryData());
        const auto found = m_promoCandidates.find(hash);
        if (found == m_promoCandidates.end()) {
          continue;
        }

        const RaytraceBuffer& positions = blasEntry->modifiedGeometryData.positionBuffer;

        lodclusters_remix::PromotionEntry promoEntry;
        promoEntry.probeVa = found->second.probeVa;
        promoEntry.captureVa = positions.getDeviceAddress() + positions.offsetFromSlice();
        promoEntry.captureStrideBytes = positions.stride();
        promoEntry.captureVertexCount = positions.stride() > 0 && positions.length() > positions.offsetFromSlice()
          ? uint32_t((positions.length() - positions.offsetFromSlice()) / positions.stride()) : 0;
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

    // ---- DIAG (PromoLimbo): a candidate that uploaded a probe and entered the
    // state machine but gets NO solve entry this frame cannot advance - and a
    // candidate at GateScheduled has no updatePromotionStates handler, so if its
    // gate is never emitted (its geometry is not in the Path B slot list this
    // frame) it stalls SILENTLY forever (no promote/reject/stuck log). This is the
    // "static pillar that never reaches Path A" class. Report the stuck population
    // and the discriminator: inPathB=0 means it is rendering classic/culled (never
    // solved); inPathB=1 means it is on Path B but the emit skipped it. Throttled
    // per geometry (600 frames). Off the hot path - only pre-promotion candidates.
    {
      std::unordered_set<uint64_t> pathBHashesThisFrame;
      for (const size_t t : { size_t(Tlas::Opaque), size_t(Tlas::Unordered) }) {
        for (const ClusterSlot& s : m_slotsB[t]) {
          const BlasEntry* be = s.instance->getBlas();
          if (be != nullptr) {
            // resolve to the stable candidate key so the check matches m_promoCandidates
            // (the raw draw hash churns per frame and would never match)
            const uint64_t ck = resolvePromoCandidateKey(be->input.getGeometryData());
            pathBHashesThisFrame.insert(ck != 0 ? ck
              : be->input.getGeometryData().getHashForRule(RtxOptions::geometryAssetHashRule()));
          }
        }
      }
      static std::mutex s_limboMx;
      static std::unordered_map<uint64_t, uint32_t> s_limboLog;
      const uint32_t frameNow = m_device->getCurrentFrameId();
      std::lock_guard<std::mutex> lk(s_limboMx);
      for (const auto& e : m_promoCandidates) {
        const PromotionCandidate& c = e.second;
        if (c.phase == PromotionCandidate::Phase::Promoted
            || c.phase == PromotionCandidate::Phase::Rejected
            || c.routeHash != 0) {
          continue;  // resolved, or rest-referenced (per-instance path)
        }
        if (emitted.count(e.first) != 0) {
          continue;  // got a solve entry this frame (advancing normally)
        }
        uint32_t& last = s_limboLog[e.first];
        if (last == 0u || frameNow - last > 600u) {
          last = frameNow;
          const bool inB = pathBHashesThisFrame.count(e.first) != 0;
          const char* ph = c.phase == PromotionCandidate::Phase::Probing ? "Probing"
                         : c.phase == PromotionCandidate::Phase::GateScheduled ? "GateScheduled"
                         : "GateRunning";
          // rigidStreak + lastSolveFrame from the state readback are the decisive
          // discriminator: solveFrame==0 (or never advancing) => this candidate has
          // NEVER been drawn Path B (off-screen; benign, resolves when drawn). A
          // recent solveFrame with rigidStreak stuck at 0/1 while frameNow climbs =>
          // it IS being drawn but on NON-CONTIGUOUS frames, so the kernel's
          // contiguity check (lastFrame+1==frameId) keeps restarting the streak at 1
          // and it can never reach the gate (the real "on-screen but never promotes"
          // bug - intermittent/multi-pass draw).
          uint32_t rigidStreak = 0, lastSolveFrame = 0;
          if (m_promoStatesValid && c.stateSlot < m_promoStates.size()) {
            rigidStreak = m_promoStates[c.stateSlot].rigidStreak;
            lastSolveFrame = m_promoStates[c.stateSlot].lastFrame;
          }
          // residentPathA: this candidate's geometry is ALSO resident in the regular
          // (non-promotion) Path A table. If a captured candidate is drawn without
          // its capture flag on later frames it falls through to the ladder and
          // renders regular Path A at its INPUT-space position (wrong transform),
          // orphaning the promotion candidate here forever. residentPathA=1 +
          // never-advancing solveFrame is that bug's signature.
          const bool residentPathA = m_geometryIdByHash.count(e.first) != 0;
          Logger::info(str::format("[PromoLimbo] geometry 0x", std::hex, e.first, std::dec,
                                   " uploaded but NOT solved this frame (phase ", ph,
                                   ", inPathB ", inB, ", residentPathA ", residentPathA,
                                   ", rigidStreak ", rigidStreak,
                                   ", lastSolveFrame ", lastSolveFrame, ", frameNow ", frameNow,
                                   ", restState ", uint32_t(c.restState), ")"));
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
    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // [GenTrace] 1s-throttled heartbeat helper: answers "why isn't residency
    // growing right now?" every frame the pipeline is not idle. A transition
    // (pending count changed) always logs immediately; otherwise throttled.
    const bool genTraceThrottleOk =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_lastGenTraceLog).count() >= 1000;

    if (m_pendingGeometryHashes.empty() && !(needsCapacityGrowth && !m_residentGeometryHashes.empty())) {
      // Nothing queued to append. Distinguish "workers still grinding" (discovery
      // outpaced processing) from "truly idle" (game drew nothing new). This is
      // the state during the observed 19s residency stall - it tells us whether
      // the stall is the mesh optimiser/processing backlog or simply no intake.
      const ClusterLodGeometryProvider::Stats st = m_provider->getStats();
      const bool workersBusy = st.pending > 0;
      const bool transition = m_lastGenTracePending != 0;  // was pending last frame, now drained/idle
      if ((workersBusy || transition) && (genTraceThrottleOk || transition)) {
        Logger::info(str::format("[GenTrace] IDLE-append: pending 0, ",
                                 workersBusy ? "PROVIDER STILL PROCESSING" : "provider idle (no new intake)",
                                 " | provider: submitted ", st.submitted, " processed ", st.processed,
                                 " inFlightQueue ", st.pending, " deforming ", st.deforming,
                                 " failed ", st.failed,
                                 " | residentGens ", m_residentGeometryHashes.size(),
                                 " frame ", currentFrame));
        m_lastGenTraceLog = std::chrono::steady_clock::now();
      }
      m_lastGenTracePending = 0;
      return 0.0;
    }

    // batch generation updates. Cache-hit batches take the fast lane (P4c,
    // plan 7.7): a .nvsngeo load costs milliseconds, so the full cooldown
    // would delay the classic->cluster flip with nothing to amortize.
    uint32_t cooldown = uint32_t(std::max(1, ClusterLodOptions::Render::generationCooldownFrames()));
    if (m_pendingHasCacheHit) {
      cooldown = std::min(cooldown, uint32_t(std::max(1, ClusterLodOptions::Render::cacheHitCooldownFrames())));
    }
    if (m_generationCount > 0 && currentFrame - m_lastGenerationFrame < cooldown) {
      // [GenTrace] append is READY but held by the cooldown gate. Reports how
      // many frames remain and how long this batch has now waited - if this is
      // what owns the stall, the fix is a cooldown knob.
      m_genTraceDeferrals++;
      const uint32_t framesWaited = currentFrame - m_genTraceEnqueuedFrame;
      const uint32_t framesLeft = cooldown - (currentFrame - m_lastGenerationFrame);
      const bool transition = m_lastGenTracePending != m_pendingGeometryHashes.size();
      if (genTraceThrottleOk || transition) {
        Logger::info(str::format("[GenTrace] COOLDOWN-hold: pending ", m_pendingGeometryHashes.size(),
                                 (m_pendingHasCacheHit ? " (has cache-hit, fast lane)" : ""),
                                 " | cooldown ", cooldown, " frames, ", framesLeft, " left",
                                 " | batch waited ", framesWaited, " frames over ", m_genTraceDeferrals,
                                 " deferrals | lastGen frame ", m_lastGenerationFrame,
                                 " now ", currentFrame));
        m_lastGenTraceLog = std::chrono::steady_clock::now();
      }
      m_lastGenTracePending = m_pendingGeometryHashes.size();
      return 0.0;
    }

    // chrono: generation events are the frame hitches of the cluster pipeline
    // (appends should stay O(new); rebuilds device-idle) - every event logs
    // its wall time + how long acquiring the submission lock took
    const std::chrono::steady_clock::time_point generationStart = std::chrono::steady_clock::now();

    const lodclusters_remix::ProcessorConfig processorConfig = buildProcessorConfig();
    const std::string configDigest = lodclusters_remix::getConfigCacheDigestUtf8(processorConfig);

    // [GenTrace] the cooldown has elapsed and we WILL touch the generation this
    // frame. Report which route and why: an incremental append is O(new) and
    // cheap; a full rebuild is a device-wait-idle swap (the ~1s hitch seen for
    // gen 1). If a rebuild happens mid-session it is one of the three reasons
    // below - this line names the culprit instead of guessing.
    {
      const bool hasGen = m_renderSystem->hasGeneration();
      const bool digestMatch = (configDigest == m_generationConfigDigest);
      const bool willAppend = hasGen && !needsCapacityGrowth && digestMatch && !m_pendingGeometryHashes.empty();
      const uint32_t framesWaited = currentFrame - m_genTraceEnqueuedFrame;
      Logger::info(str::format("[GenTrace] GENERATION-EVENT frame ", currentFrame,
                               ": route=", willAppend ? "APPEND(O(new))" : "FULL-REBUILD(device-idle swap)",
                               " | pending ", m_pendingGeometryHashes.size(),
                               " batch waited ", framesWaited, " frames / ", m_genTraceDeferrals, " deferrals",
                               " | rebuildReason=",
                               willAppend ? "n/a"
                                 : (!hasGen ? "no-generation(bootstrap)"
                                    : needsCapacityGrowth ? "instance-capacity-growth"
                                    : !digestMatch ? "SceneConfig-digest-changed"
                                    : "pending-empty"),
                               needsCapacityGrowth ? str::format(" [reqCap ", requestedCapacity,
                                                                 " > cur ", m_renderSystem->getMaxRenderInstances(),
                                                                 ", overflow ", m_frameOverflowCount, "]") : std::string()));
    }

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
                                 " in ", elapsedMs(generationStart), " ms (lock wait ", appendLockWaitMs, " ms)",
                                 " | [GenTrace] batch latency ", currentFrame - m_genTraceEnqueuedFrame,
                                 " frames (", m_genTraceDeferrals, " deferrals)"));

        m_pendingGeometryHashes.clear();
        m_pendingHasCacheHit = false;
        m_lastGenerationFrame = currentFrame;
        m_lastGenTracePending = 0;
        m_genTraceDeferrals = 0;

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
    m_lastGenTracePending = 0;
    m_genTraceDeferrals = 0;

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

    // promoDiag: emit EVERY SECOND, ALWAYS (wall-clock, not frame-throttled and
    // not gated on dispatch/candidates) so the data lands in the log before any
    // crash. Aggregates are the latest from updatePromotionStates.
    //   probeZeroGuard    = candidates that hit the probeVa==0 guard (the 4b half
    //                       expected load-bearing; nonzero => it IS the real fix)
    //   stateSlotOob      = the other guard half (expected 0 by CPU construction)
    //   maxAffineNonRigid = worst ||A^T A - I||_F; persistently high on a mesh
    //                       that should be rigid => the Umeyama upgrade is needed
    {
      const auto nowTp = std::chrono::steady_clock::now();
      if (nowTp - m_lastPromoDiagLog >= std::chrono::seconds(1)) {
        m_lastPromoDiagLog = nowTp;
        Logger::info(str::format("[ClusterLOD][promoDiag] probeZeroGuard ", m_diagProbeZeroSlots,
                                 ", degenSlots ", m_diagDegenSlots,
                                 ", maxAffineNonRigid ", m_diagMaxAffineNonRigid,
                                 " (worst geom 0x", std::hex, m_diagWorstGeom, std::dec,
                                 " refVar ", m_diagWorstRefVar,
                                 " sampleN ", m_diagWorstSampleN, ")",
                                 ", reasons[coincid ", m_diagReasonHist[1],
                                 " rankDef ", m_diagReasonHist[2],
                                 " nonOrtho ", m_diagReasonHist[3],
                                 " refVar0 ", m_diagReasonHist[4],
                                 " scaleOvf ", m_diagReasonHist[5],
                                 " nonFin ", m_diagReasonHist[6], "]",
                                 ", statesValid ", (m_promoStatesValid ? 1 : 0)));

        // PATH-A TIMING digest: worst first-sight -> Path A latency among meshes that
        // made it, and the meshes STILL WAITING (never reached Path A) with how long
        // they have been waiting. This is the churn-proof answer to "how long does it
        // take, and which ones never promote" - keyed by stable topology, material
        // hash named so the worst offender is pickable.
        if (!m_promoPathATiming.empty()) {
          uint32_t reached = 0, waiting = 0;
          float worstReached = 0.0f, worstWaiting = 0.0f;
          uint64_t worstReachedMat = 0, worstWaitingMat = 0, worstWaitingGeom = 0;
          const auto nowT = std::chrono::steady_clock::now();
          for (const auto& e : m_promoPathATiming) {
            if (e.second.pathAFrame != 0) {
              reached++;
              if (e.second.secondsToPathA > worstReached) {
                worstReached = e.second.secondsToPathA;
                worstReachedMat = e.second.materialHash;
              }
            } else {
              waiting++;
              const float w = float(std::chrono::duration_cast<std::chrono::milliseconds>(nowT - e.second.firstSeen).count() / 1000.0);
              if (w > worstWaiting) {
                worstWaiting = w;
                worstWaitingMat = e.second.materialHash;
                worstWaitingGeom = e.second.lastGeomHash;
              }
            }
          }
          Logger::info(str::format("[PathATiming] meshes ", m_promoPathATiming.size(),
                                   ", reached Path A ", reached, " (worst ", worstReached,
                                   "s, mat 0x", std::hex, worstReachedMat, std::dec,
                                   "), STILL WAITING ", waiting, " (longest ", worstWaiting,
                                   "s and counting, mat 0x", std::hex, worstWaitingMat,
                                   " geom 0x", worstWaitingGeom, std::dec, ")"));
        }
      }
    }

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

    size_t drainedThisFrame = 0;
    size_t drainedCacheHits = 0;
    const size_t pendingBeforeDrain = m_pendingGeometryHashes.size();
    for (const ClusterLodGeometryProvider::ReadyGeometry& ready : m_provider->drainReadyGeometries()) {
      m_pendingGeometryHashes.push_back(ready.hash);
      m_pendingHasCacheHit |= ready.fromCache;
      drainedThisFrame++;
      if (ready.fromCache) {
        drainedCacheHits++;
      }
    }

    // [GenTrace] mark the frame the pending queue first became non-empty so the
    // "how long has this batch been waiting" heartbeat below is meaningful.
    if (pendingBeforeDrain == 0 && drainedThisFrame > 0) {
      m_genTraceEnqueuedFrame = currentFrame;
      m_genTraceDeferrals = 0;
    }
    if (drainedThisFrame > 0) {
      const ClusterLodGeometryProvider::Stats st = m_provider->getStats();
      Logger::info(str::format("[GenTrace] drained ", drainedThisFrame, " ready geometr(y/ies) (",
                               drainedCacheHits, " cache-hit) -> pending now ", m_pendingGeometryHashes.size(),
                               " | provider: submitted ", st.submitted, " processed ", st.processed,
                               " inFlightQueue ", st.pending, " deforming ", st.deforming,
                               " | frame ", currentFrame));
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
    const bool skinned = blasEntry->input.getSkinningState().numBones > 0 && geometryData.numBonesPerVertex > 0;
    const bool captured = blasEntry->input.preCaptureVertexData != nullptr;
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const bool updatedInPlace = blasEntry->frameLastUpdated == currentFrame
                             && blasEntry->frameLastUpdated != blasEntry->frameCreated;

    // DIAG (RenderRoute): the ACTUAL per-frame routing outcome for a traced material.
    // The "PROMOTED" verdict only means the candidate passed the rigidity gate; this
    // shows whether the instance really renders Path A this frame or falls to Path B
    // / classic (the layer the user actually sees). Filtered by traceMaterialHash,
    // throttled per geometry.
    const uint64_t routeMat = blasEntry->input.getMaterialData().getHash();
    const bool traceRoute = clusterLodPromoTraceMatchesMaterial(routeMat);
    // code: 0 pinned, 1 established, 2 Path B, 3 classic. Log on OUTCOME CHANGE
    // (transition) plus a slow heartbeat, so the steady state and the exact frame a
    // mesh flips Path B->Path A are both visible - a coarse throttle only sampled
    // pre-promotion frames and hid the flip.
    auto logRoute = [&](const char* outcome, uint32_t code) {
      if (!traceRoute) {
        return;
      }
      const uint64_t gk = geometryData.getHashForRule(RtxOptions::geometryAssetHashRule());
      const uint64_t candKey = resolvePromoCandidateKey(geometryData);
      const uint64_t key = candKey != 0 ? candKey : gk;
      static std::mutex s_mx;
      static std::unordered_map<uint64_t, uint32_t> s_lastCode;   // code+1 (0 = unseen)
      static std::unordered_map<uint64_t, uint32_t> s_lastFrame;
      std::lock_guard<std::mutex> lk(s_mx);
      uint32_t& lastCode = s_lastCode[key];
      uint32_t& lastFrame = s_lastFrame[key];
      const bool changed = lastCode != code + 1u;
      if (changed || currentFrame - lastFrame > 120u) {
        Logger::info(str::format("[RenderRoute] mat 0x", std::hex, routeMat, " geomDraw 0x", gk,
                                 " cand 0x", candKey, std::dec, " captured ", captured,
                                 " updatedInPlace ", updatedInPlace, " frame ", currentFrame,
                                 (changed && lastCode != 0u ? " CHANGED -> " : " -> "), outcome));
        lastCode = code + 1u;
        lastFrame = currentFrame;
      }
    };

    // PATH-A TIMING (churn-proof): track first sight -> first actual Path A render by
    // STABLE topology key (the geometry hash churns per frame and can't be tracked).
    // markPathA() is called at the two Path A return points below; the report is in
    // onFrameBegin. Independent of the trace-material filter - covers every mesh.
    const uint64_t pathATopoKey = (captured && !skinned && ClusterLodOptions::Promotion::enable())
      ? ClusterLodGeometryProvider::makeTopologyKey(geometryData) : 0;
    if (pathATopoKey != 0) {
      auto& t = m_promoPathATiming[pathATopoKey];
      if (t.firstFrame == 0) {
        t.firstFrame = currentFrame != 0 ? currentFrame : 1;
        t.firstSeen = std::chrono::steady_clock::now();
      }
      t.materialHash = blasEntry->input.getMaterialData().getHash();
      t.lastGeomHash = geometryData.getHashForRule(RtxOptions::geometryAssetHashRule());
    }
    auto markPathA = [&]() {
      if (pathATopoKey == 0) {
        return;
      }
      auto& t = m_promoPathATiming[pathATopoKey];
      if (t.pathAFrame == 0) {
        t.pathAFrame = currentFrame;
        t.secondsToPathA = float(elapsedMs(t.firstSeen) / 1000.0);
        Logger::info(str::format("[PathATiming] mesh reached Path A after ", t.secondsToPathA,
                                 "s (mat 0x", std::hex, t.materialHash, " geom 0x", t.lastGeomHash,
                                 " topo 0x", pathATopoKey, std::dec, ")"));
      }
    };

    if (skinned || captured || updatedInPlace) {
      // ---- pinned Path A fast-path ----
      // Once an instance has PROMOTED, it is identified by its stable BlasEntry*,
      // NOT the asset hash. The draw-call cache keeps the same BlasEntry across
      // camera moves, but this game's captured-draw asset hash is unstable
      // frame-to-frame, so the m_geometryIdByHash lookup in the establish path
      // below MISSES on every camera-move frame and dropped the mesh back to Path B
      // (all-cyan in the Path Class view). Route straight off the cached
      // residentGeometryId, deliberately IGNORING updatedInPlace: a changed asset
      // hash on an already-rigid promoted instance is the transform moving, not
      // deformation. Genuine deformation is still caught by the promotion solve
      // (which sets slot.demoted), so the pin releases on real deform.
      if (!skinned && ClusterLodOptions::Promotion::enable()
          && m_renderSystem != nullptr && m_renderSystem->hasGeneration()) {
        auto pinIt = m_promoSlotByBlas.find(blasEntry);
        if (pinIt != m_promoSlotByBlas.end()
            && !pinIt->second.demoted
            && pinIt->second.residentGeometryId != ~0u
            && pinIt->second.blasFrameCreated == blasEntry->frameCreated) {
          const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
          if (usedSlots < m_renderSystem->getMaxRenderInstances()
              && (!ClusterLodOptions::Render::routeTrivialToClassic()
                  || m_trivialGeometryIds.count(pinIt->second.residentGeometryId) == 0)) {
            outGeometryId = kPromotedTag | (pinIt->second.stateSlot << kPromotedSlotShift) | pinIt->second.residentGeometryId;
            markPathA();
            logRoute("Path A (pinned)", 0u);
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
        // stable candidate key: the draw's own asset hash churns per frame on this
        // game's captured geometry, so a direct-hash lookup here matched a Promoted
        // candidate only on the rare frames the churn recurred - which is why a
        // promoted pillar still took ~2 minutes to actually route Path A. Resolve
        // through topology so EVERY frame's draw finds its promoted candidate.
        const uint64_t geometryHash = resolvePromoCandidateKey(blasEntry->input.getGeometryData());
        const auto candidate = m_promoCandidates.find(geometryHash);
        if (candidate != m_promoCandidates.end()
            && candidate->second.phase == PromotionCandidate::Phase::Promoted) {
          // rest-referenced candidates render the clusters built from their CAPTURED
          // rest pose (space-tagged hash); everything else uses the object-space id
          // (the stable candidate key, which m_geometryIdByHash is keyed by).
          const uint64_t residencyHash = candidate->second.routeHash != 0
                                       ? candidate->second.routeHash : geometryHash;
          const auto found = m_geometryIdByHash.find(residencyHash);
          // DIAG (PathARoute): a geometry can PROMOTE (verdict) yet its instances still
          // render Path B when a per-geometry condition here fails. The prime suspect is
          // MISSING residency - m_geometryIdByHash miss = the Path A cluster/generation
          // data for this geometry is not resident (not clusterized yet / evicted), so a
          // "promoted but not on Path A" building has no route. Log the reason once/geom.
          {
            static std::mutex s_m;
            static std::unordered_set<uint64_t> s_seen;
            std::lock_guard<std::mutex> lk(s_m);
            if (s_seen.insert(geometryHash).second) {
              const bool resident = found != m_geometryIdByHash.end();
              const bool trivialBlocked = resident && ClusterLodOptions::Render::routeTrivialToClassic()
                                        && m_trivialGeometryIds.count(found->second) != 0;
              const char* why = !resident ? "NOT RESIDENT (Path A cluster data missing/evicted)"
                              : (found->second > kPromotedGeometryMask) ? "geomId exceeds promoted mask"
                              : trivialBlocked ? "routed trivial->classic"
                              : "routable (should reach Path A)";
              Logger::info(str::format("[PathARoute] geometry 0x", std::hex, geometryHash, std::dec,
                                       " PROMOTED but ", why, " (resident ", resident,
                                       ", geomId ", (resident ? int64_t(found->second) : int64_t(-1)),
                                       ", promotedMask ", kPromotedGeometryMask, ")"));
            }
          }
          if (found != m_geometryIdByHash.end()
              && found->second <= kPromotedGeometryMask
              && (!ClusterLodOptions::Render::routeTrivialToClassic() || m_trivialGeometryIds.count(found->second) == 0)) {
            const uint32_t usedSlots = uint32_t(m_slots[Tlas::Opaque].size() + m_slots[Tlas::Unordered].size());
            if (usedSlots >= m_renderSystem->getMaxRenderInstances()) {
              // DIAG (PathARoute): render-instance capacity is the drop (was silent)
              ONCE(Logger::warn(str::format("[PathARoute] render-instance capacity ",
                                            m_renderSystem->getMaxRenderInstances(),
                                            " exhausted - promoted instances drop to Path B (first geometry 0x",
                                            std::hex, geometryHash, std::dec, ")")));
            }
            if (usedSlots < m_renderSystem->getMaxRenderInstances()) {
              // per-INSTANCE promotion state slot (plan R21: every captured
              // instance's buffer carries its own transform)
              auto slotIt = m_promoSlotByBlas.find(blasEntry);
              if (slotIt == m_promoSlotByBlas.end()
                  && m_promoNextStateSlot < lodclusters_remix::ClusterRenderSystem::kPromotionSlotCapacity) {
                PromoInstance promoInstance;
                promoInstance.stateSlot = m_promoNextStateSlot++;
                slotIt = m_promoSlotByBlas.emplace(blasEntry, promoInstance).first;
              } else if (slotIt != m_promoSlotByBlas.end()
                         && slotIt->second.blasFrameCreated != blasEntry->frameCreated) {
                // recycled BlasEntry address: this map is never GC'd, so a freed
                // entry's address coming back carries the OLD tenant's state
                // (demoted flag, rest phase, pin) into a brand-new instance -
                // reset everything but keep the slot (its stale GPU temporal
                // sample is one isolated spike, which persistence now absorbs)
                slotIt->second.blasFrameCreated = blasEntry->frameCreated;
                slotIt->second.demoted = false;
                slotIt->second.sweepPending = false;
                slotIt->second.restPhase = PromoInstance::RestPhase::None;
                slotIt->second.restGateFrames = 0;
                slotIt->second.restStuckFrames = 0;
                slotIt->second.restLastSolveFrame = 0;
                slotIt->second.residentGeometryId = ~0u;
                slotIt->second.geometryHash = 0;
              }
              // rest-referenced candidates promote PER-INSTANCE: an instance
              // routes Path A only after ITS OWN solve+gate against the shared
              // rest reference passed - a divergent-shape sibling stays Path B
              // instead of promoting wrongly and demote-flapping
              const bool restGated = candidate->second.routeHash != 0
                && (slotIt == m_promoSlotByBlas.end()
                    || slotIt->second.restPhase != PromoInstance::RestPhase::Promoted);
              // per-instance demotion: a demoted instance falls through to
              // Path B below while its siblings stay promoted; its slot keeps
              // solving (buildPromotionEntries) so it can re-promote
              if (slotIt != m_promoSlotByBlas.end() && !slotIt->second.demoted && !restGated) {
                // Cache the stable identity so the pinned fast-path above can route
                // this instance every subsequent frame WITHOUT the churning-hash lookup.
                slotIt->second.residentGeometryId = found->second;
                slotIt->second.geometryHash = geometryHash;
                slotIt->second.blasFrameCreated = blasEntry->frameCreated;
                outGeometryId = kPromotedTag | (slotIt->second.stateSlot << kPromotedSlotShift) | found->second;
                markPathA();
                logRoute("Path A (established)", 1u);
                return true;
              }
              // DIAG (PathARoute): routable geometry whose INSTANCE still dropped here.
              // The only two ways to reach this line: the per-instance slot is DEMOTED
              // (solve flagged non-rigid for this instance's own capture) or the slot
              // pool is exhausted. Names the last silent drop between "routable" and
              // the screen. Throttled per geometry per 300 frames.
              {
                static std::mutex s_m2;
                static std::unordered_map<uint64_t, uint32_t> s_lastLogFrame;
                std::lock_guard<std::mutex> lk2(s_m2);
                uint32_t& last = s_lastLogFrame[geometryHash];
                if (currentFrame - last > 300u || last == 0u) {
                  last = currentFrame;
                  const char* why2 = (slotIt != m_promoSlotByBlas.end() && slotIt->second.demoted)
                                   ? "instance DEMOTED (per-instance solve non-rigid)"
                                   : restGated
                                   ? "REST instance verdict pending/failed (per-instance rest reference)"
                                   : "promo state slot pool exhausted";
                  Logger::info(str::format("[PathARoute] geometry 0x", std::hex, geometryHash, std::dec,
                                           " instance dropped to Path B: ", why2,
                                           " (stateSlot ", (slotIt != m_promoSlotByBlas.end() ? int64_t(slotIt->second.stateSlot) : int64_t(-1)),
                                           ")"));
                }
              }
            }
          }
        }
      }

      const bool routedPathB = isClusterTemplateInstance(instance, blasEntry, outGeometryId);
      // DIAG (PromoClassic): a CAPTURED promotion candidate that is NOT promoted
      // and does NOT route Path B renders CLASSIC - so buildPromotionEntries never
      // sees it in m_slotsB and its candidate solve never runs (silent limbo, the
      // "static pillar never reaches Path A" class). Name the exact reason
      // isClusterTemplateInstance declined. Throttled per geometry (600 frames).
      if (!routedPathB && captured && !skinned
          && ClusterLodOptions::Promotion::enable() && !m_promoCandidates.empty()) {
        const uint64_t gh = resolvePromoCandidateKey(geometryData);
        const auto candIt = m_promoCandidates.find(gh);
        if (candIt != m_promoCandidates.end()
            && candIt->second.phase != PromotionCandidate::Phase::Promoted
            && candIt->second.phase != PromotionCandidate::Phase::Rejected) {
          static std::mutex s_classicMx;
          static std::unordered_map<uint64_t, uint32_t> s_classicLog;
          std::lock_guard<std::mutex> lk(s_classicMx);
          uint32_t& last = s_classicLog[gh];
          if (last == 0u || currentFrame - last > 600u) {
            last = currentFrame;
            const bool templSys = m_templateSystemMT != nullptr;
            const bool registered = templSys
              && m_animatedGeometryByKey.count(ClusterLodGeometryProvider::makeTopologyKey(geometryData)) != 0;
            const bool posDefined = blasEntry->modifiedGeometryData.positionBuffer.defined();
            Logger::info(str::format("[PromoClassic] geometry 0x", std::hex, gh, std::dec,
                                     " captured candidate renders CLASSIC (not Path B) - templateSys ", templSys,
                                     ", registeredAnimated ", registered, ", positionsDefined ", posDefined,
                                     ", animatedEnable ", ClusterLodOptions::Animated::enable(),
                                     ", animMapSize ", m_animatedGeometryByKey.size(), ")"));
          }
        }
      }
      logRoute(routedPathB ? "Path B (cluster template)" : "classic (fell through - no cluster route)", routedPathB ? 2u : 3u);
      return routedPathB;
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
    // DIAG: conservative valid-slot count from the capture origin (bounds the
    // correspondence scan's reads; underestimate is safe, it only skips offsets)
    framePose.positionsCount = positions.stride() > 0 && positions.length() > positions.offsetFromSlice()
      ? uint32_t((positions.length() - positions.offsetFromSlice()) / positions.stride()) : 0;
    framePose.positionsBufferOffset = positions.offset() + positions.offsetFromSlice();
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

  void ClusterLodManager::processRestCaptureRequests(Rc<DxvkContext> ctx) {
    if (m_restCaptureRequests.empty()) {
      return;
    }
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    // frames the copy must retire before the host reads the staging mapping
    constexpr uint32_t kReadbackLagFrames = 4;

    for (auto it = m_restCaptureRequests.begin(); it != m_restCaptureRequests.end();) {
      RestCaptureRequest& request = *it;

      if (request.copyFrame == ~0u) {
        // record the one-time copy (dxvk tracks the barriers/lifetimes)
        ctx->copyBuffer(request.staging, 0, request.source,
                        request.sourceOffset,
                        VkDeviceSize(request.vertexCount) * request.strideBytes);
        request.copyFrame = currentFrame;
        ++it;
        continue;
      }
      if (currentFrame - request.copyFrame < kReadbackLagFrames) {
        ++it;
        continue;
      }

      // copy retired: assemble the rest snapshot from the readback + retained topology
      const uint8_t* mapped = (const uint8_t*) request.staging->mapPtr(0);
      RetainedTopology topo;
      bool haveTopo = false;
      {
        std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
        const auto topoIt = m_promoTopologyByHash.find(request.geometryHash);
        if (topoIt != m_promoTopologyByHash.end()) {
          topo = topoIt->second;
          haveTopo = true;
        }
      }
      if (mapped != nullptr && haveTopo && m_provider != nullptr) {
        lodclusters_remix::GeometrySnapshot restSnap;
        // space-tagged rest hash: distinct clusters/.nvsngeo/residency identity
        constexpr uint64_t kRestSpaceTag = 0x9E3779B97F4A7C15ull;
        restSnap.geometryHash = request.geometryHash ^ kRestSpaceTag;
        restSnap.promoKeyHash = request.geometryHash;
        restSnap.isRestCapture = true;
        restSnap.name = topo.name + "_rest";
        restSnap.indices = std::move(topo.indices);
        restSnap.indicesHash = topo.indicesHash;
        restSnap.topologyKey = topo.topologyKey;
        restSnap.vertexCount = request.vertexCount;
        restSnap.positions.resize(size_t(request.vertexCount) * 3);
        for (uint32_t v = 0; v < request.vertexCount; v++) {
          const float* src = (const float*) (mapped + size_t(v) * request.strideBytes);
          restSnap.positions[size_t(v) * 3 + 0] = src[0];
          restSnap.positions[size_t(v) * 3 + 1] = src[1];
          restSnap.positions[size_t(v) * 3 + 2] = src[2];
        }
        restSnap.verticesHash = XXH3_64bits(restSnap.positions.data(),
                                            restSnap.positions.size() * sizeof(float));
        Logger::info(str::format("[ClusterLOD] rest-capture: geometry 0x", std::hex, request.geometryHash,
                                 " read back -> rest snapshot 0x", restSnap.geometryHash, std::dec,
                                 " (", request.vertexCount, " verts) queued for clusterization"));
        m_provider->enqueueRestSnapshot(std::move(restSnap));
      } else {
        Logger::warn(str::format("[ClusterLOD] rest-capture: geometry 0x", std::hex, request.geometryHash, std::dec,
                                 " readback unusable (mapped ", mapped != nullptr,
                                 ", topo ", haveTopo, ") - stays Path B"));
      }
      it = m_restCaptureRequests.erase(it);
    }
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

    // REST-CAPTURE readbacks: record freshly-staged copies; drain finished ones
    // (past the frames-in-flight window) into rest snapshots for the provider.
    processRestCaptureRequests(ctx);

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
    frameParams.promotionTemporalEpsilon = std::max(0.0f, ClusterLodOptions::Promotion::temporalEpsilon());
    frameParams.promotionDemoteHysteresis = std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis());
    frameParams.promotionCorrespondenceScan = ClusterLodOptions::Promotion::correspondenceScan();
    // DIAG raw dump: resolve the traced hash to its promo state slot (if a candidate)
    frameParams.promotionDumpStateSlot = ~0u;
    {
      const std::string& dumpHashStr = ClusterLodOptions::Promotion::dumpGeometryHash();
      if (!dumpHashStr.empty()) {
        uint64_t dumpHash = 0;
        try { dumpHash = std::stoull(dumpHashStr, nullptr, 16); } catch (...) { dumpHash = 0; }
        const auto it = dumpHash != 0 ? m_promoCandidates.find(dumpHash) : m_promoCandidates.end();
        if (it != m_promoCandidates.end()) {
          frameParams.promotionDumpStateSlot = it->second.stateSlot;
        }
      }
    }

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

    // Vulkan requires regionCount > 0. Unlike the Path B copy (guarded by the
    // countB==0 early-return), this Path A copy has no upstream guarantee that
    // any cluster slots exist this frame - a frame with no Path-A opaque/
    // unordered slots and no SSS duplicates yields an empty region list. Submit
    // nothing rather than trip VUID-vkCmdCopyBuffer-regionCount-arraylength.
    // No copy == no write, so the resource-tracking is scoped with it.
    if (!regions.empty()) {
      m_device->vkd()->vkCmdCopyBuffer(cmd, sourceBuffer, instanceBufferSlice.handle, uint32_t(regions.size()), regions.data());
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(instanceBuffer);
    }
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

}  // namespace dxvk
