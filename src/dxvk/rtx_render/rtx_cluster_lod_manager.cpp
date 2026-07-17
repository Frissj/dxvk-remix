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
#include <map>
#include <mutex>
#include <unordered_set>

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

    // [DrawCoverage] stamp this captured draw (CS thread, BEFORE any Remix visibility
    // culling) by its STABLE topology key, so the promotion diagnostics can separate
    // genuine off-screen (game never submits the draw -> no stamp) from a cluster-side
    // drop (drawn here but never solved). One count per distinct frame per topology.
    {
      const RasterGeometry& gd = drawCallState.getGeometryData();
      const bool captured = drawCallState.preCaptureVertexData != nullptr;
      const bool skinned = drawCallState.getSkinningState().numBones > 0 && gd.numBonesPerVertex > 0;
      if (captured && !skinned) {
        const uint64_t topo = ClusterLodGeometryProvider::makeTopologyKey(gd);
        if (topo != 0) {
          const uint32_t fr = m_device->getCurrentFrameId();
          std::lock_guard<std::mutex> lk(m_promoDrawMutex);
          uint32_t& lastF = m_promoDrawnFrameByTopo[topo];
          if (lastF != fr) {
            lastF = fr;
            m_promoDrawnCountByTopo[topo]++;
          }
        }
      }
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
      // Option 1: FULL-referenced-set centered covariance of the reference
      // (xx,yy,zz,xy,xz,yz; /n, CPU doubles). MODE_EIGEN compares eig(capture
      // cov) to eig(A*refCov*A^T), A = last-RIGID affine M (A carries the
      // anisotropic placement bake a direct refCov comparison cannot).
      float refCov[6];
      float refCovValid;  // 1.0 when populated
      float _pad1;
    };
    static_assert(sizeof(ProbeHeader) == 128, "kernel mirrors this layout");

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

    // Option 1: FULL-referenced-set centered covariance of the reference, in
    // doubles (fp32 one-pass second moments cancel catastrophically on meshes
    // away from the origin - same failure the kernel's Mcov centering fixed).
    // (cx,cy,cz) is already the referenced-set mean. MODE_EIGEN compares
    // eig(A*this*A^T) to eig(the per-frame capture cov), A = last-RIGID M.
    double refCov[6] = {};
    for (uint32_t r = 0; r < refCount; r++) {
      const uint32_t v = referenced[r];
      const double dx = positions[v * 3 + 0] - cx;
      const double dy = positions[v * 3 + 1] - cy;
      const double dz = positions[v * 3 + 2] - cz;
      refCov[0] += dx * dx; refCov[1] += dy * dy; refCov[2] += dz * dz;
      refCov[3] += dx * dy; refCov[4] += dx * dz; refCov[5] += dy * dz;
    }
    for (int i = 0; i < 6; i++) {
      refCov[i] /= double(refCount);
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
    for (int i = 0; i < 6; i++) {
      header.refCov[i] = float(refCov[i]);
    }
    header.refCovValid = 1.0f;
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
        // Match the BASE candidate too: an OWN REST CAPTURE snapshot carries a
        // SALTED geometryHash (restHash ^ classQ*mix, see restSnap.geometryHash),
        // so keying on geometryHash alone dumps ref[] only for the SHARED/base probe
        // build, never the own-reference (self-misfit) build we actually need. A rest
        // capture's promoKeyHash is the original base candidate hash - match that so a
        // single base-hash dumpGeometryHash emits ref[] for BOTH probes, tagged apart.
        const bool refDumpMatch = dumpHash != 0
            && (dumpHash == snapshot.geometryHash
                || (snapshot.isRestCapture && dumpHash == snapshot.promoKeyHash));
        if (refDumpMatch) {
          Logger::info(str::format("[PromoDump] ref geometry 0x", std::hex, snapshot.geometryHash, std::dec,
                                   snapshot.isRestCapture ? " (OWN rest q" : " (SHARED",
                                   snapshot.isRestCapture ? str::format(snapshot.promoClassQ, ")") : std::string(")"),
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
      pending.classQ = snapshot.promoClassQ;       // [ShapeClass] class-scoped rest probes land on their class
      pending.classSubId = snapshot.promoClassSubId;  // ... and on the right identity-by-fit sibling
      pending.restored = snapshot.promoRestored;   // [PromoRefs] sidecar adoptions skip the class-wipe
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

  int32_t ClusterLodManager::quantizeEigClass(float lam1Hat, float lam2Hat) {
    // Content id = the trace-normalized eigenvalue pair snapped to an eigenEpsilon
    // grid. The pair is a rigid- and uniform-scale-INVARIANT shape descriptor, so
    // the same piece at any placement yields the same cell EXACTLY - no merge
    // tolerance (the old capSig key was scale-variant + noisy, hence the 1.5/16
    // hack). eigenEpsilon is the ONE shape-precision constant (also the gate/demote
    // bar): "shapes within one cell are the same to promotion quality".
    if (!(lam1Hat >= 0.0f) || !(lam2Hat >= 0.0f)) {
      return INT32_MIN;  // no eigen key yet (degenerate / no sweep landed)
    }
    const float grid = std::max(1e-4f, ClusterLodOptions::Promotion::eigenEpsilon());
    // lam1Hat in [1/3,1], lam2Hat in [0,1/2] -> q1 in [0,~1024], q2 in [0,~512]
    const int32_t q1 = int32_t(std::lround(std::min(1.0f, lam1Hat) / grid));
    const int32_t q2 = int32_t(std::lround(std::min(1.0f, lam2Hat) / grid));
    return q1 * 4096 + q2;  // pack (q2 < 4096 for any grid >= 1e-4 * ... comfortably)
  }

  ClusterLodManager::RestClassState* ClusterLodManager::resolveRestClass(uint64_t candidateHash,
                                                                         int32_t classQ,
                                                                         int32_t subId,
                                                                         bool createIfMissing) {
    // EXACT match on the quantized eigen-key cell (+ subId cursor). The key is a
    // stable invariant, so equality is the right test - no nearest-merge tolerance.
    // subId disambiguates the rare case of two different shapes sharing a cell
    // (the eigen pair is necessary for shape identity, the gate fit is sufficient).
    if (classQ == INT32_MIN) {
      return nullptr;  // no key -> no class
    }
    auto matchIn = [&](std::vector<RestClassState>& classes) -> RestClassState* {
      for (RestClassState& c : classes) {
        if (c.classQ == classQ && c.subId == subId) {
          return &c;
        }
      }
      return nullptr;
    };
    if (!createIfMissing) {
      const auto it = m_restClassesByCandidate.find(candidateHash);
      return it == m_restClassesByCandidate.end() ? nullptr : matchIn(it->second);
    }
    std::vector<RestClassState>& classes = m_restClassesByCandidate[candidateHash];
    if (RestClassState* hit = matchIn(classes)) {
      return hit;
    }
    classes.emplace_back();
    RestClassState& fresh = classes.back();
    fresh.classQ = classQ;
    fresh.subId = subId;
    Logger::info(str::format("[ShapeClass] geom 0x", std::hex, candidateHash, std::dec,
                             " new content class cell ", fresh.classQ,
                             subId != 0 ? str::format(" sibling ", subId, " (identity-by-fit split)") : std::string(),
                             " (", classes.size(), " classes total)"));
    return &fresh;
  }

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
          if (pending.routeHash != 0 && pending.classQ != INT32_MIN) {
            // [ShapeClass] CLASS-scoped rest probe: this reference was captured
            // FROM a specific content class (the shared one failed it). Adopt it
            // onto that class; the candidate's shared reference stays untouched
            // for the classes it does fit.
            std::vector<RestClassState>& classes = m_restClassesByCandidate[pending.geometryHash];
            RestClassState* cls = nullptr;
            for (RestClassState& c : classes) {
              if (c.classQ == pending.classQ && c.subId == pending.classSubId) {
                cls = &c;
                break;
              }
            }
            if (cls == nullptr) {
              // class vanished meanwhile (content left the scene) - keep the
              // reference under a fresh entry so returning content finds it
              classes.emplace_back();
              cls = &classes.back();
              cls->classQ = pending.classQ;
              cls->subId = pending.classSubId;
            }
            if (m_templateSystemMT != nullptr && cls->probeVa != 0) {
              m_templateSystemMT->freePromotionProbe(cls->probeVa);
            }
            cls->probeVa = pending.probeVa;
            cls->routeHash = pending.routeHash;
            cls->vertexCount = pending.vertexCount;
            cls->ref = RestClassState::Ref::Own;
            cls->phase = RestClassState::Phase::Probing;
            cls->gateFrames = 0;
            cls->stuckFrames = 0;
            cls->gateStateSlot = ~0u;
            cls->rejectedFrame = 0;
            cls->captureStaged = false;
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, pending.geometryHash,
                                     std::dec, " class q", pending.classQ, " sub ", pending.classSubId,
                                     " re-referenced to OWN REST CAPTURE 0x", std::hex, pending.routeHash, std::dec));
          }
          else if (pending.routeHash != 0) {
            // candidate-level (first/shared) rest reference
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
            // class verdicts start fresh against the new shared reference; any
            // class-scoped own probes from the old era are freed with it.
            // [PromoRefs] EXCEPT for sidecar restores: the restored candidate-
            // level + sibling references are one internally-consistent set, and
            // the worker pool may adopt them out of order - wiping here would
            // destroy siblings that adopted first (and free their probes).
            if (!pending.restored) {
              auto classesIt = m_restClassesByCandidate.find(pending.geometryHash);
              if (classesIt != m_restClassesByCandidate.end()) {
                for (RestClassState& c : classesIt->second) {
                  if (m_templateSystemMT != nullptr && c.probeVa != 0) {
                    m_templateSystemMT->freePromotionProbe(c.probeVa);
                  }
                }
                m_restClassesByCandidate.erase(classesIt);
              }
            }
            // instance state from the object-space probe era is void - residency
            // moved to the rest hash and the reference content changed
            for (auto& slotEntry : m_promoSlotByBlas) {
              if (slotEntry.second.geometryHash == pending.geometryHash) {
                slotEntry.second.sweepPending = false;
                slotEntry.second.eigenSuspect = false;
                slotEntry.second.residentGeometryId = ~0u;
                slotEntry.second.contentClassQ = INT32_MIN;  // reclassify vs the new reference
                slotEntry.second.classSubId = 0;
                slotEntry.second.classSubId = 0;
                slotEntry.second.lastClassifiedFrame = 0;
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
          // [SolveDump] M + per-validation (ref,cap,dev) for the same traced slot.
          // For each validation sample the kernel emits the residual VECTOR (dev =
          // fitted-cap). Two things settle the self-misfit's nature:
          //  - is dev a CONSISTENT direction across samples? -> a transform/
          //    translation error (solve-math). scattered -> pairing/shape.
          //  - does each cap match ITS OWN ref under M, or a DIFFERENT sample's
          //    cap? We test the latter: for each validation cap, find the nearest
          //    OTHER validation cap and report whether |dev| exceeds it (a
          //    permutation puts cap nearer some other sample's fitted point).
          const uint32_t nF = lodclusters_remix::ClusterRenderSystem::promotionSolveDumpFloatCount();
          std::vector<float> sd(nF);
          // shader writes this buffer only on a STEADY self-misfit (residual
          // 0.05..0.6); guard here against a stale buffer with the same band
          if (m_renderSystem->readPromotionSolveDump(sd.data()) && sd[14] > 0.05f && sd[14] < 0.6f) {
            const uint32_t sampleCount = uint32_t(sd[12]);
            const uint32_t valCount = std::min<uint32_t>(uint32_t(sd[13]), 16u);
            Logger::info(str::format("[SolveDump] geom 0x", ClusterLodOptions::Promotion::dumpGeometryHash(),
                                     " M row0 (", sd[0], ", ", sd[1], ", ", sd[2], ", ", sd[3], ")"));
            Logger::info(str::format("[SolveDump] M row1 (", sd[4], ", ", sd[5], ", ", sd[6], ", ", sd[7],
                                     ") row2 (", sd[8], ", ", sd[9], ", ", sd[10], ", ", sd[11], ")"));
            Logger::info(str::format("[SolveDump] sampleCount ", sampleCount, " valCount ", valCount,
                                     " residualRel ", sd[14], " dirCoh ", sd[15]));
            // mean dev vector (consistent-direction test) + per-sample detail
            float mdx = 0.0f, mdy = 0.0f, mdz = 0.0f, sumMag = 0.0f;
            for (uint32_t i = 0; i < valCount; i++) {
              const float* e = &sd[16 + i * 10];
              const float dmag = std::sqrt(e[7] * e[7] + e[8] * e[8] + e[9] * e[9]);
              mdx += e[7]; mdy += e[8]; mdz += e[9]; sumMag += dmag;
              // nearest OTHER validation cap to THIS cap (permutation probe)
              float bestOther = std::numeric_limits<float>::max();
              int32_t bestJ = -1;
              for (uint32_t j = 0; j < valCount; j++) {
                if (j == i) { continue; }
                const float* o = &sd[16 + j * 10];
                const float ddx = e[4] - o[4], ddy = e[5] - o[5], ddz = e[6] - o[6];
                const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (d2 < bestOther) { bestOther = d2; bestJ = int32_t(j); }
              }
              Logger::info(str::format("[SolveDump] v", i, " idx ", uint32_t(e[0]),
                                       " ref (", e[1], ", ", e[2], ", ", e[3], ")",
                                       " cap (", e[4], ", ", e[5], ", ", e[6], ")",
                                       " |dev| ", dmag, " nearestOtherCap ", std::sqrt(bestOther),
                                       " (v", bestJ, ")"));
            }
            const float mMag = std::sqrt(mdx * mdx + mdy * mdy + mdz * mdz);
            const float coh = sumMag > 1e-9f ? mMag / sumMag : 0.0f;
            Logger::info(str::format("[SolveDump] devVec coherence ", coh,
                                     " (~1 = systematic transform error -> solve-math; ~0 = scattered"
                                     " -> pairing/shape) meanDevVec (", mdx / std::max(1u, valCount), ", ",
                                     mdy / std::max(1u, valCount), ", ", mdz / std::max(1u, valCount), ")"));
          }
        }
      }
    }

    // [CapSigDump] EVERY frame (un-throttled, small budget): the actual verts capSig
    // sampled + this frame's captureVa + capSigVar, for the traced instance. Consecutive
    // lines cover consecutive frames, so the captureVa ping-pong is visible and we can see
    // whether the two buffers hold the same instance at the same scale (=> capSig compute
    // bug) or different content (=> the double-buffered read is inconsistent).
    if (!ClusterLodOptions::Promotion::dumpGeometryHash().empty()) {
      static uint32_t s_capSigDumpBudget = 400u;
      const uint32_t nF = lodclusters_remix::ClusterRenderSystem::promotionSolveDumpFloatCount();
      if (s_capSigDumpBudget > 0u && nF >= 276u) {
        std::vector<float> sd(nF);
        if (m_renderSystem->readPromotionSolveDump(sd.data()) && sd[179] > 0.0f) {
          --s_capSigDumpBudget;
          uint32_t vaLo = 0, vaHi = 0;
          std::memcpy(&vaLo, &sd[176], 4);
          std::memcpy(&vaHi, &sd[177], 4);
          const uint64_t captureVa = (uint64_t(vaHi) << 32) | uint64_t(vaLo);
          const uint32_t sigN = uint32_t(sd[179]);
          Logger::info(str::format("[CapSigDump] geom 0x", ClusterLodOptions::Promotion::dumpGeometryHash(),
                                   " frame ", m_device->getCurrentFrameId(),
                                   " captureVa 0x", std::hex, captureVa, std::dec,
                                   " capSigVar ", sd[178], " sigN ", sigN,
                                   " v0 (", sd[180], ", ", sd[181], ", ", sd[182], ")",
                                   " v1 (", sd[183], ", ", sd[184], ", ", sd[185], ")",
                                   " v2 (", sd[186], ", ", sd[187], ", ", sd[188], ")",
                                   " v3 (", sd[189], ", ", sd[190], ", ", sd[191], ")"));
        }
        // [EigMetric] A-vs-M confirmation (kernel dump floats 200..220): the drift
        // computed with lastRigidM (sd[200]) vs with the current fitted M (sd[201]).
        // driftFit ~0 while drift large => the drift is a stale/rigid-A metric artifact
        // (same shape, wrong transform), NOT a real shape difference. The two 3x3
        // linear matrices follow so we can see HOW A differs from M (rigid vs affine,
        // stale scale). sd[202] is the sweep frame (0/uninitialized => no eigen sweep
        // hit the traced slot this readback).
        if (nF >= 300u) {
          std::vector<float> em(nF);
          if (m_renderSystem->readPromotionSolveDump(em.data()) && em[278] > 0.0f) {
            Logger::info(str::format("[EigMetric] geom 0x", ClusterLodOptions::Promotion::dumpGeometryHash(),
                                     " sweepFrame ", uint32_t(em[278]),
                                     " VERDICT ", em[298], " (usedM ", (em[297] > 0.5f ? 1 : 0),
                                     ", meanDev ", em[299], ") | driftLastRigidM ", em[276],
                                     " driftFitM ", em[277],
                                     " | M [", em[279], " ", em[280], " ", em[281], " / ",
                                     em[282], " ", em[283], " ", em[284], " / ",
                                     em[285], " ", em[286], " ", em[287], "]",
                                     " | A(lastRigid) [", em[288], " ", em[289], " ", em[290], " / ",
                                     em[291], " ", em[292], " ", em[293], " / ",
                                     em[294], " ", em[295], " ", em[296], "]"));
          }
        }
      }
    }

    const uint32_t rigidFrames = uint32_t(std::max(1, ClusterLodOptions::Promotion::rigidFrames()));
    const float epsilon = std::max(1e-5f, ClusterLodOptions::Promotion::residualEpsilon());
    const uint32_t gateLag = uint32_t(std::max(2, ClusterLodOptions::Promotion::gateLagFrames()));
    // Option 1: all promotion GATES (candidate + class) judge with the
    // permutation-invariant eigen verdict instead of the index-paired full-mesh
    // residual - the residual REJECTED genuinely rigid meshes whenever the
    // engine re-batched the capture's vertex order between probe and gate,
    // which kept classes un-Promoted and restGated whole buildings to Path B.
    const float eigenEps = std::max(0.0f, ClusterLodOptions::Promotion::eigenEpsilon());

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
          // scanVerdict [24:26]: 0 = offset 0 best / no signal, 1 = a nonzero
          // offset IMPROVED the fit (partial), 2 = a nonzero offset COLLAPSED the
          // mismatch to near-zero (a clean rigid pairing exists there). Verdict 2
          // == this mesh IS rigid and would promote if the solve read capture at
          // scanOff instead of 0 - the decisive "fixable index skew" signal.
          const uint32_t scanVerdict = (state.diagGuard >> 24) & 0x3u;
          const char* scanVerdictStr = scanVerdict == 2u ? "COLLAPSE(fixable-skew)"
                                      : scanVerdict == 1u ? "improved(partial)"
                                      : "none";
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
                                   ", scanOff ", scanOff, ", scanScore ", scanScore,
                                   ", scanVerdict ", scanVerdictStr, ", refl ", reflected,
                                   ", verts ", candidate.vertexCount, ")"));
        }

        // [ScanProbe] PERIODIC (not one-shot) correspondence-scan readout for a
        // single targeted candidate (rtx.clusterLod.promotion.dumpGeometryHash =
        // the stuck geometry's candidate key). Lets us watch whether scanOff is
        // stable at one nonzero value (a fixed shared-buffer baseVertex skew ->
        // applying the offset fixes it) or varies frame to frame (a moving
        // correspondence -> the probe basis itself is wrong). Throttled 120 frames.
        if (!ClusterLodOptions::Promotion::dumpGeometryHash().empty()) {
          uint64_t dumpKey = 0;
          const std::string& dh = ClusterLodOptions::Promotion::dumpGeometryHash();
          try { dumpKey = std::stoull(dh, nullptr, 16); } catch (...) { dumpKey = 0; }
          if (dumpKey != 0 && entry.first == dumpKey
              && (m_scanProbeLastFrame == 0u || m_device->getCurrentFrameId() - m_scanProbeLastFrame >= 120u)) {
            m_scanProbeLastFrame = m_device->getCurrentFrameId();
            const uint32_t sIdx = (state.diagGuard >> 2) & 0x3Fu;
            const uint32_t sVerdict = (state.diagGuard >> 24) & 0x3u;
            const uint32_t sScoreQ = (state.diagGuard >> 27) & 0x1Fu;
            Logger::info(str::format("[ScanProbe] geom 0x", std::hex, entry.first, std::dec,
                                     " phase ", uint32_t(candidate.phase),
                                     " residual ", state.residualRel, " (eps ", epsilon, ")",
                                     " | scanOff ", kPromoScanOffsets[sIdx],
                                     " verdict ", sVerdict, " (0=none 1=improved 2=COLLAPSE)",
                                     " off0mismatchQ ", sScoreQ,
                                     " rigidStreak ", state.rigidStreak,
                                     " frame ", m_device->getCurrentFrameId()));
          }
        }
        break;

      case PromotionCandidate::Phase::GateRunning:
        if (++candidate.gateFrames >= gateLag) {
          // Option 1: eigen gate verdict (mode-2 entry emitted at GateScheduled).
          // eigDrift < 0 or eigFrame == 0 = no verdict landed -> reschedule below.
          // The candidate's own stateSlot receives ONLY gate entries (instance
          // sweeps run on per-instance slots), so no freshness mark is needed.
          if (state.eigFrame != 0u && state.eigDrift >= 0.0f && state.eigDrift <= eigenEps) {
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
            // [DrawCoverage] frames the GAME actually submitted this geometry's draw
            // (topology-keyed, pre-culling). drawn ~ solved => genuinely off-screen;
            // drawn >> solved => drawn on-screen but the cluster path dropped it.
            uint32_t drawnFrames = 0;
            {
              std::lock_guard<std::mutex> lk(m_promoDrawMutex);
              auto dIt = m_promoDrawnCountByTopo.find(candidate.topologyKey);
              if (dIt != m_promoDrawnCountByTopo.end()) { drawnFrames = dIt->second; }
            }
            const double drawnCov = elapsedFrames > 0 ? double(drawnFrames) / double(elapsedFrames) : 0.0;
            const bool drawnNotSolved = drawnFrames > candidate.solveCount + candidate.solveCount / 2u + 2u;
            Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex, entry.first, std::dec,
                                     " PROMOTED to Path A (eigen gate drift ", state.eigDrift, ")",
                                     " | [PromoLat] ", elapsedSec, "s over ", elapsedFrames, " frames",
                                     " | solved ", candidate.solveCount, " (coverage ", coverage,
                                     ") | drawn ", drawnFrames, " (drawnCov ", drawnCov,
                                     ") | probing ", probingFrames, " frames, gate ", candidate.gateFrames,
                                     " frames | streakResets ", candidate.streakResets,
                                     " | verdict=", drawnNotSolved ? "OVER-CULLED(drawn but not solved)"
                                       : drawnCov < 0.5 ? "OFF-SCREEN-BOUND(game not drawing it)"
                                       : candidate.streakResets > 3 ? "STREAK-THRASH(non-rigid spikes)"
                                       : "solve-bound"));
          } else if (state.eigFrame != 0u && state.eigDrift > eigenEps) {
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
                                     " gate REJECTED (eigen drift ", state.eigDrift,
                                     ", fullResidual ", state.gateResidualRel,
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
        break;  // direction 2: routing + demote are per CONTENT CLASS (below)

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

      // ---- [ShapeClass] classify the content currently behind this slot ----
      // Content class = the capture's trace-normalized eigenvalue pair (eigLam1Hat,
      // eigLam2Hat) quantized to an eigenEpsilon grid (quantizeEigClass). It is a
      // rigid- and uniform-scale-INVARIANT shape descriptor, so the SAME piece at any
      // placement lands on the SAME cell exactly - classes match by exact cell, no
      // tolerance. A class change re-routes: a non-Promoted class restGates the
      // instance to Path B (see getGeometryId), so class STABILITY is what keeps a
      // building on Path A.
      //
      // HISTORY (why the key changed - do not revert to capSig):
      //  * The old key was capSig = round(log2(spread)*16), a SCALE-VARIANT signature.
      //    Being scale-variant it (a) split the same piece across placements and (b)
      //    was inherently noisy (the pooled/double-buffered capture holds content at
      //    discrete different levels), so ONE static building scattered across ~26
      //    phantom classes in a session (measured 0x878d), each re-probing/gating
      //    independently and never settling -> permanent Path B. It needed a 1.5/16
      //    "nearest-merge" tolerance purely to paper over that noise - a hack with no
      //    principled basis.
      //  * The eigen pair is the FIX: an invariant, so it is bit-stable per piece and
      //    needs no merge tolerance (exact-cell match). RULED OUT and not worth
      //    retrying: de-noising capSig (barrier, |det(A)|^(2/3)*refVar reformulation,
      //    band-widening) - the signal was fundamentally scale-variant, not just noisy.
      //  * FREEZE retained: once classified into a Promoted class, hold it; a genuine
      //    content change arrives as contentClassQ = INT32_MIN via getGeometryId's
      //    content-rebind reset (the STABLE candidate-key change). This guards a
      //    Promoted instance against re-routing on a single transient garbage capture.
      bool classSwapped = false;
      const bool freshSolve = state.lastFrame != 0u && state.lastFrame != promoInstance.lastClassifiedFrame;
      if (freshSolve) {
        promoInstance.lastClassifiedFrame = state.lastFrame;
        // Content id = the capture's trace-normalized eigenvalue pair quantized to
        // the eigenEpsilon grid (quantizeEigClass). Stable rigid/uniform-scale
        // invariant (unlike the old scale-variant capSig), so a cell change is a
        // genuine SHAPE change - matched EXACTLY, no bucket tolerance. The eigen key
        // comes from the staggered eigen sweep and persists in the status buffer
        // between sweeps, so it is available (stable) every frame once a sweep lands.
        const int32_t newQ = quantizeEigClass(state.eigLam1Hat, state.eigLam2Hat);
        if (newQ != INT32_MIN) {
          const bool unclassified = promoInstance.contentClassQ == INT32_MIN;
          // FREEZE: once classified into a currently-Promoted class, HOLD it - a
          // genuine content change arrives as contentClassQ = INT32_MIN via
          // getGeometryId's content-rebind reset (the STABLE candidate-key change),
          // which re-derives cleanly through the `unclassified` branch. This guards
          // a Promoted instance against re-routing on a single transient garbage
          // capture (mid-upload rename) that momentarily reads a different cell.
          const RestClassState* curClsFreeze = unclassified ? nullptr
            : resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ,
                               promoInstance.classSubId, false);
          const bool classFrozen = !unclassified
                                 && curClsFreeze != nullptr
                                 && curClsFreeze->phase == RestClassState::Phase::Promoted;
          if (unclassified) {
            promoInstance.contentClassQ = newQ;
          } else if (classFrozen) {
            promoInstance.pendingClassQ = INT32_MIN;
            promoInstance.pendingClassCount = 0;
          } else if (newQ != promoInstance.contentClassQ) {
            // cell change (genuine shape change) - confirm on 2 CONSECUTIVE sweeps
            // before committing (one divergent read is a transient garbage capture)
            if (promoInstance.pendingClassQ == newQ) {
              if (++promoInstance.pendingClassCount >= 2u) {
                classSwapped = true;
                m_swapCommitted++;
                promoInstance.residentGeometryId = ~0u;  // old cell's residency is stale
                // direction 2: routing follows the NEW cell's class verdict
                // automatically (getGeometryId reads the slot's current class),
                // so a swap needs no per-slot demote clearing - if the new cell
                // is Promoted the slot routes Path A next frame, else Path B.
                Logger::info(str::format("[ShapeClass] slot ", promoInstance.stateSlot,
                                         " geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                         " content class SWAP cell ", promoInstance.contentClassQ, " -> ", newQ,
                                         " (lam ", state.eigLam1Hat, ",", state.eigLam2Hat,
                                         ", frame ", m_device->getCurrentFrameId(), ")"));
                promoInstance.contentClassQ = newQ;
                promoInstance.classSubId = 0;  // new cell - re-enter the sibling chain at 0
                promoInstance.pendingClassQ = INT32_MIN;
                promoInstance.pendingClassCount = 0;
              }
            } else {
              if (promoInstance.pendingClassQ != INT32_MIN) {
                m_swapPendingAbandoned++;  // [SwapDebounce] pending cell replaced before confirm = transient
              }
              promoInstance.pendingClassQ = newQ;
              promoInstance.pendingClassCount = 1;
            }
          } else {
            if (promoInstance.pendingClassQ != INT32_MIN) {
              m_swapPendingAbandoned++;  // [SwapDebounce] returned to committed cell = transient
            }
            promoInstance.pendingClassQ = INT32_MIN;
            promoInstance.pendingClassCount = 0;
          }
        }
      }
      (void) classSwapped;

      // ---- [ShapeClass] class-keyed verdicts ----
      // This slot contributes EVIDENCE (rigid streak, residual, gate result) to
      // whichever content class its CURRENT capture classifies into; the phase
      // machine and verdicts live on the CLASS. The BlasEntry<->content binding
      // is unstable on churning-hash games, so any slot may speak for any class
      // over time - and a verdict keyed by content survives every rebind.
      // Two populations feed it:
      //  - rest-world slots (rest-referenced candidates): all phases.
      //  - ALL slots of a PROMOTED non-rest candidate (direction 2). Routing is
      //    now class-governed, so EVERY piece behind a promoted candidate needs
      //    its own class to reach Promoted before its slots route Path A - not
      //    just the divergent (misfitting) ones. The fitting content's class
      //    gates against the candidate's own object-space probe and promotes;
      //    divergent content earns its own rest reference exactly as before.
      {
        const auto candIt = m_promoCandidates.find(promoInstance.geometryHash);
        const bool restWorld = promoInstance.isRestWorld
                            && candIt != m_promoCandidates.end()
                            && candIt->second.routeHash != 0
                            && candIt->second.restState == PromotionCandidate::RestState::Referenced;
        // direction 2: any slot of a PROMOTED non-rest candidate feeds its content
        // class - the class is the routing authority, so the fitting content must
        // classify and gate-promote too (formerly only demoted/misfitting slots
        // fed classes, which left the fitting content routing off a per-slot flag).
        const bool memberOfPromoted = !promoInstance.isRestWorld
                            && candIt != m_promoCandidates.end()
                            && candIt->second.phase == PromotionCandidate::Phase::Promoted;
        // class CREATION is gated on temporal calm: continuously-deforming content
        // sweeps through buckets and would mint a class per bucket visited (observed:
        // 15 classes on one pulsing animator) - junk no verdict can ever serve.
        // Existing classes still MATCH regardless (evidence/verdicts continue).
        const bool tempCalm = state.temporalDeformRel <= ClusterLodOptions::Promotion::temporalEpsilon();
        RestClassState* cls = nullptr;
        if ((restWorld || memberOfPromoted) && promoInstance.contentClassQ != INT32_MIN) {
          cls = resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ,
                                 promoInstance.classSubId, false);
          if (cls == nullptr && promoInstance.classSubId != 0) {
            // stale cursor (sibling chain gone for this bucket) - rehome at 0
            promoInstance.classSubId = 0;
            cls = resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, 0, false);
          }
          if (cls == nullptr && tempCalm) {
            cls = resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, 0, true);
          }
        }

        if (cls != nullptr) {
          switch (cls->phase) {
          case RestClassState::Phase::Probing:
            if (freshSolve && !classSwapped) {  // swap frames carry cross-content temporal state - not evidence
              // Option 1: a permutation-invariant eigen verdict (companion sweep
              // above) is the primary "does this content fit the reference?"
              // signal - the rigidStreak path is per-index and rarely builds
              // under capture re-batching. eigDrift >= 0 = a real verdict landed.
              const bool eigClean = state.eigFrame != 0u && state.eigDrift >= 0.0f
                                 && state.eigDrift <= eigenEps;
              const bool eigDiff  = state.eigFrame != 0u && state.eigDrift > eigenEps;
              if (eigClean || state.rigidStreak >= rigidFrames) {
                cls->phase = RestClassState::Phase::GateScheduled;
                cls->gateStateSlot = promoInstance.stateSlot;
                cls->lastGateTickFrame = m_device->getCurrentFrameId();
                cls->stuckFrames = 0;
              } else if (eigDiff
                         && state.temporalDeformRel <= ClusterLodOptions::Promotion::temporalEpsilon()) {
                // static but not fitting this class's current reference. CLASS
                // threshold (restClassStuckFrames, not the candidate's 120):
                // classes accrue only during their content's dwell windows, so
                // the candidate-scale threshold starved them (1 request per
                // session while 198 static demotes cycled).
                if (++cls->stuckFrames
                    >= uint32_t(std::max(4, ClusterLodOptions::Promotion::restClassStuckFrames()))) {
                  cls->stuckFrames = 0;
                  if (cls->ref == RestClassState::Ref::CandidateProbe) {
                    // the SHARED reference cannot fit this content class - give the
                    // class its OWN reference (readback staged at the next emit from
                    // any class member)
                    cls->ref = RestClassState::Ref::Requested;
                    cls->captureStaged = false;
                    Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                             promoInstance.geometryHash, std::dec,
                                             " class q", cls->classQ, " static but misfits the shared reference (residual ",
                                             state.residualRel, ") - requesting CLASS rest capture"));
                  } else if (cls->ref == RestClassState::Ref::Own) {
                    // Misfits the sibling reference it is currently judged against.
                    // This is NOT "content deforms in place": the [RestCapProbe]
                    // chain proved captures faithful and content static - it means
                    // THIS content is a DIFFERENT SHAPE than the member whose pose
                    // the reference captured (the spread signature cannot separate
                    // non-affine-equivalent contents; verified per-content-constant
                    // residuals across sessions, refl 0, affine used). Identity is
                    // decided by FIT: advance this instance's cursor to the bucket's
                    // next sibling - an existing one first (its reference may be this
                    // content's), else mint a new sibling that runs the full ladder
                    // (shared probe -> own capture). The current entry stays: its
                    // reference fits the member it was captured from.
                    const int32_t nextSub = cls->subId + 1;
                    const int32_t curQ = cls->classQ;
                    const int32_t curSub = cls->subId;
                    RestClassState* nextCls = resolveRestClass(promoInstance.geometryHash,
                                                               promoInstance.contentClassQ, nextSub, false);
                    if (nextCls != nullptr) {
                      promoInstance.classSubId = nextSub;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " misfit by fit (eigDrift ", state.eigDrift,
                                               ", residual ", state.residualRel,
                                               ") - instance advances to EXISTING sibling ", nextSub));
                      cls = nextCls;  // post-switch checks read the instance's CURRENT class
                    } else if (nextSub < std::max(1, ClusterLodOptions::Promotion::restClassMaxRefs())) {
                      promoInstance.classSubId = nextSub;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " misfit by fit (eigDrift ", state.eigDrift,
                                               ", residual ", state.residualRel,
                                               ") - identity-by-fit SPLIT: minting sibling ", nextSub));
                      // creates via emplace_back - every pointer into the vector is
                      // INVALID after this call; re-resolve for the post-switch reads
                      resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, nextSub, true);
                      cls = resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, nextSub, false);
                      if (cls != nullptr) {
                        // skip the CandidateProbe re-proof: the advancing content
                        // already misfit the shared reference (its demote stream)
                        // and every earlier sibling - re-proving costs a full
                        // stuck ladder for zero information. Worst case is one
                        // redundant readback, never a wrong verdict (a capture
                        // still only promotes through its own full-mesh gate).
                        cls->ref = RestClassState::Ref::Requested;
                        cls->captureStaged = false;
                      }
                    } else {
                      // sibling cap: park THIS entry on the retry cooldown (as the old
                      // terminal reject did) and rewind the instance to sibling 0
                      cls->phase = RestClassState::Phase::Rejected;
                      cls->rejectedFrame = m_device->getCurrentFrameId();
                      promoInstance.classSubId = 0;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " misfits ALL ", ClusterLodOptions::Promotion::restClassMaxRefs(),
                                               " sibling refs (residual ", state.residualRel,
                                               ") - REJECTED, retries after ",
                                               ClusterLodOptions::Promotion::restRejectRetryFrames(), " frames"));
                    }
                  }
                  // Ref::Requested: reference in flight - keep waiting
                }
              } else {
                cls->stuckFrames = 0;
              }
            }
            break;

          case RestClassState::Phase::GateRunning:
            // only the slot whose capture the gate actually read judges it
            if (promoInstance.stateSlot == cls->gateStateSlot) {
              cls->lastGateTickFrame = m_device->getCurrentFrameId();
              if (++cls->gateFrames >= gateLag) {
                // Option 1: eigen gate verdict. The judging INSTANCE slot also
                // receives per-instance eigen sweeps, so require eigFrame to have
                // ADVANCED past the mark recorded at gate emission (gateEigMark) -
                // else a stale sweep verdict would be misread as the gate's.
                const bool eigGateFresh = state.eigFrame != 0u && state.eigFrame != cls->gateEigMark;
                if (eigGateFresh && state.eigDrift >= 0.0f && state.eigDrift <= eigenEps) {
                  cls->phase = RestClassState::Phase::Promoted;
                  m_statsPromoted++;
                  Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                           promoInstance.geometryHash, std::dec,
                                           " class q", cls->classQ, " PROMOTED to Path A (eigen gate drift ",
                                           state.eigDrift, ", ref ",
                                           cls->ref == RestClassState::Ref::Own ? "own" : "shared", ")"));
                  // first class to pass promotes the candidate (routing gate)
                  if (candIt->second.phase != PromotionCandidate::Phase::Promoted) {
                    candIt->second.phase = PromotionCandidate::Phase::Promoted;
                  }
                } else if (eigGateFresh && state.eigDrift > eigenEps && state.residualRel > epsilon) {
                  // CONTRADICTED: the gate slot's own sparse solve fails too - the
                  // content behind the slot changed mid-gate. Not a class verdict.
                  cls->phase = RestClassState::Phase::Probing;
                  cls->gateFrames = 0;
                  cls->gateStateSlot = ~0u;
                } else if (eigGateFresh && state.eigDrift > eigenEps) {
                  // clean gate failure on consistent content
                  if (cls->ref == RestClassState::Ref::CandidateProbe) {
                    // shared reference misfits this class's FULL mesh (sparse fit,
                    // full fail = partial divergence) - class needs its own reference
                    cls->ref = RestClassState::Ref::Requested;
                    cls->captureStaged = false;
                    cls->phase = RestClassState::Phase::Probing;
                    cls->gateFrames = 0;
                    cls->gateStateSlot = ~0u;
                    Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                             promoInstance.geometryHash, std::dec,
                                             " class q", cls->classQ, " gate misfits the shared reference (eigen drift ",
                                             state.eigDrift, ") - requesting CLASS rest capture"));
                  } else {
                    // Clean full-mesh failure vs this sibling's OWN reference: same
                    // identity-by-fit consequence as the sparse case - the judging
                    // member is a DIFFERENT SHAPE than the captured one. The entry
                    // returns to Probing (its reference may fit other members);
                    // THIS instance advances along the sibling chain.
                    cls->phase = RestClassState::Phase::Probing;
                    cls->gateFrames = 0;
                    cls->gateStateSlot = ~0u;
                    const uint32_t rr_refl = (state.diagGuard >> 26) & 0x1u;
                    const int32_t nextSub = cls->subId + 1;
                    const int32_t curQ = cls->classQ;
                    const int32_t curSub = cls->subId;
                    Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                             promoInstance.geometryHash, std::dec,
                                             " class q", curQ, " sub ", curSub, " gate misfit by fit",
                                             " | [RestReject] eigenDrift ", state.eigDrift,
                                             ", gate ", state.gateResidualRel,
                                             ", sparse ", state.residualRel,
                                             ", meanDev ", state.meanDevRel,
                                             ", dirCoh ", state.dirCoherence,
                                             ", tDeform ", state.temporalDeformRel,
                                             ", refl ", rr_refl));
                    RestClassState* nextCls = resolveRestClass(promoInstance.geometryHash,
                                                               promoInstance.contentClassQ, nextSub, false);
                    if (nextCls != nullptr) {
                      promoInstance.classSubId = nextSub;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " - instance advances to EXISTING sibling ", nextSub));
                      cls = nextCls;
                    } else if (nextSub < std::max(1, ClusterLodOptions::Promotion::restClassMaxRefs())) {
                      promoInstance.classSubId = nextSub;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " - identity-by-fit SPLIT: minting sibling ", nextSub));
                      // creates via emplace_back - re-resolve for the post-switch reads
                      resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, nextSub, true);
                      cls = resolveRestClass(promoInstance.geometryHash, promoInstance.contentClassQ, nextSub, false);
                      if (cls != nullptr) {
                        // skip the CandidateProbe re-proof (see the sparse mint site)
                        cls->ref = RestClassState::Ref::Requested;
                        cls->captureStaged = false;
                      }
                    } else {
                      cls->phase = RestClassState::Phase::Rejected;
                      cls->rejectedFrame = m_device->getCurrentFrameId();
                      promoInstance.classSubId = 0;
                      Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                               promoInstance.geometryHash, std::dec,
                                               " class q", curQ, " sub ", curSub,
                                               " misfits ALL ", ClusterLodOptions::Promotion::restClassMaxRefs(),
                                               " sibling refs (gate) - REJECTED, retries after ",
                                               ClusterLodOptions::Promotion::restRejectRetryFrames(), " frames"));
                    }
                  }
                } else {
                  // gate never accumulated (slot skipped that frame) - reschedule
                  cls->phase = RestClassState::Phase::GateScheduled;
                  cls->gateFrames = 0;
                }
              }
            }
            break;

          case RestClassState::Phase::Rejected: {
            const uint32_t retry = uint32_t(std::max(0, ClusterLodOptions::Promotion::restRejectRetryFrames()));
            if (retry > 0 && cls->rejectedFrame != 0
                && m_device->getCurrentFrameId() - cls->rejectedFrame >= retry) {
              cls->phase = RestClassState::Phase::Probing;
              cls->gateFrames = 0;
              cls->stuckFrames = 0;
              cls->gateStateSlot = ~0u;
              cls->rejectedFrame = 0;
            }
            break;
          }

          default:
            break;  // GateScheduled pairs in buildPromotionEntries; Promoted routes
          }
        }

        // pre-promotion REST-world slots skip the demote logic below (their
        // solves legitimately read non-rigid while a class probes); members of
        // a PROMOTED class fall through - they render Path A and demote per
        // slot. Divergent-of-promoted slots ALWAYS fall through: the demote
        // logic's re-promote-on-streak is exactly how they return to Path A
        // once their solves (against their class's own reference) fit.
        if (promoInstance.isRestWorld
            && (cls == nullptr || cls->phase != RestClassState::Phase::Promoted)) {
          continue;
        }

        // ---- direction 2: per-CLASS eigen demote, ATTRIBUTED TO THE MEASURED CELL ----
        // ROOT CAUSE of the phantom-demote flood: an eigen sweep produces two things
        // about ONE measured capture - its IDENTITY (lamHat -> cell) and its VERDICT
        // (drift). They are the same measurement and must never be split. The code
        // applied the verdict to the slot's `contentClassQ`, a SEPARATE pointer the
        // classification freeze/debounce pins to a stale cell whenever content
        // multiplexes under a stable BlasEntry (this game does constantly - 2249
        // swaps). So a sweep of piece P (lamHat 0.896 = cell 184325, drift 0.37) had
        // its drift charged to whatever stale cell the slot was frozen on (139280,
        // 135185, ...) - condemning cells whose real content that sweep never touched.
        // FIX: a sweep's verdict may only move the class the sweep ACTUALLY measured.
        // If the measured cell != the slot's committed class, this sweep is about a
        // DIFFERENT piece flowing through the slot - consume the bookkeeping but do
        // NOT touch this class. Its own content's sweeps drive its verdict.
        if (state.eigFrame != 0u && state.eigFrame != promoInstance.lastEigenFrame) {
          promoInstance.lastEigenFrame = state.eigFrame;
          promoInstance.sweepPending = false;   // the in-flight eigen sweep landed
          const float eigEps = std::max(0.0f, ClusterLodOptions::Promotion::eigenEpsilon());
          const uint32_t demoteSweeps = uint32_t(std::max(1, ClusterLodOptions::Promotion::eigenDemoteSweeps()));
          // the cell THIS sweep measured (its own lamHat). The verdict describes this
          // content; it may only touch `cls` when `cls` IS this content's class.
          const int32_t measuredQ = quantizeEigClass(state.eigLam1Hat, state.eigLam2Hat);
          const bool sweepMeasuredThisClass = cls != nullptr
            && measuredQ != INT32_MIN && measuredQ == promoInstance.contentClassQ;
          // is this class currently routing its members to Path A? rest candidates
          // route on phase==Promoted; non-rest route by default and drop to B only
          // when driftDemoted (see getGeometryId). Only an up class can demote.
          const bool restCand = candIt != m_promoCandidates.end() && candIt->second.routeHash != 0;
          const bool clsRoutesA = cls != nullptr
            && (restCand ? cls->phase == RestClassState::Phase::Promoted : !cls->driftDemoted);
          // HYSTERESIS: re-promote below eigEps, demote only above eigEps*demoteHysteresis.
          // A class whose drift sits STEADILY in the band between the two is a genuine
          // minor content difference (a piece ~5% off the shared reference - measured,
          // static, rigid), NOT deformation; the OLD promotion kept those on Path A and
          // demoting them on a single boundary-crossing sweep is exactly the A<->B flap
          // observed (q110611 demoting 5x on drift 0.022/0.038/0.113). Only SUSTAINED,
          // clearly-large drift (real deformation / a genuinely different shape) demotes;
          // any sweep at-or-below the band resets the streak so it must be CONSECUTIVE.
          const float eigDemote = eigEps * std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis());
          if (sweepMeasuredThisClass && state.eigDrift >= 0.0f) {  // <0 = no prediction/degenerate, no verdict
            if (state.eigDrift <= eigEps) {
              promoInstance.eigenSuspect = false;
              if (cls != nullptr) {
                cls->eigenDriftStreak = 0;      // clearly clean - reset the streak
                cls->driftDemoted = false;      // and route Path A again (non-rest)
              }
            } else if (state.eigDrift > eigDemote
                       && state.temporalDeformRel > (std::max(0.0f, ClusterLodOptions::Promotion::temporalEpsilon())
                                                     * std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis()))) {
              // DEMOTE REQUIRES ACTUAL DEFORMATION, not merely drift from the shared
              // reference. Path B exists for geometry whose vertices deform per frame;
              // a STABLE piece must stay on Path A even if its shape is far from the
              // shared reference (it earns its own reference instead). The eigen drift-
              // vs-reference is a SPATIAL misfit signal - it is large and noisy for a
              // static piece that simply differs from the reference (q110611/q159755:
              // steady, in-cell, drift 0.06-0.46 = a different STATIC shape). Gating on
              // temporalDeformRel (which is ~0 for rigid geometry, moving OR still, and
              // spikes only when the shape itself changes frame to frame) means only a
              // genuinely deforming class can demote - stable geometry never leaves A.
              // Keep suspicion armed while up so the streak accrues at readback cadence.
              promoInstance.eigenSuspect = clsRoutesA;
              if (cls != nullptr) {
                cls->eigenDriftStreak++;
                if (clsRoutesA && cls->eigenDriftStreak >= demoteSweeps) {
                  // non-rest: flip driftDemoted (routes B, keeps its proven object-
                  // space residency for instant re-promote on a clean sweep). rest:
                  // drop the phase so it re-gates against its own captured reference.
                  if (restCand) {
                    cls->phase = RestClassState::Phase::Probing;
                    cls->gateFrames = 0;
                    cls->gateStateSlot = ~0u;
                  } else {
                    cls->driftDemoted = true;
                  }
                  cls->eigenDriftStreak = 0;
                  Logger::info(str::format("[ClusterLOD] promotion: geometry 0x", std::hex,
                                           promoInstance.geometryHash, std::dec,
                                           " class q", cls->classQ, " sub ", cls->subId,
                                           " DEMOTED to Path B - DEFORMING (tDeform ", state.temporalDeformRel,
                                           ", eigen drift ", state.eigDrift, " > ", eigDemote,
                                           ") on ", demoteSweeps, " consecutive sweep(s) (lamHat ",
                                           state.eigLam1Hat, ",", state.eigLam2Hat,
                                           ", sweepFrame ", state.eigFrame, ", ",
                                           (restCand ? "rest" : "non-rest"), ")"));
                }
              }
            } else {
              // Stays on Path A. Either the hysteresis band (eigEps < drift <= eps*hyst,
              // a minor offset) OR large drift with LOW tDeform (a static piece that
              // differs from the shared reference but is NOT deforming). Neither is a
              // reason to leave Path A - hold routing state and reset the streak so only
              // sustained, genuinely-deforming drift can ever demote.
              promoInstance.eigenSuspect = clsRoutesA;
              if (cls != nullptr) {
                cls->eigenDriftStreak = 0;
              }
            }
          } else {
            promoInstance.eigenSuspect = false;  // no prediction yet - cadence retries
          }
        } else if (promoInstance.sweepPending && ++promoInstance.sweepLagFrames > 4u * gateLag) {
          // eigen sweep verdict lost (slot left view / entry dropped mid-flight):
          // release the in-flight guard so the cadence can schedule a fresh one
          promoInstance.sweepPending = false;
        }

        // per-index signals (residual/tDeform) NEVER demote - they are POISONED by
        // the engine re-batching the capture's vertex ORDER (a permuted buffer
        // explodes residual AND tDeform on a perfectly static mesh). They only
        // SCHEDULE a permutation-invariant eigen sweep to verify; the set-wise
        // verdict above is the sole demote authority.
        const float tempEpsDemote = std::max(0.0f, ClusterLodOptions::Promotion::temporalEpsilon())
                                  * std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis());
        const float epsDemote = epsilon * std::max(1.0f, ClusterLodOptions::Promotion::demoteHysteresis());
        if ((state.flags & 4u) != 0
            && state.temporalDeformRel > tempEpsDemote && state.residualRel > epsDemote
            && !promoInstance.eigenSuspect && !promoInstance.sweepPending) {
          promoInstance.eigenSuspect = true;
          static std::mutex s_susMx;
          static std::unordered_map<uint64_t, uint32_t> s_susLast;
          const uint32_t frameNow = m_device->getCurrentFrameId();
          std::lock_guard<std::mutex> lk(s_susMx);
          uint32_t& last = s_susLast[promoInstance.geometryHash];
          if (last == 0u || frameNow - last > 10u) {
            last = frameNow;
            Logger::info(str::format("[ClusterLOD] promotion: instance (slot ", promoInstance.stateSlot,
                                     ", geom 0x", std::hex, promoInstance.geometryHash, std::dec,
                                     ") SUSPECT (tDeform ", state.temporalDeformRel,
                                     ", residual ", state.residualRel,
                                     ") - eigen sweep scheduled to verify (no demote on per-index signals)"));
          }
        }
      }
    }

    // [ShapeClass] wedge guard: a class stuck in GateScheduled/GateRunning whose
    // owning slot's content swapped away (or left the scene) would wait forever -
    // no slot classifies into it to tick the gate. Reset stale gates to Probing.
    {
      const uint32_t frameNow = m_device->getCurrentFrameId();
      const uint32_t staleAfter = 4u * uint32_t(std::max(2, ClusterLodOptions::Promotion::gateLagFrames()));
      for (auto& candClasses : m_restClassesByCandidate) {
        for (RestClassState& c : candClasses.second) {
          if ((c.phase == RestClassState::Phase::GateScheduled || c.phase == RestClassState::Phase::GateRunning)
              && c.lastGateTickFrame != 0 && frameNow - c.lastGateTickFrame > staleAfter) {
            c.phase = RestClassState::Phase::Probing;
            c.gateFrames = 0;
            c.gateStateSlot = ~0u;
          }
        }
      }
    }

    // [ShapeClass] periodic histogram (1s throttle): per candidate, the distinct
    // content classes its live slots currently hold. Multi-class candidates are
    // exactly the population the shared rest reference CANNOT serve (each class
    // needs its own reference); single-class candidates validate that the
    // bucketing is stable (no boundary flicker = no spurious classes).
    {
      const auto nowT = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(nowT - m_lastShapeClassLog).count() >= 1000) {
        m_lastShapeClassLog = nowT;
        // [SwapDebounce] running totals (only when nonzero - quiet when healthy):
        // the committed-vs-abandoned ratio decides the debounce's fate (see .h)
        if (m_swapPendingAbandoned + m_swapCommitted > 0) {
          Logger::info(str::format("[SwapDebounce] totals: committed ", m_swapCommitted,
                                   ", abandoned(transients) ", m_swapPendingAbandoned));
        }
        std::unordered_map<uint64_t, std::map<int32_t, uint32_t>> classesByGeom;
        for (const auto& slotEntry : m_promoSlotByBlas) {
          if (slotEntry.second.contentClassQ != INT32_MIN && slotEntry.second.geometryHash != 0) {
            classesByGeom[slotEntry.second.geometryHash][slotEntry.second.contentClassQ]++;
          }
        }
        for (const auto& g : classesByGeom) {
          if (g.second.size() < 2) {
            continue;  // single-class candidates are the healthy default - stay quiet
          }
          std::string buckets;
          for (const auto& c : g.second) {
            buckets += str::format(" q", c.first, "x", c.second);
          }
          Logger::info(str::format("[ShapeClass] geom 0x", std::hex, g.first, std::dec,
                                   " holds ", g.second.size(), " content classes:", buckets));
        }
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

        // REST-referenced candidate ([ShapeClass]): per-slot SOLVES feed the
        // class verdict layer. Every slot solves its own capture every frame -
        // that solve simultaneously (a) classifies the slot's current content,
        // (b) builds rigidity evidence for that class, and (c) provides the M
        // the class gate scores. Gates and rest-capture readbacks are per
        // CLASS, staged from whichever slot currently holds that content.
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
            instanceIt = m_promoSlotByBlas.emplace(blasEntry, promoInstance).first;
          }
          PromoInstance& promoInstance = instanceIt->second;
          if (promoInstance.blasFrameCreated != blasEntry->frameCreated) {
            // recycled BlasEntry address = fresh capture content on the same slot
            promoInstance.blasFrameCreated = blasEntry->frameCreated;
            promoInstance.geometryHash = hash;
            promoInstance.sweepPending = false;
            promoInstance.eigenSuspect = false;
            promoInstance.residentGeometryId = ~0u;
            promoInstance.contentClassQ = INT32_MIN;  // fresh content - reclassify
            promoInstance.classSubId = 0;
            promoInstance.classSubId = 0;
            promoInstance.lastClassifiedFrame = 0;
          }
          promoInstance.isRestWorld = true;
          promoInstance.geometryHash = hash;

          // the slot's class (from its last classified solve; null until the
          // first solve classifies it). Find-only: emits never create classes.
          RestClassState* cls = promoInstance.contentClassQ != INT32_MIN
            ? resolveRestClass(hash, promoInstance.contentClassQ, promoInstance.classSubId, false)
            : nullptr;

          if (cls != nullptr && cls->phase == RestClassState::Phase::Promoted) {
            // routes Path A (class-governed); the promoted Path A emit below handles its solve
            continue;
          }

          if (!emittedInstanceSlots.insert(promoInstance.stateSlot).second) {
            continue;  // instances sharing a BlasEntry share capture content + slot
          }

          // NOTE (VA-pin reverted): capture ADDRESS is not identity (double-
          // buffering + content rebinds). Content classes are - see [ShapeClass].
          lodclusters_remix::PromotionEntry instEntry;
          // solve against the class's OWN reference once it has one, else the shared one
          instEntry.probeVa = (cls != nullptr && cls->ref == RestClassState::Ref::Own && cls->probeVa != 0)
            ? cls->probeVa : candidate.probeVa;
          instEntry.captureVa = m_framePoses[framePoseIndex].positionsAddress;
          instEntry.captureStrideBytes = m_framePoses[framePoseIndex].positionsStrideBytes;
          instEntry.captureVertexCount = m_framePoses[framePoseIndex].positionsCount;
          instEntry.stateSlot = promoInstance.stateSlot;
          instEntry.patchSlot = 0xFFFFFFFFu;
          m_framePromoEntries.push_back(instEntry);

          // Option 1: companion eigen sweep for a Probing/rest-world member so the
          // classifier has a permutation-invariant shape verdict to schedule its
          // gate on (the rigidStreak path below is per-index and rarely builds
          // under the engine's capture re-batching). Staggered like the promoted
          // sweep so it costs ~1/interval of members per frame.
          {
            const uint32_t eigInterval = uint32_t(std::max(0, ClusterLodOptions::Promotion::fullSweepIntervalFrames()));
            if (eigInterval > 0 && candidate.probeVa != 0
                && ((m_device->getCurrentFrameId() + promoInstance.stateSlot) % eigInterval) == 0) {
              lodclusters_remix::PromotionEntry eigEntry = instEntry;
              eigEntry.mode = 2;  // PROMO_MODE_EIGEN
              m_framePromoEntries.push_back(eigEntry);
            }
          }

          promoInstance.lastCaptureVa = instEntry.captureVa;  // [RestGateTrace] address is diagnostic only

          if (cls != nullptr) {
            // CLASS rest-capture readback: this slot holds the class's content -
            // stage the one-time copy of ITS capture as the class's reference
            if (cls->ref == RestClassState::Ref::Requested && !cls->captureStaged) {
              uint32_t topoVertexCount = 0;
              {
                std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
                const auto topoIt = m_promoTopologyByHash.find(hash);
                if (topoIt != m_promoTopologyByHash.end()) {
                  topoVertexCount = topoIt->second.vertexCount;
                }
              }
              const FramePose& pose = m_framePoses[framePoseIndex];
              if (topoVertexCount > 0 && pose.positionsCount >= topoVertexCount
                  && pose.positionsStrideBytes >= 3 * sizeof(float)) {
                DxvkBufferCreateInfo stagingInfo;
                stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
                stagingInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
                stagingInfo.size = VkDeviceSize(topoVertexCount) * pose.positionsStrideBytes;
                RestCaptureRequest request;
                request.geometryHash = hash;
                request.classQ = cls->classQ;
                request.classSubId = cls->subId;
                request.source = pose.positionsBuffer;
                request.sourceOffset = pose.positionsBufferOffset;
                request.strideBytes = pose.positionsStrideBytes;
                request.vertexCount = topoVertexCount;
                request.stateSlot = promoInstance.stateSlot;  // [RestCapProbe] same-frame solve reads this buffer
                request.staging = m_device->createBuffer(stagingInfo,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  DxvkMemoryStats::Category::RTXBuffer, "promo class rest-capture readback");
                m_restCaptureRequests.push_back(std::move(request));
                cls->captureStaged = true;
              } else {
                // topology missing / pose too small: this class cannot be
                // rest-referenced - park it as Rejected (retry-cooled)
                cls->ref = RestClassState::Ref::CandidateProbe;
                cls->phase = RestClassState::Phase::Rejected;
                cls->rejectedFrame = m_device->getCurrentFrameId();
                ONCE(Logger::warn(str::format("[ClusterLOD] class rest-capture: geometry 0x", std::hex, hash, std::dec,
                                              " q", cls->classQ, " has no retained topology / undersized pose")));
              }
            }

            // same-frame gate pairing (solves -> barrier -> gates in recordPromotion):
            // the scheduled slot's own capture scores the class gate
            if (cls->phase == RestClassState::Phase::GateScheduled
                && (cls->gateStateSlot == promoInstance.stateSlot || cls->gateStateSlot == ~0u)) {
              cls->gateStateSlot = promoInstance.stateSlot;
              lodclusters_remix::PromotionEntry gateEntry = instEntry;
              gateEntry.mode = 2;  // Option 1: eigen gate (permutation-invariant)
              gateEntry.vertexCount = (cls->ref == RestClassState::Ref::Own && cls->vertexCount != 0)
                ? cls->vertexCount : candidate.vertexCount;
              m_framePromoEntries.push_back(gateEntry);
              cls->phase = RestClassState::Phase::GateRunning;
              cls->gateFrames = 0;
              cls->lastGateTickFrame = m_device->getCurrentFrameId();
              cls->gateEigMark = (m_promoStatesValid && promoInstance.stateSlot < m_promoStates.size())
                ? m_promoStates[promoInstance.stateSlot].eigFrame : 0;  // verdict = eigFrame advances past this
            }
          }
          continue;
        }

        // DEMOTED promoted-instance rendering Path B: keep solving ITS OWN
        // slot so a rebuilt rigid streak re-promotes it (per-instance
        // demotion; dedup by state slot - instances sharing a BlasEntry share
        // capture content and therefore a slot).
        // SOURCE = modifiedGeometryData, the SAME buffer the post-promotion
        // Path A solve judges. Re-proving on the pose buffer while the
        // promoted solve reads modifiedGeometryData let the two disagree
        // (rebind churn): pose fits -> re-promote -> modifiedGeometryData
        // misfits -> demote, forever (observed: the same slots demoting 3x,
        // 237 demotes/87 re-promotes in one session). Judging the buffer that
        // will actually be judged makes re-promotion self-consistent.
        if (candidate.phase == PromotionCandidate::Phase::Promoted && candidate.probeVa != 0) {
          const auto instanceIt = m_promoSlotByBlas.find(blasEntry);
          if (instanceIt != m_promoSlotByBlas.end()) {
            PromoInstance& pathBInst = instanceIt->second;
            // [ShapeClass] this Path-B slot's current content class. It solves
            // against the class's OWN reference once the class earns one; its
            // solves + eigen sweeps are how the class reaches (or returns to)
            // Promoted - the promotion path for every piece behind this candidate.
            RestClassState* cls = pathBInst.contentClassQ != INT32_MIN
              ? resolveRestClass(hash, pathBInst.contentClassQ, pathBInst.classSubId, false)
              : nullptr;
            // direction 2: emit for any slot routing Path B under a promoted
            // candidate so it keeps solving/sweeping and its class can (re)promote.
            // That is a class not yet Promoted (rest, still gating) OR a drift-
            // demoted class (non-rest that deformed - phase may still be Promoted,
            // so driftDemoted must be checked or it would get no sweeps and never
            // recover). Slots routing Path A are served by the promoted emit below.
            // dedup by state slot (instances sharing a BlasEntry share content+slot).
            if ((cls == nullptr || cls->phase != RestClassState::Phase::Promoted || cls->driftDemoted)
                && emittedInstanceSlots.insert(pathBInst.stateSlot).second) {
            const RaytraceBuffer& rePositions = blasEntry->modifiedGeometryData.positionBuffer;
            lodclusters_remix::PromotionEntry probeEntry;
            probeEntry.probeVa = (cls != nullptr && cls->ref == RestClassState::Ref::Own && cls->probeVa != 0)
              ? cls->probeVa : candidate.probeVa;
            probeEntry.captureVa = rePositions.getDeviceAddress() + rePositions.offsetFromSlice();
            probeEntry.captureStrideBytes = rePositions.stride();
            probeEntry.captureVertexCount = rePositions.stride() > 0 && rePositions.length() > rePositions.offsetFromSlice()
              ? uint32_t((rePositions.length() - rePositions.offsetFromSlice()) / rePositions.stride()) : 0;
            probeEntry.stateSlot = pathBInst.stateSlot;
            probeEntry.patchSlot = 0xFFFFFFFFu;
            m_framePromoEntries.push_back(probeEntry);

            // Option 1: demoted slots ALSO run the eigen sweep on the stagger
            // (or immediately on suspicion). A clean set-wise verdict is their
            // primary re-promotion path: per-index solves keep misfitting on
            // permuted captures, so the rigid-streak route alone left demoted
            // static instances stuck on Path B (streaks need CONTIGUOUS clean
            // frames the permutation alternation denies). The streak route
            // stays as a second, faster path when correspondence holds.
            const uint32_t eigInterval = uint32_t(std::max(0, ClusterLodOptions::Promotion::fullSweepIntervalFrames()));
            if (eigInterval > 0 && !pathBInst.sweepPending
                && (pathBInst.eigenSuspect
                    || ((m_device->getCurrentFrameId() + pathBInst.stateSlot) % eigInterval) == 0)) {
              lodclusters_remix::PromotionEntry eigEntry = probeEntry;
              eigEntry.mode = 2;  // PROMO_MODE_EIGEN
              m_framePromoEntries.push_back(eigEntry);
              pathBInst.sweepPending = true;
              pathBInst.sweepLagFrames = 0;
            }

            if (cls != nullptr) {
              // class rest-capture readback: source = the SAME buffer the solves
              // judge (modifiedGeometryData), so the reference captures exactly
              // the content the class verdicts are about
              if (cls->ref == RestClassState::Ref::Requested && !cls->captureStaged) {
                uint32_t topoVertexCount = 0;
                {
                  std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
                  const auto topoIt = m_promoTopologyByHash.find(hash);
                  if (topoIt != m_promoTopologyByHash.end()) {
                    topoVertexCount = topoIt->second.vertexCount;
                  }
                }
                if (topoVertexCount > 0 && probeEntry.captureVertexCount >= topoVertexCount
                    && rePositions.stride() >= 3 * sizeof(float)) {
                  DxvkBufferCreateInfo stagingInfo;
                  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                  stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
                  stagingInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
                  stagingInfo.size = VkDeviceSize(topoVertexCount) * rePositions.stride();
                  RestCaptureRequest request;
                  request.geometryHash = hash;
                  request.classQ = cls->classQ;
                  request.classSubId = cls->subId;
                  request.source = rePositions.buffer();
                  request.sourceOffset = rePositions.offset() + rePositions.offsetFromSlice();
                  request.strideBytes = rePositions.stride();
                  request.vertexCount = topoVertexCount;
                  request.stateSlot = pathBInst.stateSlot;  // [RestCapProbe] same-frame solve reads this buffer
                  request.staging = m_device->createBuffer(stagingInfo,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    DxvkMemoryStats::Category::RTXBuffer, "promo class rest-capture readback");
                  m_restCaptureRequests.push_back(std::move(request));
                  cls->captureStaged = true;
                } else {
                  cls->ref = RestClassState::Ref::CandidateProbe;
                  cls->phase = RestClassState::Phase::Rejected;
                  cls->rejectedFrame = m_device->getCurrentFrameId();
                  ONCE(Logger::warn(str::format("[ClusterLOD] class rest-capture: geometry 0x", std::hex, hash, std::dec,
                                                " q", cls->classQ, " (demoted path) no topology / undersized capture")));
                }
              }
              // same-frame class gate pairing (solves -> barrier -> gates)
              if (cls->phase == RestClassState::Phase::GateScheduled
                  && (cls->gateStateSlot == pathBInst.stateSlot || cls->gateStateSlot == ~0u)) {
                cls->gateStateSlot = pathBInst.stateSlot;
                lodclusters_remix::PromotionEntry gateEntry = probeEntry;
                gateEntry.mode = 2;  // Option 1: eigen gate (permutation-invariant)
                gateEntry.vertexCount = (cls->ref == RestClassState::Ref::Own && cls->vertexCount != 0)
                  ? cls->vertexCount : candidate.vertexCount;
                m_framePromoEntries.push_back(gateEntry);
                cls->phase = RestClassState::Phase::GateRunning;
                cls->gateFrames = 0;
                cls->lastGateTickFrame = m_device->getCurrentFrameId();
                cls->gateEigMark = (m_promoStatesValid && pathBInst.stateSlot < m_promoStates.size())
                  ? m_promoStates[pathBInst.stateSlot].eigFrame : 0;  // verdict = eigFrame advances past this
              }
            }
            }  // close the class-not-Promoted emit gate
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
            request.stateSlot = candidate.stateSlot;  // [RestCapProbe] same-frame solve reads this buffer
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
          gateEntry.mode = 2;  // Option 1: eigen gate (permutation-invariant)
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

        // [ShapeClass] a member of an own-referenced class solves against ITS
        // class's probe - the M must map the class's reference (whose clusters
        // it renders) into the capture, not the candidate's shared reference.
        uint64_t promotedProbeVa = found->second.probeVa;
        uint32_t promotedGateVerts = found->second.vertexCount;
        RestClassState* promotedCls = nullptr;
        {
          const auto slotInfoIt = m_promoSlotByBlas.find(blasEntry);
          if (slotInfoIt != m_promoSlotByBlas.end() && slotInfoIt->second.contentClassQ != INT32_MIN) {
            promotedCls = resolveRestClass(hash, slotInfoIt->second.contentClassQ,
                                           slotInfoIt->second.classSubId, false);
            if (promotedCls != nullptr && promotedCls->ref == RestClassState::Ref::Own && promotedCls->probeVa != 0) {
              promotedProbeVa = promotedCls->probeVa;
              if (promotedCls->vertexCount != 0) {
                promotedGateVerts = promotedCls->vertexCount;
              }
            }
          }
        }

        lodclusters_remix::PromotionEntry promoEntry;
        promoEntry.probeVa = promotedProbeVa;
        promoEntry.captureVa = positions.getDeviceAddress() + positions.offsetFromSlice();
        promoEntry.captureStrideBytes = positions.stride();
        promoEntry.captureVertexCount = positions.stride() > 0 && positions.length() > positions.offsetFromSlice()
          ? uint32_t((positions.length() - positions.offsetFromSlice()) / positions.stride()) : 0;
        promoEntry.stateSlot = (slot.geometryId >> kPromotedSlotShift) & 0x1FFFu;
        promoEntry.patchSlot = flatIndex;
        m_framePromoEntries.push_back(promoEntry);

        // [BindTrace] per-frame capture-binding identity for the traced geometry.
        // The tDeform=5.0 false-demotes are on SAME-hash slots that fit rest-pose
        // at ~2% each frame - so the captured verts we read for that slot cannot be
        // a rigid/animated version of the same mesh; the read is inconsistent. This
        // trace settles which of the three roots it is, by following one slot across
        // consecutive frames (post-process: group by slot):
        //   captureVa moves frame-to-frame  -> shared-buffer aliasing (same VA slot
        //                                       reused by different draws)
        //   blas ptr / blasFC moves         -> BLAS rebuilt each frame (the pointer
        //                                       identity the promo state keys on churns)
        //   all three stable but tDeform hi -> BLAS geometry updated IN PLACE (dynamic
        //                                       mesh) -> per-BLAS-slot state is the wrong
        //                                       identity for this game (needs content key)
        // Traced hash only; self-limited line budget so it can never flood the log.
        {
          const std::string& btHashStr = ClusterLodOptions::Promotion::dumpGeometryHash();
          if (!btHashStr.empty()) {
            uint64_t btHash = 0;
            try { btHash = std::stoull(btHashStr, nullptr, 16); } catch (...) { btHash = 0; }
            static uint32_t s_bindTraceBudget = 2000u;
            if (btHash != 0 && hash == btHash && s_bindTraceBudget > 0u) {
              --s_bindTraceBudget;
              Logger::info(str::format("[BindTrace] geom 0x", std::hex, hash, std::dec,
                                       " frame ", m_device->getCurrentFrameId(),
                                       " stateSlot ", promoEntry.stateSlot,
                                       " patchSlot ", promoEntry.patchSlot,
                                       " blas ", reinterpret_cast<uintptr_t>(blasEntry),
                                       " blasFC ", (blasEntry != nullptr ? blasEntry->frameCreated : 0u),
                                       " captureVa 0x", std::hex, promoEntry.captureVa, std::dec,
                                       " stride ", promoEntry.captureStrideBytes,
                                       " vtxCount ", promoEntry.captureVertexCount,
                                       " geomId 0x", std::hex, slot.geometryId, std::dec, ")"));
            }
          }
        }

        // [ShapeClass] class service for a promoted slot whose class is still
        // working toward its own reference (misfit dwells happen while the
        // instance routes Path A between demote windows - their solves flow
        // through HERE, so the staging/gate pairing must exist here too, or
        // the class wedges in Requested/GateScheduled)
        if (promotedCls != nullptr && promotedCls->phase != RestClassState::Phase::Promoted) {
          if (promotedCls->ref == RestClassState::Ref::Requested && !promotedCls->captureStaged) {
            uint32_t topoVertexCount = 0;
            {
              std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
              const auto topoIt = m_promoTopologyByHash.find(hash);
              if (topoIt != m_promoTopologyByHash.end()) {
                topoVertexCount = topoIt->second.vertexCount;
              }
            }
            if (topoVertexCount > 0 && promoEntry.captureVertexCount >= topoVertexCount
                && positions.stride() >= 3 * sizeof(float)) {
              DxvkBufferCreateInfo stagingInfo;
              stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
              stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
              stagingInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
              stagingInfo.size = VkDeviceSize(topoVertexCount) * positions.stride();
              RestCaptureRequest request;
              request.geometryHash = hash;
              request.classQ = promotedCls->classQ;
              request.classSubId = promotedCls->subId;
              request.source = positions.buffer();
              request.sourceOffset = positions.offset() + positions.offsetFromSlice();
              request.strideBytes = positions.stride();
              request.vertexCount = topoVertexCount;
              request.stateSlot = promoEntry.stateSlot;  // [RestCapProbe] same-frame solve reads this buffer
              request.staging = m_device->createBuffer(stagingInfo,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "promo class rest-capture readback");
              m_restCaptureRequests.push_back(std::move(request));
              promotedCls->captureStaged = true;
            }
          }
          if (promotedCls->phase == RestClassState::Phase::GateScheduled
              && (promotedCls->gateStateSlot == promoEntry.stateSlot || promotedCls->gateStateSlot == ~0u)) {
            promotedCls->gateStateSlot = promoEntry.stateSlot;
            lodclusters_remix::PromotionEntry gateEntry = promoEntry;
            gateEntry.patchSlot = 0xFFFFFFFFu;
            gateEntry.mode = 2;  // Option 1: eigen gate (permutation-invariant)
            gateEntry.vertexCount = promotedGateVerts;
            m_framePromoEntries.push_back(gateEntry);
            promotedCls->phase = RestClassState::Phase::GateRunning;
            promotedCls->gateFrames = 0;
            promotedCls->lastGateTickFrame = m_device->getCurrentFrameId();
            promotedCls->gateEigMark = (m_promoStatesValid && promoEntry.stateSlot < m_promoStates.size())
              ? m_promoStates[promoEntry.stateSlot].eigFrame : 0;  // verdict = eigFrame advances past this
          }
        }

        // periodic EIGEN sweep (Option 1, replaces the R20 full-residual sweep):
        // permutation-invariant shape verdict over the full referenced capture
        // set, on the same stagger - plus IMMEDIATELY when the per-frame solve
        // flagged suspicion (eigenSuspect: residual/tDeform fired, which on this
        // game is usually the re-batched vertex ORDER, not deformation). The
        // verdict (updatePromotionStates) demotes only on genuine shape drift.
        const uint32_t sweepInterval = uint32_t(std::max(0, ClusterLodOptions::Promotion::fullSweepIntervalFrames()));
        if (sweepInterval > 0 && found->second.probeVa != 0) {
          const auto instanceIt = m_promoSlotByBlas.find(blasEntry);
          if (instanceIt != m_promoSlotByBlas.end() && !instanceIt->second.sweepPending
              && (instanceIt->second.eigenSuspect
                  || ((m_device->getCurrentFrameId() + instanceIt->second.stateSlot) % sweepInterval) == 0)) {
            lodclusters_remix::PromotionEntry sweepEntry = promoEntry;
            sweepEntry.patchSlot = 0xFFFFFFFFu;
            sweepEntry.mode = 2;  // PROMO_MODE_EIGEN
            sweepEntry.vertexCount = promotedGateVerts;  // informational (eigen uses probe.vertexCount)
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
          // [DrawCoverage] frames the GAME submitted this geometry's draw (pre-culling).
          // drawnFrames 0 => genuinely off-screen (benign). drawnFrames climbing while
          // lastSolveFrame is stuck => drawn on-screen but the cluster path never
          // solves it - the real "on-screen but never promotes" bug.
          uint32_t drawnFrames = 0;
          {
            std::lock_guard<std::mutex> lk(m_promoDrawMutex);
            auto dIt = m_promoDrawnCountByTopo.find(c.topologyKey);
            if (dIt != m_promoDrawnCountByTopo.end()) { drawnFrames = dIt->second; }
          }
          Logger::info(str::format("[PromoLimbo] geometry 0x", std::hex, e.first, std::dec,
                                   " uploaded but NOT solved this frame (phase ", ph,
                                   ", inPathB ", inB, ", residentPathA ", residentPathA,
                                   ", drawnFrames ", drawnFrames,
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

    // direction 2: does an instance of promoted candidate `candHash` holding the
    // content class (classQ, subId) route Path A this frame? NON-rest content
    // routes A by DEFAULT (the candidate itself is proven rigid) and drops to B
    // only when its class is driftDemoted - so a freshly promoted building appears
    // on Path A immediately, with no per-class gate latency. REST candidates still
    // gate on the class reaching Promoted (they render their own captured pose).
    auto routesPathA = [this](uint64_t candHash, int32_t classQ, int32_t subId) -> bool {
      const auto cIt = m_promoCandidates.find(candHash);
      if (cIt == m_promoCandidates.end() || cIt->second.phase != PromotionCandidate::Phase::Promoted) {
        return false;
      }
      RestClassState* c = (classQ != INT32_MIN)
        ? resolveRestClass(candHash, classQ, subId, false) : nullptr;
      if (cIt->second.routeHash != 0) {
        return c != nullptr && c->phase == RestClassState::Phase::Promoted;  // rest: needs own gate
      }
      return c == nullptr || !c->driftDemoted;  // non-rest: A by default, B only if drift-demoted
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
      // deformation. Genuine deformation is still caught by the promotion solve,
      // which un-promotes the CONTENT CLASS (direction 2) - the pin below reads
      // the slot's current class and releases the moment that class is no longer
      // Promoted, so real deform still drops the mesh to Path B.
      if (!skinned && ClusterLodOptions::Promotion::enable()
          && m_renderSystem != nullptr && m_renderSystem->hasGeneration()) {
        auto pinIt = m_promoSlotByBlas.find(blasEntry);
        // direction 2: the pin holds while the slot's CONTENT CLASS still routes
        // Path A (permissive for non-rest, gated for rest - see routesPathA). A
        // class that drift-demotes releases the pin the same frame; the slot then
        // re-routes through the establish path (to Path B) below.
        if (pinIt != m_promoSlotByBlas.end()
            && pinIt->second.geometryHash != 0
            && routesPathA(pinIt->second.geometryHash, pinIt->second.contentClassQ,
                           pinIt->second.classSubId)
            && pinIt->second.residentGeometryId != ~0u
            && pinIt->second.blasFrameCreated == blasEntry->frameCreated
            // content-rebind guard: a stable BlasEntry* can bind DIFFERENT content across
            // frames (draw-call cache material-match reuse) with frameCreated unchanged;
            // routing off the cached residentGeometryId would render the OLD content's Path A
            // clusters. Release the pin when the STABLE resolved candidate key no longer
            // matches the cached one (short-circuited so the key is only resolved for an
            // otherwise-valid pin). The establish path + getGeometryId reset then re-derive.
            && pinIt->second.geometryHash != 0
            && pinIt->second.geometryHash == resolvePromoCandidateKey(blasEntry->input.getGeometryData())) {
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
          // [ShapeClass] this instance's CONTENT CLASS decides its route: the
          // class's verdict (not a per-BlasEntry one - the binding is unstable),
          // and the class's residency when it owns its own rest reference.
          RestClassState* routeCls = nullptr;
          {
            const auto slotPre = m_promoSlotByBlas.find(blasEntry);
            if (slotPre != m_promoSlotByBlas.end()
                && slotPre->second.blasFrameCreated == blasEntry->frameCreated
                && slotPre->second.contentClassQ != INT32_MIN) {
              routeCls = resolveRestClass(geometryHash, slotPre->second.contentClassQ,
                                          slotPre->second.classSubId, false);
            }
          }
          // rest-referenced candidates render the clusters built from their CAPTURED
          // rest pose (space-tagged hash) - the CLASS's own one when it has it;
          // everything else uses the object-space id (the stable candidate key).
          const uint64_t residencyHash =
              (routeCls != nullptr && routeCls->phase == RestClassState::Phase::Promoted
               && routeCls->routeHash != 0)
            ? routeCls->routeHash
            : (candidate->second.routeHash != 0 ? candidate->second.routeHash : geometryHash);
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
                // (rest phase, pin) into a brand-new instance - reset everything
                // but keep the slot (its stale GPU temporal sample is one isolated
                // spike, which persistence now absorbs)
                slotIt->second.blasFrameCreated = blasEntry->frameCreated;
                slotIt->second.sweepPending = false;
                slotIt->second.eigenSuspect = false;
                slotIt->second.residentGeometryId = ~0u;
                slotIt->second.geometryHash = 0;
                slotIt->second.contentClassQ = INT32_MIN;  // fresh content - reclassify
                slotIt->second.classSubId = 0;
                slotIt->second.classSubId = 0;
                slotIt->second.lastClassifiedFrame = 0;
                routeCls = nullptr;  // the class was resolved from the OLD tenant's classification
              } else if (slotIt != m_promoSlotByBlas.end()
                         && slotIt->second.geometryHash != 0
                         && slotIt->second.geometryHash != geometryHash) {
                // [ShapeClass] CONTENT REBIND under a STABLE BlasEntry* (same frameCreated).
                // The draw-call cache reuses one BlasEntry across frames on a material match
                // even when the geometry differs (DrawCallCache::get loose match), and a
                // continuously-visible entry is never GC'd - so frameCreated is stamped ONCE
                // and the blasFrameCreated reset above NEVER fires for the ~4/34 slots that
                // carry 2 geom hashes under one pointer. Detect the genuine content change by
                // the STABLE resolved candidate key (resolvePromoCandidateKey - topology
                // resolved, immune to this game's per-frame asset-hash churn) changing, and
                // reset the content-derived state so the frozen class (updatePromotionStates)
                // re-derives for the NEW content instead of the pin/establish routing the OLD
                // content's stale Path A clusters. This IS the class freeze's release signal.
                slotIt->second.sweepPending = false;
                slotIt->second.eigenSuspect = false;
                slotIt->second.residentGeometryId = ~0u;
                slotIt->second.contentClassQ = INT32_MIN;  // new content - reclassify
                slotIt->second.classSubId = 0;
                slotIt->second.classSubId = 0;
                slotIt->second.lastClassifiedFrame = 0;
                // [ShapeClass] verify the handoff's ~4/34 "2 geom hashes under one BlasEntry"
                // claim and how often the freeze actually releases here. Logs the OLD->NEW
                // stable candidate key BEFORE the overwrite (blasFC constant proves the
                // recycled-BlasEntry reset above did NOT fire for this content change).
                Logger::info(str::format("[ShapeClass] slot ", slotIt->second.stateSlot,
                                         " CONTENT REBIND geom 0x", std::hex, slotIt->second.geometryHash,
                                         " -> 0x", geometryHash, std::dec,
                                         " (blasFC ", blasEntry->frameCreated,
                                         ", frame ", currentFrame, ") - class released"));
                slotIt->second.geometryHash = geometryHash;  // adopt the new content key
                routeCls = nullptr;  // the class was resolved from the OLD content's classification
              }
              // [ShapeClass] direction 2: routing follows the slot's CONTENT CLASS,
              // not a per-slot flag - so a slot multiplexing pieces routes by
              // whichever piece's class it currently holds instead of thrashing.
              // NON-rest content routes Path A by DEFAULT (immediately on candidate
              // promotion, no per-class gate latency) and drops to B only when the
              // class drift-demotes; REST content gates on class==Promoted. The
              // routesPathA applies the same rule the pin uses, re-resolving the
              // slot's class (so it reflects any content-rebind reset just above).
              const bool classGated = !routesPathA(geometryHash,
                                                   slotIt != m_promoSlotByBlas.end() ? slotIt->second.contentClassQ : INT32_MIN,
                                                   slotIt != m_promoSlotByBlas.end() ? slotIt->second.classSubId : 0);

              // [RouteTrace] per-frame Path-A-vs-Path-B verdict for the traced geom.
              // This is the VISIBLE pop the [DEFORMING] demote metric missed: capSig
              // jitter (~15%) crosses the fine content-class bands (class = round(
              // log2(capSig)*16) => 4.4% bands, see updatePromotionStates) so the
              // class reclassifies, and a non-Promoted class restGates the instance
              // to Path B mid-flight -> it pops A<->B every few frames. Logs the whole
              // capSig -> class -> restGate -> route chain so we can confirm the pop
              // and whether the capSig jitter is buffer-noise (join by slot+frame with
              // [BindTrace]'s captureVa). Traced hash only; self-limited line budget.
              {
                const std::string& rtHashStr = ClusterLodOptions::Promotion::dumpGeometryHash();
                if (!rtHashStr.empty() && slotIt != m_promoSlotByBlas.end()) {
                  uint64_t rtHash = 0;
                  try { rtHash = std::stoull(rtHashStr, nullptr, 16); } catch (...) { rtHash = 0; }
                  static uint32_t s_routeTraceBudget = 3000u;
                  if (rtHash != 0 && geometryHash == rtHash && s_routeTraceBudget > 0u) {
                    --s_routeTraceBudget;
                    const uint32_t rtSlot = slotIt->second.stateSlot;
                    const float rtCapSig = (m_promoStatesValid && rtSlot < m_promoStates.size())
                      ? m_promoStates[rtSlot].capSig : -1.0f;
                    const int32_t rtPhase = routeCls != nullptr ? int32_t(routeCls->phase) : -1;
                    const bool rtRouteA = !classGated;
                    Logger::info(str::format("[RouteTrace] geom 0x", std::hex, geometryHash, std::dec,
                                             " frame ", currentFrame, " slot ", rtSlot,
                                             " capSig ", rtCapSig,
                                             " classQ ", slotIt->second.contentClassQ,
                                             " subId ", slotIt->second.classSubId,
                                             " clsPhase ", rtPhase,
                                             " classGated ", classGated,
                                             " route ", (rtRouteA ? "PATH_A" : "PATH_B")));
                  }
                }
              }

              // [ClassAlias] confirm/deny capSig aliasing (the shape-signature decision).
              // For the traced candidate, dump each instance's CLASS identity (classQ +
              // subId), its capSig, its residual against the class reference, and a RAW
              // object-space shape descriptor (bbox extents ex,ey,ez). Read as: instances
              // that share (classQ, subId) but show DIFFERENT extents AND high residual =
              // the 1-D capSig band is grouping non-corresponding shapes under ONE
              // reference and the sibling chain is not splitting them -> the shape-signature
              // class key is warranted. blasEntry is LIVE here (draw thread), so reading its
              // bbox is safe. Raw values, no threshold; throttled per slot per 10 frames.
              {
                const std::string& caHashStr = ClusterLodOptions::Promotion::dumpGeometryHash();
                if (!caHashStr.empty() && slotIt != m_promoSlotByBlas.end()) {
                  uint64_t caHash = 0;
                  try { caHash = std::stoull(caHashStr, nullptr, 16); } catch (...) { caHash = 0; }
                  if (caHash != 0 && geometryHash == caHash) {
                    const uint32_t caSlot = slotIt->second.stateSlot;
                    static std::mutex s_caMutex;
                    static std::unordered_map<uint32_t, uint32_t> s_caLast;  // slot -> last logged frame
                    std::lock_guard<std::mutex> caLk(s_caMutex);
                    uint32_t& caLastFrame = s_caLast[caSlot];
                    if (caLastFrame == 0u || currentFrame - caLastFrame > 10u) {
                      caLastFrame = currentFrame;
                      const auto& caBox = blasEntry->input.getGeometryData().boundingBox;
                      const bool caValid = caBox.isValid();
                      const float caEx = caValid ? (caBox.maxPos.x - caBox.minPos.x) : 0.0f;
                      const float caEy = caValid ? (caBox.maxPos.y - caBox.minPos.y) : 0.0f;
                      const float caEz = caValid ? (caBox.maxPos.z - caBox.minPos.z) : 0.0f;
                      const float caCapSig = (m_promoStatesValid && caSlot < m_promoStates.size())
                        ? m_promoStates[caSlot].capSig : -1.0f;
                      const float caResid = (m_promoStatesValid && caSlot < m_promoStates.size())
                        ? m_promoStates[caSlot].residualRel : -1.0f;
                      Logger::info(str::format("[ClassAlias] geom 0x", std::hex, geometryHash, std::dec,
                                               " frame ", currentFrame, " slot ", caSlot,
                                               " classQ ", slotIt->second.contentClassQ,
                                               " subId ", slotIt->second.classSubId,
                                               " capSig ", caCapSig,
                                               " residual ", caResid,
                                               " ext (", caEx, ", ", caEy, ", ", caEz, ")"));
                    }
                  }
                }
              }

              // direction 2: route by the content class. A slot whose current
              // class is Promoted renders Path A; anything else falls through to
              // Path B below and keeps solving (buildPromotionEntries) so its
              // class can (re)promote. No per-slot demote flag is involved.
              if (slotIt != m_promoSlotByBlas.end() && !classGated) {
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
              // Under direction 2 the drop is either the content class not (yet)
              // Promoted - unclassified, still gating, or un-promoted by drift - or
              // the state-slot pool being exhausted. Names the last silent drop
              // between "routable" and the screen.
              {
                static std::mutex s_m2;
                static std::unordered_map<uint64_t, uint32_t> s_lastLogFrame;
                std::lock_guard<std::mutex> lk2(s_m2);
                uint32_t& last = s_lastLogFrame[geometryHash];
                if (currentFrame - last > 300u || last == 0u) {
                  last = currentFrame;
                  const char* why2 = classGated
                                   ? "content-class not Promoted ([ShapeClass] pending/gating/drift-demoted)"
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
        // [RestCapProbe] pair this copy with a SAME-FRAME snapshot of the solve
        // kernel's view of the same capture buffer: the slot's solve (dispatched
        // later this frame, in the same exec cmd stream) writes its sampled
        // capture positions to promoLastSampleBuffer, and the frameParams fill
        // later in dispatchBuild routes this frame's dump copy into our staging.
        // At drain, bit-comparing the two separates an UNFAITHFUL copy (tear/
        // ordering/rename - solve saw different bytes than we copied) from a
        // faithful copy of content that genuinely held that pose this frame.
        if (request.stateSlot != ~0u && !m_restCapProbeInFlight) {
          DxvkBufferCreateInfo sampleInfo;
          sampleInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          sampleInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
          sampleInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
          sampleInfo.size = VkDeviceSize(64) * 3 * sizeof(float);
          request.sampleStaging = m_device->createBuffer(sampleInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "promo rest-capture probe sample view");
          const DxvkBufferSliceHandle slice = request.sampleStaging->getSliceHandle();
          m_restCapProbeInFlight = true;
          m_restCapProbeSlotPending = request.stateSlot;
          m_restCapProbeTargetPending = slice.handle;
          m_restCapProbeTargetOffsetPending = slice.offset;
        }
        ++it;
        continue;
      }
      if (currentFrame - request.copyFrame < kReadbackLagFrames) {
        ++it;
        continue;
      }

      // copy retired: assemble the rest snapshot from the readback + retained topology
      if (request.sampleStaging != nullptr) {
        m_restCapProbeInFlight = false;  // [RestCapProbe] the armed probe drains with this request
      }
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
        // space-tagged rest hash: distinct clusters/.nvsngeo/residency identity.
        // Class-scoped references ([ShapeClass]) salt the class in so every
        // content class of one candidate gets its own clusters + residency.
        constexpr uint64_t kRestSpaceTag = 0x9E3779B97F4A7C15ull;
        uint64_t restHash = request.geometryHash ^ kRestSpaceTag;
        if (request.classQ != INT32_MIN) {
          restHash ^= (uint64_t(uint32_t(request.classQ)) * 0xBF58476D1CE4E5B9ull) | 1ull;
          // identity-by-fit siblings share classQ but hold DIFFERENT content -
          // each needs its own clusters/cache/residency identity (subId 0 keeps
          // the pre-sibling hash so existing caches stay valid)
          if (request.classSubId != 0) {
            restHash ^= (uint64_t(uint32_t(request.classSubId)) * 0x94D049BB133111EBull) | 1ull;
          }
        }
        restSnap.geometryHash = restHash;
        restSnap.promoKeyHash = request.geometryHash;
        restSnap.promoClassQ = request.classQ;
        restSnap.promoClassSubId = request.classSubId;
        restSnap.isRestCapture = true;
        restSnap.name = request.classQ != INT32_MIN
          ? (request.classSubId != 0
               ? str::format(topo.name, "_rest_q", request.classQ, "_s", request.classSubId)
               : str::format(topo.name, "_rest_q", request.classQ))
          : (topo.name + "_rest");
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

        // ---- [RestCapProbe] copy-fidelity + weld-structure verdict ----
        // Welds = exact-duplicate position triples. Affine-invariant: ANY faithful
        // single-pose capture of this content reproduces the base capture's weld
        // pattern, so fewer welded verts = per-vertex corruption or mid-deform
        // content - never a scaled/rotated sibling placement.
        {
          uint32_t weldGroups = 0, weldVerts = 0;
          {
            std::unordered_map<uint64_t, uint32_t> counts;
            counts.reserve(size_t(request.vertexCount) * 2);
            for (uint32_t v = 0; v < request.vertexCount; v++) {
              counts[XXH3_64bits(&restSnap.positions[size_t(v) * 3], 3 * sizeof(float))]++;
            }
            for (const auto& kv : counts) {
              if (kv.second >= 2) {
                weldGroups++;
                weldVerts += kv.second;
              }
            }
          }
          uint32_t baseGroups = ~0u, baseVerts = 0;
          {
            std::lock_guard<std::mutex> topoLock(m_promoTopologyMutex);
            const auto weldIt = m_promoTopologyByHash.find(request.geometryHash);
            if (weldIt != m_promoTopologyByHash.end()) {
              if (request.classQ == INT32_MIN) {
                // the BASE capture defines the weld baseline for later class captures
                weldIt->second.weldGroups = weldGroups;
                weldIt->second.weldVerts = weldVerts;
              }
              baseGroups = weldIt->second.weldGroups;
              baseVerts = weldIt->second.weldVerts;
            }
          }

          if (request.sampleStaging != nullptr) {
            const float* sv = (const float*) request.sampleStaging->mapPtr(0);
            if (sv != nullptr) {
              // Solve-view membership: the solve kernel and this request's transfer
              // copy read the SAME buffer range in the SAME frame's cmd stream, so a
              // faithful copy contains every solve-sampled position BIT-exactly.
              // Kernel writes lastSample[threadId] only for threadId < sampleCount,
              // so tail slots hold stale earlier frames - misses strictly AFTER the
              // matched prefix are expected; misses INSIDE the prefix are not.
              std::unordered_set<uint64_t> posSet;
              posSet.reserve(size_t(request.vertexCount) * 2);
              for (uint32_t v = 0; v < request.vertexCount; v++) {
                posSet.insert(XXH3_64bits(&restSnap.positions[size_t(v) * 3], 3 * sizeof(float)));
              }
              char svMap[65];
              uint32_t svMatches = 0, svMisses = 0, svZeros = 0;
              int32_t lastMatch = -1, firstMiss = -1;
              for (uint32_t i = 0; i < 64; i++) {
                const float* s = &sv[i * 3];
                if (s[0] == 0.0f && s[1] == 0.0f && s[2] == 0.0f) {
                  svMap[i] = 'z';
                  svZeros++;
                } else if (posSet.count(XXH3_64bits(s, 3 * sizeof(float))) != 0) {
                  svMap[i] = 'M';
                  svMatches++;
                  lastMatch = int32_t(i);
                } else {
                  svMap[i] = '?';
                  svMisses++;
                  if (firstMiss < 0) {
                    firstMiss = int32_t(i);
                  }
                }
              }
              svMap[64] = '\0';
              // hint only: ring-lagged sampleCount of the slot's recent solves
              const uint32_t sampleHint = (m_promoStatesValid && request.stateSlot < m_promoStates.size())
                ? ((m_promoStates[request.stateSlot].diagGuard >> 8) & 0xFFu) : 0u;
              Logger::info(str::format("[RestCapProbe] geometry 0x", std::hex, request.geometryHash, std::dec,
                                       " classQ ", request.classQ, " slot ", request.stateSlot,
                                       " copyFrame ", request.copyFrame, " verts ", request.vertexCount,
                                       " | solveView ", svMap,
                                       " (M ", svMatches, " / ? ", svMisses, " / z ", svZeros,
                                       ", sampleHint ", sampleHint, ")"));
              // raw detail for the first misses: distance to the nearest copied
              // position names the magnitude (tear ~ inter-frame motion; garbage ~ huge)
              uint32_t detailed = 0;
              for (uint32_t i = 0; i < 64 && detailed < 4; i++) {
                if (svMap[i] != '?') {
                  continue;
                }
                const float* s = &sv[i * 3];
                float bestD2 = std::numeric_limits<float>::max();
                for (uint32_t v = 0; v < request.vertexCount; v++) {
                  const float dx = restSnap.positions[size_t(v) * 3 + 0] - s[0];
                  const float dy = restSnap.positions[size_t(v) * 3 + 1] - s[1];
                  const float dz = restSnap.positions[size_t(v) * 3 + 2] - s[2];
                  bestD2 = std::min(bestD2, dx * dx + dy * dy + dz * dz);
                }
                Logger::info(str::format("[RestCapProbe] miss[", i, "] sv (", s[0], ", ", s[1], ", ", s[2],
                                         ") nearestCopyDist ", std::sqrt(bestD2)));
                detailed++;
              }
              Logger::info(str::format("[RestCapProbe] welds copy ", weldGroups, " groups / ", weldVerts,
                                       " verts | base ", int32_t(baseGroups), " groups / ",
                                       (baseGroups == ~0u ? -1 : int32_t(baseVerts)), " verts"));
              const bool headMiss = firstMiss >= 0 && firstMiss < lastMatch;
              const char* verdict =
                  svMatches == 0                                 ? "NO OVERLAP - copy and solve saw DIFFERENT content (rebind/full tear)"
                : headMiss                                       ? "UNFAITHFUL COPY - solve-view mismatch inside the sample prefix (tear/ordering)"
                : (baseGroups != ~0u && weldVerts < baseVerts)   ? "copy FAITHFUL to solve view; content WELD-SPLIT on copy frame (not at rest)"
                :                                                  "copy FAITHFUL to solve view; welds intact vs base";
              Logger::info(str::format("[RestCapProbe] verdict: ", verdict));
            } else {
              Logger::warn("[RestCapProbe] sample staging not mappable - probe skipped");
            }
          } else if (request.classQ != INT32_MIN) {
            // un-probed class capture (another probe held the single slot): the weld
            // comparison alone still separates corrupt from faithful
            Logger::info(str::format("[RestCapProbe] geometry 0x", std::hex, request.geometryHash, std::dec,
                                     " classQ ", request.classQ, " (no solve-view probe) welds copy ",
                                     weldGroups, " groups / ", weldVerts, " verts | base ",
                                     int32_t(baseGroups), " groups / ",
                                     (baseGroups == ~0u ? -1 : int32_t(baseVerts)), " verts"));
          }
        }

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
        // [CapSigDump] prefer a PROMOTED INSTANCE of the traced hash over the candidate:
        // the candidate is a probe-only slot with a stable capture, but the popping is on
        // instance slots whose captureVa ping-pongs (double buffer). Trace one of those so
        // the capSig-vertex dump compares the two ping-pong buffers of a real instance.
        if (dumpHash != 0) {
          for (const auto& kv : m_promoSlotByBlas) {
            if (kv.second.geometryHash == dumpHash && kv.second.contentClassQ != INT32_MIN) {
              frameParams.promotionDumpStateSlot = kv.second.stateSlot;
              break;
            }
          }
        }
      }
    }
    // [RestCapProbe] a rest-capture copy recorded THIS frame (processRestCaptureRequests
    // above) overrides the config-driven dump for one frame: the solve-view snapshot
    // must be of the SAME frame as the capture copy, and it lands in the request's own
    // staging - the internal ring (and the config [PromoDump] cadence) is untouched.
    if (m_restCapProbeSlotPending != ~0u) {
      frameParams.promotionDumpStateSlot = m_restCapProbeSlotPending;
      frameParams.promotionDumpTargetBuffer = m_restCapProbeTargetPending;
      frameParams.promotionDumpTargetOffset = m_restCapProbeTargetOffsetPending;
      m_restCapProbeSlotPending = ~0u;
      m_restCapProbeTargetPending = VK_NULL_HANDLE;
      m_restCapProbeTargetOffsetPending = 0;
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
