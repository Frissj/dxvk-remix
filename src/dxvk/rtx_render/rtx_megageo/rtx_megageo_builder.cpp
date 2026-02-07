/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_megageo_builder.h"
#include "../rtx_context.h"
#include "../rtx_camera.h"
#include "../../util/log/log.h"
#include "../../../util/util_error.h"  // For DxvkError exception handling
#include <algorithm>
#include <cmath>
#include <chrono>

// Enable chrono timing for performance profiling (set to 1 to enable)
#define RTXMG_CHRONO_TIMING 1
#include "nvrhi_adapter/nvrhi_dxvk_texture.h"
#include "nvrhi_adapter/nvrhi_dxvk_buffer.h"
#include "scene/rtxmg_scene.h"
#include "scene/instance.h"
#include "subdivision/subdivision_surface.h"
#include "subdivision/topology_cache.h"
#include "subdivision/shape.h"
#include "cluster_builder/fill_instance_descs_params.h"

// OpenSubdiv includes for topology building (using osd_lite like the sample)
#include "osd_lite/opensubdiv/far/topologyDescriptor.h"
#include "osd_lite/opensubdiv/far/topologyRefinerFactory.h"
#include "osd_lite/opensubdiv/sdc/types.h"
#include "osd_lite/opensubdiv/sdc/options.h"
#include "osd_lite/opensubdiv/tmr/topologyMap.h"
#include "osd_lite/opensubdiv/tmr/surfaceTableFactory.h"

#include "nvrhi_adapter/nvrhi_dxvk_command_list.h"

#include "rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_RTX_MEGAGEO_BUILDER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace dxvk {

// Wraps a SubdivisionSurface as a DxvkResource so it can be tracked on the
// DXVK command list. When the GPU fence signals completion, this holder is
// destroyed, which frees the SubdivisionSurface and all its GPU buffers.
class DxvkSubdSurfaceHolder : public DxvkResource {
public:
  DxvkSubdSurfaceHolder(std::unique_ptr<SubdivisionSurface> surface)
    : m_surface(std::move(surface)) {}
private:
  std::unique_ptr<SubdivisionSurface> m_surface;
};

  // Use OpenSubdiv namespace (from osd_lite, matching the sample)
  // Only use the versioned namespace to avoid ambiguity
  using namespace OpenSubdiv::OPENSUBDIV_VERSION;

  RtxMegaGeoBuilder::RtxMegaGeoBuilder(
    const Rc<DxvkDevice>& device,
    const Rc<RtxContext>& rtxContext)
    : m_device(device)
    , m_rtxContext(rtxContext)
  {
  }

  RtxMegaGeoBuilder::~RtxMegaGeoBuilder() {
    // Shut down worker threads
    if (!m_workerThreads.empty()) {
      RTXMG_LOG(str::format("RTX MegaGeo: Shutting down ", m_workerThreads.size(), " worker threads"));
      m_workerShouldExit.store(true);
      m_workerCV.notify_all();
      for (auto& thread : m_workerThreads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      m_workerThreads.clear();
      RTXMG_LOG("RTX MegaGeo: All worker threads shut down");
    }

    // Cleanup BLAS handles
    for (auto& [id, surface] : m_surfaces) {
      if (surface.blas != VK_NULL_HANDLE) {
        // BLAS cleanup will be handled by DXVK's resource management
        surface.blas = VK_NULL_HANDLE;
      }
    }

    // NVRHI adapter cleanup
    if (m_nvrhiDevice) {
      delete m_nvrhiDevice;
      m_nvrhiDevice = nullptr;
    }
  }

  bool RtxMegaGeoBuilder::initialize() {
    if (m_initialized) {
      Logger::warn("RtxMegaGeoBuilder::initialize() called multiple times");
      return true;
    }

    RTXMG_LOG("Initializing RTX Mega Geometry Builder...");

    // Create NVRHI adapter
    m_nvrhiDevice = new NvrhiDxvkDevice(
      m_device,
      m_rtxContext,  // RtxContext inherits from DxvkContext
      m_rtxContext);

    if (!m_nvrhiDevice) {
      Logger::err("Failed to create NVRHI device adapter");
      return false;
    }

    // Create NVRHI command list
    m_commandList = m_nvrhiDevice->createCommandList();
    if (!m_commandList) {
      Logger::err("Failed to create NVRHI command list");
      return false;
    }

    // Create ClusterAccelBuilder
    RTXMG_LOG("RTX MegaGeo: Creating ClusterAccelBuilder...");
    RTXMG_LOG("RTX MegaGeo: Note: Cluster acceleration requires VK_NV_cluster_acceleration_structure extension");
    RTXMG_LOG("RTX MegaGeo: This extension is only available on NVIDIA RTX GPUs with latest drivers");

    try {
      m_clusterBuilder = std::make_unique<ClusterAccelBuilder>(m_nvrhiDevice, m_rtxContext.ptr());
      RTXMG_LOG("RTX MegaGeo: ClusterAccelBuilder created successfully");
    } catch (const std::exception& e) {
      Logger::err(str::format("ClusterAccelBuilder creation failed: ", e.what()));
      Logger::err("RTX MegaGeo: This likely means VK_NV_cluster_acceleration_structure extension is not available");
      return false;
    }

    // Create ClusterAccels storage
    m_clusterAccels = std::make_unique<ClusterAccels>();

    // Create RTXMGScene for managing subdivision surfaces and instances
    m_scene = std::make_unique<RTXMGScene>(m_nvrhiDevice);
    m_scene->Initialize();

    // Create TopologyCache for each worker thread (thread-safe parallel processing)
    TopologyCache::Options topoCacheOptions{
      .isoLevelSharp = 3,
      .isoLevelSmooth = 2,
      .useTerminalNodes = true
    };
    m_topologyCaches.reserve(m_numWorkerThreads);
    for (uint32_t i = 0; i < m_numWorkerThreads; ++i) {
      m_topologyCaches.push_back(std::make_unique<TopologyCache>(topoCacheOptions));
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Scene and ", m_numWorkerThreads, " topology caches initialized"));

    // Create HIZ buffer (for visibility culling)
    // Note: HIZ buffer will be created on-demand with appropriate size when needed
    // The static Create method requires size parameters that aren't known at init time

    // Create Z buffer (depth buffer interface)
    // Note: Z buffer will be created on-demand when needed
    // The static Create method requires size parameters that aren't known at init time

    // Create scratch buffer for cluster operations
    // Start with a reasonable default size (16 MB), will grow if needed
    const uint64_t initialScratchSize = 16 * 1024 * 1024;
    nvrhi::BufferDesc scratchDesc;
    scratchDesc.byteSize = initialScratchSize;
    scratchDesc.debugName = "RTX MG Cluster Scratch";
    scratchDesc.canHaveUAVs = true;
    scratchDesc.canHaveRawViews = true;
    scratchDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    scratchDesc.keepInitialState = true;
    m_scratchBuffer = m_nvrhiDevice->createBuffer(scratchDesc);

    if (!m_scratchBuffer) {
      Logger::err("RTX MegaGeo: Failed to create scratch buffer");
      return false;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Created scratch buffer (", initialScratchSize / 1024, " KB)"));

    // Start async worker threads for subdivision surface creation
    RTXMG_LOG(str::format("RTX MegaGeo: Starting ", m_numWorkerThreads, " worker threads"));
    m_workerThreads.reserve(m_numWorkerThreads);
    for (uint32_t i = 0; i < m_numWorkerThreads; ++i) {
      m_workerThreads.emplace_back(&RtxMegaGeoBuilder::workerThreadFunc, this, i);
    }

    m_initialized = true;
    RTXMG_LOG("RTX Mega Geometry Builder initialized successfully");
    return true;
  }

  void RtxMegaGeoBuilder::workerThreadFunc(uint32_t threadIndex) {
    RTXMG_LOG(str::format("RTX MegaGeo: Worker thread ", threadIndex, " started"));

    while (!m_workerShouldExit.load()) {
      PendingSurface pending;
      bool hasPending = false;

      {
        std::unique_lock<std::mutex> lock(m_pendingMutex);
        m_workerCV.wait(lock, [this] {
          return !m_pendingSurfaces.empty() || m_workerShouldExit.load();
        });

        if (m_workerShouldExit.load() && m_pendingSurfaces.empty()) {
          break;
        }

        if (!m_pendingSurfaces.empty()) {
          pending = std::move(m_pendingSurfaces.front());
          m_pendingSurfaces.pop();
          hasPending = true;
        }
      }

      if (hasPending) {
        uint32_t numFaces = pending.shape ? pending.shape->nvertsPerFace.size() : 0;
        RTXMG_LOG(str::format("RTX MegaGeo: Worker[", threadIndex, "] processing surface ", pending.surfaceId, " (", numFaces, " faces)"));

        try {
          // Just validate and return the Shape - SubdivisionSurface will be created on main thread
          // The Shape object already contains all the topology data needed
          RTXMG_LOG(str::format("RTX MegaGeo: Worker[", threadIndex, "] completed surface ", pending.surfaceId));

          // Move to completed queue
          {
            std::lock_guard<std::mutex> lock(m_completedMutex);
            m_completedSurfaces.push({
              pending.surfaceId,
              std::move(pending.shape),
              pending.topologyHash
            });
          }
        } catch (const std::exception& e) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to process surface ", pending.surfaceId, ": ", e.what()));
        } catch (...) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to process surface ", pending.surfaceId, ": unknown exception"));
        }
      }
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Worker thread ", threadIndex, " exiting"));
  }

  bool RtxMegaGeoBuilder::createSubdivisionSurface(
    const SubdivisionSurfaceDesc& desc,
    uint32_t& surfaceId)
  {
    if (!m_initialized) {
      Logger::err("RtxMegaGeoBuilder not initialized - call initialize() first");
      return false;
    }

    // Validate input
    if (desc.numFaces == 0 || desc.numVertices == 0) {
      Logger::err("Invalid subdivision surface: numFaces or numVertices is zero");
      return false;
    }

    if (!desc.faceVertexIndices || !desc.controlPoints) {
      Logger::err("Invalid subdivision surface: missing required data");
      return false;
    }

    // Allocate surface ID
    surfaceId = m_nextSurfaceId++;

    // Convert descriptor to Shape format
    auto shape = convertDescToShape(desc);
    if (!shape) {
      Logger::err(str::format("Failed to convert descriptor to Shape for surface ", surfaceId));
      return false;
    }

    // Queue for async creation on worker thread
    RTXMG_LOG(str::format("RTX MegaGeo: Queuing surface ", surfaceId, " for async creation (", desc.numFaces, " faces)"));

    {
      std::lock_guard<std::mutex> lock(m_pendingMutex);
      m_pendingSurfaces.push({
        surfaceId,
        std::move(shape),
        0 // topologyHash - caller can set this if needed
      });
    }
    m_workerCV.notify_one();

    // Create placeholder entry
    RTXMGSubdivisionSurfaceEntry surface;
    // Store only value-type data (raw pointers in desc become dangling after caller returns)
    surface.debugName = desc.debugName ? desc.debugName : "";
    surface.numVertices = desc.numVertices;
    surface.numFaces = desc.numFaces;
    surface.isolationLevel = desc.isolationLevel;
    surface.tessellationScale = desc.tessellationScale;
    surface.isDirty = true;
    surface.isReady = false; // Will be set to true when worker completes
    surface.m_hasDisplacementMaterial = desc.enableDisplacement;
    surface.displacementScale = desc.displacementScale;

    // Store placeholder in map
    m_surfaces[surfaceId] = std::move(surface);

    RTXMG_LOG(str::format("Queued subdivision surface ", surfaceId,
                             " for async creation (", desc.numFaces, " faces, ",
                             desc.numVertices, " vertices)"));

    return true;
  }

  bool RtxMegaGeoBuilder::updateSubdivisionSurface(
    uint32_t surfaceId,
    const SubdivisionSurfaceDesc& desc)
  {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      Logger::err(str::format("Surface ", surfaceId, " not found"));
      return false;
    }

    RTXMGSubdivisionSurfaceEntry& surface = it->second;

    surface.isDirty = true;
    surface.isReady = false;

    // Recreate Shape and SubdivisionSurface with updated data
    auto shape = convertDescToShape(desc);
    if (!shape) {
      Logger::err(str::format("Failed to convert descriptor to Shape for surface ", surfaceId));
      return false;
    }

    std::vector<std::unique_ptr<Shape>> keyFrames;
    surface.subdivSurface = std::make_unique<SubdivisionSurface>(
      *m_topologyCaches[0],
      std::move(shape),
      keyFrames,
      nullptr,
      m_commandList.Get());

    RTXMG_LOG(str::format("Updated subdivision surface ", surfaceId));
    return true;
  }

  void RtxMegaGeoBuilder::removeSubdivisionSurface(uint32_t surfaceId) {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return;
    }

    RTXMGSubdivisionSurfaceEntry& surface = it->second;

    // Cleanup BLAS
    if (surface.blas != VK_NULL_HANDLE) {
      // BLAS cleanup handled by DXVK
      surface.blas = VK_NULL_HANDLE;
    }

    // Null out the raw pointer in m_subdMeshes to prevent dangling pointer access.
    // The m_surfaceToMeshIndex mapping was already erased by the caller, so no new
    // instances will reference this mesh index. But we null it for safety.
    auto meshIt = m_surfaceToMeshIndex.find(surfaceId);
    if (meshIt != m_surfaceToMeshIndex.end() && m_scene) {
      m_scene->NullifySubdMesh(meshIt->second);
    }

    // Move SubdivisionSurface to pending list. In buildClusterBlas, these will be
    // tracked on the DXVK command list so the GPU fence keeps them alive until all
    // in-flight GPU work completes. This replaces frame-counting which gets out of
    // sync with the GPU under heavy cluster loads.
    if (surface.subdivSurface) {
      m_pendingGpuDestructions.push_back(std::move(surface.subdivSurface));
      Logger::info(str::format("RTX MegaGeo: Surface ", surfaceId,
          " queued for GPU fence-based deferred destruction"));
    }

    m_surfaces.erase(it);
    RTXMG_LOG(str::format("Removed subdivision surface ", surfaceId));
  }

  bool RtxMegaGeoBuilder::buildClusterBlas(
    const Rc<RtxContext>& context,
    const Rc<DxvkImageView>& depthBuffer,
    const RtCamera& rtCamera,
    const std::unordered_map<uint32_t, Matrix4>& instanceTransforms)
  {
    // Store instance transforms for use when setting localToWorld
    m_instanceTransforms = instanceTransforms;
    static uint32_t s_frameCounter = 0;
    s_frameCounter++;
    RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - FRAME ", s_frameCounter, " Entry"));

    // Update last-seen frame for surfaces that have transforms this frame
    for (const auto& [surfaceId, _] : instanceTransforms) {
      m_surfaceLastSeenFrame[surfaceId] = s_frameCounter;
    }

    // Prune surfaces that haven't had a transform for 120 frames
    {
      constexpr uint32_t kStaleFrameThreshold = 120;
      std::vector<uint32_t> staleSurfaceIds;
      for (const auto& [surfaceId, surface] : m_surfaces) {
        auto it = m_surfaceLastSeenFrame.find(surfaceId);
        uint32_t lastSeen = (it != m_surfaceLastSeenFrame.end()) ? it->second : 0;
        if (s_frameCounter > lastSeen + kStaleFrameThreshold) {
          staleSurfaceIds.push_back(surfaceId);
        }
      }
      for (uint32_t surfaceId : staleSurfaceIds) {
        RTXMG_LOG(str::format("RTX MegaGeo: Pruning stale surface ", surfaceId,
            " (last seen frame ", m_surfaceLastSeenFrame[surfaceId], ", current ", s_frameCounter, ")"));
        m_surfaceToMeshIndex.erase(surfaceId);
        m_surfaceLastSeenFrame.erase(surfaceId);
        removeSubdivisionSurface(surfaceId);
      }
      if (!staleSurfaceIds.empty()) {
        Logger::info(str::format("RTX MegaGeo: Pruned ", staleSurfaceIds.size(), " stale surfaces, ", m_surfaces.size(), " remaining"));
      }
    }

    // Track pending SubdivisionSurface destructions on the DXVK command list.
    // The GPU fence will hold them alive until all in-flight GPU work completes.
    if (!m_pendingGpuDestructions.empty()) {
      Rc<DxvkCommandList> cmdList = context->getCommandList();
      for (auto& surface : m_pendingGpuDestructions) {
        Rc<DxvkResource> holder = new DxvkSubdSurfaceHolder(std::move(surface));
        cmdList->trackResource<DxvkAccess::Read>(std::move(holder));
      }
      Logger::info(str::format("RTX MegaGeo: Tracked ", m_pendingGpuDestructions.size(),
          " surfaces on GPU command list for fence-based destruction"));
      m_pendingGpuDestructions.clear();
    }

    // Reset scratch buffers at start of frame - DXVK ensures GPU is done with previous frame
    if (m_commandList) {
      RTXMG_LOG(str::format("RTX MegaGeo: FRAME ", s_frameCounter, " - calling clearState"));
      m_commandList->clearState();
      RTXMG_LOG(str::format("RTX MegaGeo: FRAME ", s_frameCounter, " - clearState done"));
    }

    if (!m_initialized) {
      Logger::err("RtxMegaGeoBuilder not initialized");
      return false;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - Surface count: ", m_surfaces.size()));

    if (m_surfaces.empty()) {
      // No surfaces to tessellate
      return true;
    }

    // Update tessellation camera from RTX Remix camera
    // Use actual world space camera position for distance-based LOD calculations
    // Control points are stored in world space, so we need world space camera position
    dxvk::Vector3 camPos = rtCamera.getPosition(true);
    Vector3 actualCameraPos(camPos.x, camPos.y, camPos.z);
    m_tessellationCamera.SetEye(actualCameraPos);

    // Extract forward direction from view matrix for lookat
    Matrix4d worldToView = rtCamera.getWorldToView(true);
    float fwdX = -static_cast<float>(worldToView[0][2]);
    float fwdY = -static_cast<float>(worldToView[1][2]);
    float fwdZ = -static_cast<float>(worldToView[2][2]);
    Vector3 forward(fwdX, fwdY, fwdZ);
    m_tessellationCamera.SetLookat(actualCameraPos + forward);
    float upX = static_cast<float>(worldToView[0][1]);
    float upY = static_cast<float>(worldToView[1][1]);
    float upZ = static_cast<float>(worldToView[2][1]);
    Vector3 up(upX, upY, upZ);
    m_tessellationCamera.SetUp(up);
    // Extract FOV and aspect from projection matrix
    Matrix4d projMat = rtCamera.getViewToProjection();
    float proj11 = static_cast<float>(projMat[1][1]);
    float proj00 = static_cast<float>(projMat[0][0]);
    float fovY = 2.0f * std::atan(1.0f / proj11) * (180.0f / 3.14159265f);
    float aspectRatio = proj11 / proj00;
    m_tessellationCamera.SetFovY(fovY);
    m_tessellationCamera.SetAspectRatio(aspectRatio);
    // Get near/far planes directly from RtCamera - use actual game values
    float zNear = rtCamera.getNearPlane();
    float zFar = rtCamera.getFarPlane();
    // Only prevent degenerate values that would break the projection matrix
    if (zNear <= 0.0f) zNear = 0.001f;  // Tiny near plane to support close objects
    if (zFar <= zNear) zFar = 1000000.0f;  // Very large far plane - no artificial limit
    m_tessellationCamera.SetNear(zNear);
    m_tessellationCamera.SetFar(zFar);

    // CRITICAL: Initialize cluster templates BEFORE binding any image views (like HiZ)
    // The sync Downloads in template initialization close/reopen the command list
    // which destroys any bound resources. Must happen before updateHiZBuffer!
    {
      uint32_t maxGeometryCountPerMesh = m_scene ?
          static_cast<uint32_t>(m_scene->GetSceneGraph()->GetMaxGeometryCountPerMesh()) : 1;
      RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - Ensuring templates initialized, maxGeoPerMesh=", maxGeometryCountPerMesh));
      m_clusterBuilder->EnsureTemplatesInitialized(maxGeometryCountPerMesh, m_commandList.Get());
      RTXMG_LOG("RTX MegaGeo: buildClusterBlas - Templates initialized");
    }

    // HiZ culling DISABLED: With s_megaGeoOnly=true, only cluster geometry produces depth.
    // This creates a feedback loop where HiZ oscillates between aggressive/no culling each frame,
    // causing geometry to appear and disappear randomly.
    // TODO: Re-enable when non-cluster geometry also provides depth, or use a rasterization pre-pass.
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - HiZ culling disabled (feedback loop prevention)");

    // Configure tessellator
    TessellatorConfig config;
    config.enableVertexNormals = true;
    config.enableLogging = false;  // DISABLED - causes GPU readback stalls (~1 second per frame)
    config.zbuffer = nullptr;  // Disable HiZ culling to prevent oscillation
    config.camera = &m_tessellationCamera;
    config.isolationLevel = TessellatorConfig::kMaxIsolationLevel;  // Match sample: use max isolation level (6) for correct patch point evaluation
    config.disableSubdivision = false;

    // Dynamic memory sizing based on GPU-reported demand from previous frame.
    //
    // The GPU shader (compute_cluster_tiling.hlsl) handles overflow gracefully:
    //   - desiredClusters: unclamped atomic counter tracking ACTUAL demand
    //   - clusters: clamped counter that stays within maxClusters
    //   - When desired > max, excess clusters return early (no crash, no OOB)
    //
    // Strategy: grow-only high-water mark. We never shrink allocations because:
    //   1. Shrinking causes constant reallocation churn (buffer destroy/create every frame)
    //   2. The peak demand for this game is tiny vs the 4GB sample defaults
    //   3. Reallocation triggers driver overhead and risks use-after-free with deferred destruction
    {
      constexpr uint32_t kMaxClusters = 1u << 21;          // 2M (sample default cap)
      constexpr size_t kMaxVertexBytes = 1024ull << 20;     // 1GB
      constexpr size_t kMaxClasBytes = 3076ull << 20;       // 3GB
      constexpr uint32_t kInitialClusters = 1u << 16;       // 64K (conservative first frame)
      constexpr size_t kInitialVertexBytes = 64ull << 20;    // 64MB
      constexpr size_t kInitialClasBytes = 128ull << 20;     // 128MB
      constexpr float kHeadroom = 2.0f;

      // Compute demand-based target from previous frame's stats
      uint32_t demandClusters = kInitialClusters;
      size_t demandClasBytes = kInitialClasBytes;
      size_t demandVertexBytes = kInitialVertexBytes;

      if (m_clusterStats.desired.m_numClusters > 0) {
        demandClusters = uint32_t(std::min(double(kMaxClusters),
            double(m_clusterStats.desired.m_numClusters) * kHeadroom));
        demandClasBytes = size_t(std::min(double(kMaxClasBytes),
            double(m_clusterStats.desired.m_clasSize) * kHeadroom));
        demandVertexBytes = size_t(std::min(double(kMaxVertexBytes),
            double(m_clusterStats.desired.m_vertexBufferSize) * kHeadroom));
      }

      // Grow-only: high-water mark never shrinks, so buffers are allocated once and reused
      m_hwmClusters = std::max(m_hwmClusters, demandClusters);
      m_hwmClasBytes = std::max(m_hwmClasBytes, demandClasBytes);
      m_hwmVertexBytes = std::max(m_hwmVertexBytes, demandVertexBytes);

      config.memorySettings.maxClusters = std::min(kMaxClusters, m_hwmClusters);
      config.memorySettings.clasBufferBytes = std::min(kMaxClasBytes, m_hwmClasBytes);
      config.memorySettings.vertexBufferBytes = std::min(kMaxVertexBytes, m_hwmVertexBytes);
    }

    // Set viewport size from depth buffer - CRITICAL for tessellation rate calculations
    // Without this, viewportSize is {0,0} and edge tessellation rates are all 0, causing clusters=0
    if (depthBuffer != nullptr) {
      VkExtent3D extent = depthBuffer->mipLevelExtent(0);
      config.viewportSize = { extent.width, extent.height };
      RTXMG_LOG(str::format("RTX MegaGeo: viewportSize set to ", extent.width, "x", extent.height));
    } else {
      // Fallback to a reasonable default if no depth buffer
      config.viewportSize = { 1920, 1080 };
      RTXMG_LOG("RTX MegaGeo: No depth buffer, using default viewportSize 1920x1080");
    }

    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - About to call BuildAccel");

    // Initialize TopologyCache GPU buffers (plansBuffer, subpatchTreesArraysBuffer, etc.)
    // This must be called before BuildAccel to ensure TopologyMap buffers are available
    // for FillInstanceClusters shader bindings
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - Initializing TopologyCache device data");
    m_topologyCaches[0]->InitDeviceData(m_commandList.Get());
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - TopologyCache device data initialized");

    // PROPER FIX: Rebuild instance list each frame from active transforms
    // This ensures only surfaces with valid transforms this frame are rendered
    // Prevents "sticking to screen" bug from identity transforms on inactive surfaces
    RTXMG_LOG(str::format("RTX MegaGeo: Rebuilding instances - m_instanceTransforms.size()=", m_instanceTransforms.size(),
        " m_surfaceToMeshIndex.size()=", m_surfaceToMeshIndex.size()));

    if (m_scene) {
      // Clear instances from previous frame
      m_scene->ClearInstances();
      m_surfaceToInstanceIndex.clear();

      uint32_t instancesCreated = 0;
      uint32_t meshesNotReady = 0;
      uint32_t meshesSkippedDegenerate = 0;

      // Rebuild instances only for surfaceIds that have transforms this frame
      // IMPORTANT: Sort by surfaceId for deterministic instance ordering.
      // When dirty tracking skips BuildAccel, m_surfaceToInstanceIndex must match
      // the blasPtrsBuffer from the previous BuildAccel. Unordered_map iteration
      // order is nondeterministic, so we sort to ensure consistent instance indices.
      std::vector<uint32_t> sortedSurfaceIds;
      sortedSurfaceIds.reserve(m_instanceTransforms.size());
      for (const auto& [surfaceId, _] : m_instanceTransforms) {
        sortedSurfaceIds.push_back(surfaceId);
      }
      std::sort(sortedSurfaceIds.begin(), sortedSurfaceIds.end());

      for (uint32_t surfaceId : sortedSurfaceIds) {
        const Matrix4& transform = m_instanceTransforms[surfaceId];
        auto meshIt = m_surfaceToMeshIndex.find(surfaceId);
        if (meshIt == m_surfaceToMeshIndex.end()) {
          // Mesh not ready yet (async creation in progress)
          meshesNotReady++;
          continue;
        }

        uint32_t meshIndex = meshIt->second;

        // Verify mesh exists
        if (meshIndex >= m_scene->GetSubdMeshes().size()) {
          Logger::warn(str::format("RTX MegaGeo: Invalid meshIndex ", meshIndex, " for surfaceId ", surfaceId));
          continue;
        }

        // Check for degenerate/invalid AABB that could cause GPU crashes
        SubdivisionSurface* meshForCheck = m_scene->GetSubdMeshes()[meshIndex];
        if (meshForCheck) {
          const box3& aabb = meshForCheck->m_aabb;
          float extentX = aabb.m_maxs[0] - aabb.m_mins[0];
          float extentY = aabb.m_maxs[1] - aabb.m_mins[1];
          float extentZ = aabb.m_maxs[2] - aabb.m_mins[2];
          bool isDegenerate = (extentX <= 0.0f || extentY <= 0.0f || extentZ <= 0.0f);
          bool isInvalid = std::isnan(aabb.m_mins[0]) || std::isnan(aabb.m_maxs[0]) ||
                           std::isinf(aabb.m_mins[0]) || std::isinf(aabb.m_maxs[0]);
          if (isDegenerate || isInvalid) {
            meshesSkippedDegenerate++;
            continue;  // Skip to prevent GPU crashes
          }
        }

        // Create instance for this surface with the proper transform
        Instance instance;
        instance.meshID = meshIndex;
        instance.localToWorld = transform;  // Use actual transform, NOT identity

        // Create MeshInfo for instance
        auto meshInfo = std::make_shared<MeshInfo>();
        SubdivisionSurface* mesh = m_scene->GetSubdMeshes()[meshIndex];
        const Shape* shape = mesh ? mesh->GetShape() : nullptr;
        meshInfo->name = shape ? shape->filepath.string() : "RTXMGSurface";
        meshInfo->geometryCount = 1;

        auto geomInfo = std::make_shared<GeometryInfo>();
        geomInfo->globalGeometryIndex = meshIndex;
        geomInfo->name = meshInfo->name;
        meshInfo->geometries.push_back(geomInfo);

        instance.meshInstance = std::make_shared<MeshInstance>(meshInfo);

        // Record mapping BEFORE adding (so instanceIndex is correct)
        uint32_t instanceIndex = static_cast<uint32_t>(m_scene->GetSubdMeshInstances().size());
        m_surfaceToInstanceIndex[surfaceId] = instanceIndex;

        m_scene->AddInstance(instance);
        instancesCreated++;

        // Log first few for debugging - show full matrix to verify layout
#if RTXMG_VERBOSE_LOGGING
        if (instancesCreated <= 3) {
          RTXMG_LOG(str::format("RTX MegaGeo: Created instance surfaceId=", surfaceId,
              " meshIdx=", meshIndex, " instanceIdx=", instanceIndex));
          // Log full matrix to understand layout (row-major: row3 = translation, column-major: col3 = translation)
          RTXMG_LOG(str::format("  row0=(", transform.data[0][0], ",", transform.data[0][1], ",", transform.data[0][2], ",", transform.data[0][3], ")"));
          RTXMG_LOG(str::format("  row1=(", transform.data[1][0], ",", transform.data[1][1], ",", transform.data[1][2], ",", transform.data[1][3], ")"));
          RTXMG_LOG(str::format("  row2=(", transform.data[2][0], ",", transform.data[2][1], ",", transform.data[2][2], ",", transform.data[2][3], ")"));
          RTXMG_LOG(str::format("  row3=(", transform.data[3][0], ",", transform.data[3][1], ",", transform.data[3][2], ",", transform.data[3][3], ")"));
          // Also show what col3 looks like (in case it's column-major)
          RTXMG_LOG(str::format("  col3=(", transform.data[0][3], ",", transform.data[1][3], ",", transform.data[2][3], ",", transform.data[3][3], ")"));
        }
#endif
      }

      RTXMG_LOG(str::format("RTX MegaGeo: Rebuilt ", instancesCreated, " instances (",
          meshesNotReady, " not ready, ", meshesSkippedDegenerate, " skipped degenerate)"));
    } else {
      RTXMG_LOG("RTX MegaGeo: Scene is null, cannot rebuild instances");
    }

    // Dirty tracking: skip expensive BuildAccel if nothing changed
    {
      bool needsRebuild = m_forceRebuild;

      if (!needsRebuild) {
        const float cameraPosThreshold = 0.01f;
        const float cameraFwdThreshold = 0.001f;
        const float fovThreshold = 0.001f;

        float posDelta = length(actualCameraPos - m_prevCameraPosition);
        float fwdDelta = length(forward - m_prevCameraForward);

        if (posDelta > cameraPosThreshold ||
            fwdDelta > cameraFwdThreshold ||
            std::abs(fovY - m_prevFovY) > fovThreshold) {
          needsRebuild = true;
        }
      }

      if (!needsRebuild) {
        if (m_instanceTransforms.size() != m_prevBuildTransforms.size()) {
          needsRebuild = true;
        } else {
          for (const auto& [surfaceId, transform] : m_instanceTransforms) {
            auto it = m_prevBuildTransforms.find(surfaceId);
            if (it == m_prevBuildTransforms.end()) {
              needsRebuild = true;
              break;
            }
            const auto& prev = it->second;
            bool same = true;
            for (int r = 0; r < 4 && same; r++)
              for (int c = 0; c < 4 && same; c++)
                if (std::abs(transform.data[r][c] - prev.data[r][c]) > 1e-6f)
                  same = false;
            if (!same) { needsRebuild = true; break; }
          }
        }
      }

      m_prevCameraPosition = actualCameraPos;
      m_prevCameraForward = forward;
      m_prevFovY = fovY;
      m_prevBuildTransforms = m_instanceTransforms;
      m_forceRebuild = false;

      if (!needsRebuild) {
        RTXMG_LOG("RTX MegaGeo: Skipping BuildAccel - nothing changed");
        return true;
      }
    }

    // Chrono timing for BuildAccel
#if RTXMG_CHRONO_TIMING
    static float s_buildTimeMs = 0.0f;
    auto buildStart = std::chrono::high_resolution_clock::now();
#endif

    try {
      // Actually call ClusterAccelBuilder::BuildAccel
      // The command list wraps RTX Remix's DxvkContext, which is always "open"
      // Commands are recorded immediately and will be submitted by the rendering pipeline
      // Do NOT call open() or close() here - let RTX Remix manage command submission
      m_clusterBuilder->BuildAccel(*m_scene, config, *m_clusterAccels, m_clusterStats, m_frameIndex++, m_commandList.Get());
      RTXMG_LOG("RTX MegaGeo: buildClusterBlas - BuildAccel completed");
    } catch (const dxvk::DxvkError& e) {
      // DxvkError doesn't inherit from std::exception, so catch it separately
      Logger::err(str::format("RTX MegaGeo: Failed to build cluster BLAS: ", e.message()));
      Logger::err("RTX MegaGeo: This is likely a memory allocation failure - try reducing memory settings");
      return false;
    } catch (const std::exception& e) {
      Logger::err(str::format("RTX MegaGeo: Failed to build cluster BLAS: ", e.what()));
      Logger::err("RTX MegaGeo: This likely means VK_NV_cluster_acceleration_structure extension is not available");
      Logger::err("RTX MegaGeo: Cluster-based tessellation requires NVIDIA RTX GPU with latest drivers");
      return false;
    } catch (...) {
      Logger::err("RTX MegaGeo: Failed to build cluster BLAS: unknown error");
      return false;
    }

#if RTXMG_CHRONO_TIMING
    auto buildEnd = std::chrono::high_resolution_clock::now();
    auto buildDuration = std::chrono::duration_cast<std::chrono::microseconds>(buildEnd - buildStart);
    float frameTimeMs = buildDuration.count() * 0.001f;
    s_buildTimeMs = s_buildTimeMs * 0.9f + frameTimeMs * 0.1f; // Smoothed average
    RTXMG_LOG(str::format(">>> RTXMG CHRONO: BuildAccel frame=", frameTimeMs, "ms smoothed=", s_buildTimeMs, "ms"));
#endif

    // Mark all surfaces as ready
    // Note: BLAS addresses will be patched directly on GPU by AccelManager::patchClusterBlasAddresses()
    // The blasPtrsBuffer contains GPU-side addresses that will be copied to the instance buffer
    if (m_clusterAccels->blasBuffer) {
      RTXMG_LOG("RTX MegaGeo: Cluster BLAS built successfully");
      RTXMG_LOG(str::format("RTX MegaGeo: blasPtrsBuffer available: ", (m_clusterAccels->blasPtrsBuffer.Get() != nullptr)));

      // Log buffer sizes for debugging GPU crashes
      RTXMG_LOG(str::format("RTX MegaGeo: Buffer sizes - clusterShadingData: ",
          m_clusterAccels->clusterShadingDataBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterShadingDataBuffer.GetNumElements(), " elements)"));
      RTXMG_LOG(str::format("RTX MegaGeo: Buffer sizes - clusterVertexPositions: ",
          m_clusterAccels->clusterVertexPositionsBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterVertexPositionsBuffer.GetNumElements(), " elements)"));
      RTXMG_LOG(str::format("RTX MegaGeo: Buffer sizes - clusterVertexNormals: ",
          m_clusterAccels->clusterVertexNormalsBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterVertexNormalsBuffer.GetNumElements(), " elements)"));
      RTXMG_LOG(str::format("RTX MegaGeo: numClusters=", m_clusterStats.allocated.m_numClusters,
          " numTriangles=", m_clusterStats.allocated.m_numTriangles));

      // CRITICAL DEBUG: Log desired vs allocated to understand if tessellation is producing geometry
      Logger::info(str::format("RTX MegaGeo STATS: desired clusters=", m_clusterStats.desired.m_numClusters,
          " tris=", m_clusterStats.desired.m_numTriangles,
          " | allocated clusters=", m_clusterStats.allocated.m_numClusters,
          " tris=", m_clusterStats.allocated.m_numTriangles));
      Logger::info(str::format("RTX MegaGeo STATS: desired blasSize=", m_clusterStats.desired.m_blasSize,
          " clasSize=", m_clusterStats.desired.m_clasSize,
          " vertBufSize=", m_clusterStats.desired.m_vertexBufferSize));
      Logger::info(str::format("RTX MegaGeo STATS: allocated blasSize=", m_clusterStats.allocated.m_blasSize,
          " clasSize=", m_clusterStats.allocated.m_clasSize,
          " vertBufSize=", m_clusterStats.allocated.m_vertexBufferSize));
      if (m_clusterStats.allocated.m_numTriangles == 0) {
        Logger::err("RTX MegaGeo STATS: *** ZERO TRIANGLES PRODUCED! Tessellation is not generating geometry ***");
        Logger::err(str::format("RTX MegaGeo STATS: Config: tessMode=", (int)config.tessMode,
            " isolationLevel=", config.isolationLevel,
            " fineTessRate=", config.fineTessellationRate,
            " coarseTessRate=", config.coarseTessellationRate,
            " viewportSize=", config.viewportSize.x, "x", config.viewportSize.y,
            " disableSubdiv=", config.disableSubdivision ? "YES" : "NO"));
        Logger::err(str::format("RTX MegaGeo STATS: Scene: meshes=", m_scene ? m_scene->GetSubdMeshes().size() : 0,
            " instances=", m_scene ? m_scene->GetSubdMeshInstances().size() : 0));
      }

      // DIAGNOSTIC: Log per-surface info to identify rendering issues
      RTXMG_LOG("=== RTXMG SURFACE DIAGNOSTICS ===");
      for (auto& [id, surface] : m_surfaces) {
        if (surface.subdivSurface) {
          const Shape* shape = surface.subdivSurface->GetShape();
          const box3& aabb = surface.subdivSurface->m_aabb;
          uint32_t numSurfaces = surface.subdivSurface->SurfaceCount();
          uint32_t numVertices = surface.subdivSurface->NumVertices();

          // Check if surface has valid geometry
          bool hasValidGeometry = (numSurfaces > 0 && numVertices > 0);

          // Check AABB validity - look for degenerate or suspicious bounds
          float extentX = aabb.m_maxs[0] - aabb.m_mins[0];
          float extentY = aabb.m_maxs[1] - aabb.m_mins[1];
          float extentZ = aabb.m_maxs[2] - aabb.m_mins[2];
          bool aabbDegenerate = (extentX <= 0.0f || extentY <= 0.0f || extentZ <= 0.0f);
          bool aabbHuge = (extentX > 10000.0f || extentY > 10000.0f || extentZ > 10000.0f);
          bool aabbTiny = (extentX < 0.001f && extentY < 0.001f && extentZ < 0.001f);

          // Check for NaN/Inf in AABB
          bool aabbInvalid = std::isnan(aabb.m_mins[0]) || std::isnan(aabb.m_maxs[0]) ||
                            std::isinf(aabb.m_mins[0]) || std::isinf(aabb.m_maxs[0]);

          // Check transform for this surface
          auto transformIt = m_instanceTransforms.find(id);
          bool hasTransform = (transformIt != m_instanceTransforms.end());

          // Log warning for problematic surfaces
          if (!hasValidGeometry || aabbDegenerate || aabbHuge || aabbTiny || aabbInvalid || !hasTransform) {
            Logger::warn(str::format("RTXMG DIAG: Surface ", id, " ISSUES:"));
            if (!hasValidGeometry) Logger::warn(str::format("  - No geometry: surfaces=", numSurfaces, " verts=", numVertices));
            if (aabbDegenerate) Logger::warn("  - AABB degenerate (zero or negative extent)");
            if (aabbHuge) Logger::warn(str::format("  - AABB huge: extent=(", extentX, ",", extentY, ",", extentZ, ")"));
            if (aabbTiny) Logger::warn(str::format("  - AABB tiny: extent=(", extentX, ",", extentY, ",", extentZ, ")"));
            if (aabbInvalid) Logger::warn("  - AABB contains NaN/Inf");
            if (!hasTransform) Logger::warn("  - No transform this frame (won't render)");
            Logger::warn(str::format("  AABB: min=(", aabb.m_mins[0], ",", aabb.m_mins[1], ",", aabb.m_mins[2],
                ") max=(", aabb.m_maxs[0], ",", aabb.m_maxs[1], ",", aabb.m_maxs[2], ")"));
          } else {
            // Log healthy surfaces too for comparison
            RTXMG_LOG(str::format("RTXMG DIAG: Surface ", id, " OK: ", numSurfaces, " patches, ", numVertices, " verts, extent=(",
                extentX, ",", extentY, ",", extentZ, ")"));
          }

          // Check control point positions against AABB (sample first few)
          if (shape && shape->verts.size() > 0) {
            uint32_t outOfBoundsCount = 0;
            float maxOutOfBounds = 0.0f;
            for (size_t i = 0; i < std::min(shape->verts.size(), size_t(100)); ++i) {
              const Vector3& v = shape->verts[i];
              bool outX = (v.x < aabb.m_mins[0] - 0.01f || v.x > aabb.m_maxs[0] + 0.01f);
              bool outY = (v.y < aabb.m_mins[1] - 0.01f || v.y > aabb.m_maxs[1] + 0.01f);
              bool outZ = (v.z < aabb.m_mins[2] - 0.01f || v.z > aabb.m_maxs[2] + 0.01f);
              if (outX || outY || outZ) {
                outOfBoundsCount++;
                float distX = std::max(aabb.m_mins[0] - v.x, v.x - aabb.m_maxs[0]);
                float distY = std::max(aabb.m_mins[1] - v.y, v.y - aabb.m_maxs[1]);
                float distZ = std::max(aabb.m_mins[2] - v.z, v.z - aabb.m_maxs[2]);
                maxOutOfBounds = std::max(maxOutOfBounds, std::max(distX, std::max(distY, distZ)));
              }
            }
            if (outOfBoundsCount > 0) {
              Logger::warn(str::format("RTXMG DIAG: Surface ", id, " has ", outOfBoundsCount,
                  " control points outside AABB (max dist=", maxOutOfBounds, ")"));
            }
          }
        }
      }
      RTXMG_LOG("=== END RTXMG DIAGNOSTICS ===");

#if RTXMG_LOG_RTX_MEGAGEO_BUILDER // DIAGNOSTIC: BLAS address + vertex position readback (causes GPU stalls)
      {
        static uint32_t s_readbackCount = 0;
        if (s_readbackCount < 3) {
          // BLAS address readback
          Logger::info("RTX MegaGeo: Performing BLAS address readback verification...");
          Logger::info(str::format("RTX MegaGeo: blasPtrsBuffer bytes=",
              m_clusterAccels->blasPtrsBuffer.GetBytes(),
              " elements=", m_clusterAccels->blasPtrsBuffer.GetNumElements()));
          bool blasOk = downloadBlasAddresses();
          Logger::info(str::format("RTX MegaGeo BLAS READBACK: result=", blasOk ? "OK (has non-zero)" : "FAIL (all zero or empty)"));

          // Vertex position readback - check if GPU vertex data is valid
          if (m_clusterAccels->clusterVertexPositionsBuffer.GetBytes() > 0) {
            auto vertexPositions = m_clusterAccels->clusterVertexPositionsBuffer.Download(m_commandList.Get());
            uint32_t totalVerts = static_cast<uint32_t>(vertexPositions.size());
            uint32_t zeroVerts = 0, nanVerts = 0, infVerts = 0, sentinelVerts = 0;
            float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
            float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
            const float kSentinel = 10000.0f;
            for (uint32_t i = 0; i < totalVerts; ++i) {
              float x = vertexPositions[i].x, y = vertexPositions[i].y, z = vertexPositions[i].z;
              if (x == 0.0f && y == 0.0f && z == 0.0f) zeroVerts++;
              else if (x == kSentinel && y == kSentinel && z == kSentinel) sentinelVerts++;
              if (std::isnan(x) || std::isnan(y) || std::isnan(z)) nanVerts++;
              if (std::isinf(x) || std::isinf(y) || std::isinf(z)) infVerts++;
              if (!std::isnan(x) && !std::isinf(x) && x != kSentinel) { minX = std::min(minX, x); maxX = std::max(maxX, x); }
              if (!std::isnan(y) && !std::isinf(y) && y != kSentinel) { minY = std::min(minY, y); maxY = std::max(maxY, y); }
              if (!std::isnan(z) && !std::isinf(z) && z != kSentinel) { minZ = std::min(minZ, z); maxZ = std::max(maxZ, z); }
            }
            uint32_t realData = totalVerts - zeroVerts - sentinelVerts - nanVerts - infVerts;
            Logger::info(str::format("RTX MegaGeo VERTEX READBACK: ", totalVerts, " total, ",
                sentinelVerts, " SENTINEL(never written), ", zeroVerts, " zero(shader wrote 0), ",
                realData, " real data, ", nanVerts, " NaN, ", infVerts, " Inf"));
            Logger::info(str::format("RTX MegaGeo VERTEX READBACK: bounds min=(", minX, ",", minY, ",", minZ,
                ") max=(", maxX, ",", maxY, ",", maxZ, ")"));
            // Find first non-zero vertex and log its location
            uint32_t firstNonZero = UINT32_MAX;
            uint32_t lastNonZero = 0;
            for (uint32_t i = 0; i < totalVerts; ++i) {
              float x = vertexPositions[i].x, y = vertexPositions[i].y, z = vertexPositions[i].z;
              if (x != 0.0f || y != 0.0f || z != 0.0f) {
                if (firstNonZero == UINT32_MAX) firstNonZero = i;
                lastNonZero = i;
              }
            }
            Logger::info(str::format("RTX MegaGeo VERTEX READBACK: firstNonZero=", firstNonZero,
                " lastNonZero=", lastNonZero, " populated range=", (lastNonZero >= firstNonZero ? lastNonZero - firstNonZero + 1 : 0)));
            if (firstNonZero != UINT32_MAX) {
              uint32_t logStart = (firstNonZero > 5) ? firstNonZero - 5 : 0;
              for (uint32_t i = logStart; i < std::min<uint32_t>(logStart + 20, totalVerts); ++i) {
                Logger::info(str::format("RTX MegaGeo VERTEX[", i, "] = (", vertexPositions[i].x, ",",
                    vertexPositions[i].y, ",", vertexPositions[i].z, ")"));
              }
              uint32_t mid = firstNonZero + (lastNonZero - firstNonZero) / 2;
              Logger::info(str::format("RTX MegaGeo VERTEX READBACK: mid-range samples around ", mid, ":"));
              uint32_t midStart = (mid > 5) ? mid - 5 : 0;
              for (uint32_t i = midStart; i < std::min<uint32_t>(midStart + 10, totalVerts); ++i) {
                Logger::info(str::format("RTX MegaGeo VERTEX[", i, "] = (", vertexPositions[i].x, ",",
                    vertexPositions[i].y, ",", vertexPositions[i].z, ")"));
              }
            }
          } else {
            Logger::warn("RTX MegaGeo VERTEX READBACK: clusterVertexPositionsBuffer is empty!");
          }

          s_readbackCount++;
        }
      }
#endif

      // Mark all surfaces as ready - BLAS addresses will be patched on GPU
      for (auto& [id, surface] : m_surfaces) {
        if (surface.isDirty && surface.subdivSurface) {
          surface.isDirty = false;
          surface.isReady = true;
          // Note: blasAddress is left as 0 - patching happens on GPU via patchClusterBlasAddresses()
          RTXMG_LOG(str::format("RTX MegaGeo: Surface ", id, " marked ready"));
        }
      }
    }

    // Collect statistics from ClusterStatistics
    m_stats.numClusters = m_clusterStats.allocated.m_numClusters;
    m_stats.numDesiredClusters = m_clusterStats.desired.m_numClusters;
    m_stats.numTriangles = m_clusterStats.allocated.m_numTriangles;
    m_stats.numVertices = 0; // Not directly available in ClusterStatistics
    m_stats.clasMemoryBytes = m_clusterStats.allocated.m_clasSize;
    m_stats.cullRatio = (m_stats.numDesiredClusters > 0) ?
      1.0f - (float)m_stats.numClusters / (float)m_stats.numDesiredClusters : 0.0f;

    RTXMG_LOG(str::format("RTX MegaGeo: Built cluster BLAS: ", m_stats.numClusters, " clusters, ",
                            m_stats.numTriangles, " triangles"));

    // Always log RTXMG stats prominently so user can verify it's working
    static uint32_t frameLogCounter = 0;
    if ((frameLogCounter++ % 60) == 0) { // Log every 60 frames (~1 second)
      if (m_stats.numClusters > 0) {
#if RTXMG_CHRONO_TIMING
        RTXMG_LOG(str::format(">>> RTXMG: ", m_stats.numClusters, " clusters, ",
                                  m_stats.numTriangles, " tris, ", m_surfaces.size(), " surfaces, ",
                                  s_buildTimeMs, "ms <<<"));
#else
        RTXMG_LOG(str::format(">>> RTXMG ACTIVE: ", m_stats.numClusters, " clusters, ",
                                  m_stats.numTriangles, " triangles from ", m_surfaces.size(), " surfaces <<<"));
#endif
      } else {
        Logger::warn(str::format(">>> RTXMG: 0 clusters (desired: ", m_stats.numDesiredClusters, ") from ", m_surfaces.size(), " surfaces - CHECK TESSELLATION <<<"));
      }
    }

    // Log cluster buffer status after build
    {
      static uint32_t s_bufDiagCount = 0;
      if (s_bufDiagCount < 5) {
        auto shadingBuf = getClusterShadingDataBuffer();
        auto posBuf = getClusterVertexPositionsBuffer();
        auto normBuf = getClusterVertexNormalsBuffer();
        RTXMG_LOG(str::format("RTX MegaGeo POST-BUILD[", s_bufDiagCount, "]: shadingBuf=", (void*)shadingBuf.Get(),
            " posBuf=", (void*)posBuf.Get(), " normBuf=", (void*)normBuf.Get()));

        // Log BLAS addresses for first few surfaces
        for (auto& [surfId, surfData] : m_surfaces) {
          if (s_bufDiagCount == 0) {
            VkDeviceAddress blasAddr = getSurfaceBlasAddress(surfId);
            RTXMG_LOG(str::format("RTX MegaGeo POST-BUILD: surface[", surfId, "] blasAddr=0x", std::hex, blasAddr, std::dec));
            if (surfId >= 5) break; // Only log first few
          }
        }
        s_bufDiagCount++;
      }
    }

    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - returning true");
    return true;
  }

  VkAccelerationStructureKHR RtxMegaGeoBuilder::getSurfaceBlas(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return VK_NULL_HANDLE;
    }
    return it->second.blas;
  }

  VkDeviceAddress RtxMegaGeoBuilder::getSurfaceBlasAddress(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return 0;
    }
    return it->second.blasAddress;
  }

  bool RtxMegaGeoBuilder::isSurfaceReady(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return false;
    }
    return it->second.isReady;
  }

  nvrhi::IBuffer* RtxMegaGeoBuilder::getBlasPointersBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->blasPtrsBuffer.Get();
  }

  uint32_t RtxMegaGeoBuilder::getInstanceIndexForSurface(uint32_t surfaceId) const {
    auto it = m_surfaceToInstanceIndex.find(surfaceId);
    if (it == m_surfaceToInstanceIndex.end()) {
      return UINT32_MAX;
    }
    return it->second;
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterShadingDataBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterShadingDataBuffer.Get();
  }

  uint32_t RtxMegaGeoBuilder::getClusterCount() const {
    return m_stats.numClusters;
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterVertexPositionsBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterVertexPositionsBuffer.Get();
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterVertexNormalsBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterVertexNormalsBuffer.Get();
  }

  // Helper to check if an NVRHI buffer has a valid underlying DxvkBuffer
  static bool isNvrhiBufferReady(nvrhi::IBuffer* buffer) {
    if (!buffer) return false;
    NvrhiDxvkBuffer* nvrhiBuf = static_cast<NvrhiDxvkBuffer*>(buffer);
    if (!nvrhiBuf) return false;
    const Rc<DxvkBuffer>& dxvkBuf = nvrhiBuf->getDxvkBuffer();
    return dxvkBuf.ptr() != nullptr;
  }

  bool RtxMegaGeoBuilder::hasValidBuffers() const {
    if (!m_clusterAccels) {
      ONCE(RTXMG_LOG("RTX MegaGeo hasValidBuffers: m_clusterAccels is null"));
      return false;
    }

    // Check that the underlying DxvkBuffers are actually valid, not just the NVRHI handles
    // This matches the validation done in rtx_context.cpp when binding buffers
    bool shadingValid = isNvrhiBufferReady(m_clusterAccels->clusterShadingDataBuffer.Get());
    bool posValid = isNvrhiBufferReady(m_clusterAccels->clusterVertexPositionsBuffer.Get());
    bool normValid = isNvrhiBufferReady(m_clusterAccels->clusterVertexNormalsBuffer.Get());

    static uint32_t s_logCounter = 0;
    if ((s_logCounter++ % 100) == 0) {
      RTXMG_LOG(str::format("RTX MegaGeo hasValidBuffers: shading=", shadingValid,
                               " pos=", posValid, " norm=", normValid));
    }

    return shadingValid && posValid && normValid;
  }

  void RtxMegaGeoBuilder::processCompletedSurfaces() {
    std::queue<CompletedSurface> completed;

    {
      std::lock_guard<std::mutex> lock(m_completedMutex);
      completed.swap(m_completedSurfaces);
    }

    while (!completed.empty()) {
      CompletedSurface& comp = completed.front();

      auto it = m_surfaces.find(comp.surfaceId);
      if (it != m_surfaces.end()) {
        RTXMGSubdivisionSurfaceEntry& surface = it->second;

        RTXMG_LOG(str::format("RTX MegaGeo: Creating SubdivisionSurface for surface ", comp.surfaceId, " on main thread"));

        // Create SubdivisionSurface on main thread with proper GPU resources
        m_commandList->open();

        try {
          std::vector<std::unique_ptr<Shape>> keyFrames;
          surface.subdivSurface = std::make_unique<SubdivisionSurface>(
            *m_topologyCaches[0], // Use first topology cache on main thread
            std::move(comp.shape),
            keyFrames,
            nullptr, // descriptorTableManager - may need to provide this
            m_commandList.Get());

          m_commandList->close();
          m_nvrhiDevice->executeCommandList(m_commandList.Get());

          // Add to RTXMGScene - get the actual index for meshID
          uint32_t meshIndex = static_cast<uint32_t>(m_scene->GetSubdMeshes().size());
          m_scene->AddSubdMesh(surface.subdivSurface.get());

          // Record the PERSISTENT mapping from surfaceId to mesh index
          // Instances will be rebuilt each frame in buildClusterBlas based on active transforms
          m_surfaceToMeshIndex[comp.surfaceId] = meshIndex;
          RTXMG_LOG(str::format("RTX MegaGeo: Mapping surfaceId ", comp.surfaceId, " to mesh index ", meshIndex));

          surface.isReady = true;

          const Shape* shape = surface.subdivSurface ? surface.subdivSurface->GetShape() : nullptr;
          uint32_t numFaces = shape ? shape->nvertsPerFace.size() : 0;
          RTXMG_LOG(str::format("RTX MegaGeo: Integrated completed surface ", comp.surfaceId,
                                   " (", numFaces, " faces)"));
        } catch (const std::exception& e) {
          m_commandList->close();
          Logger::err(str::format("RTX MegaGeo: Failed to create SubdivisionSurface for surface ", comp.surfaceId, ": ", e.what()));
        }
      }

      completed.pop();
    }
  }

  std::unique_ptr<Shape> RtxMegaGeoBuilder::convertDescToShape(const SubdivisionSurfaceDesc& desc)
  {
    RTXMG_LOG("RTX MegaGeo: Converting SubdivisionSurfaceDesc to Shape");

    // Validate input pointers
    if (!desc.controlPoints || !desc.faceVertexIndices || !desc.faceVertexCounts) {
      Logger::err("RTX MegaGeo: Invalid desc - null pointers");
      return nullptr;
    }

    // Log first few values to verify data is valid
    if (desc.numVertices > 0) {
      const Vector3& firstVert = desc.controlPoints[0];
      RTXMG_LOG(str::format("RTX MegaGeo: First vertex: (", firstVert.x, ", ", firstVert.y, ", ", firstVert.z, ")"));
    }
    if (desc.numFaces > 0 && desc.faceVertexCounts[0] > 0) {
      uint32_t firstIndex = desc.faceVertexIndices[0];
      RTXMG_LOG(str::format("RTX MegaGeo: First face index: ", firstIndex));
    }

    auto shape = std::make_unique<Shape>();

    // Set scheme to Catmull-Clark
    shape->scheme = Scheme::kCatmark;

    // Set filepath as name (Shape doesn't have a separate name field)
    if (desc.debugName) {
      shape->filepath = desc.debugName;
    }

    // Copy vertex positions
    shape->verts.resize(desc.numVertices);
    for (uint32_t i = 0; i < desc.numVertices; ++i) {
      shape->verts[i] = desc.controlPoints[i];
    }
    RTXMG_LOG(str::format("RTX MegaGeo: Copied ", desc.numVertices, " vertices"));

    // Copy UVs if present
    if (desc.texcoords) {
      shape->uvs.resize(desc.numVertices);
      for (uint32_t i = 0; i < desc.numVertices; ++i) {
        shape->uvs[i] = desc.texcoords[i];
      }
    }

    // Build face vertex counts and indices
    shape->nvertsPerFace.resize(desc.numFaces);
    uint32_t totalIndices = 0;
    for (uint32_t i = 0; i < desc.numFaces; ++i) {
      shape->nvertsPerFace[i] = static_cast<int>(desc.faceVertexCounts[i]);
      totalIndices += desc.faceVertexCounts[i];
    }

    shape->faceverts.resize(totalIndices);
    for (uint32_t i = 0; i < totalIndices; ++i) {
      uint32_t vertexIndex = desc.faceVertexIndices[i];

      // Validate vertex index is within bounds - fail immediately if invalid
      if (vertexIndex >= desc.numVertices) {
        Logger::err(str::format("RTX MegaGeo: Invalid vertex index ", vertexIndex,
                                " at position ", i, " (numVertices=", desc.numVertices, ")"));
        Logger::err("RTX MegaGeo: This indicates corrupted mesh data or incorrect index buffer format");
        return nullptr;
      }

      shape->faceverts[i] = static_cast<int>(vertexIndex);
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Copied ", totalIndices, " face indices"));

    // Setup face UV indices (faceuvs) if UVs are present
    // Since UVs are per-vertex, face UV indices are the same as face vertex indices
    if (desc.texcoords) {
      shape->faceuvs = shape->faceverts; // Copy the same indices
    }

    // Setup material bindings (required by OpenSubdiv)
    shape->mtlbind.resize(desc.numFaces, 0); // All faces use material 0
    shape->mtls.push_back(std::make_unique<Shape::material>()); // Create default material

    // Create single subshape for all faces
    Shape::Subshape subshape;
    subshape.startFaceIndex = 0;
    subshape.mtlBind = 0; // Single material
    shape->subshapes.push_back(subshape);

    // Map all faces to subshape 0
    shape->faceToSubshapeIndex.resize(desc.numFaces, 0);

    // Compute bounding box
    if (desc.numVertices > 0 && desc.controlPoints != nullptr) {
      Vector3 minPt = desc.controlPoints[0];
      Vector3 maxPt = desc.controlPoints[0];
      for (uint32_t i = 1; i < desc.numVertices; ++i) {
        const Vector3& p = desc.controlPoints[i];
        minPt.x = std::min(minPt.x, p.x);
        minPt.y = std::min(minPt.y, p.y);
        minPt.z = std::min(minPt.z, p.z);
        maxPt.x = std::max(maxPt.x, p.x);
        maxPt.y = std::max(maxPt.y, p.y);
        maxPt.z = std::max(maxPt.z, p.z);
      }
      // Ensure AABB has minimum extent to avoid degenerate acceleration structures
      // Flat geometry (floors, walls) can have zero extent in one dimension which causes GPU hangs
      const float kMinExtent = 0.001f;
      for (int i = 0; i < 3; ++i) {
        float extent = (&maxPt.x)[i] - (&minPt.x)[i];
        if (extent < kMinExtent) {
          float center = ((&maxPt.x)[i] + (&minPt.x)[i]) * 0.5f;
          (&minPt.x)[i] = center - kMinExtent * 0.5f;
          (&maxPt.x)[i] = center + kMinExtent * 0.5f;
        }
      }

      shape->aabb.m_mins[0] = minPt.x;
      shape->aabb.m_mins[1] = minPt.y;
      shape->aabb.m_mins[2] = minPt.z;
      shape->aabb.m_maxs[0] = maxPt.x;
      shape->aabb.m_maxs[1] = maxPt.y;
      shape->aabb.m_maxs[2] = maxPt.z;
      RTXMG_LOG(str::format("convertToShape: Computed aabb min=(", minPt.x, ",", minPt.y, ",", minPt.z,
          ") max=(", maxPt.x, ",", maxPt.y, ",", maxPt.z, ")"));
    } else {
      Logger::warn(str::format("convertToShape: Cannot compute aabb - numVertices=", desc.numVertices,
          " controlPoints=", (void*)desc.controlPoints));
    }

    RTXMG_LOG(str::format("Converted to Shape: ", desc.numFaces, " faces, ", desc.numVertices, " vertices"));
    return shape;
  }

  void RtxMegaGeoBuilder::updateHiZBuffer(const Rc<DxvkImageView>& depthBuffer) {
    // Integrate RTX Remix depth buffer with RTX MG HiZ system
    // This follows the SDK pattern from zbuffer.cpp:154-161

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Entry");

    if (!m_commandList) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - No command list, returning");
      return;
    }

    if (depthBuffer == nullptr) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - No depth buffer, returning");
      return;
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Getting image info");
    const Rc<DxvkImage>& image = depthBuffer->image();
    const DxvkImageCreateInfo& imageInfo = image->info();

    // Check if depth buffer has valid content (not UNDEFINED layout)
    if (imageInfo.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Depth buffer layout is UNDEFINED, skipping");
      return;
    }

    // Create ZBuffer on first use (when we know the depth buffer size)
    if (!m_zBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo: Creating ZBuffer with size ",
        imageInfo.extent.width, "x", imageInfo.extent.height));

      uint2 bufferSize = { imageInfo.extent.width, imageInfo.extent.height };

      // Create ZBuffer following reference sample pattern from zbuffer.cpp:85-119
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Calling ZBuffer::Create");
      m_zBuffer = ZBuffer::Create(bufferSize, m_nvrhiDevice, m_commandList.Get());

      if (!m_zBuffer) {
        Logger::err("RTX MegaGeo: Failed to create ZBuffer");
        return;
      }

      RTXMG_LOG("RTX MegaGeo: ZBuffer and HiZ hierarchy created successfully");
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Creating scoped marker");
    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeo::updateHiZBuffer");

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Creating texture wrapper");
    // Copy depth buffer to ZBuffer current texture
    // Note: ZBuffer doesn't have Read() method in our adapter, so we directly copy
    nvrhi::TextureDesc depthDesc;
    depthDesc.width = imageInfo.extent.width;
    depthDesc.height = imageInfo.extent.height;
    depthDesc.depth = imageInfo.extent.depth;
    depthDesc.arraySize = imageInfo.numLayers;
    depthDesc.mipLevels = imageInfo.mipLevels;
    depthDesc.format = static_cast<nvrhi::Format>(imageInfo.format);
    depthDesc.debugName = "RTX Remix Depth Buffer";

    nvrhi::TextureHandle depthTexture = new dxvk::NvrhiDxvkTexture(depthDesc, image);

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Copying texture");
    // Copy to ZBuffer's current texture if available
    nvrhi::TextureHandle zbufferTex = m_zBuffer->GetCurrent();
    if (zbufferTex) {
      m_commandList->copyTexture(zbufferTex.Get(), depthTexture.Get());
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Reducing HiZ hierarchy");
    // Reduce ZBuffer hierarchy to build HiZ mip chain
    // This performs min/max reduction across mip levels for efficient culling
    m_zBuffer->ReduceHierarchy(m_commandList.Get());

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Complete");
    // The HiZ buffer is now ready for use in compute_cluster_tiling shader
    // It will be bound via the TessellatorConfig::zbuffer pointer
  }

  void RtxMegaGeoBuilder::showImguiSettings() {
#ifdef IMGUI_ENABLED
    static const char* kColorModeNames[] = {
      "Base Color",
      "Surface Normal",
      "Tex Coord",
      "Material",
      "Geometry Index",
      "Surface Index",
      "Cluster ID",
      "MicroTri ID",
      "Cluster UV",
      "MicroTri Area",
      "Topology Quality"
    };
    static_assert(std::size(kColorModeNames) == static_cast<size_t>(ColorMode::COLOR_MODE_COUNT),
                  "ColorMode names must match enum count");

    ImGui::Checkbox("Enable Tessellation", &m_enableTessellation);
    if (!m_enableTessellation) {
      ImGui::TextDisabled("(Subdivision surfaces disabled)");
      return;
    }

    ImGui::Separator();

    // Micro triangle visualization
    if (ImGui::Checkbox("Show Micro Triangles", &m_showMicroTriangles)) {
      // When micro triangles are shown, override color mode
      if (m_showMicroTriangles) {
        m_colorMode = ColorMode::COLOR_BY_MICROTRI_ID;
      }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Toggle micro triangle visualization mode with a unique color per triangle id.");
    }

    // Wireframe mode
    ImGui::Checkbox("Wireframe", &m_wireframeMode);
    if (m_wireframeMode) {
      ImGui::SliderFloat("Wireframe Thickness", &m_wireframeThickness, 0.0f, 5.0f, "%.1f");
    }

    // Color mode selection (disabled when showing micro triangles)
    if (!m_showMicroTriangles) {
      int colorModeInt = static_cast<int>(m_colorMode);
      if (ImGui::Combo("Color Mode", &colorModeInt, kColorModeNames, static_cast<int>(ColorMode::COLOR_MODE_COUNT))) {
        m_colorMode = static_cast<ColorMode>(colorModeInt);
      }
    }

    ImGui::Separator();
    ImGui::Text("Tessellation Statistics:");

    // Display statistics
    ImGui::Text("Surfaces: %u", static_cast<uint32_t>(m_surfaces.size()));
    ImGui::Text("Clusters: %u / %u", m_stats.numClusters, m_stats.numDesiredClusters);
    ImGui::Text("Triangles: %u", m_stats.numTriangles);
    ImGui::Text("Vertices: %u", m_stats.numVertices);

    // Memory usage
    const float clasMemoryMB = m_stats.clasMemoryBytes / (1024.0f * 1024.0f);
    ImGui::Text("CLAS Memory: %.2f MB", clasMemoryMB);

    // Culling ratio
    if (m_stats.cullRatio > 0.0f) {
      ImGui::Text("Cull Ratio: %.1f%%", m_stats.cullRatio * 100.0f);
    }

    // Per-surface statistics
    if (ImGui::TreeNode("Per-Surface Stats")) {
      for (const auto& [id, surface] : m_surfaces) {
        std::string label = str::format("Surface ", id, ": ", surface.debugName.empty() ? "unnamed" : surface.debugName.c_str());
        if (ImGui::TreeNode(label.c_str())) {
          ImGui::Text("Vertices: %u", surface.numVertices);
          ImGui::Text("Faces: %u", surface.numFaces);
          ImGui::Text("Isolation Level: %u", surface.isolationLevel);
          ImGui::Text("Tessellation Scale: %.2f", surface.tessellationScale);
          ImGui::Text("Status: %s", surface.isReady ? "Ready" : (surface.isDirty ? "Dirty" : "Building"));
          if (surface.m_hasDisplacementMaterial) {
            ImGui::Text("Displacement: Enabled (scale=%.2f)", surface.displacementScale);
          }
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
#endif
  }

  void RtxMegaGeoBuilder::patchClusterBlasAddresses(
    VkBuffer instanceBuffer,
    VkDeviceSize instanceBufferOffset,
    const std::vector<InstancePatchMapping>& mappings)
  {
    if (mappings.empty()) {
      RTXMG_LOG("RTX MegaGeo: patchClusterBlasAddresses - no mappings to patch");
      return;
    }

    if (!m_clusterAccels || m_clusterAccels->blasPtrsBuffer.GetBytes() == 0) {
      Logger::warn("RTX MegaGeo: patchClusterBlasAddresses - no blasPtrsBuffer available");
      return;
    }

    if (!m_nvrhiDevice || !m_commandList) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddresses - NVRHI not initialized");
      return;
    }

    if (instanceBuffer == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddresses - null instance buffer");
      return;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: GPU-side patching ", mappings.size(), " cluster BLAS addresses"));

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeoBuilder::patchClusterBlasAddresses");

    // Create or resize mappings buffer
    const size_t mappingsSize = mappings.size() * sizeof(ClusterInstanceMapping);
    if (!m_patchMappingsBuffer || m_patchMappingsBuffer->getDesc().byteSize < mappingsSize) {
      nvrhi::BufferDesc mappingsDesc;
      mappingsDesc.byteSize = mappingsSize;
      mappingsDesc.structStride = sizeof(ClusterInstanceMapping);
      mappingsDesc.debugName = "ClusterInstanceMappings";
      mappingsDesc.initialState = nvrhi::ResourceStates::ShaderResource;
      mappingsDesc.keepInitialState = true;
      m_patchMappingsBuffer = m_nvrhiDevice->createBuffer(mappingsDesc);
    }

    // Upload mappings to GPU
    std::vector<ClusterInstanceMapping> gpuMappings(mappings.size());
    for (size_t i = 0; i < mappings.size(); ++i) {
      gpuMappings[i].remixInstanceIndex = mappings[i].remixInstanceIndex;
      gpuMappings[i].rtxmgInstanceIndex = mappings[i].rtxmgInstanceIndex;
    }
    m_commandList->writeBuffer(m_patchMappingsBuffer, gpuMappings.data(), mappingsSize);

    // Create wrapper for the instance buffer
    // We need to create a temporary wrapper that references the external VkBuffer
    nvrhi::BufferDesc instanceBufDesc;
    instanceBufDesc.byteSize = mappings.size() * sizeof(VkAccelerationStructureInstanceKHR) * 4; // Conservative
    instanceBufDesc.structStride = sizeof(VkAccelerationStructureInstanceKHR);
    instanceBufDesc.debugName = "InstanceBufferWrapper";
    instanceBufDesc.canHaveUAVs = true;
    instanceBufDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

    // Get the DxvkBuffer from the VkBuffer handle
    // Since we have the raw VkBuffer, we need to use it directly via the device address
    // Create a binding set that uses the blasPtrsBuffer and dispatches with device addresses

    // Set up binding set
    // Bindings:
    //   t0 = mappings buffer (SRV)
    //   t1 = blasPtrsBuffer (SRV)
    //   u0 = instance buffer (UAV) - need to create wrapper

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_patchMappingsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterAccels->blasPtrsBuffer.Get()));

    // CPU download path for cases where only a raw VkBuffer handle is available.
    // The GPU-optimized path (patchClusterBlasAddressesGPU) is the primary method
    // used by AccelManager when an NVRHI buffer wrapper is available.
    std::vector<nvrhi::GpuVirtualAddress> blasAddresses = m_clusterAccels->blasPtrsBuffer.Download(m_commandList.Get());

    if (blasAddresses.empty()) {
      Logger::err("RTX MegaGeo: Failed to download BLAS addresses from GPU");
      return;
    }

    // Store downloaded addresses for use by AccelManager
    m_downloadedBlasAddresses.resize(blasAddresses.size());
    for (size_t i = 0; i < blasAddresses.size(); ++i) {
      m_downloadedBlasAddresses[i] = static_cast<VkDeviceAddress>(blasAddresses[i]);
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Downloaded ", blasAddresses.size(), " BLAS addresses"));

    // Log first few addresses for debugging
    for (size_t i = 0; i < std::min<size_t>(3, blasAddresses.size()); ++i) {
      RTXMG_LOG(str::format("RTX MegaGeo: BLAS[", i, "] = 0x", std::hex, m_downloadedBlasAddresses[i]));
    }
  }

  void RtxMegaGeoBuilder::patchClusterBlasAddressesGPU(
    nvrhi::IBuffer* instanceBuffer,
    uint32_t instanceBufferOffset,
    const std::vector<InstancePatchMapping>& mappings)
  {
    if (mappings.empty()) {
      return;
    }

    if (!m_clusterAccels || m_clusterAccels->blasPtrsBuffer.GetBytes() == 0) {
      Logger::warn("RTX MegaGeo: patchClusterBlasAddressesGPU - no blasPtrsBuffer available");
      return;
    }

    if (!m_nvrhiDevice || !m_commandList) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddressesGPU - NVRHI not initialized");
      return;
    }

    // Per-frame diagnostic logging for GPU patching
    static uint32_t s_gpuPatchLogCount = 0;
    static uint32_t s_prevMappings = 0;
    static uint32_t s_prevBlasElements = 0;

    uint32_t blasElements = m_clusterAccels->blasPtrsBuffer.GetNumElements();
    uint32_t maxRtxmgIdx = 0;
    for (const auto& m : mappings) {
      if (m.rtxmgInstanceIndex > maxRtxmgIdx) maxRtxmgIdx = m.rtxmgInstanceIndex;
    }

    // Abort on OOB - dispatching with out-of-bounds indices would read garbage
    // BLAS addresses from blasPtrsBuffer, causing GPU crash during ray tracing
    if (maxRtxmgIdx >= blasElements) {
      Logger::err(str::format("RTX MegaGeo GPU PATCH: *** OUT OF BOUNDS *** maxRtxmgIdx=", maxRtxmgIdx,
          " >= blasPtrsBuffer elements=", blasElements, " mappings=", mappings.size(),
          " frame=", s_gpuPatchLogCount));
      return;
    }

    // Log every frame - compact single line
    bool countsChanged = (mappings.size() != s_prevMappings || blasElements != s_prevBlasElements);
    Logger::info(str::format("RTX MegaGeo GPU PATCH[", s_gpuPatchLogCount, "]: mappings=", mappings.size(),
        " blasPtrsElements=", blasElements, " maxRtxmgIdx=", maxRtxmgIdx,
        countsChanged ? " ***CHANGED***" : ""));

    s_prevMappings = uint32_t(mappings.size());
    s_prevBlasElements = blasElements;
    s_gpuPatchLogCount++;

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeoBuilder::patchClusterBlasAddressesGPU");

    // Create params buffer on first use
    if (!m_patchParamsBuffer) {
      nvrhi::BufferDesc paramsDesc;
      paramsDesc.byteSize = 256; // Align to 256 for constant buffer requirements
      paramsDesc.debugName = "PatchClusterBlasAddressParams";
      paramsDesc.isConstantBuffer = true;
      paramsDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
      paramsDesc.keepInitialState = true;
      m_patchParamsBuffer = m_nvrhiDevice->createBuffer(paramsDesc);
    }

    // Create or resize mappings buffer
    const size_t mappingsSize = mappings.size() * sizeof(ClusterInstanceMapping);
    if (!m_patchMappingsBuffer || m_patchMappingsBuffer->getDesc().byteSize < mappingsSize) {
      nvrhi::BufferDesc mappingsDesc;
      mappingsDesc.byteSize = mappingsSize;
      mappingsDesc.structStride = sizeof(ClusterInstanceMapping);
      mappingsDesc.debugName = "ClusterInstanceMappings";
      mappingsDesc.initialState = nvrhi::ResourceStates::ShaderResource;
      mappingsDesc.keepInitialState = true;
      m_patchMappingsBuffer = m_nvrhiDevice->createBuffer(mappingsDesc);
    }

    // Upload mappings to GPU
    std::vector<ClusterInstanceMapping> gpuMappings(mappings.size());
    for (size_t i = 0; i < mappings.size(); ++i) {
      gpuMappings[i].remixInstanceIndex = mappings[i].remixInstanceIndex + instanceBufferOffset;
      gpuMappings[i].rtxmgInstanceIndex = mappings[i].rtxmgInstanceIndex;
    }
    m_commandList->writeBuffer(m_patchMappingsBuffer, gpuMappings.data(), mappingsSize);

    // Write params to constant buffer
    PatchClusterBlasAddressParams params = {};
    params.numMappings = static_cast<uint32_t>(mappings.size());
    params.instanceBufferStride = sizeof(VkAccelerationStructureInstanceKHR);
    m_commandList->writeBuffer(m_patchParamsBuffer, &params, sizeof(params));

    // Set up binding set
    // Note: u0 uses StructuredBuffer_UAV to match the Slang shader's RWStructuredBuffer<VkInstanceRaw>
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_patchMappingsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterAccels->blasPtrsBuffer.Get()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, instanceBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_patchParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_nvrhiDevice, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_patchBlasAddressesBL, bindingSet)) {
      Logger::err("RTX MegaGeo: Failed to create binding set for patch_cluster_blas_addresses");
      return;
    }

    // Create compute pipeline on first use
    if (!m_patchBlasAddressesPSO) {
      nvrhi::ShaderHandle shader = m_clusterBuilder->getShaderFactory().CreateShader(
        "cluster_builder/patch_cluster_blas_addresses.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

      if (!shader) {
        Logger::err("RTX MegaGeo: Failed to create patch_cluster_blas_addresses shader");
        return;
      }

      auto pipelineDesc = nvrhi::ComputePipelineDesc()
          .setComputeShader(shader)
          .addBindingLayout(m_patchBlasAddressesBL);

      m_patchBlasAddressesPSO = m_nvrhiDevice->createComputePipeline(pipelineDesc);
      if (!m_patchBlasAddressesPSO) {
        Logger::err("RTX MegaGeo: Failed to create patch_cluster_blas_addresses pipeline");
        return;
      }
    }

    // Set compute state and dispatch
    auto state = nvrhi::ComputeState()
        .setPipeline(m_patchBlasAddressesPSO)
        .addBindingSet(bindingSet);

    // Log GPU virtual addresses of all bound resources for crash investigation
    {
      auto* mappingsBuf = static_cast<NvrhiDxvkBuffer*>(m_patchMappingsBuffer.Get());
      auto* blasPtrsBuf = static_cast<NvrhiDxvkBuffer*>(m_clusterAccels->blasPtrsBuffer.Get());
      auto* instBuf = static_cast<NvrhiDxvkBuffer*>(instanceBuffer);
      auto* paramsBuf = static_cast<NvrhiDxvkBuffer*>(m_patchParamsBuffer.Get());
      Logger::info(str::format("RTX MegaGeo: GPU PATCH BIND ADDRS: mappings=0x", std::hex,
          mappingsBuf->getDxvkBuffer()->getDeviceAddress(),
          " blasPtrs=0x", blasPtrsBuf->getDxvkBuffer()->getDeviceAddress(),
          " instance=0x", instBuf->getDxvkBuffer()->getDeviceAddress(),
          " params=0x", paramsBuf->getDxvkBuffer()->getDeviceAddress(), std::dec,
          " paramsSize=", paramsBuf->getDesc().byteSize));
    }
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - Setting compute state"));
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - blasPtrsBuffer bytes=", m_clusterAccels->blasPtrsBuffer.GetBytes(),
        " numElements=", m_clusterAccels->blasPtrsBuffer.GetNumElements()));
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - instanceBuffer byteSize=", instanceBuffer->getDesc().byteSize));

    m_commandList->setComputeState(state);

    // Dispatch - one thread per mapping
    uint32_t numGroups = (params.numMappings + kFillInstanceDescsThreads - 1) / kFillInstanceDescsThreads;
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - dispatching numGroups=", numGroups, " for ", params.numMappings, " mappings"));
    m_commandList->dispatch(numGroups, 1, 1);

    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - dispatch complete for ", mappings.size(), " instances"));
  }

  VkDeviceAddress RtxMegaGeoBuilder::getDownloadedBlasAddress(uint32_t rtxmgInstanceIndex) const {
    if (rtxmgInstanceIndex >= m_downloadedBlasAddresses.size()) {
      return 0;
    }
    return m_downloadedBlasAddresses[rtxmgInstanceIndex];
  }

  bool RtxMegaGeoBuilder::downloadBlasAddresses() {
    if (!m_clusterAccels || m_clusterAccels->blasPtrsBuffer.GetBytes() == 0) {
      Logger::warn("RTX MegaGeo: downloadBlasAddresses - no blasPtrsBuffer available");
      return false;
    }

    if (!m_commandList) {
      Logger::err("RTX MegaGeo: downloadBlasAddresses - no command list");
      return false;
    }

    std::vector<nvrhi::GpuVirtualAddress> blasAddresses = m_clusterAccels->blasPtrsBuffer.Download(m_commandList.Get());

    if (blasAddresses.empty()) {
      Logger::err("RTX MegaGeo: downloadBlasAddresses - download returned empty");
      return false;
    }

    m_downloadedBlasAddresses.resize(blasAddresses.size());
    uint32_t nonZeroCount = 0;
    for (size_t i = 0; i < blasAddresses.size(); ++i) {
      m_downloadedBlasAddresses[i] = static_cast<VkDeviceAddress>(blasAddresses[i]);
      if (m_downloadedBlasAddresses[i] != 0) nonZeroCount++;
    }

    Logger::info(str::format("RTX MegaGeo BLAS READBACK: Downloaded ", blasAddresses.size(), " BLAS addresses, ",
        nonZeroCount, " non-zero, ", (blasAddresses.size() - nonZeroCount), " ZERO"));

    // Log first few and any zero entries for debugging
    for (size_t i = 0; i < std::min<size_t>(10, blasAddresses.size()); ++i) {
      Logger::info(str::format("RTX MegaGeo BLAS[", i, "] = ", m_downloadedBlasAddresses[i],
          (m_downloadedBlasAddresses[i] == 0 ? " *** ZERO ***" : "")));
    }
    // Also log any zero addresses beyond the first 10
    if (blasAddresses.size() > 10) {
      for (size_t i = 10; i < blasAddresses.size(); ++i) {
        if (m_downloadedBlasAddresses[i] == 0) {
          Logger::warn(str::format("RTX MegaGeo BLAS[", i, "] = 0 *** ZERO ***"));
        }
      }
    }

    return nonZeroCount > 0;
  }

} // namespace dxvk
