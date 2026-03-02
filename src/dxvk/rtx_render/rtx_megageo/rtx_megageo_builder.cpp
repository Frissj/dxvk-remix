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
#include "../../dxvk_device.h"         // For DxvkDevice, adapter(), memoryProperties()
#include "../../dxvk_adapter.h"        // For DxvkAdapterMemoryInfo, getMemoryHeapInfo()
#include "cluster_builder/cluster_lod_builder.h"
#include "cluster_builder/meshlet_template_builder.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>

// Enable chrono timing for performance profiling (set to 1 to enable)
#define RTXMG_CHRONO_TIMING 0
#include "nvrhi_adapter/nvrhi_dxvk_texture.h"
#include "nvrhi_adapter/nvrhi_dxvk_buffer.h"
#include "cluster_builder/fill_instance_descs_params.h"

#include "nvrhi_adapter/nvrhi_dxvk_command_list.h"

#include "rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_RTX_MEGAGEO_BUILDER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace dxvk {

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

    // Start async worker threads for cluster mesh LOD DAG building
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
      PendingClusterMesh pendingMesh;
      bool hasPendingMesh = false;

      {
        std::unique_lock<std::mutex> lock(m_pendingMutex);
        m_workerCV.wait(lock, [this] {
          std::lock_guard<std::mutex> meshLock(m_pendingClusterMeshMutex);
          return !m_pendingClusterMeshes.empty() || m_workerShouldExit.load();
        });

        if (m_workerShouldExit.load()) {
          std::lock_guard<std::mutex> meshLock(m_pendingClusterMeshMutex);
          if (m_pendingClusterMeshes.empty())
            break;
        }
      }

      // Grab a pending mesh from the queue
      {
        std::lock_guard<std::mutex> meshLock(m_pendingClusterMeshMutex);
        if (!m_pendingClusterMeshes.empty()) {
          pendingMesh = std::move(m_pendingClusterMeshes.front());
          m_pendingClusterMeshes.pop();
          hasPendingMesh = true;
        }
      }

      if (hasPendingMesh) {
        Logger::info(str::format("RTX MegaGeo: Worker[", threadIndex, "] building LOD DAG for cluster mesh ",
            pendingMesh.surfaceId, " (", pendingMesh.indices.size() / 3, " triangles)"));

        try {
          auto lodData = ClusterLODBuilder::build(
              pendingMesh.indices.data(),
              pendingMesh.indices.size(),
              pendingMesh.vertexPositions.data(),
              pendingMesh.vertexCount,
              12); // packed float3 stride

          {
            std::lock_guard<std::mutex> lock(m_completedClusterMeshMutex);
            m_completedClusterMeshes.push({
              pendingMesh.surfaceId,
              std::move(lodData),
              pendingMesh.debugName
            });
          }

          Logger::info(str::format("RTX MegaGeo: Worker[", threadIndex, "] completed LOD DAG for cluster mesh ", pendingMesh.surfaceId));
        } catch (const std::exception& e) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to build LOD DAG for mesh ",
              pendingMesh.surfaceId, ": ", e.what()));
        } catch (...) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to build LOD DAG for mesh ",
              pendingMesh.surfaceId, ": unknown error"));
        }
      }
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Worker thread ", threadIndex, " exiting"));
  }

  bool RtxMegaGeoBuilder::createClusterMesh(
    const TriangleMeshDesc& desc,
    uint32_t& surfaceId)
  {
    if (!m_initialized) {
      Logger::err("RtxMegaGeoBuilder not initialized - call initialize() first");
      return false;
    }

    if (!desc.indices || desc.indexCount == 0 || !desc.vertexPositions || desc.vertexCount == 0) {
      Logger::err("Invalid cluster mesh: missing indices or vertices");
      return false;
    }

    if (desc.indexCount % 3 != 0) {
      Logger::err(str::format("Invalid cluster mesh: indexCount ", desc.indexCount, " not divisible by 3"));
      return false;
    }

    surfaceId = m_nextSurfaceId++;

    Logger::info(str::format("RTX MegaGeo: Creating cluster mesh ", surfaceId,
        " (", desc.indexCount / 3, " triangles, ", desc.vertexCount, " vertices)",
        desc.debugName ? str::format(" name='", desc.debugName, "'") : ""));

    // Create placeholder entry
    RTXMGClusterMeshEntry entry;
    entry.debugName = desc.debugName ? desc.debugName : "";
    entry.isReady = false;
    entry.templatesBuilt = false;
    m_clusterMeshes[surfaceId] = std::move(entry);

    // Copy data for async processing
    PendingClusterMesh pending;
    pending.surfaceId = surfaceId;
    pending.vertexCount = desc.vertexCount;
    pending.vertexPositionsStride = desc.vertexPositionsStride;
    pending.debugName = desc.debugName ? desc.debugName : "";

    // Copy indices
    pending.indices.assign(desc.indices, desc.indices + desc.indexCount);

    // Copy vertex positions (packed float3)
    pending.vertexPositions.resize(desc.vertexCount * 3);
    size_t srcStrideFloats = desc.vertexPositionsStride / sizeof(float);
    for (size_t i = 0; i < desc.vertexCount; ++i) {
      pending.vertexPositions[i * 3 + 0] = desc.vertexPositions[i * srcStrideFloats + 0];
      pending.vertexPositions[i * 3 + 1] = desc.vertexPositions[i * srcStrideFloats + 1];
      pending.vertexPositions[i * 3 + 2] = desc.vertexPositions[i * srcStrideFloats + 2];
    }

    // Queue for async processing
    {
      std::lock_guard<std::mutex> lock(m_pendingClusterMeshMutex);
      m_pendingClusterMeshes.push(std::move(pending));
    }
    m_workerCV.notify_one();

    return true;
  }

  bool RtxMegaGeoBuilder::isSurfaceReady(uint32_t surfaceId) const {
    auto it = m_clusterMeshes.find(surfaceId);
    if (it == m_clusterMeshes.end())
      return false;
    return it->second.templatesBuilt;
  }

  VkDeviceAddress RtxMegaGeoBuilder::getSurfaceBlasAddress(uint32_t surfaceId) const {
    auto instIt = m_surfaceToInstanceIndex.find(surfaceId);
    if (instIt == m_surfaceToInstanceIndex.end())
      return 0;
    uint32_t instanceIndex = instIt->second;
    if (instanceIndex >= m_downloadedBlasAddresses.size())
      return 0;
    return m_downloadedBlasAddresses[instanceIndex];
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

    // Update last-seen frame for surfaces that have transforms this frame
    for (const auto& [surfaceId, _] : instanceTransforms) {
      m_surfaceLastSeenFrame[surfaceId] = s_frameCounter;
    }

    // VRAM capacity monitoring — log warnings when approaching total VRAM limits
    {
      VkDeviceSize totalVramUsed = 0;
      VkDeviceSize totalVramBudget = 0;
      const VkPhysicalDeviceMemoryProperties memory = m_device->adapter()->memoryProperties();
      const DxvkAdapterMemoryInfo memHeapInfo = m_device->adapter()->getMemoryHeapInfo();
      for (uint32_t i = 0; i < memory.memoryHeapCount; i++) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
          totalVramUsed += memHeapInfo.heaps[i].memoryAllocated;
          totalVramBudget += memHeapInfo.heaps[i].memoryBudget;
        }
      }

      if (totalVramBudget > 0) {
        if (totalVramUsed > totalVramBudget * 95 / 100) {
          Logger::err(str::format("RTX MegaGeo: CRITICAL VRAM usage ",
              totalVramUsed / (1024 * 1024), " MB / ", totalVramBudget / (1024 * 1024),
              " MB (", totalVramUsed * 100 / totalVramBudget, "%)"));
        } else if (totalVramUsed > totalVramBudget * 90 / 100) {
          Logger::warn(str::format("RTX MegaGeo: High VRAM usage ",
              totalVramUsed / (1024 * 1024), " MB / ", totalVramBudget / (1024 * 1024),
              " MB (", totalVramUsed * 100 / totalVramBudget, "%)"));
        }

        // Periodic monitoring every 60 frames
        if (s_frameCounter % 60 == 0) {
          RTXMG_LOG(str::format("RTX MegaGeo: VRAM ",
              totalVramUsed / (1024 * 1024), " MB / ", totalVramBudget / (1024 * 1024),
              " MB (", totalVramUsed * 100 / totalVramBudget, "%) - ",
              m_clusterMeshes.size(), " cluster meshes"));
        }
      }
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

    // =====================================================================
    // Process completed cluster mesh LOD DAG builds (async worker thread)
    // =====================================================================
    {
      std::queue<CompletedClusterMesh> completed;
      {
        std::lock_guard<std::mutex> lock(m_completedClusterMeshMutex);
        completed.swap(m_completedClusterMeshes);
      }
      while (!completed.empty()) {
        CompletedClusterMesh& comp = completed.front();
        auto it = m_clusterMeshes.find(comp.surfaceId);
        if (it != m_clusterMeshes.end() && comp.lodData) {
          it->second.lodData = std::move(comp.lodData);
          it->second.isReady = true;
          it->second.debugName = comp.debugName;
          Logger::info(str::format("RTX MegaGeo: Cluster mesh ", comp.surfaceId,
              " LOD DAG ready (", it->second.lodData->clusters.size(), " clusters)"));
        }
        completed.pop();
      }
    }

    // =====================================================================
    // Build per-meshlet CLAS templates for ready cluster meshes (one-time GPU work)
    // =====================================================================
    for (auto& [surfaceId, entry] : m_clusterMeshes) {
      if (entry.isReady && !entry.templatesBuilt && entry.lodData) {
        Logger::info(str::format("RTX MegaGeo: Building meshlet templates for cluster mesh ", surfaceId));

        // Use a large maxGeometryIndex to allow global cluster indexing during instantiation.
        // The template's maxGeometryIndex must be >= any geometryIndex produced during instantiation
        // (per Vulkan spec). We use global cluster indices (0..totalClusters-1) as geometry indices,
        // so the template must accommodate the maximum possible cluster count.
        // 16383 supports up to 16384 clusters per frame — well above typical usage.
        uint32_t maxGeomIdx = 16383;

        entry.templates = MeshletTemplateBuilder::build(
            *entry.lodData,
            m_nvrhiDevice,
            m_commandList.Get(),
            maxGeomIdx);

        if (entry.templates && entry.templates->isBuilt) {
          entry.templatesBuilt = true;
          Logger::info(str::format("RTX MegaGeo: Meshlet templates built for cluster mesh ", surfaceId,
              " (", entry.templates->numTemplates, " templates)"));
        } else {
          Logger::err(str::format("RTX MegaGeo: Failed to build meshlet templates for cluster mesh ", surfaceId));
        }
      }
    }

    // =====================================================================
    // Meshlet BuildAccel path (if we have cluster meshes with templates)
    // =====================================================================
    {
      // Collect meshlet instances from cluster meshes that have active transforms
      std::vector<ClusterAccelBuilder::MeshletInstance> meshletInstances;
      std::vector<uint32_t> meshletSurfaceIds; // Track surfaceId per instance

      for (const auto& [surfaceId, entry] : m_clusterMeshes) {
        if (!entry.templatesBuilt || !entry.lodData || !entry.templates)
          continue;

        // Check if this surface has a transform this frame
        auto transformIt = m_instanceTransforms.find(surfaceId);
        if (transformIt == m_instanceTransforms.end())
          continue;

        ClusterAccelBuilder::MeshletInstance mi;
        mi.meshIndex = 0; // Not used in meshlet path
        mi.templates = entry.templates.get();
        mi.lodData = entry.lodData.get();
        mi.surfaceId = surfaceId;

        // Extract 3x4 row-major transform from Matrix4
        const Matrix4& xform = transformIt->second;
        mi.localToWorld[0] = xform.data[0][0]; mi.localToWorld[1] = xform.data[0][1]; mi.localToWorld[2] = xform.data[0][2];
        mi.localToWorld[3] = xform.data[1][0]; mi.localToWorld[4] = xform.data[1][1]; mi.localToWorld[5] = xform.data[1][2];
        mi.localToWorld[6] = xform.data[2][0]; mi.localToWorld[7] = xform.data[2][1]; mi.localToWorld[8] = xform.data[2][2];
        mi.localToWorld[9] = xform.data[3][0]; mi.localToWorld[10] = xform.data[3][1]; mi.localToWorld[11] = xform.data[3][2];

        meshletInstances.push_back(mi);
        meshletSurfaceIds.push_back(surfaceId);
      }

      if (!meshletInstances.empty()) {
        // Extract camera data for LOD selection
        dxvk::Vector3 camPos = rtCamera.getPosition(true);
        float cameraPos[3] = { camPos.x, camPos.y, camPos.z };
        Matrix4d projMat = rtCamera.getViewToProjection();
        float cameraProj = static_cast<float>(projMat[1][1]); // cot(fovy/2)
        float cameraNear = rtCamera.getNearPlane();
        if (cameraNear <= 0.0f) cameraNear = 0.001f;
        float errorThreshold = 0.01f; // ~1% screen space error

        // Record instance index mapping for BLAS address patching
        m_surfaceToInstanceIndex.clear();
        for (uint32_t i = 0; i < meshletSurfaceIds.size(); ++i) {
          m_surfaceToInstanceIndex[meshletSurfaceIds[i]] = i;
        }

        try {
          m_clusterBuilder->BuildAccelMeshlet(
              meshletInstances,
              cameraPos,
              cameraProj,
              cameraNear,
              errorThreshold,
              *m_clusterAccels,
              m_clusterStats,
              m_frameIndex++,
              m_commandList.Get());

          // Populate TessellationStats from ClusterStatistics so the accel manager
          // knows we have valid geometry (numTriangles > 0 gates TLAS inclusion).
          m_stats.numClusters = m_clusterStats.allocated.m_numClusters;
          m_stats.numTriangles = m_clusterStats.allocated.m_numTriangles;
          m_stats.numDesiredClusters = m_clusterStats.desired.m_numClusters;
          m_stats.clasMemoryBytes = m_clusterStats.allocated.m_clasSize;

        } catch (const dxvk::DxvkError& e) {
          Logger::err(str::format("RTX MegaGeo: Meshlet BuildAccel failed: ", e.message()));
        } catch (const std::exception& e) {
          Logger::err(str::format("RTX MegaGeo: Meshlet BuildAccel failed: ", e.what()));
        } catch (...) {
          Logger::err("RTX MegaGeo: Meshlet BuildAccel failed: unknown error");
        }
      }
    }

    return true;
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

  void RtxMegaGeoBuilder::updateHiZBuffer(const Rc<DxvkImageView>& depthBuffer) {
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

    if (imageInfo.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Depth buffer layout is UNDEFINED, skipping");
      return;
    }

    // Create ZBuffer on first use (when we know the depth buffer size)
    if (!m_zBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo: Creating ZBuffer with size ",
        imageInfo.extent.width, "x", imageInfo.extent.height));

      uint2 bufferSize = { imageInfo.extent.width, imageInfo.extent.height };
      m_zBuffer = ZBuffer::Create(bufferSize, m_nvrhiDevice, m_commandList.Get());

      if (!m_zBuffer) {
        Logger::err("RTX MegaGeo: Failed to create ZBuffer");
        return;
      }

      RTXMG_LOG("RTX MegaGeo: ZBuffer and HiZ hierarchy created successfully");
    }

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeo::updateHiZBuffer");

    nvrhi::TextureDesc depthDesc;
    depthDesc.width = imageInfo.extent.width;
    depthDesc.height = imageInfo.extent.height;
    depthDesc.depth = imageInfo.extent.depth;
    depthDesc.arraySize = imageInfo.numLayers;
    depthDesc.mipLevels = imageInfo.mipLevels;
    depthDesc.format = static_cast<nvrhi::Format>(imageInfo.format);
    depthDesc.debugName = "RTX Remix Depth Buffer";

    nvrhi::TextureHandle depthTexture = new dxvk::NvrhiDxvkTexture(depthDesc, image);

    nvrhi::TextureHandle zbufferTex = m_zBuffer->GetCurrent();
    if (zbufferTex) {
      m_commandList->copyTexture(zbufferTex.Get(), depthTexture.Get());
    }

    m_zBuffer->ReduceHierarchy(m_commandList.Get());

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Complete");
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
      ImGui::TextDisabled("(Cluster meshes disabled)");
      return;
    }

    ImGui::Separator();

    // Micro triangle visualization
    if (ImGui::Checkbox("Show Micro Triangles", &m_showMicroTriangles)) {
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
    ImGui::Text("Cluster Mesh Statistics:");

    // Display statistics
    ImGui::Text("Cluster Meshes: %u", static_cast<uint32_t>(m_clusterMeshes.size()));
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

    // Per-mesh statistics
    if (ImGui::TreeNode("Per-Mesh Stats")) {
      for (const auto& [id, entry] : m_clusterMeshes) {
        std::string label = str::format("Mesh ", id, ": ", entry.debugName.empty() ? "unnamed" : entry.debugName.c_str());
        if (ImGui::TreeNode(label.c_str())) {
          ImGui::Text("LOD Ready: %s", entry.isReady ? "Yes" : "No");
          ImGui::Text("Templates Built: %s", entry.templatesBuilt ? "Yes" : "No");
          if (entry.lodData) {
            ImGui::Text("Clusters: %zu", entry.lodData->clusters.size());
          }
          if (entry.templates) {
            ImGui::Text("Templates: %u", entry.templates->numTemplates);
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

    // CPU download path for cases where only a raw VkBuffer handle is available.
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

    uint32_t blasElements = m_clusterAccels->blasPtrsBuffer.GetNumElements();
    uint32_t maxRtxmgIdx = 0;
    for (const auto& m : mappings) {
      if (m.rtxmgInstanceIndex > maxRtxmgIdx) maxRtxmgIdx = m.rtxmgInstanceIndex;
    }

    // Check if we have downloaded BLAS addresses for verification
    if (!m_downloadedBlasAddresses.empty()) {
      uint32_t zeroAddrs = 0, nonZeroAddrs = 0;
      for (size_t i = 0; i < std::min(m_downloadedBlasAddresses.size(), (size_t)blasElements); i++) {
        if (m_downloadedBlasAddresses[i] == 0) zeroAddrs++;
        else nonZeroAddrs++;
      }
    }

    // Abort on OOB - dispatching with out-of-bounds indices would read garbage
    if (maxRtxmgIdx >= blasElements) {
      Logger::err(str::format("RTX MegaGeo GPU PATCH: *** OUT OF BOUNDS *** maxRtxmgIdx=", maxRtxmgIdx,
          " >= blasPtrsBuffer elements=", blasElements, " mappings=", mappings.size()));
      return;
    }

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeoBuilder::patchClusterBlasAddressesGPU");

    // Barrier: BLAS build (on this same NVRHI cmd list) wrote blasPtrsBuffer.
    // The patching shader needs to read it. Without this barrier, the GPU may
    // execute the patching dispatch before the BLAS build output is visible.
    m_commandList->bufferBarrier(m_clusterAccels->blasPtrsBuffer,
        nvrhi::ResourceStates::AccelStructWrite, nvrhi::ResourceStates::ShaderResource);

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

    {
      static uint32_t s_patchLogCount = 0;
      static uint32_t s_lastMappingCount = 0;
      // Reset when mapping count changes (level transition)
      if (params.numMappings != s_lastMappingCount) {
        s_patchLogCount = 0;
        s_lastMappingCount = params.numMappings;
      }
      if (s_patchLogCount < 2) {
        s_patchLogCount++;
        Logger::warn(str::format("RTX MegaGeo PATCH-GPU[", s_patchLogCount, "]: ",
            " blasPtrsBuffer bytes=", m_clusterAccels->blasPtrsBuffer.GetBytes(),
            " numElements=", m_clusterAccels->blasPtrsBuffer.GetNumElements(),
            " instanceBuffer byteSize=", instanceBuffer->getDesc().byteSize,
            " numMappings=", params.numMappings,
            " instanceBufferStride=", params.instanceBufferStride));

        // Log the BLAS ptr buffer GPU address and instance buffer address
        Logger::warn(str::format("RTX MegaGeo PATCH-GPU: blasPtrsAddr=0x", std::hex,
            m_clusterAccels->blasPtrsBuffer.GetGpuVirtualAddress(), std::dec,
            " instanceBufAddr=0x", std::hex,
            static_cast<NvrhiDxvkBuffer*>(instanceBuffer)->getDxvkBuffer()->getDeviceAddress(), std::dec));

        // Log individual mappings
        Logger::warn(str::format("RTX MegaGeo PATCH-GPU: mappings (first 10 of ", params.numMappings, "):"));
        for (uint32_t i = 0; i < std::min(params.numMappings, 10u); ++i) {
          Logger::warn(str::format("  mapping[", i, "] remixInst=", mappings[i].remixInstanceIndex,
              " rtxmgInst=", mappings[i].rtxmgInstanceIndex));
        }
      }
    }

    m_commandList->setComputeState(state);

    // Dispatch - one thread per mapping
    uint32_t numGroups = (params.numMappings + kFillInstanceDescsThreads - 1) / kFillInstanceDescsThreads;
    Logger::warn(str::format("RTX MegaGeo PATCH-GPU: dispatching ", numGroups, " groups for ", params.numMappings, " mappings"));
    m_commandList->dispatch(numGroups, 1, 1);
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

    // Log ALL BLAS addresses for debugging
    Logger::info(str::format("RTX MegaGeo: downloadBlasAddresses total=", blasAddresses.size(),
        " nonZero=", nonZeroCount, " zero=", blasAddresses.size() - nonZeroCount));
    for (size_t i = 0; i < blasAddresses.size(); ++i) {
      Logger::info(str::format("RTX MegaGeo: BLAS[", i, "] = 0x", std::hex, m_downloadedBlasAddresses[i],
          (m_downloadedBlasAddresses[i] == 0 ? " *** ZERO ***" : "")));
    }

    return nonZeroCount > 0;
  }

} // namespace dxvk
