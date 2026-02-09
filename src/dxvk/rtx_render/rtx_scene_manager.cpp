/*
* Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <mutex>
#include <vector>

#include "rtx_asset_replacer.h"
#include "rtx_scene_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_buffer.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_xess.h"
#include "rtx_megageo/rtx_megageo_builder.h"

#include <assert.h>

#include "../d3d9/d3d9_state.h"
#include "vulkan/vulkan_core.h"

#include "rtx_game_capturer.h"
#include "rtx_matrix_helpers.h"
#include "rtx_intersection_test.h"

#include "dxvk_scoped_annotation.h"
#include "rtx_lights_data.h"
#include "rtx_light_utils.h"

#include "../util/util_globaltime.h"

namespace dxvk {
  SceneManager::SceneManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    , m_instanceManager(device, this)
    , m_accelManager(device)
    , m_lightManager(device)
    , m_graphManager()
    , m_rayPortalManager(device, this)
    , m_drawCallCache(device)
    , m_bindlessResourceManager(device)
    , m_pReplacer(new AssetReplacer())
    , m_terrainBaker(new TerrainBaker())
    , m_cameraManager(device)
    , m_uniqueObjectSearchDistance(RtxOptions::uniqueObjectDistance()) {
    InstanceEventHandler instanceEvents(this);
    instanceEvents.onInstanceAddedCallback = [this](RtInstance& instance) { onInstanceAdded(instance); };
    instanceEvents.onInstanceUpdatedCallback = [this](RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, bool hasTransformChanged, bool hasVerticesChanged, bool isFirstUpdateThisFrame) { onInstanceUpdated(instance, drawCall, material, hasTransformChanged, hasVerticesChanged, isFirstUpdateThisFrame); };
    instanceEvents.onInstanceDestroyedCallback = [this](RtInstance& instance) { onInstanceDestroyed(instance); };
    m_instanceManager.addEventHandler(instanceEvents);
    
    if (env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME") != "") {
      m_beginUsdExportFrameNum = stoul(env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME"));
    }
  }

  SceneManager::~SceneManager() {
  }

  bool SceneManager::areAllReplacementsLoaded() const {
    return m_pReplacer->areAllReplacementsLoaded();
  }

  std::vector<Mod::State> SceneManager::getReplacementStates() const {
    return m_pReplacer->getReplacementStates();
  }

  void SceneManager::initialize(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();
    m_pReplacer->initialize(ctx);
  }

  void SceneManager::logStatistics() {
    if (m_opacityMicromapManager.get()) {
      m_opacityMicromapManager->logStatistics();
    }
  }

  Vector3 SceneManager::getSceneUp() {
    return RtxOptions::zUp() ? Vector3(0.f, 0.f, 1.f) : Vector3(0.f, 1.f, 0.f);
  }

  Vector3 SceneManager::getSceneForward() {
    return RtxOptions::zUp() ? Vector3(0.f, 1.f, 0.f) : Vector3(0.f, 0.f, 1.f);
  }

  Vector3 SceneManager::calculateSceneRight() {
    const Vector3 up = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    return RtxOptions::leftHandedCoordinateSystem() ? cross(up, forward) : cross(forward, up);
  }

  Vector3 SceneManager::worldToSceneOrientedVector(const Vector3& worldVector) {
    return RtxOptions::zUp() ? worldVector : Vector3(worldVector.x, worldVector.z, worldVector.y);
  }

  Vector3 SceneManager::sceneToWorldOrientedVector(const Vector3& sceneVector) {
    // Same transform applies to and from
    return worldToSceneOrientedVector(sceneVector);
  }

  float SceneManager::getTotalMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
  
    const bool temporalUpscaling = RtxOptions::isDLSSOrRayReconstructionEnabled() || RtxOptions::isXeSSEnabled() || RtxOptions::isTAAEnabled();
    
    float totalUpscaleMipBias = 0.0f;
    
    if (temporalUpscaling) {
      if (RtxOptions::isXeSSEnabled()) {
        // XeSS uses the new formula from the XeSS developer guide
        totalUpscaleMipBias = -log2(resourceManager.getUpscaleRatio());
        
        // Add XeSS-specific mip bias when XeSS is active
        DxvkXeSS& xess = m_device->getCommon()->metaXeSS();
        if (xess.isActive()) {
          float xessMipBias = xess.calcRecommendedMipBias();
          totalUpscaleMipBias += xessMipBias;
        }
      } else {
        // Restore original behavior for DLSS, TAA, and other upscalers
        totalUpscaleMipBias = log2(resourceManager.getUpscaleRatio()) + RtxOptions::upscalingMipBias();
      }
    }
    
    return totalUpscaleMipBias + RtxOptions::nativeMipBias();
  }

  float SceneManager::getCalculatedUpscalingMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
    
    const bool temporalUpscaling = RtxOptions::isXeSSEnabled();
    if (!temporalUpscaling) {
      return 0.0f;
    }
    
    float calculatedUpscalingBias = -log2(resourceManager.getUpscaleRatio());
    return calculatedUpscalingBias;
  }

  void SceneManager::clear(Rc<DxvkContext> ctx, bool needWfi) {
    ScopedCpuProfileZone();

    auto& textureManager = m_device->getCommon()->getTextureManager();

    // Only clear once after the scene disappears, to avoid adding a WFI on every frame through clear().
    if (needWfi) {
      if (ctx.ptr())
        ctx->flushCommandList();
      m_device->waitForIdle();
    }

    // We still need to clear caches even if the scene wasn't rendered
    m_bufferCache.clear();
    m_surfaceMaterialCache.clear();
    m_preCreationSurfaceMaterialMap.clear();
    m_surfaceMaterialExtensionCache.clear();
    m_volumeMaterialCache.clear();
    
    // Called before instance manager's clear, so that it resets all tracked instances in Opacity Micromap manager at once
    if (m_opacityMicromapManager.get())
      m_opacityMicromapManager->clear();
    
    m_instanceManager.clear();
    m_lightManager.clear();
    m_graphManager.clear();
    m_rayPortalManager.clear();
    m_drawCallCache.clear();
    textureManager.clear();

    m_previousFrameSceneAvailable = false;
  }

  void SceneManager::garbageCollection() {
    ScopedCpuProfileZone();

    const size_t oldestFrame = m_device->getCurrentFrameId() - RtxOptions::numFramesToKeepGeometryData();
    // ClusterBlas entries are expensive to recreate (subdivision surface extraction + topology cache).
    // Keep them alive 10x longer than regular BLAS to survive brief offscreen periods.
    const size_t clusterOldestFrame = m_device->getCurrentFrameId() - RtxOptions::numFramesToKeepGeometryData() * 10;
    const uint32_t currentBufferTableSize = m_bufferCache.getActiveCount();
    auto blasEntryGarbageCollection = [&](auto& iter, auto& entries) -> void {
      if (iter->second.frameLastTouched < oldestFrame) {
        if (iter->second.isClusterBlas()) {
          // TEMP: Never GC cluster BLASes to test if GC is causing crash
          ++iter;
          return;
        }
        // Log buffer indices for GC'd entries to trace stale descriptor access
        const auto& geo = iter->second.modifiedGeometryData;
        if (iter->second.isClusterBlas()) {
          Logger::info(str::format("GC ClusterBlas: posIdx=", geo.positionBufferIndex,
              " normIdx=", geo.normalBufferIndex,
              " texIdx=", geo.texcoordBufferIndex,
              " col0Idx=", geo.color0BufferIndex,
              " idxIdx=", geo.indexBufferIndex,
              " prevPosIdx=", geo.previousPositionBufferIndex,
              " curBufTableSize=", currentBufferTableSize,
              " frameLastTouched=", iter->second.frameLastTouched,
              " curFrame=", m_device->getCurrentFrameId(),
              " histBuf0=0x", std::hex, (geo.historyBuffer[0] != nullptr ? (uint64_t)geo.historyBuffer[0]->getSliceHandle().handle : 0),
              " idxCacheBuf=0x", (geo.indexCacheBuffer != nullptr ? (uint64_t)geo.indexCacheBuffer->getSliceHandle().handle : 0), std::dec));
        } else {
          Logger::info(str::format("GC TriangleBlas: posIdx=", geo.positionBufferIndex,
              " idxIdx=", geo.indexBufferIndex,
              " curBufTableSize=", currentBufferTableSize,
              " frameLastTouched=", iter->second.frameLastTouched,
              " histBuf0=0x", std::hex, (geo.historyBuffer[0] != nullptr ? (uint64_t)geo.historyBuffer[0]->getSliceHandle().handle : 0),
              " idxCacheBuf=0x", (geo.indexCacheBuffer != nullptr ? (uint64_t)geo.indexCacheBuffer->getSliceHandle().handle : 0), std::dec));
        }
        onSceneObjectDestroyed(iter->second);
        iter = entries.erase(iter);
      } else {
        ++iter;
      }
    };

    // Garbage collection for BLAS/Scene objects
    //
    // When anti-culling is enabled, we need to check if any instances are outside frustum. Because in such
    // case the life of the instances will be extended and we need to keep the BLAS as well.
    if (!RtxOptions::AntiCulling::isObjectAntiCullingEnabled()) {
      auto& entries = m_drawCallCache.getEntries();
      if (m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
        for (auto iter = entries.begin(); iter != entries.end(); ) {
          blasEntryGarbageCollection(iter, entries);
        }
      }
    }
    else { // Implement anti-culling BLAS/Scene object GC
      fast_unordered_cache<const RtInstance*> outsideFrustumInstancesCache;

      auto& entries = m_drawCallCache.getEntries();
      for (auto iter = entries.begin(); iter != entries.end();) {
        bool isAllInstancesInCurrentBlasInsideFrustum = true;
        for (const RtInstance* instance : iter->second.getLinkedInstances()) {
          const Matrix4 objectToView = getCamera().getWorldToView(false) * instance->getTransform();

          bool isInsideFrustum = true;
          // Check for camera cut. Anti-Culling should NOT be enabled during a camera cut.
          // In some cases, we can't reliably detect a camera cut (e.g., when the game doesn't set up the View Matrix),
          // so we must disable Anti-Culling to prevent visual corruption.
          if (!getCamera().isCameraCut() && m_isAntiCullingSupported) {
            if (RtxOptions::needsMeshBoundingBox()) {
              const AxisAlignedBoundingBox& boundingBox = instance->getBlas()->input.getGeometryData().boundingBox;
              if (RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCulling()) {
                isInsideFrustum = boundingBoxIntersectsFrustumSAT(
                  getCamera(),
                  boundingBox.minPos,
                  boundingBox.maxPos,
                  objectToView,
                  RtxOptions::AntiCulling::Object::enableInfinityFarFrustum());
              } else {
                isInsideFrustum = boundingBoxIntersectsFrustum(getCamera().getFrustum(), boundingBox.minPos, boundingBox.maxPos, objectToView);
              }
            }
            else {
              // Fallback to check object center under view space
              auto getViewSpacePosition = [](const Matrix4& objectToView) -> float3 {
                return float3(objectToView[3][0], objectToView[3][1], objectToView[3][2]);
              };
              isInsideFrustum = getCamera().getFrustum().CheckSphere(getViewSpacePosition(objectToView), 0);
            }
          }

          // Only GC the objects inside the frustum to anti-frustum culling, this could cause significant performance impact
          // For the objects which can't be handled well with this algorithm, we will need game specific hash to force keeping them
          if (isInsideFrustum && !instance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling)) {
            instance->markAsInsideFrustum();
          } else {
            instance->markAsOutsideFrustum();
            isAllInstancesInCurrentBlasInsideFrustum = false;

            // Anti-Culling GC extension:
            // Eliminate duplicated instances that are outside of the game frustum.
            // This is used to handle cases:
            //   1. The game frustum is different to our frustum
            //   2. The game culling method is NOT frustum culling

            const XXH64_hash_t antiCullingHash = instance->calculateAntiCullingHash();

            auto it = outsideFrustumInstancesCache.find(antiCullingHash);
            if (it == outsideFrustumInstancesCache.end()) {
              // No duplication, just cache the current instance
              outsideFrustumInstancesCache[antiCullingHash] = instance;
            } else {
              const RtInstance* cachedInstance = it->second;
              if (instance->getId() != cachedInstance->getId()) {
                // Only keep the instance that is latest updated
                if (instance->getFrameLastUpdated() < cachedInstance->getFrameLastUpdated()) {
                  instance->markAsInsideFrustum();
                } else {
                  cachedInstance->markAsInsideFrustum();
                  it->second = instance;
                }
              }
            }
          }
        }

        // If all instances in current BLAS are inside the frustum, then use original GC logic to recycle BLAS Objects
        if (isAllInstancesInCurrentBlasInsideFrustum &&
            m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
          blasEntryGarbageCollection(iter, entries);
        } else { // If any instances are outside of the frustum in current BLAS, we need to keep the entity
          ++iter;
        }
      }
    }

    // Perform GC on the other managers
    m_instanceManager.garbageCollection();
    m_accelManager.garbageCollection();
    m_lightManager.garbageCollection(getCamera());
    m_rayPortalManager.garbageCollection();
  }

  void SceneManager::onDestroy() {
    m_accelManager.onDestroy();
    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onDestroy();
    }
  }

  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, RaytraceGeometry& inOutGeometry) {
    ScopedCpuProfileZone();
    ObjectCacheState result = ObjectCacheState::KBuildBVH;
    const RasterGeometry& input = drawCallState.getGeometryData();

    // Determine the optimal object state for this geometry
    if (!isNew) {
      // This is a geometry we've seen before, that requires updating
      //  'inOutGeometry' has valid historical data
      if (input.hashes[HashComponents::Indices] == inOutGeometry.hashes[HashComponents::Indices]) {
        // Check if the vertex positions have changed, requiring a BVH refit
        if (input.hashes[HashComponents::VertexPosition] == inOutGeometry.hashes[HashComponents::VertexPosition]
         && input.hashes[HashComponents::VertexShader] == inOutGeometry.hashes[HashComponents::VertexShader]
         && drawCallState.getSkinningState().boneHash == inOutGeometry.lastBoneHash) {
          result = ObjectCacheState::kUpdateInstance;
        } else {
          result = ObjectCacheState::kUpdateBVH;
        }
      }
    }

    // Copy the input directly to the output as a starting point for our modified geometry data
    RaytraceGeometry output = inOutGeometry;

    output.lastBoneHash = drawCallState.getSkinningState().boneHash;

    // Update draw parameters
    output.cullMode = input.cullMode;
    output.frontFace = input.frontFace;

    // Copy the hashes over
    output.hashes = input.hashes;

    if (!input.positionBuffer.defined()) {
      ONCE(Logger::err("processGeometryInfo: no position data on input detected"));
      return ObjectCacheState::kInvalid;
    }

    if (input.vertexCount == 0) {
      ONCE(Logger::err("processGeometryInfo: input data is violating some assumptions"));
      return ObjectCacheState::kInvalid;
    }

    // Set to 1 if inspection of the GeometryData structures contents on CPU is desired
    #define DEBUG_GEOMETRY_MEMORY 0
    // NOTE: Subdivision surfaces require CPU-accessible index buffers to extract quad topology
    // Make buffers HOST_VISIBLE for potential subdivision surface candidates
    // Be permissive: even if original indices aren't divisible by 6 (triangle strips),
    // they might become quad-compatible after strip-to-list conversion
    const bool isPotentialCandidate = (input.indexCount > 0 &&
                                       input.vertexCount >= 16 && input.vertexCount <= 65536);
    const bool needsCpuAccess = DEBUG_GEOMETRY_MEMORY || isPotentialCandidate;
    // NOTE: Don't combine HOST_VISIBLE with DEVICE_LOCAL - causes alignment issues
    const VkMemoryPropertyFlags memoryProperty = needsCpuAccess ?
      (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) :
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Assume we won't need this, and update the value if required
    output.previousPositionBuffer = RaytraceBuffer();

    const size_t vertexStride = (input.isVertexDataInterleaved() && input.areFormatsGpuFriendly()) ? input.positionBuffer.stride() : RtxGeometryUtils::computeOptimalVertexStride(input);

    switch (result) {
      case ObjectCacheState::KBuildBVH: {
        // Set up the ideal vertex params, if input vertices are interleaved, it's safe to assume the positionBuffer stride is the vertex stride
        output.vertexCount = input.vertexCount;

        const size_t vertexBufferSize = output.vertexCount * vertexStride;

        // Set up the ideal index params
        output.indexCount = input.isTopologyRaytraceReady() ? input.indexCount : RtxGeometryUtils::getOptimalTriangleListSize(input);
        const VkIndexType indexBufferType = input.isTopologyRaytraceReady() ? input.indexBuffer.indexType() : RtxGeometryUtils::getOptimalIndexFormat(output.vertexCount);
        const size_t indexStride = (indexBufferType == VK_INDEX_TYPE_UINT16) ? 2 : 4;

        // Make sure we're not stomping something else...
        assert(output.indexCacheBuffer == nullptr && output.historyBuffer[0] == nullptr);

        // Create a index buffer and vertex buffer we can use for raytracing.
        DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

        info.size = align(output.indexCount * indexStride, CACHE_LINE_SIZE);
        output.indexCacheBuffer = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Index Cache Buffer");

        if (!RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output)) {
          ONCE(Logger::err("processGeometryInfo: failed to cache index data on GPU"));
          return ObjectCacheState::kInvalid;
        }

        output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);

        info.size = align(vertexBufferSize, CACHE_LINE_SIZE);
        output.historyBuffer[0] = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output);

        break;
      }
      case ObjectCacheState::kUpdateBVH: {
        bool invalidateHistory = false;

        // Stride changed, so we must recreate the previous buffer and use identical data
        if (output.historyBuffer[0]->info().size != align(vertexStride * input.vertexCount, CACHE_LINE_SIZE)) {
          auto desc = output.historyBuffer[0]->info();
          desc.size = align(vertexStride * input.vertexCount, CACHE_LINE_SIZE);
          output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

          // Invalidate the current buffer
          output.historyBuffer[1] = nullptr;

          // Mark this object for realignment
          invalidateHistory = true;
        }

        // Use the previous updates vertex data for previous position lookup
        std::swap(output.historyBuffer[0], output.historyBuffer[1]);

        if (output.historyBuffer[0].ptr() == nullptr) {
          // First frame this object has been dynamic need to allocate a 2nd frame of data to preserve history.
          output.historyBuffer[0] = m_device->createBuffer(output.historyBuffer[1]->info(), memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
        } 

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output);

        // Sometimes, we need to invalidate history, do that here by copying the current buffer to the previous..
        if (invalidateHistory) {
          ctx->copyBuffer(output.historyBuffer[1], 0, output.historyBuffer[0], 0, output.historyBuffer[1]->info().size);
        }

        // Assign the previous buffer using the last slice (copy most params from the position, just change buffer)
        output.previousPositionBuffer = RaytraceBuffer(DxvkBufferSlice(output.historyBuffer[1], 0, output.positionBuffer.length()), output.positionBuffer.offsetFromSlice(), output.positionBuffer.stride(), output.positionBuffer.vertexFormat());
        break;
      }
      default:
        break;
    }

    // Update color buffer in BVH with DrawCallState
    // The user can disable/enable color buffer for specific materials, so we manually sync the DrawCallState and BVH here to keep the color buffer in BVH updated.
    // Note, we don't setup kUpdateBVH because it's too waste to update all buffers if only the color buffer needs to be updated.
    if (output.color0Buffer.defined() && !drawCallState.geometryData.color0Buffer.defined()) {
      // Remove the color buffer in BVH if the color buffer from drawcall is removed by ignoreBakedLighting
      output.color0Buffer = RaytraceBuffer();
    } else if (!output.color0Buffer.defined() && drawCallState.geometryData.color0Buffer.defined()) {
      // Write the color buffer back to BVH if the color buffer is enabled again
      const DxvkBufferSlice slice = DxvkBufferSlice(output.historyBuffer[0]);
      const auto& colorBuffer = drawCallState.geometryData.color0Buffer;
      output.color0Buffer = RaytraceBuffer(slice, colorBuffer.offsetFromSlice(), colorBuffer.stride(), colorBuffer.vertexFormat());
    }

    // Update buffers in the cache
    updateBufferCache(output);

    // Finalize our modified geometry data to the output
    inOutGeometry = output;

    return result;
  }


  void SceneManager::onFrameEnd(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();

    manageTextureVram();

    if (m_enqueueDelayedClear || m_pReplacer->checkForChanges(ctx)) {
      clear(ctx, true);
      m_enqueueDelayedClear = false;
    }

    m_cameraManager.onFrameEnd();
    m_instanceManager.onFrameEnd();
    m_previousFrameSceneAvailable = RtxOptions::enablePreviousTLAS();

    m_bufferCache.clear();
    {
      std::lock_guard lock { m_drawCallMeta.mutex };
      const uint8_t curTick = m_drawCallMeta.ticker;
      const uint8_t nextTick = (m_drawCallMeta.ticker + 1) % m_drawCallMeta.MaxTicks;

      m_drawCallMeta.ready[curTick] = true;

      m_drawCallMeta.infos[nextTick].clear();
      m_drawCallMeta.ready[nextTick] = false;
      m_drawCallMeta.ticker = nextTick;
    }

    m_terrainBaker->onFrameEnd(ctx);

    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onFrameEnd();
    }
    
    m_drawCallCache.logFrameStats();

    m_activePOMCount = 0;
    m_startInMediumMaterialIndex = BINDING_INDEX_INVALID;
    m_startInMediumMaterialIndex_inCache = UINT32_MAX;

    if (m_uniqueObjectSearchDistance != RtxOptions::uniqueObjectDistance()) {
      m_uniqueObjectSearchDistance = RtxOptions::uniqueObjectDistance();
      m_drawCallCache.rebuildSpatialMaps();
    }

    // Not currently safe to cache these across frames (due to texture indices and rtx options potentially changing)
    m_preCreationSurfaceMaterialMap.clear();

    m_thinOpaqueMaterialExist = false;
    m_sssMaterialExist = false;

    // execute graph updates after all garbage collection is complete (to avoid updating graphs that will just be deleted)
    // RtxOptions will still be pending, so any changes to them will apply next frame.
    m_graphManager.update(ctx);

    // Clear replacement material hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameReplacementMaterialHashes();
    
    // Clear mesh hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameMeshHashes();
  }

  void SceneManager::onFrameEndNoRTX() {
    m_cameraManager.onFrameEnd();
    m_instanceManager.onFrameEnd();
    manageTextureVram();
  }

  std::unordered_set<XXH64_hash_t> uniqueHashes;


  void SceneManager::submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData) {
    ScopedCpuProfileZone();

    // Process any completed async subdivision surfaces (once per frame)
    static uint32_t lastFrameProcessed = 0;
    uint32_t currentFrame = m_device->getCurrentFrameId();
    if (currentFrame != lastFrameProcessed) {
      Logger::warn(str::format("CHECKPOINT: submitDrawState NEW FRAME ", currentFrame));
      lastFrameProcessed = currentFrame;
      RtxMegaGeoBuilder* megaGeoBuilder = m_accelManager.getMegaGeoBuilder();
      if (megaGeoBuilder) {
        Logger::warn("CHECKPOINT: pre-processCompletedSurfaces");
        megaGeoBuilder->processCompletedSurfaces();
        Logger::warn("CHECKPOINT: post-processCompletedSurfaces");
      }
    }

    if (m_bufferCache.getTotalCount() >= kBufferCacheLimit && m_bufferCache.getActiveCount() >= kBufferCacheLimit) {
      ONCE(Logger::info("[RTX-Compatibility-Info] This application is pushing more unique buffers than is currently supported - some objects may not raytrace."));
      return;
    }

    if (input.getFogState().mode != D3DFOG_NONE) {
      XXH64_hash_t fogHash = input.getFogState().getHash();
      if (m_fogStates.find(fogHash) == m_fogStates.end()) {
        // Only do anything if we haven't seen this fog before.
        m_fogStates[fogHash] = input.getFogState();

        MaterialData* pFogReplacement = m_pReplacer->getReplacementMaterial(fogHash);
        if (pFogReplacement) {
          // Track this replacement material hash for hash checking
          trackReplacementMaterialHash(fogHash);
          // Fog has been replaced by a translucent material to start the camera in,
          // meaning that it was being used to indicate 'underwater' or something similar.
          if (pFogReplacement->getType() != MaterialDataType::Translucent) {
            Logger::warn(str::format("Fog replacement materials must be translucent.  Ignoring material for ", std::hex, m_fog.getHash()));
          } else {
            uint32_t id = UINT32_MAX;
            createSurfaceMaterial(*pFogReplacement, input, &id);
            assert(id != UINT32_MAX);
            m_startInMediumMaterialIndex_inCache = id;
          }
        } else if (m_fog.mode == D3DFOG_NONE) {
          // render the first unreplaced fog.
          m_fog = input.getFogState();
        }
      }
    }


    const XXH64_hash_t activeReplacementHash = input.getHash(RtxOptions::geometryAssetHashRule());
    
    // Track this mesh hash for mesh hash checking
    trackMeshHash(activeReplacementHash);
    
    std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForMesh(activeReplacementHash);

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash0);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash0: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash1);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash1: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    MaterialData renderMaterialData = determineMaterialData(overrideMaterialData, input);

    if (pReplacements != nullptr) {
      drawReplacements(ctx, &input, pReplacements, renderMaterialData);
    } else {
      processDrawCallState(ctx, input, renderMaterialData, nullptr, nullptr);
    }
  }

  MaterialData SceneManager::determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input) {
    // First see if we have an explicit override
    if (overrideMaterialData != nullptr) {
      return *overrideMaterialData;
    } 

    // test if any direct material replacements exist
    MaterialData* pReplacementMaterial = m_pReplacer->getReplacementMaterial(input.getMaterialData().getHash());
    if (pReplacementMaterial != nullptr) {
      // Make a copy - dont modify the replacement data.
      MaterialData renderMaterialData = *pReplacementMaterial;
      // merge in the input material from game
      renderMaterialData.mergeLegacyMaterial(input.getMaterialData());
      return renderMaterialData;
    }

    // Detect meshes that would have unstable hashes due to the vertex hash using vertex data from a shared vertex buffer.
    // TODO: Once the vertex hash only uses vertices referenced by the index buffer, this should be removed.
    const bool highlightUnsafeAnchor = RtxOptions::useHighlightUnsafeAnchorMode() && input.getGeometryData().indexBuffer.defined() && input.getGeometryData().vertexCount > input.getGeometryData().indexCount;
    if (highlightUnsafeAnchor) {
      const static MaterialData sHighlightMaterialData(OpaqueMaterialData(TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(),
                                                                          0.f, 1.f, Vector3(0.2f, 0.2f, 0.2f), 1.0f, 0.1f, 0.1f, Vector3(0.46f, 0.26f, 0.31f), true, 1, 1, 0, false, false, 200.f, true, false, BlendType::kAlpha, false, AlphaTestType::kAlways, 0, 0.0f, 0.0f, Vector3(), 0.0f, Vector3(), 0.0f, false, Vector3(), 0.0f, 0.0f,
                                                                          lss::Mdl::Filter::Nearest, lss::Mdl::WrapMode::Repeat, lss::Mdl::WrapMode::Repeat));
      return sHighlightMaterialData;
    }

    // Check if a Ray Portal override is needed
    size_t rayPortalTextureIndex;
    if (RtxOptions::getRayPortalTextureIndex(input.getMaterialData().getHash(), rayPortalTextureIndex)) {
      assert(rayPortalTextureIndex < maxRayPortalCount);
      assert(rayPortalTextureIndex < std::numeric_limits<uint8_t>::max());

      MaterialData renderMaterialData = input.getMaterialData().as<RayPortalMaterialData>();
      renderMaterialData.getRayPortalMaterialData().setRayPortalIndex(rayPortalTextureIndex);
      return renderMaterialData;
    }

    // Standard legacy material conversion
    return input.getMaterialData().as<OpaqueMaterialData>();
  }

  void SceneManager::createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance) {
    const float effectLightIntensity = RtxOptions::effectLightIntensity();
    if (effectLightIntensity <= 0.f)
      return;

    const RasterGeometry& geometryData = input.getGeometryData();

    const GeometryBufferData bufferData(geometryData);
    
    if (!bufferData.indexData && geometryData.indexCount > 0 || !bufferData.positionData)
      return;

    // Find centroid of point cloud.
    Vector3 centroid = Vector3();
    uint32_t counter = 0;
    if (geometryData.indexCount > 0) {
      for (uint32_t i = 0; i < geometryData.indexCount; i++) {
        const uint16_t index = bufferData.getIndex(i);
        centroid += bufferData.getPosition(index);
        ++counter;
      }
    } else {
      for (uint32_t i = 0; i < geometryData.vertexCount; i++) {
        centroid += bufferData.getPosition(i);
        ++counter;
      }
    }
    centroid /= (float) counter;
    
    const Vector4 renderingPos = input.getTransformData().objectToView * Vector4(centroid.x, centroid.y, centroid.z, 1.0f);
    // Note: False used in getViewToWorld since the renderingPos of the object is defined with respect to the game's object to view
    // matrix, not our freecam's, and as such we want to convert it back to world space using the matching matrix.
    const Vector4 worldPos{ getCamera().getViewToWorld(false) * Vector4d{ renderingPos } };

    RtLightShaping shaping{};

    float lightRadius = std::max(RtxOptions::effectLightRadius(), 1e-3f);
    const Vector3 lightPosition { worldPos.x, worldPos.y, worldPos.z };
    Vector3 lightRadiance;
    if (RtxOptions::effectLightPlasmaBall()) {
      // Todo: Make these options more configurable via config options.
      const double timeMilliseconds = static_cast<double>(GlobalTime::get().absoluteTimeMs());
      const double animationPhase = sin(timeMilliseconds * 0.006) * 0.5 + 0.5;
      lightRadiance = lerp(Vector3(1.f, 0.921f, 0.738f), Vector3(1.f, 0.521f, 0.238f), animationPhase);
    } else {
      const D3DCOLORVALUE originalColor = input.getMaterialData().getLegacyMaterial().Diffuse;
      lightRadiance = Vector3(originalColor.r, originalColor.g, originalColor.b) * RtxOptions::effectLightColor();
    }
    const float surfaceArea = 4.f * kPi * lightRadius * lightRadius;
    const float radianceFactor = 1e5f * effectLightIntensity / surfaceArea;
    lightRadiance *= radianceFactor;

    RtLight rtLight(RtSphereLight(lightPosition, lightRadiance, lightRadius, shaping));
    rtLight.isDynamic = true;

    m_lightManager.addLight(rtLight, input, RtLightAntiCullingType::MeshReplacement);
  }

  void SceneManager::drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, MaterialData& renderMaterialData) {
    ScopedCpuProfileZone();
    // TODO: Ideally we should create and track `replacementInstance` based on the draw call.  It currently relies on the
    // `findSimilarInstance` function of the first RtInstance created for the draw call, which is pretty clumsy.
    // We also should be tracking and garbage collecting the entire draw call together,
    // rather than doing each instance separately.
    ReplacementInstance* replacementInstance = nullptr;

    // Detect replacements of meshes that would have unstable hashes due to the vertex hash using vertex data from a shared vertex buffer.
    // TODO: Once the vertex hash only uses vertices referenced by the index buffer, this should be removed.
    const bool highlightUnsafeReplacement = RtxOptions::useHighlightUnsafeReplacementMode() &&
        input->getGeometryData().indexBuffer.defined() && input->getGeometryData().vertexCount > input->getGeometryData().indexCount;
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto& replacement = (*pReplacements)[i];
      RtInstance* instance = nullptr;
      if (replacement.includeOriginal) {
        DrawCallState newDrawCallState(*input);
        newDrawCallState.categories = replacement.categories.applyCategoryFlags(newDrawCallState.categories);
        instance = processDrawCallState(ctx, newDrawCallState, renderMaterialData);
      } else if (replacement.type == AssetReplacement::eMesh) {
        DrawCallTransforms transforms = input->getTransformData();
        
        transforms.objectToWorld = transforms.objectToWorld * replacement.replacementToObject;
        transforms.objectToView = transforms.objectToView * replacement.replacementToObject;

        if (!replacement.instancesToObject.empty()) {
          transforms.instancesToObject = &replacement.instancesToObject;
        } else {
          transforms.instancesToObject = nullptr;
        }
        
        // Mesh replacements dont support these.
        transforms.textureTransform = Matrix4();
        transforms.texgenMode = TexGenMode::None;

        DrawCallState newDrawCallState(*input);
        newDrawCallState.geometryData = replacement.geometry->data; // Note: Geometry Data replaced
        newDrawCallState.transformData = transforms;
        newDrawCallState.categories = replacement.categories.applyCategoryFlags(newDrawCallState.categories);

        // Note: Material Data replaced if a replacement is specified in the Mesh Replacement
        if (replacement.materialData != nullptr) {
          renderMaterialData = *replacement.materialData;
        }
        if (highlightUnsafeReplacement) {
          const static MaterialData sHighlightMaterialData(OpaqueMaterialData(TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(),
              0.f, 1.f, Vector3(0.2f, 0.2f, 0.2f), 1.f, 0.1f, 0.1f, Vector3(1.f, 0.f, 0.f), true, 1, 1, 0, false, false, 200.f, true, false, BlendType::kAlpha, false, AlphaTestType::kAlways, 0, 0.0f, 0.0f, Vector3(), 0.0f, Vector3(), 0.0f, false, Vector3(), 0.0f, 0.0f,
              lss::Mdl::Filter::Nearest, lss::Mdl::WrapMode::Repeat, lss::Mdl::WrapMode::Repeat));
          if ((GlobalTime::get().absoluteTimeMs()) / 200 % 2 == 0) {
            renderMaterialData = sHighlightMaterialData;
          }
        }

        const RtxParticleSystemDesc* pParticleSystemDesc = replacement.particleSystem.has_value() ? &replacement.particleSystem.value() : nullptr;

        RtInstance* existingInstance = replacementInstance ? replacementInstance->prims[i].getInstance() : nullptr;
        // Only use findSimilarInstance if we're processing the root of a replacement - all others should just rely on the existingInstance.
        instance = processDrawCallState(ctx, newDrawCallState, renderMaterialData, existingInstance, pParticleSystemDesc);
      }
      
      if (instance != nullptr) {
        if (replacementInstance == nullptr) {
          // first mesh in this replacement, so it becomes the root.
          replacementInstance = instance->getPrimInstanceOwner().getOrCreateReplacementInstance(instance, PrimInstance::Type::Instance, i, pReplacements->size());
        }
        if (replacementInstance->prims[i].getUntyped() == nullptr) {
          // First frame, need to set the replacement instance.
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        } else if (replacementInstance->prims[i].getInstance() != instance) {
          Logger::err(str::format("ReplacementInstance: instance returned by processDrawCallState is not the same as the one stored. index: ", i,"  mesh hash: ", std::hex, input->getHash(RtxOptions::geometryAssetHashRule())));
          assert(false && "instance returned by processDrawCallState is not the same as the one stored.");
        }
      }
    }

    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eLight) {
        if (replacementInstance == nullptr) {
          // TODO(TREX-1141) if we refactor instancing to depend on the pre-replacement drawcall instead
          // of the fully processed draw call, we can remove this requirement.
          Logger::err(str::format(
              "Light prims anchored to a mesh replacement must also include actual meshes.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        if (replacement.lightData.has_value()) {
          RtLight localLight = replacement.lightData->toRtLight();
          localLight.applyTransform(input->getTransformData().objectToWorld);
          
          // Handle all non-root lights as externally tracked lights - they'll be cleaned up when the root is garbage collected.
          // For mesh replacements, the root is always a mesh, so no need to handle root lights here.
          RtLight* existingLight = replacementInstance->prims[i].getLight();
          if (existingLight != nullptr) {
            if (existingLight->getPrimInstanceOwner().getReplacementInstance() != replacementInstance) {
              ONCE(assert(false && "light in a replacementInstance believes it is owned by a different replacementInstance."));
            }
            m_lightManager.updateExternallyTrackedLight(existingLight, localLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(localLight);
            newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
          }
        }
      }
    }

    // Create graphs associated with this replacement, if they haven't already been created.
    // Graphs are cleaned up when the replacementInstance is destroyed, which happens when the 
    // root instance is destroyed.
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eGraph && replacementInstance->prims[i].getGraph() == nullptr) {
        if (!replacement.graphState.has_value()) {
          Logger::err(str::format(
              "Graph prims missing graph state in mesh replacement.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        GraphInstance* graphInstance = m_graphManager.addInstance(ctx, replacement.graphState.value());
        if (graphInstance) {
          graphInstance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, graphInstance, PrimInstance::Type::Graph);
        }
      }
    }
  }

  void SceneManager::clearFogState() {
    ImGUI::SetFogStates(m_fogStates, m_fog.getHash());
    m_fog = FogState();
    m_fogStates.clear();
  }

  void SceneManager::updateBufferCache(RaytraceGeometry& newGeoData) {
    ScopedCpuProfileZone();
    if (newGeoData.indexBuffer.defined()) {
      newGeoData.indexBufferIndex = m_bufferCache.track(newGeoData.indexBuffer);
    } else {
      newGeoData.indexBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.normalBuffer.defined()) {
      newGeoData.normalBufferIndex = m_bufferCache.track(newGeoData.normalBuffer);
    } else {
      newGeoData.normalBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.color0Buffer.defined()) {
      newGeoData.color0BufferIndex = m_bufferCache.track(newGeoData.color0Buffer);
    } else {
      newGeoData.color0BufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.texcoordBuffer.defined()) {
      newGeoData.texcoordBufferIndex = m_bufferCache.track(newGeoData.texcoordBuffer);
    } else {
      newGeoData.texcoordBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.positionBuffer.defined()) {
      newGeoData.positionBufferIndex = m_bufferCache.track(newGeoData.positionBuffer);
    } else {
      newGeoData.positionBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.previousPositionBuffer.defined()) {
      newGeoData.previousPositionBufferIndex = m_bufferCache.track(newGeoData.previousPositionBuffer);
    } else {
      newGeoData.previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
    }
  }

  SceneManager::ObjectCacheState SceneManager::onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    // RTX Mega Geometry: Try to create subdivision surface BEFORE processGeometryInfo
    // This is critical because processGeometryInfo queues GPU upload which makes the
    // original buffer data inaccessible (isPendingGpuWrite becomes true)
    
    // DEBUG FEATURE: When enabled, meshes that don't qualify for MegaGeo subdivision will not be rendered at all
    // This helps debug which meshes are being rejected by the subdivision system
    static const bool s_debugRejectNonCandidates = false; // DISABLED - subdivision is disabled
    
    // Store rejection state for debug mode (used later to hide the instance)
    pBlas->isMegaGeoDebugRejected = false;
    
    const bool isMegaGeoCandidate = tryCreateSubdivisionSurface(ctx, drawCallState, pBlas);
    
    if (!isMegaGeoCandidate && s_debugRejectNonCandidates) {
      // Debug mode: Mark this BLAS as rejected so we can hide its instance later
      pBlas->isMegaGeoDebugRejected = true;
    }

    // This is a new object.
    ObjectCacheState result = processGeometryInfo<true>(ctx, drawCallState, pBlas->modifiedGeometryData);

    assert(result == ObjectCacheState::KBuildBVH);

    pBlas->frameLastUpdated = m_device->getCurrentFrameId();

    return result;
  }
  
  SceneManager::ObjectCacheState SceneManager::onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    if (pBlas->frameLastTouched == m_device->getCurrentFrameId()) {
      pBlas->cacheMaterial(drawCallState.getMaterialData());
      return SceneManager::ObjectCacheState::kUpdateInstance;
    }

    // ClusterBlas: skip triangle geometry processing, just update input state
    if (pBlas->isClusterBlas()) {
      pBlas->clearMaterialCache();
      pBlas->input = drawCallState;
      return SceneManager::ObjectCacheState::kUpdateInstance;
    }

    // TODO: If mesh is static, no need to do any of the below, just use the existing modifiedGeometryData and set result to kInstanceUpdate.
    ObjectCacheState result = processGeometryInfo<false>(ctx, drawCallState, pBlas->modifiedGeometryData);

    // We dont expect to hit the rebuild path here - since this would indicate an index buffer or other topological change, and that *should* trigger a new scene object (since the hash would change)
    assert(result != ObjectCacheState::KBuildBVH);

    if (result == ObjectCacheState::kUpdateBVH)
      pBlas->frameLastUpdated = m_device->getCurrentFrameId();
    
    pBlas->clearMaterialCache();
    pBlas->input = drawCallState; // cache the draw state for the next time.
    return result;
  }
  
  void SceneManager::onSceneObjectDestroyed(const BlasEntry& blas) {
    for (RtInstance* instance : blas.getLinkedInstances()) {
      instance->markForGarbageCollection();
      instance->markAsUnlinkedFromBlasEntryForGarbageCollection();
    }
  }

  void SceneManager::onInstanceAdded(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->linkInstance(&instance);
    }
  }

  void SceneManager::onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, const bool hasTransformChanged, const bool hasVerticesChanged, const bool isFirstUpdateThisFrame) {
    auto capturer = m_device->getCommon()->capturer();
    if (hasTransformChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
    }

    if (hasVerticesChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::PositionsUpdate);
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::NormalsUpdate);
    }

    // Create and bind the RT material
    const RtSurfaceMaterial& surfaceMaterial = createSurfaceMaterial(material, drawCall);

    if(isFirstUpdateThisFrame) {
      m_instanceManager.bindMaterial(instance, surfaceMaterial);
    }

    // Update portal
    if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
      m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
    }
  }

  void SceneManager::onInstanceDestroyed(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    // Some BLAS were cleared in the SceneManager::garbageCollection().
    // When a BLAS is destroyed, all instances that linked to it will be automatically unlinked. In such case we don't need to
    // call onInstanceDestroyed to double unlink the instances.
    // Note: This case often happens when BLAS are destroyed faster than instances. (e.g. numFramesToKeepGeometryData >= numFramesToKeepInstances)
    if (pBlas != nullptr && !instance.isUnlinkedForGC()) {
      pBlas->unlinkInstance(&instance);
    }
  }

  // Helper to populate the texture cache with this resource (and patch sampler if required for texture)
  void SceneManager::trackTexture(const TextureRef &inputTexture,
                                  uint32_t& textureIndex,
                                  bool hasTexcoords,
                                  bool async,
                                  uint16_t samplerFeedbackStamp) {
    // If no texcoords, no need to bind the texture
    if (!hasTexcoords) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to bind a texture to a mesh without UVs.  Was this intended?")));
      return;
    }

    auto& textureManager = m_device->getCommon()->getTextureManager();
    textureManager.addTexture(inputTexture, samplerFeedbackStamp, async, textureIndex);
  }

  RtInstance* SceneManager::processDrawCallState(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, MaterialData& renderMaterialData, RtInstance* existingInstance, const RtxParticleSystemDesc* pParticleSystemDesc) {
    ScopedCpuProfileZone();

    if (renderMaterialData.getIgnored()) {
      return nullptr;
    }

    ObjectCacheState result = ObjectCacheState::kInvalid;
    BlasEntry* pBlas = nullptr;
    if (m_drawCallCache.get(drawCallState, &pBlas) == DrawCallCache::CacheState::kExisted) {
      result = onSceneObjectUpdated(ctx, drawCallState, pBlas);
    } else {
      result = onSceneObjectAdded(ctx, drawCallState, pBlas);
    }
    
    assert(pBlas != nullptr);
    assert(result != ObjectCacheState::kInvalid);

    // Update the input state, so we always have a reference to the original draw call state
    pBlas->frameLastTouched = m_device->getCurrentFrameId();

    if (drawCallState.getSkinningState().numBones > 0 &&
        drawCallState.getGeometryData().numBonesPerVertex > 0 &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSkinning(drawCallState, pBlas->modifiedGeometryData);
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
    }

    // Note: The material data can be modified in instance manager
    RtInstance* instance = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, existingInstance);

    // DEBUG: Set to true to hide ALL non-MegaGeo geometry (only ClusterBlas surfaces visible)
    // Waits until ClusterBlas instances exist before hiding, to avoid empty TLAS crash
    static const bool s_megaGeoOnly = false;
    static bool s_hasSeenClusterBlas = false;
    if (pBlas->isClusterBlas()) {
      s_hasSeenClusterBlas = true;
    }
    if (instance && s_megaGeoOnly && s_hasSeenClusterBlas && !pBlas->isClusterBlas()) {
      instance->setHidden(true);
    }

    // DEBUG FEATURE: Hide instances for meshes rejected by MegaGeo debug mode
    if (instance && pBlas->isMegaGeoDebugRejected) {
      instance->setHidden(true);
    }

    // Check if a light should be created for this Material
    if (instance && RtxOptions::shouldConvertToLight(drawCallState.getMaterialData().getHash())) {
      createEffectLight(ctx, drawCallState, instance);
    }

    const bool objectPickingActive = m_device->getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();

    if (objectPickingActive && instance && g_allowMappingLegacyHashToObjectPickingValue) {
      auto meta = DrawCallMetaInfo {};
      {
        XXH64_hash_t h;
        h = drawCallState.getMaterialData().getColorTexture().getImageHash();
        if (h != kEmptyHash) {
          meta.legacyTextureHash = h;
        }
        h = drawCallState.getMaterialData().getColorTexture2().getImageHash();
        if (h != kEmptyHash) {
          meta.legacyTextureHash2 = h;
        }
      }

      {
        std::lock_guard lock { m_drawCallMeta.mutex };
        auto [iter, isNew] = m_drawCallMeta.infos[m_drawCallMeta.ticker].emplace(instance->surface.objectPickingValue, meta);
        ONCE_IF_FALSE(isNew, Logger::warn(
          "Found multiple draw calls with the same \'objectPickingValue\'. "
          "Ignoring further MetaInfo-s, some objects might be not be available through object picking"));
      }
    }

    // Priority ordering for particle system descriptors is: Mesh, Material, Texture.  This matches the implementation in toolkit.
    // By this point, pParticleSystemDesc will contain the information from a mesh replacement (if one exists), so we just handle
    // materials replacements, and texture taggin categories below.
    if (!pParticleSystemDesc) {
      pParticleSystemDesc = renderMaterialData.getParticleSystemDesc();
    }
    if (!pParticleSystemDesc && drawCallState.categories.test(InstanceCategories::ParticleEmitter)) {
      pParticleSystemDesc = &RtxParticleSystemManager::createGlobalParticleSystemDesc();
    }
    if (instance && pParticleSystemDesc) {
      RtxParticleSystemManager& particleSystem = device()->getCommon()->metaParticleSystem();
      particleSystem.spawnParticles(ctx.ptr(), *pParticleSystemDesc, instance->getVectorIdx(), drawCallState, renderMaterialData);

      if (pParticleSystemDesc->hideEmitter) {
        instance->setHidden(true);
      }
    }

    return instance; 
  }

  const RtSurfaceMaterial& SceneManager::createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                               const DrawCallState& drawCallState,
                                                               uint32_t* out_indexInCache) {
    ScopedCpuProfileZone();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();
    const auto renderMaterialDataType = renderMaterialData.getType();

    // We're going to use this to create a modified sampler for replacement textures.
    // Legacy and replacement materials should follow same filtering but due to lack of override capability per texture
    // legacy textures use original sampler to stay true to the original intent while replacements use more advanced filtering
    // for better quality by default.
    const Rc<DxvkSampler>& samplerOverride = renderMaterialData.getSamplerOverride();
    Rc<DxvkSampler> sampler = samplerOverride;
    // If the original sampler if valid and there isnt an override sampler
    // go ahead with patching and maybe merging the sampler states
    if (samplerOverride == nullptr && drawCallState.getMaterialData().getSampler().ptr() != nullptr) {
      DxvkSamplerCreateInfo samplerInfo = drawCallState.getMaterialData().getSampler()->info(); // Use sampler create info struct as convenience
      renderMaterialData.populateSamplerInfo(samplerInfo);

      sampler = patchSampler(samplerInfo.magFilter,
                             samplerInfo.addressModeU, samplerInfo.addressModeV, samplerInfo.addressModeW,
                             samplerInfo.borderColor);
    }
    uint32_t samplerIndex = trackSampler(sampler);
    uint32_t samplerIndex2 = UINT32_MAX;
    if (renderMaterialDataType == MaterialDataType::RayPortal) {
      samplerIndex2 = trackSampler(drawCallState.getMaterialData().getSampler2());
    }

    XXH64_hash_t preCreationHash = renderMaterialData.getHash();
    preCreationHash = XXH64(&samplerIndex, sizeof(samplerIndex), preCreationHash);
    preCreationHash = XXH64(&samplerIndex2, sizeof(samplerIndex2), preCreationHash);
    preCreationHash = XXH64(&hasTexcoords, sizeof(hasTexcoords), preCreationHash);
    preCreationHash = XXH64(&drawCallState.isUsingRaytracedRenderTarget, sizeof(drawCallState.isUsingRaytracedRenderTarget), preCreationHash);

    auto iter = m_preCreationSurfaceMaterialMap.find(preCreationHash);
    if (iter != m_preCreationSurfaceMaterialMap.end()) {
      if (out_indexInCache) {
        *out_indexInCache = iter->second;
      }
      return m_surfaceMaterialCache.at(iter->second);
    }

    std::optional<RtSurfaceMaterial> surfaceMaterial;

    if (renderMaterialDataType == MaterialDataType::Opaque || drawCallState.isUsingRaytracedRenderTarget) {
      uint32_t albedoOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t tangentTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceMaterialIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceTransmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceThicknessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceSingleScatteringAlbedoTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      float anisotropy;
      float emissiveIntensity;
      Vector4 albedoOpacityConstant;
      float roughnessConstant;
      float metallicConstant;
      Vector3 emissiveColorConstant;
      bool enableEmissive;
      bool thinFilmEnable = false;
      bool alphaIsThinFilmThickness = false;
      float thinFilmThicknessConstant = 0.0f;
      float displaceIn = 0.0f;
      float displaceOut = 0.0f;
      bool isUsingRaytracedRenderTarget = drawCallState.isUsingRaytracedRenderTarget;
      uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

      Vector3 subsurfaceTransmittanceColor(0.0f, 0.0f, 0.0f);
      float subsurfaceMeasurementDistance = 0.0f;
      Vector3 subsurfaceSingleScatteringAlbedo(0.0f, 0.0f, 0.0f);
      float subsurfaceVolumetricAnisotropy = 0.0f;

      float subsurfaceRadiusScale = 0.0f;
      float subsurfaceMaxSampleRadius = 0.0f;

      bool ignoreAlphaChannel = false;

      constexpr Vector4 kWhiteModeAlbedo = Vector4(0.7f, 0.7f, 0.7f, 1.0f);

      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();

      if (RtxOptions::useWhiteMaterialMode()) {
        albedoOpacityConstant = kWhiteModeAlbedo;
        metallicConstant = 0.f;
        roughnessConstant = 1.f;
      } else {
        if (opaqueMaterialData.getAlbedoOpacityTexture().getManagedTexture() != nullptr) {
          samplerFeedbackStamp = opaqueMaterialData.getAlbedoOpacityTexture().getManagedTexture()->m_samplerFeedbackStamp;
        }

        trackTexture(opaqueMaterialData.getAlbedoOpacityTexture(), albedoOpacityTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getRoughnessTexture(), roughnessTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getMetallicTexture(), metallicTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

        albedoOpacityConstant.xyz() = opaqueMaterialData.getAlbedoConstant();
        albedoOpacityConstant.w = opaqueMaterialData.getOpacityConstant();
        metallicConstant = opaqueMaterialData.getMetallicConstant();
        roughnessConstant = opaqueMaterialData.getRoughnessConstant();
      }

      trackTexture(opaqueMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getTangentTexture(), tangentTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getHeightTexture(), heightTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

      emissiveIntensity = opaqueMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();
      emissiveColorConstant = opaqueMaterialData.getEmissiveColorConstant();
      enableEmissive = opaqueMaterialData.getEnableEmission();
      anisotropy = opaqueMaterialData.getAnisotropyConstant();
        
      thinFilmEnable = opaqueMaterialData.getEnableThinFilm();
      alphaIsThinFilmThickness = opaqueMaterialData.getAlphaIsThinFilmThickness();
      thinFilmThicknessConstant = opaqueMaterialData.getThinFilmThicknessConstant();
      displaceIn = opaqueMaterialData.getDisplaceIn();
      displaceOut = opaqueMaterialData.getDisplaceOut();

      ignoreAlphaChannel = opaqueMaterialData.getIgnoreAlphaChannel();

      subsurfaceMeasurementDistance = opaqueMaterialData.getSubsurfaceMeasurementDistance() * RtxOptions::SubsurfaceScattering::surfaceThicknessScale();

      const bool isSubsurfaceScatteringDiffusionProfile = opaqueMaterialData.getSubsurfaceDiffusionProfile();

      if ((RtxOptions::SubsurfaceScattering::enableThinOpaque()       && subsurfaceMeasurementDistance > 0.0f) ||
          (RtxOptions::SubsurfaceScattering::enableDiffusionProfile() && isSubsurfaceScatteringDiffusionProfile)) {

        subsurfaceTransmittanceColor = opaqueMaterialData.getSubsurfaceTransmittanceColor();
        subsurfaceVolumetricAnisotropy = opaqueMaterialData.getSubsurfaceVolumetricAnisotropy();

        if (isSubsurfaceScatteringDiffusionProfile) {
          // NOTE: reuse of the variable!
          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceRadius(); 
          subsurfaceMaxSampleRadius = std::max(0.F, opaqueMaterialData.getSubsurfaceMaxSampleRadius());
          subsurfaceRadiusScale = std::max(opaqueMaterialData.getSubsurfaceRadiusScale(), 1e-5f);
          assert(subsurfaceRadiusScale > 0);

          m_sssMaterialExist = true;
        } else /* if thin opaque */ {
          assert(subsurfaceMeasurementDistance > 0);

          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceSingleScatteringAlbedo();
          subsurfaceMaxSampleRadius = 0;
          subsurfaceRadiusScale = -1;
          assert(subsurfaceRadiusScale < 0);  // if < 0, then shaders assume that
                                              // this material is not SubsurfaceScatter, but just SingleScatter
                                              // same here, but <0.F

          m_thinOpaqueMaterialExist = true;
        }

        if (RtxOptions::SubsurfaceScattering::enableTextureMaps()) {
          trackTexture(opaqueMaterialData.getSubsurfaceTransmittanceTexture(), subsurfaceTransmittanceTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

          if (isSubsurfaceScatteringDiffusionProfile) {
            // NOTE: reuse of 'subsurfaceSingleScatteringAlbedoTextureIndex' variable!
            trackTexture(opaqueMaterialData.getSubsurfaceRadiusTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
          } else {
            trackTexture(opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
            trackTexture(opaqueMaterialData.getSubsurfaceThicknessTexture(), subsurfaceThicknessTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
          }
        }

        const auto subsurfaceMaterial = RtSubsurfaceMaterial{
          subsurfaceTransmittanceTextureIndex,
          subsurfaceThicknessTextureIndex,
          subsurfaceSingleScatteringAlbedoTextureIndex,
          subsurfaceTransmittanceColor,
          subsurfaceMeasurementDistance,
          subsurfaceSingleScatteringAlbedo,
          subsurfaceVolumetricAnisotropy,
          subsurfaceRadiusScale,
          subsurfaceMaxSampleRadius,
        };
        subsurfaceMaterialIndex = m_surfaceMaterialExtensionCache.track(subsurfaceMaterial);
      }

      const RtOpaqueSurfaceMaterial opaqueSurfaceMaterial{
        albedoOpacityTextureIndex, normalTextureIndex,
        tangentTextureIndex, heightTextureIndex, roughnessTextureIndex,
        metallicTextureIndex, emissiveColorTextureIndex,
        anisotropy, emissiveIntensity,
        albedoOpacityConstant,
        roughnessConstant, metallicConstant,
        emissiveColorConstant, enableEmissive,
        ignoreAlphaChannel, thinFilmEnable, alphaIsThinFilmThickness,
        thinFilmThicknessConstant, samplerIndex, displaceIn, displaceOut, 
        subsurfaceMaterialIndex, isUsingRaytracedRenderTarget,
        samplerFeedbackStamp,
      };

      if (opaqueSurfaceMaterial.hasValidDisplacement()) {
        ++m_activePOMCount;
      }

      surfaceMaterial.emplace(opaqueSurfaceMaterial);
    } else if (renderMaterialDataType == MaterialDataType::Translucent) {
      const auto& translucentMaterialData = renderMaterialData.getTranslucentMaterialData();

      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t transmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      trackTexture(translucentMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords);
      trackTexture(translucentMaterialData.getTransmittanceTexture(), transmittanceTextureIndex, hasTexcoords);
      trackTexture(translucentMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords);

      float refractiveIndex = translucentMaterialData.getRefractiveIndex();
      Vector3 transmittanceColor = translucentMaterialData.getTransmittanceColor();
      float transmittanceMeasureDistance = translucentMaterialData.getTransmittanceMeasurementDistance();
      Vector3 emissiveColorConstant = translucentMaterialData.getEmissiveColorConstant();
      bool enableEmissive = translucentMaterialData.getEnableEmission();
      float emissiveIntensity = translucentMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();
      bool isThinWalled = translucentMaterialData.getEnableThinWalled();
      float thinWallThickness = translucentMaterialData.getThinWallThickness();
      bool useDiffuseLayer = translucentMaterialData.getEnableDiffuseLayer();

      const RtTranslucentSurfaceMaterial translucentSurfaceMaterial{
        normalTextureIndex, transmittanceTextureIndex, emissiveColorTextureIndex,
        refractiveIndex,
        transmittanceMeasureDistance, transmittanceColor,
        enableEmissive, emissiveIntensity, emissiveColorConstant,
        isThinWalled, thinWallThickness, useDiffuseLayer, samplerIndex
      };

      surfaceMaterial.emplace(translucentSurfaceMaterial);
    } else if (renderMaterialDataType == MaterialDataType::RayPortal) {
      const auto& rayPortalMaterialData = renderMaterialData.getRayPortalMaterialData();

      uint32_t maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture(), maskTextureIndex, hasTexcoords, false);
      uint32_t maskTextureIndex2 = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture2(), maskTextureIndex2, hasTexcoords, false);

      uint8_t rayPortalIndex = rayPortalMaterialData.getRayPortalIndex();
      float rotationSpeed = rayPortalMaterialData.getRotationSpeed();
      bool enableEmissive = rayPortalMaterialData.getEnableEmission();
      float emissiveIntensity = rayPortalMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();

      const RtRayPortalSurfaceMaterial rayPortalSurfaceMaterial{
        maskTextureIndex, maskTextureIndex2, rayPortalIndex,
        rotationSpeed, enableEmissive, emissiveIntensity, samplerIndex, samplerIndex2
      };

      surfaceMaterial.emplace(rayPortalSurfaceMaterial);
    }

    assert(surfaceMaterial.has_value());
    assert(surfaceMaterial->validate());

    // Cache this
    const uint32_t index = m_surfaceMaterialCache.track(*surfaceMaterial);
    m_preCreationSurfaceMaterialMap[preCreationHash] = index;
    if (out_indexInCache) {
      *out_indexInCache = index;
    }
    return m_surfaceMaterialCache.at(index);
  }

  std::optional<XXH64_hash_t> SceneManager::findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue) {
    std::lock_guard lock { m_drawCallMeta.mutex };

    auto tryFindIn = [](const std::unordered_map<ObjectPickingValue, DrawCallMetaInfo>& table, ObjectPickingValue toFind)
      -> std::optional<XXH64_hash_t> {
      auto found = table.find(toFind);
      if (found != table.end()) {
        const DrawCallMetaInfo& meta = found->second;
        if (meta.legacyTextureHash != kEmptyHash) {
          return meta.legacyTextureHash;
        }
      }
      return std::nullopt;
    };

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        if (auto h = tryFindIn(m_drawCallMeta.infos[tick], objectPickingValue)) {
          return h;
        }
      }
    }
    return std::nullopt;
  }

  std::vector<ObjectPickingValue> SceneManager::gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash) {
    std::lock_guard lock { m_drawCallMeta.mutex };
    assert(texHash != kEmptyHash);

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };

    auto correspondingValues = std::vector<ObjectPickingValue> {};
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        for (const auto& [pickingValue, meta] : m_drawCallMeta.infos[tick]) {
          if (texHash == meta.legacyTextureHash) {
            correspondingValues.push_back(pickingValue);
          } else if (texHash == meta.legacyTextureHash2) {
            correspondingValues.push_back(pickingValue);
          }
        }
        break;
      }
    }
    return correspondingValues;
  }

  SceneManager::SamplerIndex SceneManager::trackSampler(Rc<DxvkSampler> sampler) {
    if (sampler == nullptr) {
      ONCE(Logger::warn("Found a null sampler. Fallback to linear-repeat"));
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VkClearColorValue {});
    }
    return m_samplerCache.track(sampler);
  }

  Rc<DxvkSampler> SceneManager::patchSampler( const VkFilter filterMode,
                                              const VkSamplerAddressMode addressModeU,
                                              const VkSamplerAddressMode addressModeV,
                                              const VkSamplerAddressMode addressModeW,
                                              const VkClearColorValue borderColor) {
    auto& resourceManager = m_device->getCommon()->getResources();
    // Create a sampler to account for DLSS lod bias and any custom filtering overrides the user has set
    return resourceManager.getSampler(
      filterMode,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      addressModeU,
      addressModeV,
      addressModeW,
      borderColor,
      getTotalMipBias(),
      RtxOptions::useAnisotropicFiltering());
  }

  void SceneManager::addLight(const D3DLIGHT9& light) {
    ScopedCpuProfileZone();
    // Attempt to convert the D3D9 light to RT

    std::optional<LightData> lightData = LightData::tryCreate(light);

    // Note: Skip adding this light if it is somehow malformed such that it could not be created.
    if (!lightData.has_value()) {
      return;
    }

    const RtLight rtLight = lightData->toRtLight();
    const std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForLight(rtLight.getInitialHash());

    if (pReplacements) {
      const Matrix4 lightTransform = LightUtils::getLightTransform(light);

      ReplacementInstance* replacementInstance = nullptr;

      // TODO(TREX-1091) to implement meshes as light replacements, replace the below loop with a call to drawReplacements.
      for (size_t i = 0; i < pReplacements->size(); i++) {
        const auto& replacement = (*pReplacements)[i];
        if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          LightData replacementLight = replacement.lightData.value();

          // Merge the d3d9 light into replacements based on overrides
          replacementLight.merge(light);

          // Convert to runtime light
          RtLight rtReplacementLight = replacementLight.toRtLight(&rtLight);

          // Transform the replacement light by the legacy light
          if (replacementLight.relativeTransform()) {
            rtReplacementLight.applyTransform(lightTransform); // note: we dont need to consider the transform of parent replacement light in this scenario, this is detected on mod load and so absolute transform is used
          }

          if (replacementInstance == nullptr) {
            // Handle the root light as a normal light.
            RtLight* newLight;

            // Setup Light Replacement for Anti-Culling
            RtLightAntiCullingType antiCullingType = RtLightAntiCullingType::Ignore;
            if (RtxOptions::AntiCulling::isLightAntiCullingEnabled() && rtLight.getType() == RtLightType::Sphere) {
              antiCullingType = RtLightAntiCullingType::LightReplacement;
            }

            // Apply the light
            newLight = m_lightManager.addLight(rtReplacementLight, antiCullingType);

            // Setup tracking for all the lights created for this replacement.
            if (newLight != nullptr) {
              // This is the first light created, so it will be the root.
              replacementInstance = newLight->getPrimInstanceOwner().getOrCreateReplacementInstance(newLight, PrimInstance::Type::Light, i, pReplacements->size());
            }
          } else {
            // Handle all non-root lights as externally tracked lights - they'll be cleaned up when the root is garbage collected.
            RtLight* existingLight = replacementInstance->prims[i].getLight();
            if (existingLight != nullptr) {
              m_lightManager.updateExternallyTrackedLight(existingLight, rtReplacementLight);
            } else {
              RtLight* newLight = m_lightManager.createExternallyTrackedLight(rtReplacementLight);
              newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
            }
          }
        } else {
          assert(false); // We don't support meshes as children of lights yet.
        }
      }
    } else {
      // This is a light coming from the game directly, so use the appropriate API for filter rules
      m_lightManager.addGameLight(light.Type, rtLight);
    }
  }

  void SceneManager::prepareSceneData(Rc<RtxContext> ctx, DxvkBarrierSet& execBarriers) {
    Logger::warn("CHECKPOINT: prepareSceneData ENTER");
    ScopedGpuProfileZone(ctx, "Build Scene");

  #ifdef REMIX_DEVELOPMENT
    if (m_device->getCurrentFrameId() == RtxOptions::dumpAllInstancesOnFrame()) {
      // Print all RtInstances for debugging
      printAllRtInstances();
    }
  #endif

    // Needs to happen before garbageCollection to avoid destroying dynamic lights
    Logger::warn("CHECKPOINT: pre-dynamicLightMatching");
    m_lightManager.dynamicLightMatching();

    Logger::warn("CHECKPOINT: pre-garbageCollection");
    garbageCollection();
    Logger::warn("CHECKPOINT: post-garbageCollection");

    m_graphManager.applySceneOverrides(ctx);
    Logger::warn("CHECKPOINT: post-applySceneOverrides");

    m_terrainBaker->prepareSceneData(ctx);
    Logger::warn("CHECKPOINT: post-terrainBaker");

    auto& textureManager = m_device->getCommon()->getTextureManager();
    m_bindlessResourceManager.prepareSceneData(ctx, textureManager.getTextureTable(), getBufferTable(), getSamplerTable());
    Logger::warn("CHECKPOINT: post-bindlessResourceManager");

    // If there are no instances, we should do nothing!
    if (m_instanceManager.getActiveCount() == 0) {
      // Clear the ray portal data before the next frame
      m_rayPortalManager.clear();
      return;
    }

    m_rayPortalManager.prepareSceneData(ctx);
    // Note: only main camera needs to be teleportation corrected as only that one is used for ray tracing & denoising
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::Main));
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::ViewModel));
    m_rayPortalManager.createVirtualCameras(m_cameraManager);
    const bool didTeleport = m_rayPortalManager.detectTeleportationAndCorrectCameraHistory(
      m_cameraManager.getCamera(CameraType::Main),
      m_cameraManager.isCameraValid(CameraType::ViewModel) ? &m_cameraManager.getCamera(CameraType::ViewModel) : nullptr);

    if (m_cameraManager.isCameraCutThisFrame()) {
      // Ignore camera cut events on teleportation so we don't flush the caches
      if (!didTeleport) {
        Logger::info(str::format("Camera cut detected on frame ", m_device->getCurrentFrameId()));
        m_enqueueDelayedClear = true;
      }
    }

    // Initialize/remove opacity micromap manager
    if (RtxOptions::getEnableOpacityMicromap()) {
      if (!m_opacityMicromapManager.get() || 
          // Reset the manager on camera cuts
          m_enqueueDelayedClear) {
        if (m_opacityMicromapManager.get())
          m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());

        m_opacityMicromapManager = std::make_unique<OpacityMicromapManager>(m_device);
        m_instanceManager.addEventHandler(m_opacityMicromapManager->getInstanceEventHandler());
        Logger::info("[RTX] Opacity Micromap: enabled");
      }
    } else if (m_opacityMicromapManager.get()) {
      m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());
      m_opacityMicromapManager = nullptr;
      Logger::info("[RTX] Opacity Micromap: disabled");
    }

    RtxParticleSystemManager& particles = m_device->getCommon()->metaParticleSystem();
    particles.simulate(ctx.ptr());

    m_instanceManager.findPortalForVirtualInstances(m_cameraManager, m_rayPortalManager);
    m_instanceManager.createViewModelInstances(ctx, m_cameraManager, m_rayPortalManager);
    m_instanceManager.createPlayerModelVirtualInstances(ctx, m_cameraManager, m_rayPortalManager);

    Logger::warn("CHECKPOINT: pre-mergeInstancesIntoBlas");
    m_accelManager.mergeInstancesIntoBlas(ctx, execBarriers, textureManager.getTextureTable(), m_cameraManager, m_instanceManager, m_opacityMicromapManager.get());
    Logger::warn("CHECKPOINT: post-mergeInstancesIntoBlas");

    // Call on the other managers to prepare their GPU data for the current scene
    Logger::warn("CHECKPOINT: pre-accelManager-prepareSceneData");
    m_accelManager.prepareSceneData(ctx, execBarriers, m_instanceManager);
    Logger::warn("CHECKPOINT: post-accelManager-prepareSceneData");
    m_lightManager.prepareSceneData(ctx, m_cameraManager);

    // Build the TLAS
    m_accelManager.buildTlas(ctx);
    Logger::warn("CHECKPOINT: post-buildTlas in prepareSceneData");

    // Todo: These updates require a lot of temporary buffer allocations and memcopies, ideally we should memcpy directly into a mapped pointer provided by Vulkan,
    // but we have to create a buffer to pass to DXVK's updateBuffer for now.
    {
      // Allocate the instance buffer and copy its contents from host to device memory
      DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      // Surface Material buffer
      if (m_surfaceMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterials");
        Logger::warn(str::format("CHECKPOINT: updateSurfaceMaterials surfaceCount=", m_accelManager.getSurfaceCount(), " matCacheCount=", m_surfaceMaterialCache.getTotalCount()));
        // Note: We duplicate the materials in the buffer so we don't have to do pointer chasing on the GPU (i.e. rather than BLAS->Surface->Material, do, BLAS->Surface, BLAS->Material)
        size_t surfaceMaterialsGPUSize = m_accelManager.getSurfaceCount() * kSurfaceMaterialGPUSize;
        if (m_startInMediumMaterialIndex_inCache != UINT32_MAX) {
          surfaceMaterialsGPUSize += kSurfaceMaterialGPUSize;
        }

        info.size = align(surfaceMaterialsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_surfaceMaterialBuffer == nullptr || info.size > m_surfaceMaterialBuffer->info().size) {
          m_surfaceMaterialBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Buffer");
        }

        std::size_t dataOffset = 0;
        uint16_t surfaceIndex = 0;
        std::vector<unsigned char> surfaceMaterialsGPUData(surfaceMaterialsGPUSize);
        const auto& orderedInstances = m_accelManager.getOrderedInstances();
        const auto& matTable = m_surfaceMaterialCache.getObjectTable();
        Logger::warn(str::format("CHECKPOINT: matLoop orderedInstances=", orderedInstances.size(), " matTableSize=", matTable.size(), " gpuSize=", surfaceMaterialsGPUSize));
        for (auto&& pInstance : orderedInstances) {
          uint32_t matIdx = pInstance->surface.surfaceMaterialIndex;
          if (matIdx >= matTable.size()) {
            Logger::err(str::format("CRASH: surfaceMaterialIndex=", matIdx, " >= matTableSize=", matTable.size(), " at surfaceIndex=", surfaceIndex, " isCluster=", pInstance->getBlas() ? pInstance->getBlas()->isClusterBlas() : false));
            surfaceIndex++;
            continue;
          }
          auto&& surfaceMaterial = matTable[matIdx];
          surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          surfaceIndex++;
        }
        Logger::warn(str::format("CHECKPOINT: matLoop done surfaceIndex=", surfaceIndex));

        if (m_startInMediumMaterialIndex_inCache != UINT32_MAX) {
          auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[m_startInMediumMaterialIndex_inCache];
          surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          m_startInMediumMaterialIndex = surfaceIndex;
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialsGPUSize);
        assert(surfaceMaterialsGPUData.size() == surfaceMaterialsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialBuffer, 0, surfaceMaterialsGPUData.size(), surfaceMaterialsGPUData.data());
        Logger::warn("CHECKPOINT: surfaceMaterial writeToBuffer done");
      }

      // Surface Material Extension Buffer
      if (m_surfaceMaterialExtensionCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterialExtensions");
        const auto surfaceMaterialExtensionsGPUSize = m_surfaceMaterialExtensionCache.getTotalCount() * kSurfaceMaterialGPUSize;

        info.size = align(surfaceMaterialExtensionsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_surfaceMaterialExtensionBuffer == nullptr || info.size > m_surfaceMaterialExtensionBuffer->info().size) {
          m_surfaceMaterialExtensionBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Extension Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> surfaceMaterialExtensionsGPUData(surfaceMaterialExtensionsGPUSize);

        uint16_t surfaceIndex = 0;
        for (auto&& surfaceMaterialExtension : m_surfaceMaterialExtensionCache.getObjectTable()) {
          surfaceMaterialExtension.writeGPUData(surfaceMaterialExtensionsGPUData.data(), dataOffset, surfaceIndex);
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialExtensionsGPUSize);
        assert(surfaceMaterialExtensionsGPUData.size() == surfaceMaterialExtensionsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionsGPUData.size(), surfaceMaterialExtensionsGPUData.data());
      }
      Logger::warn("CHECKPOINT: post-materialExtensions");

      // Volume Material buffer
      if (m_volumeMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateVolumeMaterials");
        const auto volumeMaterialsGPUSize = m_volumeMaterialCache.getTotalCount() * kVolumeMaterialGPUSize;

        info.size = align(volumeMaterialsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_volumeMaterialBuffer == nullptr || info.size > m_volumeMaterialBuffer->info().size) {
          m_volumeMaterialBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Volume Material Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> volumeMaterialsGPUData(volumeMaterialsGPUSize);

        for (auto&& volumeMaterial : m_volumeMaterialCache.getObjectTable()) {
          volumeMaterial.writeGPUData(volumeMaterialsGPUData.data(), dataOffset);
        }

        assert(dataOffset == volumeMaterialsGPUSize);
        assert(volumeMaterialsGPUData.size() == volumeMaterialsGPUSize);

        ctx->writeToBuffer(m_volumeMaterialBuffer, 0, volumeMaterialsGPUData.size(), volumeMaterialsGPUData.data());
      }
    }
    Logger::warn("CHECKPOINT: post-allMaterialUploads");

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // Update stats
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBlasCount, AccelManager::getBlasCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBufferCount, m_bufferCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxTextureCount, textureManager.getTextureTable().size());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxInstanceCount, m_instanceManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialCount, m_surfaceMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialExtensionCount, m_surfaceMaterialExtensionCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxVolumeMaterialCount, m_volumeMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxLightCount, m_lightManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSamplers, m_samplerCache.getActiveCount());

    auto capturer = m_device->getCommon()->capturer();
    if (m_device->getCurrentFrameId() == m_beginUsdExportFrameNum) {
      capturer->triggerNewCapture();
    }
    capturer->step(ctx, ctx->getCommonObjects()->getLastKnownWindowHandle());
    Logger::warn("CHECKPOINT: prepareSceneData COMPLETE");

    // Clear the ray portal data before the next frame
    m_rayPortalManager.clear();

    // Check Anti-Culling Support:
    // When the game doesn't set up the View Matrix, we must disable Anti-Culling to prevent visual corruption.
    m_isAntiCullingSupported = (getCamera().getViewToWorld() != Matrix4d());
  }

  static_assert(std::is_same_v< decltype(RtSurface::objectPickingValue), ObjectPickingValue>);

  void SceneManager::submitExternalDraw(Rc<DxvkContext> ctx, ExternalDrawState&& state) {
    if (m_externalSampler == nullptr) {
      auto s = DxvkSamplerCreateInfo {};
      {
        s.magFilter = VK_FILTER_LINEAR;
        s.minFilter = VK_FILTER_LINEAR;
        s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        s.mipmapLodBias = 0.f;
        s.mipmapLodMin = 0.f;
        s.mipmapLodMax = 0.f;
        s.useAnisotropy = VK_FALSE;
        s.maxAnisotropy = 1.f;
        s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.compareToDepth = VK_FALSE;
        s.compareOp = VK_COMPARE_OP_NEVER;
        s.borderColor = VkClearColorValue {};
        s.usePixelCoord = VK_FALSE;
      }
      m_externalSampler = m_device->createSampler(s);
    }

    {
      state.drawCall.materialData.samplers[0] = m_externalSampler;
      state.drawCall.materialData.samplers[1] = m_externalSampler;
    }
    {
      const RtCamera& rtCamera = ctx->getCommonObjects()->getSceneManager().getCameraManager()
        .getCamera(state.cameraType);
      state.drawCall.transformData.worldToView = Matrix4 { rtCamera.getWorldToView() };
      state.drawCall.transformData.viewToProjection = Matrix4 { rtCamera.getViewToProjection() };
      state.drawCall.transformData.objectToView = state.drawCall.transformData.worldToView * state.drawCall.transformData.objectToWorld;
    }

    for (const RasterGeometry& submesh : m_pReplacer->accessExternalMesh(state.mesh)) {
      state.drawCall.geometryData = submesh;
      state.drawCall.geometryData.cullMode = state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;

      const MaterialData* material = m_pReplacer->accessExternalMaterial(submesh.externalMaterial);
      if (material != nullptr) {
        state.drawCall.materialData.setHashOverride(material->getHash());
      } 

      const RtxParticleSystemDesc* pParticles = nullptr;
      if(state.optionalParticleDesc.has_value()) {
        pParticles = &state.optionalParticleDesc.value();
      }

      processDrawCallState(ctx, state.drawCall, material != nullptr ? MaterialData(*material) : LegacyMaterialData().as<OpaqueMaterialData>(), nullptr, pParticles);
    }
  }

  namespace {
    bool ifTrue_andThenSetFalse(std::atomic_bool& atomicBool) {
      bool expected = true;
      if (atomicBool.compare_exchange_strong(expected, false)) {
        return true;
      }
      return false;
    }
  } // unnamed

  void SceneManager::requestTextureVramFree() {
    m_forceFreeTextureMemory.store(true);
  }

  void SceneManager::requestVramCompaction() {
    m_forceFreeUnusedDxvkAllocatorChunks.store(true);
  }

  void SceneManager::manageTextureVram() {
    bool freeUnused = false;
    bool freeTextures = false;
    {
      if (ifTrue_andThenSetFalse(m_forceFreeTextureMemory)) {
        freeTextures = true;
        freeUnused = true;
      }
      if (ifTrue_andThenSetFalse(m_forceFreeUnusedDxvkAllocatorChunks)) {
        freeUnused = true;
      }
    }

    if (freeTextures) {
      m_device->getCommon()->getTextureManager().clear();

      if (m_opacityMicromapManager) {
        m_opacityMicromapManager->clear();
      }
    }

    if (freeUnused) {
      // DXVK doesnt free chunks for us by default (its high water mark) so force release some memory back to the system here.
      m_device->getCommon()->memoryManager().freeUnusedChunks();
    }
  }

  void SceneManager::printAllRtInstances() {
  #ifdef REMIX_DEVELOPMENT
    
    const auto& instances = m_instanceManager.getInstanceTable();
    Logger::info(str::format("=== Printing all RtInstances (", instances.size(), " total) ==="));
    
    for (size_t i = 0; i < instances.size(); ++i) {
      const RtInstance* instance = instances[i];
      if (instance != nullptr) {
        Logger::info(str::format("Instance ", i, ":"));
        instance->printDebugInfo();
      } else {
        Logger::warn(str::format("Instance ", i, ": nullptr"));
      }
    }
    
    Logger::info("=== End RtInstances Print ===");
  #endif
  }

  void SceneManager::trackReplacementMaterialHash(XXH64_hash_t materialHash) {
    if (materialHash != kEmptyHash) {
      m_currentFrameReplacementMaterialHashes[materialHash]++;
    }
  }

  bool SceneManager::isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const {
    return m_currentFrameReplacementMaterialHashes.find(materialHash) != m_currentFrameReplacementMaterialHashes.end();
  }

  uint32_t SceneManager::getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const {
    auto it = m_currentFrameReplacementMaterialHashes.find(materialHash);
    return (it != m_currentFrameReplacementMaterialHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameReplacementMaterialHashes() {
    m_currentFrameReplacementMaterialHashes.clear();
  }

  void SceneManager::trackMeshHash(XXH64_hash_t meshHash) {
    if (meshHash != kEmptyHash) {
      m_currentFrameMeshHashes[meshHash]++;
    }
  }

  bool SceneManager::isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const {
    return m_currentFrameMeshHashes.find(meshHash) != m_currentFrameMeshHashes.end();
  }

  uint32_t SceneManager::getMeshHashUsageCount(XXH64_hash_t meshHash) const {
    auto it = m_currentFrameMeshHashes.find(meshHash);
    return (it != m_currentFrameMeshHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameMeshHashes() {
    m_currentFrameMeshHashes.clear();
  }

  // =========================================================================
  // RTX Mega Geometry: Subdivision Surface Detection and Creation
  // =========================================================================

  bool SceneManager::isSubdivisionSurfaceCandidate(const RaytraceGeometry& convertedGeom, const DrawCallState& drawCallState) const {
    // Use the CONVERTED geometry data (triangle strips already converted to triangle lists by processGeometryInfo)

    // Early rejection: Must have vertex and index data
    // Note: We check indexCount > 0 instead of usesIndices() because the temporary geometry
    // passed from tryCreateSubdivisionSurface doesn't have an actual index buffer set
    if (convertedGeom.indexCount == 0 || convertedGeom.vertexCount == 0) {
      return false;
    }

    // NOTE: No topology check needed! The converted geometry is ALWAYS triangle list
    // because processGeometryInfo() already converted strips/fans to lists via generateTriangleList()

    // Subdivision surfaces require quad topology
    // In DX9/legacy games, quads are typically submitted as 2 triangles per quad
    // Check if index count is divisible by 6 (2 triangles * 3 indices per quad)
    if (convertedGeom.indexCount % 6 != 0) {
      return false;
    }

    const uint32_t potentialQuadCount = convertedGeom.indexCount / 6;

    // Sanity check: Must have at least 1 quad
    if (potentialQuadCount == 0) {
      return false;
    }

    // Check for USD replacement mesh with subdivision metadata
    // NOTE: This would be extended when USD integration is complete
    // For now, we check the AssetReplacer for subdivision scheme metadata
    if (m_pReplacer) {
      XXH64_hash_t meshHash = drawCallState.getHash(RtxOptions::geometryAssetHashRule());
      // TODO: Query AssetReplacer for subdivision scheme attribute
      // const AssetReplacement* replacement = m_pReplacer->getReplacementForHash(meshHash);
      // if (replacement && replacement->hasSubdivisionScheme()) {
      //   return true;
      // }
    }

    // Check mesh hash allow-list (configuration-based detection)
    // TODO: Add rtx.conf option for subdivision surface mesh hash allow-list
    // Example: rtx.subdivisionSurface.meshHashes = "hash1,hash2,hash3"
    // This would enable specific legacy meshes to use RTX MG tessellation
    XXH64_hash_t meshHash = drawCallState.getHash(RtxOptions::geometryAssetHashRule());

    // For now, enable subdivision surfaces for meshes that:
    // 1. Have quad-compatible topology (indexCount % 6 == 0)
    // 2. Have reasonable vertex counts (not too small, not too large)
    // 3. Are tracked as replacement material (indicating they're important assets)
    const bool hasReasonableComplexity = (convertedGeom.vertexCount >= 16 && convertedGeom.vertexCount <= 65536) &&
                                         (potentialQuadCount >= 4 && potentialQuadCount <= 16384);

    if (!hasReasonableComplexity) {
      return false;
    }

    // Check if this mesh hash is in the replacement material tracking
    // (indicates it's a significant asset worth tessellating)
    const bool isTrackedReplacementMesh = isReplacementMaterialHashUsedThisFrame(meshHash) ||
                                           getMeshHashUsageCount(meshHash) > 0;

    // Quad topology verification: Sample indices to verify quad structure
    // This is a heuristic check - true verification would require reading all indices
    if (convertedGeom.indexBuffer.defined()) {
      // For now, accept as candidate if it passes all other checks
      // Full quad topology verification would happen in tryCreateSubdivisionSurface

      // TODO: Add heuristic verification by checking first few quads
      // Read first 12 indices (2 quads worth) and verify they form quads:
      // Quad pattern: [v0,v1,v2] [v0,v2,v3] where each pair shares edge v0-v2
    }

    // TEMPORARY: Accept all meshes with reasonable complexity for testing
    // TODO: Re-enable conservative check once mega geometry is proven working
    bool result = hasReasonableComplexity;  // Changed from isTrackedReplacementMesh

    // Conservative approach: Only enable for tracked replacement meshes with quad topology
    // This can be relaxed with additional configuration options
    return result;
  }

  bool SceneManager::tryCreateSubdivisionSurface(
    Rc<DxvkContext> ctx,
    const DrawCallState& drawCallState,
    BlasEntry* pBlas)
  {
    // Async subdivision surface creation - surfaces are created on a worker thread
    // to prevent freezing the render thread during heavy OpenSubdiv processing
    //
    // IMPORTANT: This function is now called BEFORE processGeometryInfo to access
    // the original vertex buffer data before GPU upload makes it inaccessible.
    // We use the ORIGINAL geometry from drawCallState, not converted geometry.

    // Use the ORIGINAL geometry from drawCallState (before GPU upload)
    // This is called before processGeometryInfo so pBlas->modifiedGeometryData isn't ready yet
    const RasterGeometry& originalGeom = drawCallState.getGeometryData();

    // Create a temporary RaytraceGeometry for candidate check
    // We need to estimate what the converted geometry would look like
    RaytraceGeometry tempConvertedGeom;
    tempConvertedGeom.vertexCount = originalGeom.vertexCount;
    // Estimate converted index count: if triangle strip, it becomes triangle list
    if (originalGeom.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP && originalGeom.indexCount >= 3) {
      tempConvertedGeom.indexCount = (originalGeom.indexCount - 2) * 3;
    } else {
      tempConvertedGeom.indexCount = originalGeom.indexCount;
    }
    // Note: We don't copy the indexBuffer because the types are incompatible (RasterBuffer vs RaytraceBuffer)
    // The candidate check uses indexCount > 0 instead of usesIndices()

    // Only attempt subdivision surface creation if this mesh is a candidate
    if (!isSubdivisionSurfaceCandidate(tempConvertedGeom, drawCallState)) {
      return false;
    }

    // Cache by TOPOLOGY only (not vertex positions) for animated meshes!
    // Topology = indices + geometry descriptor (no positions)
    HashRule topologyRule = rules::TopologicalHash;
    XXH64_hash_t topologyHash = drawCallState.getHash(topologyRule);
    auto cacheIt = m_subdivisionSurfaceCache.find(topologyHash);

    // Invalidate stale cache entries (surface was pruned from builder)
    if (cacheIt != m_subdivisionSurfaceCache.end()) {
      RtxMegaGeoBuilder* megaGeoBuilder = m_accelManager.getMegaGeoBuilder();
      if (megaGeoBuilder && !megaGeoBuilder->hasSurface(cacheIt->second)) {
        m_subdivisionSurfaceCache.erase(cacheIt);
        cacheIt = m_subdivisionSurfaceCache.end(); // treat as cache miss
      }
    }

    if (cacheIt != m_subdivisionSurfaceCache.end()) {
      // Cache hit! Topology exists, just update control points for animated meshes
      uint32_t cachedSurfaceId = cacheIt->second;

      RtxMegaGeoBuilder* megaGeoBuilder = m_accelManager.getMegaGeoBuilder();
      if (!megaGeoBuilder) {
        Logger::err("RTX MegaGeo: Cache hit but builder is null!");
        return false;
      }

      // Extract ONLY positions for animation update (cache hit path)
      // Use original geometry - called before processGeometryInfo
      const uint32_t vertexCount = originalGeom.vertexCount;
      std::vector<Vector3> controlPoints;

      // Use original position buffer via GeometryBufferData
      GeometryBufferData bufferData(originalGeom);

      if (bufferData.positionData == nullptr) {
        Logger::err("RTX MegaGeo: Cache hit but position buffer not CPU-accessible");
        return false;
      }

      controlPoints.resize(vertexCount);
      for (uint32_t v = 0; v < vertexCount; ++v) {
        controlPoints[v] = bufferData.getPosition(v);
      }

      // For topology cache hits, reuse the existing surface
      // Topology hasn't changed, so we can use the cached surface directly
      // TODO: In the future, implement proper animation support by updating only control point buffers

      pBlas->blasType = BlasEntry::Type::ClusterBlas;
      pBlas->megaGeoBuilder = megaGeoBuilder;
      pBlas->megaGeoSurfaceId = cachedSurfaceId;
      return true;
    }

    // Get or initialize RtxMegaGeoBuilder from AccelManager
    RtxMegaGeoBuilder* megaGeoBuilder = m_accelManager.getMegaGeoBuilder();
    if (!megaGeoBuilder) {
      if (!m_accelManager.initializeMegaGeoBuilder(ctx)) {
        Logger::err("RTX MegaGeo: Failed to initialize builder");
        return false;
      }
      megaGeoBuilder = m_accelManager.getMegaGeoBuilder();
      if (!megaGeoBuilder) {
        Logger::err("RTX MegaGeo: Builder is still null after initialization");
        return false;
      }
    }

    // Extract VERTEX data from ORIGINAL RasterGeometry (CPU-accessible, only on cache miss!)
    // originalGeom is already defined at the start of this function
    GeometryBufferData bufferData(originalGeom);

    // Allocate temporary storage for mesh data extraction
    std::vector<Vector3> controlPoints;
    std::vector<Vector2> texcoords;
    std::vector<Vector3> normals;
    std::vector<uint32_t> faceVertexIndices;
    std::vector<uint32_t> faceVertexCounts;

    // Use original vertex count (this is called before processGeometryInfo)
    const uint32_t vertexCount = originalGeom.vertexCount;

    // RTX Mega Geometry: Prefer originalPositionBuffer if available (contains D3D9 vertex data)
    // When vertex capture is enabled, positionBuffer points to GPU-filled capture buffer (zeros until GPU runs)
    // originalPositionBuffer contains the actual D3D9 vertex data we can read immediately
    float* positionDataPtr = bufferData.positionData;
    size_t positionStrideFloats = bufferData.positionStride;

    if (originalGeom.originalPositionBuffer.defined()) {
      void* origPtr = originalGeom.originalPositionBuffer.mapPtr(
        (size_t)originalGeom.originalPositionBuffer.offsetFromSlice());
      if (origPtr != nullptr) {
        positionDataPtr = (float*)origPtr;
        positionStrideFloats = originalGeom.originalPositionBuffer.stride() / sizeof(float);
      }
    }

    if (positionDataPtr == nullptr) {
      Logger::err("RTX MegaGeo: Position buffer not CPU-accessible, cannot create subdivision surface");
      return false;
    }

    controlPoints.resize(vertexCount);
    for (uint32_t v = 0; v < vertexCount; ++v) {
      // Read position using the determined pointer and stride
      const float* vertPtr = positionDataPtr + v * positionStrideFloats;
      controlPoints[v] = Vector3(vertPtr[0], vertPtr[1], vertPtr[2]);
    }

    // Extract texture coordinates from ORIGINAL buffer
    if (bufferData.texcoordData != nullptr) {
      texcoords.resize(vertexCount);
      for (uint32_t v = 0; v < vertexCount; ++v) {
        texcoords[v] = bufferData.getTexCoord(v);
      }
    } else {
      Logger::warn("RTX MegaGeo: texcoordData is NULL - no texcoords for subdivision surface!");
    }

    // Extract normals from ORIGINAL buffer
    if (bufferData.normalData != nullptr) {
      normals.resize(originalGeom.vertexCount);
      for (uint32_t v = 0; v < originalGeom.vertexCount; ++v) {
        // Normals in RTX Remix are stored as packed floats
        const float* normPtr = bufferData.normalData + v * bufferData.normalStride;
        normals[v] = Vector3(normPtr[0], normPtr[1], normPtr[2]);
      }
    } else {
      Logger::warn("RTX MegaGeo: normalData is NULL - no normals for subdivision surface!");
    }

    // Extract quad topology - copy index data to CPU for async processing
    // We need to copy NOW before GPU upload completes asynchronously

    const uint32_t numQuads = tempConvertedGeom.indexCount / 6;
    if (numQuads == 0 || (tempConvertedGeom.indexCount % 6) != 0) {
      Logger::warn(str::format("RTX MegaGeo: Converted index count ", tempConvertedGeom.indexCount, " not divisible by 6, not a quad mesh"));
      return false;
    }

    // Copy indices from original buffer (CPU-accessible) to our own storage
    // Original buffer contains the source data that was/will be uploaded to GPU
    if (!bufferData.indexData) {
      Logger::err("RTX MegaGeo: No CPU-accessible index data");
      return false;
    }

    // Determine index type and size
    const VkIndexType indexType = originalGeom.indexBuffer.indexType();
    const bool is16Bit = (indexType == VK_INDEX_TYPE_UINT16);
    const size_t indexStride = is16Bit ? 2 : 4;

    // Convert triangle strips/fans to triangle lists on CPU
    std::vector<uint32_t> convertedIndices;

    if (originalGeom.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
      // Already triangle list, just copy
      convertedIndices.reserve(originalGeom.indexCount);
      if (is16Bit) {
        const uint16_t* indices16 = (const uint16_t*)bufferData.indexData;
        for (uint32_t i = 0; i < originalGeom.indexCount; ++i) {
          convertedIndices.push_back(indices16[i]);
        }
      } else {
        const uint32_t* indices32 = (const uint32_t*)bufferData.indexData;
        convertedIndices.assign(indices32, indices32 + originalGeom.indexCount);
      }
    } else if (originalGeom.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) {
      // Convert triangle strip to list
      // Strip with N vertices produces (N-2) triangles
      const uint32_t numTriangles = (originalGeom.indexCount >= 3) ? (originalGeom.indexCount - 2) : 0;
      convertedIndices.reserve(numTriangles * 3);

      for (uint32_t i = 0; i < numTriangles; ++i) {
        uint32_t i0, i1, i2;
        if (is16Bit) {
          const uint16_t* indices16 = (const uint16_t*)bufferData.indexData;
          i0 = indices16[i];
          i1 = indices16[i + 1];
          i2 = indices16[i + 2];
        } else {
          const uint32_t* indices32 = (const uint32_t*)bufferData.indexData;
          i0 = indices32[i];
          i1 = indices32[i + 1];
          i2 = indices32[i + 2];
        }

        // Alternate winding order for strips
        if (i % 2 == 0) {
          convertedIndices.push_back(i0);
          convertedIndices.push_back(i1);
          convertedIndices.push_back(i2);
        } else {
          convertedIndices.push_back(i0);
          convertedIndices.push_back(i2);
          convertedIndices.push_back(i1);
        }
      }
    } else {
      Logger::warn(str::format("RTX MegaGeo: Unsupported topology type ", (uint32_t)originalGeom.topology));
      return false;
    }

    const void* indexBufferPtr = convertedIndices.data();
    const uint32_t convertedIndexCount = convertedIndices.size();

    // Edge-adjacency quad detection: build adjacency from non-degenerate triangles,
    // pair triangles sharing an edge into quads, emit remaining as triangle faces.
    // This avoids losing valid triangles to degenerate strip-separator neighbors.
    const uint32_t finalTriCount = convertedIndexCount / 3;
    if (finalTriCount == 0) {
      Logger::warn(str::format("RTX MegaGeo: After conversion, no valid triangles (indexCount=", convertedIndexCount, ")"));
      return false;
    }

    const uint32_t* indices32 = (const uint32_t*)indexBufferPtr;

    // Step 1: Collect non-degenerate triangles
    struct Tri { uint32_t v[3]; };
    std::vector<Tri> validTris;
    validTris.reserve(finalTriCount);
    uint32_t degenerateTriCount = 0;

    for (uint32_t t = 0; t < finalTriCount; ++t) {
      uint32_t i0 = indices32[t * 3];
      uint32_t i1 = indices32[t * 3 + 1];
      uint32_t i2 = indices32[t * 3 + 2];
      if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
        Logger::err(str::format("RTX MegaGeo: Invalid vertex index at triangle ", t));
        return false;
      }
      if (i0 == i1 || i0 == i2 || i1 == i2) {
        degenerateTriCount++;
        continue;
      }
      validTris.push_back({i0, i1, i2});
    }

    if (validTris.empty()) {
      Logger::err("RTX MegaGeo: No valid triangles after filtering degenerates");
      return false;
    }

    // Step 2: Build edge adjacency map (edge → triangle indices)
    auto edgeKey = [](uint32_t a, uint32_t b) -> uint64_t {
      if (a > b) std::swap(a, b);
      return ((uint64_t)a << 32) | (uint64_t)b;
    };

    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> edgeToTris;
    // Value: (tri0, tri1) where tri1 == UINT32_MAX means only 1 triangle on this edge

    for (uint32_t ti = 0; ti < (uint32_t)validTris.size(); ++ti) {
      const auto& tri = validTris[ti];
      for (int e = 0; e < 3; ++e) {
        uint64_t ek = edgeKey(tri.v[e], tri.v[(e + 1) % 3]);
        auto it = edgeToTris.find(ek);
        if (it == edgeToTris.end()) {
          edgeToTris[ek] = {ti, UINT32_MAX};
        } else if (it->second.second == UINT32_MAX) {
          it->second.second = ti; // second triangle on this edge
        }
        // If >2 triangles share an edge (non-manifold), ignore extras
      }
    }

    // Step 3: Greedily pair triangles into quads via shared edges
    std::vector<bool> used(validTris.size(), false);
    uint32_t validQuadCount = 0;
    uint32_t validTriCount = 0;

    faceVertexIndices.reserve(validTris.size() * 3);
    faceVertexCounts.reserve(validTris.size());

    for (auto& [ek, triPair] : edgeToTris) {
      uint32_t ti0 = triPair.first;
      uint32_t ti1 = triPair.second;
      if (ti1 == UINT32_MAX) continue; // boundary edge
      if (used[ti0] || used[ti1]) continue;

      const auto& t0 = validTris[ti0];
      const auto& t1 = validTris[ti1];

      // Shared edge vertices from the key
      uint32_t s0 = (uint32_t)(ek >> 32);
      uint32_t s1 = (uint32_t)(ek & 0xFFFFFFFF);

      // Find opposite vertices
      uint32_t opp0 = UINT32_MAX, opp1 = UINT32_MAX;
      for (int i = 0; i < 3; ++i) {
        if (t0.v[i] != s0 && t0.v[i] != s1) opp0 = t0.v[i];
        if (t1.v[i] != s0 && t1.v[i] != s1) opp1 = t1.v[i];
      }
      if (opp0 == UINT32_MAX || opp1 == UINT32_MAX) continue;
      if (opp0 == opp1) continue; // degenerate — both triangles same

      // Find shared edge direction in tri0's winding to produce consistent quad winding
      // tri0 has edges: v[0]→v[1], v[1]→v[2], v[2]→v[0]
      // Find which edge matches (s0,s1) or (s1,s0)
      uint32_t ea = s0, eb = s1; // default
      for (int i = 0; i < 3; ++i) {
        uint32_t vi = t0.v[i];
        uint32_t vj = t0.v[(i + 1) % 3];
        if ((vi == s0 && vj == s1) || (vi == s1 && vj == s0)) {
          ea = vi; eb = vj;
          break;
        }
      }

      // Quad perimeter: ea → opp0 → eb → opp1
      faceVertexIndices.push_back(ea);
      faceVertexIndices.push_back(opp0);
      faceVertexIndices.push_back(eb);
      faceVertexIndices.push_back(opp1);
      faceVertexCounts.push_back(4);
      validQuadCount++;

      used[ti0] = true;
      used[ti1] = true;
    }

    // Step 4: Emit remaining unpaired triangles as 3-vertex faces
    // Catmull-Clark handles non-quad faces as "irregular" — they get subdivided
    // with stencil matrices. This is slightly more expensive per-face than quads
    // but avoids losing geometry (holes in the mesh).
    for (uint32_t ti = 0; ti < (uint32_t)validTris.size(); ++ti) {
      if (!used[ti]) {
        const auto& tri = validTris[ti];
        faceVertexIndices.push_back(tri.v[0]);
        faceVertexIndices.push_back(tri.v[1]);
        faceVertexIndices.push_back(tri.v[2]);
        faceVertexCounts.push_back(3);
        validTriCount++;
      }
    }

    uint32_t totalFaces = validQuadCount + validTriCount;

    if (totalFaces == 0) {
      Logger::err("RTX MegaGeo: No valid faces found after edge-adjacency pairing");
      return false;
    }

    // Populate SubdivisionSurfaceDesc
    SubdivisionSurfaceDesc desc = {};

    std::string debugName = str::format("SubdivSurface_", std::hex, topologyHash);
    desc.debugName = debugName.c_str();

    desc.numFaces = totalFaces; // Quads + unpaired triangles
    desc.numVertices = vertexCount; // Use converted vertex count (matches converted indices)
    desc.faceVertexCounts = faceVertexCounts.data();
    desc.faceVertexIndices = faceVertexIndices.data();
    desc.controlPoints = controlPoints.data();
    desc.texcoords = texcoords.empty() ? nullptr : texcoords.data();
    desc.normals = normals.empty() ? nullptr : normals.data();

    // Material index - use 0 as placeholder
    desc.materialIndex = 0;

    // Set tessellation parameters
    desc.isolationLevel = 2; // Subdivision level (0-4, higher = smoother)
    desc.tessellationScale = 1.0f; // Adaptive tessellation multiplier

    // Check for displacement mapping
    desc.enableDisplacement = false; // RTX Remix doesn't expose displacement texture API yet
    desc.displacementTextureIndex = 0;
    desc.displacementScale = 0.0f;

    // Create subdivision surface in RtxMegaGeoBuilder
    uint32_t surfaceId = 0;
    if (!megaGeoBuilder->createSubdivisionSurface(desc, surfaceId)) {
      Logger::err("RTX Mega Geometry: Failed to create subdivision surface");
      return false;
    }

    // Mark BlasEntry as cluster-based BLAS
    pBlas->blasType = BlasEntry::Type::ClusterBlas;
    pBlas->megaGeoBuilder = megaGeoBuilder;
    pBlas->megaGeoSurfaceId = surfaceId;

    // Cache the subdivision surface ID by topology for future reuse
    m_subdivisionSurfaceCache[topologyHash] = surfaceId;

    return true;
  }

}  // namespace nvvk
