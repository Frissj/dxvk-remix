/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <assert.h>

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_scene_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_mega_geometry.h"
#include "rtx_mega_geometry_integration.h"

#include "../d3d9/d3d9_state.h"
#include "rtx_matrix_helpers.h"

#include "dxvk_scoped_annotation.h"
#include "rtx_options.h"

#include "rtx/pass/instance_definitions.h"
#include "rtx/concept/billboard.h"

#include "rtx/pass/common_binding_indices.h"

#include <numeric>  // For std::accumulate

namespace dxvk {

  // Make this static and not a member of AccelManager to make it safe updating the count from ~PooledBlas()
  static int g_blasCount = 0;

  AccelManager::AccelManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    // Note: The scratch buffer's device address must be aligned to the minimum alignment required by the Vulkan runtime, otherwise
    //    // even if scratch allocation offsets are aligned they may add to a device address which will mess up this alignment (the alignment
    //    // requirement in Vulkan applies to the scratch buffer's device address, not just an offset as the name may imply). The lack of
    //    // this alignment override created issues on Intel GPUs where the min scratch alignment is 128 bytes but the underlying buffer was
    //    // only allocated with a 64 byte alignment.
    //    // Note: This could use the value of m_scratchAlignment, but this is duplicated to avoid potential future initialization order issues.
    , m_scratchAlignment(device->properties().khrDeviceAccelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment) {
  }

  void AccelManager::clear() {
    m_blasPool.clear();
  }

  void AccelManager::garbageCollection() {
    // Can be configured per game: 'rtx.numFramesToKeepBLAS'
    // Note: keep the BLAS for at least two frames so that they're alive for previous-frame TLAS access.
    const uint32_t numFramesToKeepBLAS = std::max(RtxOptions::enablePreviousTLAS() ? 2u : 1u, RtxOptions::numFramesToKeepBLAS());

    // Remove instances past their lifetime or marked for GC explicitly
    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // Remove all pooled BLAS that haven't been used for a few frames
    for (uint32_t i = 0; i < m_blasPool.size();) {
      Rc<PooledBlas>& blas = m_blasPool[i];

      if (blas->frameLastTouched + numFramesToKeepBLAS < currentFrame) {
        // Put this BLAS to the end of the vector
        std::swap(blas, m_blasPool.back());
        // Remove the last element
        m_blasPool.pop_back();
        continue;
      }
      ++i;
    }
  }
  
  PooledBlas::PooledBlas() {
    ++g_blasCount;
    buildInfo.geometryCount = 0;
    buildInfo.pGeometries = nullptr;
  }

  PooledBlas::~PooledBlas() {
    if (buildInfo.pGeometries) {
      delete[] buildInfo.pGeometries;
      buildInfo.pGeometries = nullptr;
    }
    accelerationStructureReference = 0;
    accelStructure = nullptr;
    --g_blasCount;
  }

  // Keep a copy of the build info to validate it for potential updateability
  static void copyAccelerationStructureBuildGeometryInfo(const VkAccelerationStructureBuildGeometryInfoKHR& srcInfo, VkAccelerationStructureBuildGeometryInfoKHR& dstInfo)
  {
    const VkAccelerationStructureGeometryKHR* pGeometries = dstInfo.pGeometries;
    if (srcInfo.pGeometries) {
      if (srcInfo.geometryCount != dstInfo.geometryCount) {
        if (pGeometries) {
          delete[] pGeometries;
        }
        pGeometries = new VkAccelerationStructureGeometryKHR[srcInfo.geometryCount];
      }
      std::memcpy((void*) pGeometries, srcInfo.pGeometries, srcInfo.geometryCount * sizeof(VkAccelerationStructureGeometryKHR));

      dstInfo = srcInfo;
      dstInfo.pGeometries = pGeometries;
    }
  }

  uint32_t AccelManager::getBlasCount() {
    // Should never be negative, but just in case...
    return uint32_t(std::max(g_blasCount, 0));
  }

  bool AccelManager::BlasBucket::tryAddInstance(RtInstance* instance) {
    const uint8_t geometryInstanceMask = instance->getVkInstance().mask;
    const uint32_t geometryCustomIndexFlags = instance->getVkInstance().instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK);
    const bool geometryUsesUnorderedApproximations = instance->usesUnorderedApproximations();
    const VkGeometryInstanceFlagsKHR geometryInstanceFlags = instance->getVkInstance().flags;
    const uint32_t geometryInstanceShaderBindingTableRecordOffset = instance->getVkInstance().instanceShaderBindingTableRecordOffset;

    if (!geometries.empty()) {
      if (instanceMask != geometryInstanceMask)
        return false;
      if (instanceShaderBindingTableRecordOffset != geometryInstanceShaderBindingTableRecordOffset)
        return false;
      if (customIndexFlags != geometryCustomIndexFlags)
        return false;
      if (instanceFlags != geometryInstanceFlags)
        return false;
      if (usesUnorderedApproximations != geometryUsesUnorderedApproximations)
        return false;
    }

    BlasEntry* blasEntry = instance->getBlas();

    geometries.insert(geometries.end(), blasEntry->buildGeometries.begin(), blasEntry->buildGeometries.end());
    ranges.insert(ranges.end(), blasEntry->buildRanges.begin(), blasEntry->buildRanges.end());

    for (auto& range : blasEntry->buildRanges) {
      originalInstances.push_back(instance);
      primitiveCounts.push_back(range.primitiveCount);
    }
    instanceBillboardIndices.insert(instanceBillboardIndices.end(), instance->billboardIndices.begin(), instance->billboardIndices.end());
    indexOffsets.insert(indexOffsets.end(), instance->indexOffsets.begin(), instance->indexOffsets.end());

    instanceShaderBindingTableRecordOffset = geometryInstanceShaderBindingTableRecordOffset;
    instanceMask = geometryInstanceMask;
    customIndexFlags = geometryCustomIndexFlags;
    instanceFlags = geometryInstanceFlags;
    usesUnorderedApproximations = geometryUsesUnorderedApproximations;
    return true;
  }

  static void fillGeometryInfoFromBlasEntry(BlasEntry& blasEntry, RtInstance& instance, const OpacityMicromapManager* opacityMicromapManager) {
    ScopedCpuProfileZone();
    blasEntry.buildGeometries.clear();
    blasEntry.buildRanges.clear();
    instance.billboardIndices.clear();
    instance.indexOffsets.clear();

    const bool usesIndices = blasEntry.modifiedGeometryData.usesIndices();

    // Associate each billboard with a unique geometry entry
    // ToDo: get rid of usesIndices requirement, it's not needed to build OMMs. It's only used below
    if (usesIndices && 
        opacityMicromapManager &&
        opacityMicromapManager->isActive() &&
        OpacityMicromapManager::usesOpacityMicromap(instance) &&
        OpacityMicromapManager::usesSplitBillboardOpacityMicromap(instance)) {

      VkAccelerationStructureGeometryKHR geometry = {};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.flags = instance.getGeometryFlags();

      VkAccelerationStructureGeometryTrianglesDataKHR& triangleData = geometry.geometry.triangles;
      triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangleData.indexData.deviceAddress = blasEntry.modifiedGeometryData.indexBuffer.getDeviceAddress();
      triangleData.indexType = blasEntry.modifiedGeometryData.indexBuffer.indexType();
      triangleData.vertexData.deviceAddress = blasEntry.modifiedGeometryData.positionBuffer.getDeviceAddress() + blasEntry.modifiedGeometryData.positionBuffer.offsetFromSlice();
      triangleData.vertexStride = blasEntry.modifiedGeometryData.positionBuffer.stride();
      triangleData.vertexFormat = blasEntry.modifiedGeometryData.positionBuffer.vertexFormat();
      triangleData.maxVertex = blasEntry.modifiedGeometryData.vertexCount - 1;

      assert((blasEntry.modifiedGeometryData.calculatePrimitiveCount() & 1) == 0);
      VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
      buildRange.primitiveCount = 2;

      for (uint32_t billboardIndex = 0; billboardIndex < instance.getBillboardCount(); billboardIndex++) {
        const uint32_t kNumIndicesPerBillboardQuad = buildRange.primitiveCount * 3;
        buildRange.primitiveOffset = (billboardIndex * kNumIndicesPerBillboardQuad * blasEntry.modifiedGeometryData.indexBuffer.stride());
        blasEntry.buildGeometries.push_back(geometry);
        blasEntry.buildRanges.push_back(buildRange);
        instance.billboardIndices.push_back(billboardIndex);
        instance.indexOffsets.push_back(billboardIndex * kNumIndicesPerBillboardQuad);
      }
    }
    else {
      VkAccelerationStructureGeometryKHR geometry = {};

      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.flags = instance.getGeometryFlags();

      VkAccelerationStructureGeometryTrianglesDataKHR& triangleData = geometry.geometry.triangles;

      const bool usesIndices = blasEntry.modifiedGeometryData.usesIndices();

      triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

      if (usesIndices) {
        triangleData.indexData.deviceAddress = blasEntry.modifiedGeometryData.indexBuffer.getDeviceAddress();
        triangleData.indexType = blasEntry.modifiedGeometryData.indexBuffer.indexType();
      } else {
        triangleData.indexData.deviceAddress = 0;
        triangleData.indexType = VK_INDEX_TYPE_NONE_KHR;
      }

      triangleData.vertexData.deviceAddress = blasEntry.modifiedGeometryData.positionBuffer.getDeviceAddress() + blasEntry.modifiedGeometryData.positionBuffer.offsetFromSlice();
      triangleData.vertexStride = blasEntry.modifiedGeometryData.positionBuffer.stride();
      triangleData.vertexFormat = blasEntry.modifiedGeometryData.positionBuffer.vertexFormat();
      triangleData.maxVertex = blasEntry.modifiedGeometryData.vertexCount - 1;

      VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
      buildRange.primitiveCount = blasEntry.modifiedGeometryData.calculatePrimitiveCount();
      buildRange.primitiveOffset = 0;

      blasEntry.buildGeometries.push_back(geometry);
      blasEntry.buildRanges.push_back(buildRange);
      instance.billboardIndices.push_back(0);
      instance.indexOffsets.push_back(0);
    }
  }
  int AccelManager::getCurrentFramePrimitiveIDPrefixSumBufferID() const {
    return m_device->getCurrentFrameId() & 0x1;
  }

  uint32_t additionalAccelerationStructureFlags() {
    return RtxOptions::lowMemoryGpu() ? VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR : 0;
  }

  void AccelManager::createAndBuildIntersectionBlas(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers) {
    if (m_intersectionBlas.ptr())
      return;

    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.pNext = nullptr;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = additionalAccelerationStructureFlags();
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.ppGeometries = nullptr;

    uint32_t maxPrimitiveCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxPrimitiveCount, &sizeInfo);

    m_intersectionBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Intersection");
    
    buildInfo.dstAccelerationStructure = m_intersectionBlas->accelStructure->getAccelStructure();

    VkAabbPositionsKHR aabbPositions = { -1.f, -1.f, -1.f, 1.f, 1.f, 1.f };

    DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = sizeof(aabbPositions);

    m_aabbBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "AABB Buffer");
    // Note: don't use ctx->updateBuffer() because that will place the command on the InitBuffer, not ExecBuffer.
    ctx->getCommandList()->cmdUpdateBuffer(DxvkCmdBuffer::ExecBuffer, m_aabbBuffer->getBufferRaw(), m_aabbBuffer->getSliceHandle().offset, sizeof(aabbPositions), &aabbPositions);
    
    execBarriers.accessBuffer(
      m_aabbBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.accessBuffer(
      m_scratchBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV);

    execBarriers.recordCommands(ctx->getCommandList());
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);

    geometry.geometry.aabbs.data.deviceAddress = m_aabbBuffer->getDeviceAddress();

    const size_t requiredScratchAllocSize = sizeInfo.buildScratchSize + m_scratchAlignment;
    buildInfo.scratchData.deviceAddress = getScratchMemory(requiredScratchAllocSize)->getDeviceAddress();
    assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

    VkAccelerationStructureBuildRangeInfoKHR buildRange {};
    buildRange.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRange);

    execBarriers.accessBuffer(
      m_scratchBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV);

    execBarriers.recordCommands(ctx->getCommandList());
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_scratchBuffer);
  }

  Rc<DxvkBuffer> AccelManager::getScratchMemory(const size_t requiredScratchAllocSize) {
    if (m_scratchBuffer == nullptr || m_scratchBuffer->info().size < requiredScratchAllocSize) {
      DxvkBufferCreateInfo bufferCreateInfo {};
      bufferCreateInfo.size = requiredScratchAllocSize;
      bufferCreateInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      bufferCreateInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      m_scratchBuffer = m_device->createBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "BVH Scratch");
    }

    return m_scratchBuffer;
  }

  Rc<PooledBlas> AccelManager::createPooledBlas(size_t bufferSize, const char* name) const {
    auto newBlas = new PooledBlas();

    DxvkBufferCreateInfo bufferCreateInfo {};
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    bufferCreateInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    newBlas->accelStructure = m_device->createAccelStructure(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, name);

    newBlas->accelerationStructureReference = newBlas->accelStructure->getAccelDeviceAddress();

    return newBlas;
  }

  static void trackBlasBuildResources(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers, const BlasEntry* blasEntry) {
    ScopedCpuProfileZone();
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->modifiedGeometryData.positionBuffer.buffer());
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->modifiedGeometryData.indexBuffer.buffer());

    execBarriers.accessBuffer(
      blasEntry->modifiedGeometryData.positionBuffer.getSliceHandle(),
      blasEntry->modifiedGeometryData.positionBuffer.buffer()->info().stages,
      blasEntry->modifiedGeometryData.positionBuffer.buffer()->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.accessBuffer(
      blasEntry->modifiedGeometryData.indexBuffer.getSliceHandle(),
      blasEntry->modifiedGeometryData.indexBuffer.buffer()->info().stages,
      blasEntry->modifiedGeometryData.indexBuffer.buffer()->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.recordCommands(ctx->getCommandList());
  }

  void AccelManager::mergeInstancesIntoBlas(Rc<DxvkContext> ctx, 
                                            DxvkBarrierSet& execBarriers, 
                                            const std::vector<TextureRef>& textures,
                                            const CameraManager& cameraManager,
                                            InstanceManager& instanceManager,
                                            OpacityMicromapManager* opacityMicromapManager) {
    ScopedGpuProfileZone(ctx, "buildBLAS");

    auto& instances = instanceManager.getInstanceTable();

    // Allocate the transform buffer
    DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

    info.size = align(instances.size() * sizeof(VkTransformMatrixKHR), kBufferAlignment);

    if (m_transformBuffer == nullptr || info.size > m_transformBuffer->info().size) {
      // TODO: allocate with some spare space to make reallocations less frequent
      m_transformBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Transform Buffer");
      Logger::debug("DxvkRaytrace: Vulkan Transform Buffer Realloc");
    }

    std::vector<VkTransformMatrixKHR> instanceTransforms;
    instanceTransforms.reserve(instances.size());

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blasToBuild;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> blasRangesToBuild;

    blasToBuild.reserve(instances.size());
    blasRangesToBuild.reserve(instances.size());

    m_reorderedSurfaces.clear();
    m_reorderedSurfacesFirstIndexOffset.clear();
    for (auto& instances : m_mergedInstances) {
      instances.clear();
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    if (instances.size() > CUSTOM_INDEX_SURFACE_MASK) {
      ONCE(Logger::err("DxvkRaytrace: instances size is greater than max supported custom index value"));
    }

    if (opacityMicromapManager)
      opacityMicromapManager->onFrameStart(ctx);

    std::vector<std::unique_ptr<BlasBucket>> blasBuckets;
    blasBuckets.reserve(instances.size());

    size_t totalScratchMemory = 0;

    // NOTE: Would like to use the BLAS Linked instances here, but that misses viewmodel and virtual instances
    std::unordered_map<BlasEntry*, std::vector<RtInstance*>> uniqueBlas;

    for (RtInstance* instance : instances) {
      if (instance->isHidden()) {
        continue;
      }

      // If the instance has zero mask, do not build BLAS for it: no ray can intersect this instance.
      if (instance->getVkInstance().mask == 0) {
        
        bool needsOpacityMicromap = instance->isViewModelReference() && opacityMicromapManager;
        bool hasBillboards = instance->getBillboardCount() > 0;

        // OMM requests and billboards need a valid surface.
        // Particles on the player model generate valid billboards but their geometric instance mask is set to 0.
        if (needsOpacityMicromap || hasBillboards) {
          instance->setSurfaceIndex(m_reorderedSurfaces.size());

          m_reorderedSurfaces.push_back(instance);
          m_reorderedSurfacesFirstIndexOffset.push_back(0);
        }

        // Register OMM build request for reference ViewModel instances, which are persistent unlike the intermittent active view model instances
        if (needsOpacityMicromap) {
          opacityMicromapManager->registerOpacityMicromapBuildRequest(*instance, instanceManager, textures);
        }

        continue;
      }

      if (opacityMicromapManager) {
        opacityMicromapManager->registerOpacityMicromapBuildRequest(*instance, instanceManager, textures);
      }

      // Find the blas entry for this instance.
      // Cannot store BlasEntry* directly in the RtInstance because the entries are owned and potentially moved by the hash table.
      BlasEntry* blasEntry = instance->getBlas();
      assert(blasEntry);

      fillGeometryInfoFromBlasEntry(*blasEntry, *instance, opacityMicromapManager);

      const uint32_t minPrimsInDynamicBLAS = std::max(RtxOptions::minPrimsInDynamicBLAS(), 100u);
      const uint32_t maxPrimsForMergedBLAS = RtxOptions::maxPrimsInMergedBLAS();
      const uint32_t blasPrims = blasEntry->modifiedGeometryData.calculatePrimitiveCount();

      // Figure out if this blas should be a dynamic one
      const bool requestDynamicBlas = instance->surface.instancesToObject != nullptr ||    // Point instancer geometry is replicated many times in a scene, we want to reuse the BLAS memory for these objects
                                      blasEntry->input.getSkinningState().numBones != 0 || // Skinned meshes are always desirable to give a dynamic BLAS, since we'll want to make use of BVH update for performance reasons
                                      blasEntry->getLinkedInstances().size() > 1  ||       // Meshes that are used in instances multiple times should benefit from BLAS reuse
                                      blasEntry->dynamicBlas != nullptr ||                 // If we already have a dynamic BLAS, keep using it.
                                      blasPrims > maxPrimsForMergedBLAS ||                 // Avoid large meshes ending up in the merged BLAS which is built every frame.  # prims is proportional to build cost.
                                      RtxOptions::minimizeBlasMerging();                   // Option to attempt putting as many objects into dynamic BLAS as possible.

      const bool forceMergedBlas = (blasEntry->buildGeometries.size() > 1 ||                                       // Currently we use multiple build geometries for particle billboards, which we prefer to merge into large BLAS
                                    (!RtxOptions::minimizeBlasMerging() && blasPrims < minPrimsInDynamicBLAS) ||   // Avoid creating lots of small dynamic BLAS
                                    RtxOptions::forceMergeAllMeshes()) &&                                          // Setting to force all meshes into the merged BLAS
                                      instance->surface.instancesToObject == nullptr;                              // Never merge point instancer geometry

      // Route instances to uniqueBlas or merged BLAS based on normal criteria
      // Cluster BLASes will go through normal path and be patched on GPU afterward (NVIDIA's method)
      if (requestDynamicBlas && !forceMergedBlas) {
        // Since this loop is iterating over instances, and instances can share BLAS, we will build these later after identifying unique ones.
        uniqueBlas[blasEntry].push_back(instance);
      } else {
        // Make sure we don't double up on blas entries, this should only happen if theres a bug
        // TODO (REMIX-3996) will break the assumptions we make here about all instances in a BlasEntry having the same instancesToObject array
        assert(uniqueBlas.find(blasEntry) == uniqueBlas.end());

        if (blasEntry->dynamicBlas != nullptr && !blasEntry->dynamicBlas->isClusterBlas) {
          // Move the BLAS used by this geometry to the common pool.
          // This also ensures the dynamic blas resource that's still being used by previous TLAS is properly tracked for the next frame
          // NOTE: Skip this for cluster BLASes - they're prebuilt and should stay in place
          m_blasPool.push_back(std::move(blasEntry->dynamicBlas));
          blasEntry->dynamicBlas = nullptr;
        }

        // Calculate the device address for the current instance's transform and write the transform data
        // TODO: only do this for non-identity transforms
        VkDeviceAddress transformDeviceAddress = m_transformBuffer->getDeviceAddress() + instanceTransforms.size() * sizeof(VkTransformMatrixKHR);
        instanceTransforms.push_back(instance->getVkInstance().transform);

        for (auto& geometry : blasEntry->buildGeometries) {
          geometry.geometry.triangles.transformData.deviceAddress = transformDeviceAddress;
        }

        // Try to merge the instance into one of the blasBuckets
        bool merged = false;
        for (auto& bucket : blasBuckets) {
          if (bucket->tryAddInstance(instance)) {
            merged = true;
            break;
          }
        }

        // The instance couldn't be merged into any bucket - make a new one
        if (!merged) {
          auto newBucket = std::make_unique<BlasBucket>();
          merged = newBucket->tryAddInstance(instance);
          assert(merged);

          blasBuckets.push_back(std::move(newBucket));
        }

        // Track the lifetime and states of the source geometry buffers
        trackBlasBuildResources(ctx, execBarriers, blasEntry);
      }
    }

    // Build/Update the dynamic BLAS
    Logger::info(str::format("[AccelManager DEBUG] Building dynamic BLASes: count=", uniqueBlas.size()));
    for (const std::pair<BlasEntry*, std::vector<RtInstance*>> pair : uniqueBlas) {
      BlasEntry* blasEntry = pair.first;
      Logger::info(str::format("[AccelManager DEBUG] Processing BLAS entry, instances=", pair.second.size()));
      if (pair.second.size() == 0) {
        continue;
      }

      // Cluster BLASes may have empty buildGeometries - skip them and let GPU patching handle it
      if (blasEntry->buildGeometries.size() == 0) {
        Logger::info(str::format("[AccelManager DEBUG] Empty buildGeometries (likely cluster BLAS hash=0x",
                                std::hex, blasEntry->input.clusterBlasGeometryHash, std::dec, ") - skipping build, adding instances"));
        trackBlasBuildResources(ctx, execBarriers, blasEntry);
        for (RtInstance* inst : pair.second) {
          addBlas(inst, blasEntry, nullptr);
        }
        continue;
      }

      assert(blasEntry->buildGeometries.size() == 1); // dynamic BLAS should always have this
      assert(blasEntry->buildRanges.size() == 1); // dynamic BLAS should always have this

      // CRITICAL: Skip BLAS building for cluster BLASes - they're pre-built by RtxMegaGeometry
      // Cluster BLASes are injected via rtx_scene_manager.cpp and should not be rebuilt here
      if (blasEntry->dynamicBlas.ptr() && blasEntry->dynamicBlas->isClusterBlas) {
        Logger::info(str::format("[AccelManager] Skipping BLAS build for cluster BLAS (hash=0x",
                                std::hex, blasEntry->dynamicBlas->clusterBlasGeometryHash, std::dec, ")"));

        // Still need to add instances to TLAS
        for (RtInstance* rtInstance : pair.second) {
          if (rtInstance->surface.instancesToObject == nullptr) {
            addBlas(rtInstance, blasEntry, nullptr);
          } else {
            rtInstance->surface.surfaceIndexOfFirstInstance = m_reorderedSurfaces.size();
            for (auto& instanceToObject : *rtInstance->surface.instancesToObject) {
              addBlas(rtInstance, blasEntry, &instanceToObject);
            }
          }
        }

        ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->dynamicBlas->accelStructure);
        trackBlasBuildResources(ctx, execBarriers, blasEntry);
        continue;  // Skip regular BLAS building
      }

      // Get reference to dynamicBlas for regular (non-cluster) BLAS building
      Rc<PooledBlas>& selectedBlas = blasEntry->dynamicBlas;

      bool forceRebuild = false;
      XXH64_hash_t boundOpacityMicromapHash = kEmptyHash;
      if (opacityMicromapManager) {
        // Check validity of a built BLAS, only if:
        // We can only support OMM on dynamic BLAS whos surface is unique to that BLAS.  This is so we can benefit from instancing BLAS memory.  
        // In cases where there are multiple linked instances each with different surfaces OMM would break.
        bool ommsCompatible = pair.second.size() == 1;
        const XXH64_hash_t firstOmmHash = OpacityMicromapManager::getOpacityMicromapHash(*pair.second[0]);
        for (uint32_t i = 1; i < pair.second.size(); i++) {
          const XXH64_hash_t thisOmmHash = OpacityMicromapManager::getOpacityMicromapHash(*pair.second[i]);
          if (thisOmmHash != firstOmmHash) {
            ommsCompatible = false;
            break;
          }
        }

        if (ommsCompatible) {
          RtInstance* exemplarInstance = pair.second[0];

          // Bind opacity micromap
          // Opacity micromaps must be bound before acceleration sizes are calculated
          // Note: since opacity micromaps for this frame are scheduled later 
          //       this will only pickup Opacity Micromaps built in previous frames
          boundOpacityMicromapHash = opacityMicromapManager->tryBindOpacityMicromap(ctx, *exemplarInstance, 0, blasEntry->buildGeometries[0], instanceManager);

          if (blasEntry->dynamicBlas.ptr()) {
            // A previously built BLAS needs to be rebuild if a corresponding Opacity Micromap availability has changed
            forceRebuild = boundOpacityMicromapHash != blasEntry->dynamicBlas->opacityMicromapSourceHash;
          }
        } else if (blasEntry->dynamicBlas.ptr() && blasEntry->dynamicBlas->opacityMicromapSourceHash != kEmptyHash) {
          // If we had a OMM bound at some point, but now that OMM is invalid, force a rebuild
          forceRebuild = true;
        }
      }

      VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
      buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = 1;
      buildInfo.pGeometries = blasEntry->buildGeometries.data();

      // Calculate the build sizes for this bucket
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
      sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                               &buildInfo, &blasEntry->buildRanges[0].primitiveCount, &sizeInfo);

      // Try to reuse our dynamic BLAS if it exists

      bool build = forceRebuild || !selectedBlas.ptr() || selectedBlas->accelStructure->info().size != sizeInfo.accelerationStructureSize;

      // Validate that the selected blas is compatible with the current build info for update purposes
      bool update = blasEntry->frameLastUpdated == currentFrame;
      if (update && !build && !validateUpdateMode(selectedBlas->buildInfo, buildInfo)) {
        // If an update is requested but the BLAS is not compatible with the current build info then force a rebuild
        update = false;
        build = true;
      }

      // There is no such BLAS - create one
      if (build) {
        if (selectedBlas.ptr()) {
          // Move the BLAS used by this geometry to the common pool.
          // This also ensures the dynamic blas resource that's still being used by previous TLAS is properly tracked for the next frame
          m_blasPool.push_back(std::move(selectedBlas));
        }
        selectedBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Dynamic");
      }

      assert(selectedBlas.ptr());
      selectedBlas->frameLastTouched = currentFrame;
      blasEntry->dynamicBlas->opacityMicromapSourceHash = boundOpacityMicromapHash;

      if (update || build) {
        if (update && !build) {
          buildInfo.srcAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();
          buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        }
        // Use the selected BLAS for the build
        buildInfo.dstAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();

        // Allocate a scratch buffer slice
        const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
        buildInfo.scratchData.deviceAddress = totalScratchMemory;
        totalScratchMemory += requiredScratchAllocSize;

        assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

        // Track the lifetime of the BLAS buffers
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(selectedBlas->accelStructure);

        // Put the merged BLAS into the build queue
        blasToBuild.push_back(buildInfo);
        blasRangesToBuild.push_back(&blasEntry->buildRanges[0]);

        copyAccelerationStructureBuildGeometryInfo(buildInfo, selectedBlas->buildInfo);
      }

      for (RtInstance* rtInstance : pair.second) {
        // Append an instance of this merged BLAS to the merged instance list
        if (rtInstance->surface.instancesToObject == nullptr) {
          addBlas(rtInstance, blasEntry, nullptr);
        } else {
          // This RtInstance is a PointInstancer - it represents multiple instances on the GPU.
          // Track the starting index for this block of instances in m_reorderedSurfaces.
          rtInstance->surface.surfaceIndexOfFirstInstance = m_reorderedSurfaces.size();

          // Add the same RtInstance pointer to m_reorderedSurfaces multiple times
          // Add a separate VkAccelerationStructureInstanceKHR to m_mergedInstances each time.
          for (auto& instanceToObject : *rtInstance->surface.instancesToObject) {
            addBlas(rtInstance, blasEntry, &instanceToObject);
          }
        }
      }

      ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->dynamicBlas->accelStructure);

      // Track the lifetime and states of the source geometry buffers
      trackBlasBuildResources(ctx, execBarriers, blasEntry);
    }

    // Copy the instance transform data to the device
    if(instanceTransforms.size() > 0)
      ctx->writeToBuffer(m_transformBuffer, 0, instanceTransforms.size() * sizeof(VkTransformMatrixKHR), instanceTransforms.data());

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_transformBuffer);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transformBuffer);

    // Place a barrier on the transform buffer
    DxvkBufferSliceHandle transformBufferSlice;
    transformBufferSlice.handle = m_transformBuffer->getBufferRaw();
    execBarriers.accessBuffer(
      transformBufferSlice,
      m_transformBuffer->info().stages,
      m_transformBuffer->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // Collect all the surfaces
    for (const auto& blasBucket : blasBuckets) {
      // Store the offset to use it later during blas instance creation
      blasBucket->reorderedSurfacesOffset = static_cast<uint32_t>(m_reorderedSurfaces.size());

      // Append the bucket's instances to the reordered surface list
      m_reorderedSurfaces.insert(m_reorderedSurfaces.end(), blasBucket->originalInstances.begin(), blasBucket->originalInstances.end());
      m_reorderedSurfacesFirstIndexOffset.insert(m_reorderedSurfacesFirstIndexOffset.end(), blasBucket->indexOffsets.begin(), blasBucket->indexOffsets.end());
    }

    // Build prefix sum array
    // Collect primitive count for each surface object
    // Because we use exclusive prefix sum here, we add one more element to record the scene's total primitive count
    m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame = m_reorderedSurfacesPrimitiveIDPrefixSum;
    m_reorderedSurfacesPrimitiveIDPrefixSum.resize(m_reorderedSurfaces.size() + 1);
    m_reorderedSurfacesPrimitiveIDPrefixSum[0] = 0;
    for (uint32_t i = 0; i < m_reorderedSurfaces.size(); i++) {
      auto surface = m_reorderedSurfaces[i];
      int primitiveCount = 0;
      for (const auto& buildRange: surface->getBlas()->buildRanges) {
        primitiveCount += buildRange.primitiveCount;
      }
      m_reorderedSurfacesPrimitiveIDPrefixSum[i + 1] = primitiveCount;
    }

    // Calculate exclusive prefix sum
    uint totalPrimitiveIDOffset = 0;
    for (uint32_t i = 0; i < m_reorderedSurfacesPrimitiveIDPrefixSum.size(); i++) {
      uint primitiveCount = m_reorderedSurfacesPrimitiveIDPrefixSum[i];
      m_reorderedSurfacesPrimitiveIDPrefixSum[i] += totalPrimitiveIDOffset;
      totalPrimitiveIDOffset += primitiveCount;
    }

    buildBlases(ctx, execBarriers, cameraManager, opacityMicromapManager, instanceManager,
                textures, instances, blasBuckets, blasToBuild, blasRangesToBuild, totalScratchMemory);
  }

  void AccelManager::addBlas(RtInstance* instance, BlasEntry* blasEntry, const Matrix4* instanceToObject) {
    // Create an instance for this BLAS
    VkAccelerationStructureInstanceKHR blasInstance = instance->getVkInstance();

    // For cluster BLASes, dynamicBlas is set during injection and must be used
    // For regular BLASes, dynamicBlas is null and we keep the address from getVkInstance()
    if (blasEntry->dynamicBlas != nullptr) {
      blasInstance.accelerationStructureReference = blasEntry->dynamicBlas->accelerationStructureReference;
    }

    blasInstance.instanceCustomIndex =
      (blasInstance.instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
      uint32_t(m_reorderedSurfaces.size()) & uint32_t(CUSTOM_INDEX_SURFACE_MASK);

    if (instanceToObject) {
      // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
      // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
      const Matrix4 transform = transpose(instance->surface.objectToWorld * (*instanceToObject));
      memcpy(&blasInstance.transform, &transform, sizeof(VkTransformMatrixKHR));
    }

    // Get the instance's flags and apply the objectToWorldMirrored flag.
    if (instance->isObjectToWorldMirrored()) {
      blasInstance.flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
    }

    if (instance->usesUnorderedApproximations() && RtxOptions::enableSeparateUnorderedApproximations()) {
      m_mergedInstances[Tlas::Unordered].push_back(blasInstance);
    } else {
      m_mergedInstances[Tlas::Opaque].push_back(blasInstance);
    }

    // Append the instance to the reordered surface list
    // Note: this happens *after* the instance is appended, because the size of m_reorderedSurfaces is used above
    m_reorderedSurfaces.push_back(instance);
    m_reorderedSurfacesFirstIndexOffset.push_back(0);
  }

  void AccelManager::createBlasBuffersAndInstances(Rc<DxvkContext> ctx, 
                                                   const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                                   std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                                   std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                                   size_t& totalScratchMemory) {

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // Create or find a matching BLAS for each bucket, then build it
    for (const auto& bucket : blasBuckets) {
      // Fill out the build info
      VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
      buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = bucket->geometries.size();
      buildInfo.pGeometries = bucket->geometries.data();

      // Calculate the build sizes for this bucket
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
      sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                               &buildInfo, bucket->primitiveCounts.data(), &sizeInfo);

      // MEGA GEOMETRY INTEGRATION: Compute geometry hash for precise BLAS matching
      XXH64_hash_t geometryHash = 0;
      bool usedCachedBlas = false;  // Track if we used a cached BLAS (proper flag instead of NULL signal)
      PooledBlas* selectedBlas = nullptr;  // Declare here so it's in scope for all paths

      // CRITICAL: This Rc must stay alive for the entire loop iteration
      // If we use a cached BLAS, this holds the reference to keep it alive
      Rc<PooledBlas> cachedBlasWrapper;

      if (RtxMegaGeometry::enable() && !RtxMegaGeometry::enableTessellation()) {
        RtxMegaGeometry* megaGeom = getMegaGeometry();
        if (megaGeom) {
          // CRITICAL: RTX Remix uses "merged BLASes" where multiple instances are pre-transformed
          // to world space and combined into one BLAS (with identity TLAS transform).
          //
          // BLAS caching works best for OBJECT-SPACE geometry (transform applied at TLAS level).
          // We can only do this efficiently for SINGLE-INSTANCE buckets.
          //
          // Strategy:
          // - Single instance (1 object): Cache in object space, NO transform in hash
          // - Multiple instances (merged): Skip caching (transforms baked in, changes every frame)

          const bool isSingleInstance = (bucket->originalInstances.size() == 1);
          const bool canCache = isSingleInstance;  // Only cache single-instance BLASes

          if (canCache) {
            // Hash only geometry data (NOT transforms) for object-space BLAS
            for (const auto& geom : bucket->geometries) {
              XXH64_hash_t h = 0;
              if (geom.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                const auto& triangles = geom.geometry.triangles;
                // Hash geometry properties
                h = XXH64(&triangles.vertexFormat, sizeof(triangles.vertexFormat), h);
                h = XXH64(&triangles.vertexStride, sizeof(triangles.vertexStride), h);
                h = XXH64(&triangles.maxVertex, sizeof(triangles.maxVertex), h);
                h = XXH64(&triangles.indexType, sizeof(triangles.indexType), h);
                // Hash buffer addresses (uniquely identifies geometry)
                h = XXH64(&triangles.vertexData.deviceAddress, sizeof(triangles.vertexData.deviceAddress), h);
                if (triangles.indexType != VK_INDEX_TYPE_NONE_KHR) {
                  h = XXH64(&triangles.indexData.deviceAddress, sizeof(triangles.indexData.deviceAddress), h);
                }
              }
              geometryHash = XXH64(&h, sizeof(h), geometryHash);
            }
          } else {
            // Multi-instance merged BLAS: Skip caching entirely
            // (geometry is pre-transformed, changes whenever any instance moves)
            goto skip_cache;
          }

          // Check if we have a cached BLAS for this geometry
          auto* cachedBlas = megaGeom->lookupBLAS(geometryHash);
          if (cachedBlas) {
            // Cache HIT! Reuse the existing BLAS from mega geometry cache
            ONCE(Logger::info(str::format(
              "[AccelManager] Mega Geometry BLAS cache HIT: hash=0x", std::hex, geometryHash, std::dec,
              ", size=", cachedBlas->blasSize / 1024, "KB"
            )));

            Logger::info("[AccelManager] Cache HIT step 1: Validating resources");
            // Safety check: Verify cached BLAS has valid resources
            if (cachedBlas->blasBuffer.ptr() == nullptr) {
              Logger::err("[AccelManager] Cached BLAS has invalid buffer - falling back to rebuild");
              // Skip cache and fall through to normal BLAS creation
              goto skip_cache;
            }

            Logger::info("[AccelManager] Cache HIT step 2: Creating temporary PooledBlas for cached BLAS");
            // Create a temporary PooledBlas wrapper that is NOT added to the pool
            // Cached BLASes don't need pool management since they're already built
            cachedBlasWrapper = Rc<PooledBlas>(new PooledBlas());

            Logger::info(str::format("[AccelManager] Cache HIT: cachedBlas->blasBuffer.ptr()=", (void*)cachedBlas->blasBuffer.ptr()));
            cachedBlasWrapper->accelStructure = cachedBlas->blasBuffer;
            Logger::info(str::format("[AccelManager] Cache HIT: After assignment, cachedBlasWrapper->accelStructure.ptr()=", (void*)cachedBlasWrapper->accelStructure.ptr()));

            cachedBlasWrapper->accelerationStructureReference = cachedBlas->blasBuffer->getAccelDeviceAddress();
            cachedBlasWrapper->frameLastTouched = currentFrame;

            Logger::info(str::format("[AccelManager] Cache HIT step 3: Got address 0x", std::hex, cachedBlasWrapper->accelerationStructureReference, std::dec));

            // Initialize buildInfo with CURRENT bucket geometry
            copyAccelerationStructureBuildGeometryInfo(buildInfo, cachedBlasWrapper->buildInfo);
            cachedBlasWrapper->primitiveCounts = bucket->primitiveCounts;

            // Store raw pointer for use in labels
            selectedBlas = cachedBlasWrapper.ptr();

            Logger::info("[AccelManager] Cache HIT step 4: Jumping to cached_blas_path (skip normal pool)");
            // Skip the normal pooling search and build - use special path for cached BLASes
            usedCachedBlas = true;  // Mark that we used a cached BLAS
            goto cached_blas_path;
          }
        }
      }

      // If we didn't use a cached BLAS, skip the cached_blas_path code
      goto skip_cache;

cached_blas_path:  // Special path for cached BLASes (skip pool entirely)
      Logger::info("[AccelManager] Cached BLAS path: tracking resource for read");
      Logger::info(str::format("[AccelManager] Cached BLAS path: selectedBlas=", (void*)selectedBlas,
                              ", accelStructure=", (void*)selectedBlas->accelStructure.ptr()));

      if (!selectedBlas || selectedBlas->accelStructure.ptr() == nullptr) {
        Logger::err("[AccelManager] Cached BLAS has null accelStructure!");
        assert(false && "Cached BLAS has null accelStructure");
      }

      ctx->getCommandList()->trackResource<DxvkAccess::Read>(selectedBlas->accelStructure);
      Logger::info("[AccelManager] Cached BLAS path: resource tracked successfully");

      // Don't add to build queue - this BLAS is already built
      // Just create the TLAS instance and we're done
      goto create_tlas_instance;

skip_cache:  // Fallback label for cache failures
      // Try to find an existing BLAS that is minimally sufficient to fit this bucket of geometries
      // selectedBlas already declared above
      for (const auto& blas : m_blasPool) {
        size_t bufferSize = blas->accelStructure->info().size;
        uint32_t paddedLastTouched = blas->frameLastTouched + 1 + (RtxOptions::enablePreviousTLAS() ? 1u : 0u); /* note: +2 because frameLastTouched is unsigned and init'd with UINT32_MAX, and keep the BLAS'es for one extra frame for previous TLAS access */
        if (bufferSize >= sizeInfo.accelerationStructureSize &&
            (!selectedBlas || bufferSize < selectedBlas->accelStructure->info().size) &&
            paddedLastTouched <= currentFrame)
        {
          selectedBlas = blas.ptr();
        }
      }

      // Must ensure that if we are updating an existing blas, rather than rebuilding, the blas is compatible with our new build info
      // Cannot update a blas that contains OMM instances, this leads to sporadic device lost errors
      if (!bucket->hasOmmInstances && selectedBlas && validateUpdateMode(selectedBlas->buildInfo, buildInfo) && selectedBlas->primitiveCounts == bucket->primitiveCounts) {
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      }

      // There is no such BLAS - create one and put it into the pool
      if (!selectedBlas) {
        auto newBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Merged");

        selectedBlas = newBlas.ptr();

        m_blasPool.push_back(std::move(newBlas));
      }

blas_selected:  // Label for mega geometry cache hit path
      Logger::info(str::format("[AccelManager] At blas_selected: usedCachedBlas=", usedCachedBlas));
      assert(selectedBlas);
      selectedBlas->frameLastTouched = currentFrame;
      Logger::info("[AccelManager] After setting frameLastTouched");

      if (!usedCachedBlas) {
        // Use the selected BLAS for the build
        buildInfo.dstAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();

        if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
          // Set the src to the dst if we're updating
          buildInfo.srcAccelerationStructure = buildInfo.dstAccelerationStructure;
        }

        copyAccelerationStructureBuildGeometryInfo(buildInfo, selectedBlas->buildInfo);
        selectedBlas->primitiveCounts = bucket->primitiveCounts;

        // Allocate a scratch buffer slice
        const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
        buildInfo.scratchData.deviceAddress = totalScratchMemory;
        totalScratchMemory += requiredScratchAllocSize;

        assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

        // Track the lifetime of the BLAS buffers
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(selectedBlas->accelStructure);

        // Put the merged BLAS into the build queue
        blasToBuild.push_back(buildInfo);
        blasRangesToBuild.push_back(bucket->ranges.data());
      } else {
        // This BLAS is from cache - already built, just track it for read
        Logger::info("[AccelManager] Cached BLAS path: tracking resource");
        ctx->getCommandList()->trackResource<DxvkAccess::Read>(selectedBlas->accelStructure);
        Logger::info("[AccelManager] Cached BLAS path: resource tracked successfully");
      }

      Logger::info("[AccelManager] Proceeding to TLAS instance creation");
      // MEGA GEOMETRY INTEGRATION: Store newly created BLAS in cache
      // CRITICAL: Only cache if this is NOT already a cached BLAS (usedCachedBlas==false means it's new)
      if (!usedCachedBlas && geometryHash != 0 && RtxMegaGeometry::enable() && !RtxMegaGeometry::enableTessellation()) {
        RtxMegaGeometry* megaGeom = getMegaGeometry();
        if (megaGeom && buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
          // Only cache if this is a new build (not an update)
          auto* cachedBlas = megaGeom->lookupBLAS(geometryHash);
          if (!cachedBlas) {
            // Cache MISS - store this new BLAS for future reuse
            RtxMegaGeometry::CachedBLAS newCachedBlas;
            newCachedBlas.blasBuffer = selectedBlas->accelStructure;
            newCachedBlas.accelStructure = buildInfo.dstAccelerationStructure;
            newCachedBlas.vertexCount = 0;  // Not tracked at this level
            newCachedBlas.triangleCount = std::accumulate(bucket->primitiveCounts.begin(), bucket->primitiveCounts.end(), 0u);
            newCachedBlas.lastUsedFrame = currentFrame;
            newCachedBlas.blasSize = sizeInfo.accelerationStructureSize;
            newCachedBlas.isCompacted = false;

            megaGeom->cacheBLAS(geometryHash, newCachedBlas);

            ONCE(Logger::info(str::format(
              "[AccelManager] Mega Geometry BLAS cache MISS: hash=0x", std::hex, geometryHash, std::dec,
              ", building new BLAS (", sizeInfo.accelerationStructureSize / 1024, "KB, ", newCachedBlas.triangleCount, " triangles)"
            )));
          }
        }
      }

create_tlas_instance:  // Label for TLAS instance creation (used by cached BLAS path)
      static float identityTransform[3][4] = {
        { 1.f, 0.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f, 0.f },
        { 0.f, 0.f, 1.f, 0.f }
      };

      // Append an instance of this merged BLAS to the merged instance list
      Logger::info("[AccelManager] Creating TLAS instance");
      VkAccelerationStructureInstanceKHR instance {};
      instance.accelerationStructureReference = selectedBlas->accelerationStructureReference;

      // DEBUG: Log BLAS address to check if it's valid
      Logger::info(str::format("[AccelManager] TLAS instance using BLAS address: 0x",
                              std::hex, instance.accelerationStructureReference, std::dec));
      instance.flags = bucket->instanceFlags;
      instance.instanceShaderBindingTableRecordOffset = bucket->instanceShaderBindingTableRecordOffset;
      instance.mask = bucket->instanceMask;
      instance.instanceCustomIndex =
        (bucket->customIndexFlags & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
        (bucket->reorderedSurfacesOffset & uint32_t(CUSTOM_INDEX_SURFACE_MASK));
      memcpy(static_cast<void*>(&instance.transform.matrix[0][0]), &identityTransform[0][0], sizeof(VkTransformMatrixKHR));

      Logger::info("[AccelManager] TLAS instance created, adding to list");
      if (bucket->usesUnorderedApproximations && RtxOptions::enableSeparateUnorderedApproximations())
        m_mergedInstances[Tlas::Unordered].push_back(instance);
      else
        m_mergedInstances[Tlas::Opaque].push_back(instance);
      Logger::info("[AccelManager] TLAS instance added successfully");
    }
  }

  void AccelManager::prepareSceneData(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers, InstanceManager& instanceManager) {
    ScopedCpuProfileZone();

    bool haveInstances = false;
    for (const auto& instances : m_mergedInstances) {
      if (!instances.empty()) {
        haveInstances = true;
        break;
      }
    }

    if (!haveInstances && instanceManager.getBillboards().empty())
      return;

    createAndBuildIntersectionBlas(ctx, execBarriers);

    // Prepare billboard data and instances
    std::vector<MemoryBillboard> memoryBillboards;
    uint32_t numActiveBillboards = 0;

    // Check the enablement here - because the instance manager needs to run the billboard analysis all the time
    if (RtxOptions::enableBillboardOrientationCorrection()) {
      memoryBillboards.resize(instanceManager.getBillboards().size());
      uint32_t index = 0;

      for (const auto& billboard : instanceManager.getBillboards()) {
        if (billboard.instanceMask == 0 || !billboard.allowAsIntersectionPrimitive)
          continue;

        // Shader data
        MemoryBillboard& memory = memoryBillboards[index];
        memory.center = billboard.center;
        memory.surfaceIndex = billboard.instance->getSurfaceIndex();
        memory.materialType = (billboard.instance->getVkInstance().instanceCustomIndex >> CUSTOM_INDEX_MATERIAL_TYPE_BIT) & surfaceMaterialTypeMask;
        memory.inverseHalfWidth = 2.f / billboard.width;
        memory.inverseHalfHeight = 2.f / billboard.height;
        memory.xAxis = billboard.xAxis;
        memory.yAxis = billboard.yAxis;
        memory.xAxisUV = billboard.xAxisUV;
        memory.yAxisUV = billboard.yAxisUV;
        memory.centerUV = billboard.centerUV;
        memory.vertexColor = billboard.vertexColor;
        memory.flags = 0;
        if (billboard.isBeam)
          memory.flags |= billboardFlagIsBeam;
        if (billboard.isCameraFacing)
          memory.flags |= billboardFlagIsCameraFacing;

        // TLAS instance
        VkAccelerationStructureInstanceKHR instance {};
        instance.accelerationStructureReference = m_intersectionBlas->accelerationStructureReference;
        instance.flags = 0;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.mask = billboard.instanceMask;
        instance.instanceCustomIndex = index;

        Matrix4 transform;
        if (billboard.isBeam) {
          // Scale and orient the primitive so that its local X and Y axes match the billboard's X and Y axes,
          // and the Z axis is (obviously) orthogonal to those. Note that the beam is cylindrical, so its 'width'
          // applies to both the X and Z axes.
          transform[0] = Vector4(billboard.xAxis * billboard.width * 0.5f, 0.f);
          transform[1] = Vector4(billboard.yAxis * billboard.height * 0.5f, 0.f);
          transform[2] = Vector4(normalize(cross(billboard.xAxis, billboard.yAxis)) * billboard.width * 0.5f, 0.f);
        }
        else {
          // Note: to be fully conservative, the size of the intersection primitive should be equal to the diagonal
          // of the original particle, not its largest side. But the particle textures are usually round, so
          // the reduced size works well in practice and results in fewer unnecessary ray interactions.
          const float radius = std::max(billboard.width, billboard.height) * 0.5f;
          transform[0][0] = transform[1][1] = transform[2][2] = radius;
        }
        transform[3] = Vector4(billboard.center, 1.f);
        transform = transpose(transform);
        memcpy(instance.transform.matrix, &transform, sizeof(VkTransformMatrixKHR));

        m_mergedInstances[Tlas::Unordered].push_back(instance);

        ++index;
      }

      numActiveBillboards = index;
    }

    // Allocate the instance buffer and copy its contents from host to device memory
    DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

    // Vk instance buffer
    for (const auto& instances : m_mergedInstances) {
      info.size += instances.size();
    }
    info.size = align(info.size * sizeof(VkAccelerationStructureInstanceKHR), kBufferAlignment);

    if (m_vkInstanceBuffer == nullptr || info.size > m_vkInstanceBuffer->info().size) {
      m_vkInstanceBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Instance Buffer");
      Logger::debug("DxvkRaytrace: Vulkan AS Instance Realloc");
    }

    // Write instance data
    size_t offset = 0;
    for (const auto& instances : m_mergedInstances) {
      if (!instances.empty()) {
        const size_t size = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
        ctx->writeToBuffer(m_vkInstanceBuffer, offset, size, instances.data());
        offset += size;
      }
    }

    // Vk billboard buffer
    if (numActiveBillboards) {
      info.size = align(numActiveBillboards * sizeof(MemoryBillboard), kBufferAlignment);
      if (info.size > 0 && (m_billboardsBuffer == nullptr || info.size > m_billboardsBuffer->info().size)) {
        m_billboardsBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Billboards Buffer");
      }

      // Write billboard data
      ctx->writeToBuffer(m_billboardsBuffer, 0, numActiveBillboards * sizeof(MemoryBillboard), memoryBillboards.data());
    }
  }

  void AccelManager::buildParticleSurfaceMapping(std::vector<uint32_t>& surfaceIndexMapping) {
    // Simplify syntax for accessing the persistent containers
    auto& surfaceInfoLists = buildParticleSurfaceMappingFuncState.surfaceInfoLists;
    uint32_t& currIndex = buildParticleSurfaceMappingFuncState.currIndex;
    uint32_t& prevIndex = buildParticleSurfaceMappingFuncState.prevIndex;

    // Build surface index mapping for particle objects.
    surfaceInfoLists[currIndex].resize(m_reorderedSurfaces.size());
    std::unordered_map<uint32_t, std::vector<int>> curMaterialHashToSurfaceMap;
    for (uint32_t surfaceIndex = 0; surfaceIndex < m_reorderedSurfaces.size(); surfaceIndex++) {
      RtInstance& surface = *m_reorderedSurfaces[surfaceIndex];

      // Only record objects that use unordered approximations.
      // In some cases, objects with unorder resolve flag will generate a set of billboards, each one occupies one "Surface" entry
      // in the shaders' surface array. These entries has identical information except the "firstIndex" member.
      // See "fillGeometryInfoFromBlasEntry()" for more details in generating indexOffsets.
      // See "uploadSurfaceData()" for how the "firstIndex" is fed to the shaders surface array.
      if (surface.usesUnorderedApproximations() && m_reorderedSurfacesFirstIndexOffset[surfaceIndex] == 0) {
        const RasterGeometry& geometryData = surface.getBlas()->input.getGeometryData();

        // Need to find the closest object with the same material, so use material ID as hash value, and record bounding box's center.
        surfaceInfoLists[currIndex][surfaceIndex] = { 
          surface.surface.surfaceMaterialIndex,
          geometryData.boundingBox.getTransformedCentroid(surface.getTransform()) };

        if (surface.getBlas()->buildRanges.size() > 0 && surface.getBlas()->buildGeometries.size() > 0) {
          curMaterialHashToSurfaceMap[surface.surface.surfaceMaterialIndex].push_back(surfaceIndex);
        }
      } else {
        surfaceInfoLists[currIndex][surfaceIndex].surfaceMaterialIndex = kSurfaceInvalidSurfaceMaterialIndex;
      }
    }

    // Fix missed surface mapping by searching among objects with the same hash value, and choose the closest one.
    for (int i = 0; i < surfaceIndexMapping.size(); i++) {
      // Skip objects that have surface mapping
      if (surfaceIndexMapping[i] != BINDING_INDEX_INVALID) {
        continue;
      }

      if (i >= surfaceInfoLists[prevIndex].size()) {
        continue;
      }

      // Skip objects with different materials
      auto lastInfo = surfaceInfoLists[prevIndex][i];
      auto pCandidateList = curMaterialHashToSurfaceMap.find(lastInfo.surfaceMaterialIndex);
      if (pCandidateList == curMaterialHashToSurfaceMap.end()) {
        continue;
      }

      auto& candidateList = pCandidateList->second;
      float minDistanceSq = FLT_MAX;
      int bestSurfaceID = -1;

      // Iterate through the candidate list and find the closest one
      for (int ithCandidate = 0; ithCandidate < candidateList.size(); ithCandidate++) {
        int curSurfaceID = candidateList[ithCandidate];
        RtInstance& surface = *m_reorderedSurfaces[curSurfaceID];
        if (surface.getBlas()->buildGeometries.size() == 0) {
          continue;
        }

        // Calculate bounding box centers' distance
        const RasterGeometry& geometryData = surface.getBlas()->input.getGeometryData();
        Vector3 center = geometryData.boundingBox.getTransformedCentroid(surface.getTransform());
        float distanceSq = lengthSqr(center - lastInfo.worldPosition);
        if (distanceSq < minDistanceSq) {
          minDistanceSq = distanceSq;
          bestSurfaceID = curSurfaceID;
        }
      }

      // Use the closest surface
      if (bestSurfaceID != -1) {
        surfaceIndexMapping[i] = bestSurfaceID;
      }
    }
    // Make current previous
    std::swap(currIndex, prevIndex);
  }

  void AccelManager::uploadSurfaceData(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();
    if (m_reorderedSurfaces.empty()) {
      return;
    }

    // Simplify syntax for accessing the persistent containers
    auto& surfacesGPUData = uploadSurfaceDataFuncState.surfacesGPUData;
    auto& surfaceIndexMapping = uploadSurfaceDataFuncState.surfaceIndexMapping;

    // Surface buffer
    const auto surfacesGPUSize = m_reorderedSurfaces.size() * kSurfaceGPUSize;

    // Allocate the instance buffer and copy its contents from host to device memory
    DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = align(surfacesGPUSize, kBufferAlignment);
    if (m_surfaceBuffer == nullptr || info.size > m_surfaceBuffer->info().size) {
      m_surfaceBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Surface Buffer");
    }

    uint32_t maxPreviousSurfaceIndex = 0;

    // Write surface data
    std::size_t dataOffset = 0;
    surfacesGPUData.resize(surfacesGPUSize);

    for (uint32_t i = 0; i < m_reorderedSurfaces.size(); ++i) {
      const auto& currentInstance = *m_reorderedSurfaces[i];
      RtSurface& currentSurface = m_reorderedSurfaces[i]->surface;

      // Split instance geometry need to have their first index offset set in their corresponding surface instances
      currentSurface.firstIndex += m_reorderedSurfacesFirstIndexOffset[i];
      currentSurface.writeGPUData(surfacesGPUData.data(), dataOffset, i);
      currentSurface.firstIndex -= m_reorderedSurfacesFirstIndexOffset[i];

      // Find the size of the surface mapping buffer
      if (currentInstance.surface.instancesToObject) {
        maxPreviousSurfaceIndex = std::max(maxPreviousSurfaceIndex, uint32_t(currentInstance.getPreviousSurfaceIndex() + currentInstance.surface.instancesToObject->size()));
      } else {
        maxPreviousSurfaceIndex = std::max(maxPreviousSurfaceIndex, currentInstance.getPreviousSurfaceIndex());
      }
    }

    assert(dataOffset == surfacesGPUSize);
    assert(surfacesGPUData.size() == surfacesGPUSize);

    ctx->writeToBuffer(m_surfaceBuffer, 0, surfacesGPUData.size(), surfacesGPUData.data());

    // Allocate and initialize the surface mapping buffer
    surfaceIndexMapping.resize(maxPreviousSurfaceIndex + 1);
    std::fill(surfaceIndexMapping.begin(), surfaceIndexMapping.end(), BINDING_INDEX_INVALID);
    
    // Assign the surface indices to instances for this frame,
    // Fill the surface mapping buffer with correct indices
    for (uint32_t surfaceIndex = 0; surfaceIndex < m_reorderedSurfaces.size(); surfaceIndex++) {
      RtInstance& surface = *m_reorderedSurfaces[surfaceIndex];

      // Ensure instances have the first seen reordered surface index set which contains a non-offsetted firstIndex of the surface
      // The actual index offsetting is done in the surface instances copied to the GPU.
      // OpacityMicromap baker passes index offset to add on top of instance's surface firstIndex via a constant buffer 
      //
      if (surface.getSurfaceIndex() == BINDING_INDEX_INVALID) {
        surface.setSurfaceIndex(surfaceIndex);

        // Single RtInstance appears multiple times in m_reorderedSurfaces, want to do this for only the first appearance.
        if (surface.surface.instancesToObject) {
          assert(surfaceIndex == surface.surface.surfaceIndexOfFirstInstance);
          for (size_t i = 0; i < surface.surface.instancesToObject->size(); ++i) {
            surfaceIndexMapping[surface.getPreviousSurfaceIndex() + i] = surfaceIndex + i;
          }
          surface.setPreviousSurfaceIndex(surfaceIndex);
        }
      }

      if (surface.getBillboardCount() == 0 && !surface.surface.instancesToObject) {
        if (surface.getPreviousSurfaceIndex() != BINDING_INDEX_INVALID) {
          surfaceIndexMapping[surface.getPreviousSurfaceIndex()] = surfaceIndex;
        }
        surface.setPreviousSurfaceIndex(surfaceIndex);
      }
    }

    if (RtxOptions::trackParticleObjects()) {
      buildParticleSurfaceMapping(surfaceIndexMapping);
    }

    // Create and upload the primitive id prefix sum buffer
    auto updatePrefixSumBuffer = [&info, this, ctx](std::vector<uint32_t>& prefixSumList, Rc<DxvkBuffer>& prefixSumBuffer) {
      info.size = std::max(prefixSumList.size(), 1llu) * sizeof(prefixSumList[0]);

      if (prefixSumBuffer == nullptr || info.size > prefixSumBuffer->info().size) {
        prefixSumBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Prefixsum Buffer");
      }

      if (prefixSumList.size() > 0) {
        ctx->writeToBuffer(prefixSumBuffer, 0, prefixSumList.size() * sizeof(prefixSumList[0]), prefixSumList.data());
      }
    };

    updatePrefixSumBuffer(m_reorderedSurfacesPrimitiveIDPrefixSum, m_primitiveIDPrefixSumBuffer);
    updatePrefixSumBuffer(m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame, m_primitiveIDPrefixSumBufferLastFrame);

    // Create and upload the surface mapping buffer
    if (!surfaceIndexMapping.empty()) {
      info.size = align(surfaceIndexMapping.size() * sizeof(int), kBufferAlignment);
      if (m_surfaceMappingBuffer == nullptr || info.size > m_surfaceMappingBuffer->info().size) {
        m_surfaceMappingBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Surface Mapping Buffer");
      }

      ctx->writeToBuffer(m_surfaceMappingBuffer, 0, surfaceIndexMapping.size() * sizeof(surfaceIndexMapping[0]), surfaceIndexMapping.data());
    }
  }

  void AccelManager::buildBlases(Rc<DxvkContext> ctx,
                                 DxvkBarrierSet& execBarriers,
                                 const CameraManager& cameraManager,
                                 OpacityMicromapManager* opacityMicromapManager,
                                 const InstanceManager& instanceManager,
                                 const std::vector<TextureRef>& textures,
                                 const std::vector<RtInstance*>& instances,
                                 const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                 std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                 std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                 size_t& totalScratchMemory) {
    ScopedGpuProfileZone(ctx, "buildBLAS");
    // Upload surfaces before opacity micromap generation which reads the surface data on the GPU
    uploadSurfaceData(ctx);

    // Build and bind opacity micromaps
    if (opacityMicromapManager && opacityMicromapManager->isActive()) {
      opacityMicromapManager->buildOpacityMicromaps(ctx, textures, cameraManager.getLastCameraCutFrameId());

      // Bind opacity micromaps
      for (auto& blasBucket : blasBuckets) {
        for (uint32_t i = 0; i < blasBucket->geometries.size(); i++) {
          auto ommSourceHash = opacityMicromapManager->tryBindOpacityMicromap(ctx, *blasBucket->originalInstances[i], blasBucket->instanceBillboardIndices[i],
                                                         blasBucket->geometries[i], instanceManager);
          if (ommSourceHash != kEmptyHash) {
            blasBucket->hasOmmInstances = true;
          }
        }
      }

      opacityMicromapManager->onBlasBuild(ctx);
    }

    // Blas buffers must be created after opacity micromaps were generated to calculate correct acceleration structure sizes
    createBlasBuffersAndInstances(ctx, blasBuckets, blasToBuild, blasRangesToBuild, totalScratchMemory);

    // Make sure we have enough scratch memory for this build job
    if (totalScratchMemory > 0) {
      m_scratchBuffer = getScratchMemory(align(totalScratchMemory, m_scratchAlignment));

      execBarriers.accessBuffer(
       m_scratchBuffer->getSliceHandle(),
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV);

      ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);
    }

    // Execute all barriers generated to this point as part of:
    //  o mergeInstancesIntoBlas()
    //  o Opacity micromap generation above
    execBarriers.recordCommands(ctx->getCommandList());

    // Build the BLASes
    if (!blasToBuild.empty()) {
      // Now apply the buffer offset to the scratch address we calculated earlier
      for (auto& desc : blasToBuild) {
        desc.scratchData.deviceAddress += m_scratchBuffer->getDeviceAddress();
      }
      assert(blasToBuild.size() == blasRangesToBuild.size());
      ctx->vkCmdBuildAccelerationStructuresKHR(blasToBuild.size(), blasToBuild.data(), blasRangesToBuild.data());

      execBarriers.accessBuffer(
       m_scratchBuffer->getSliceHandle(),
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV);

      ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_scratchBuffer);
    }
  }

  void AccelManager::buildTlas(Rc<DxvkContext> ctx) {
    uint32_t frameId = ctx->getDevice()->getCurrentFrameId();
    Logger::info(str::format("[TLAS TIMING] ========== START TLAS BUILD (frame ", frameId, ") =========="));

    if (m_vkInstanceBuffer == nullptr) {
      Logger::info(str::format("[TLAS TIMING] vkInstanceBuffer is null, returning early (frame ", frameId, ")"));
      return;
    }

    ScopedGpuProfileZone(ctx, "buildTLAS");

    Logger::info(str::format("[TLAS TIMING] BLAS pool size: ", m_blasPool.size(), " BLASes"));
    Logger::info(str::format("[TLAS TIMING] Merged instances: Opaque=", m_mergedInstances[Tlas::Opaque].size(),
                            ", Unordered=", m_mergedInstances[Tlas::Unordered].size()));

    // Two barriers in one:
    // Accel build bit - to protect from BLAS builds
    // Transfer bit - to protect from updateBuffer in prepareSceneData
    Logger::info("[TLAS TIMING] Emitting pre-TLAS barrier (BLAS writes -> TLAS reads)...");
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    Logger::info("[TLAS TIMING] Pre-TLAS barrier complete");

    // CRITICAL: Pre-inject cluster BLAS from cache BEFORE patching
    // This must happen before patchClusterBlasAddresses so it can find cluster BLAS instances
    // Matches sample's approach where BLAS data is available before FillInstanceDescs
    Logger::info("[TLAS PRE-INJECTION] Checking for cached cluster BLASes to pre-inject...");
    RtxMegaGeometry* megaGeometry = getMegaGeometry();
    if (megaGeometry) {
      uint32_t preInjectionCount = 0;
      // Iterate through all surfaces to get BlasEntry objects
      for (RtInstance* rtInstance : m_reorderedSurfaces) {
        if (!rtInstance) continue;

        BlasEntry* blasEntry = rtInstance->getBlas();
        if (!blasEntry || blasEntry->dynamicBlas.ptr()) {
          continue;  // Skip if no BLAS or already has dynamicBlas
        }

        // Check if this geometry hash has cluster BLAS in cache
        XXH64_hash_t geometryHash = blasEntry->input.getHash(RtxOptions::geometryAssetHashRule());
        if (geometryHash == kEmptyHash) {
          continue;
        }

        // Try to get cached cluster BLAS (const_cast to match rtx_scene_manager.cpp pattern)
        RtxMegaGeometry::TessellatedGeometryCache* cached =
          const_cast<RtxMegaGeometry::TessellatedGeometryCache*>(
            megaGeometry->getTessellatedGeometry(geometryHash));

        if (cached && cached->hasClusterBLAS && cached->clusterBLAS.ptr()) {
          // Check if GPU work is complete
          bool isSafeToReuse = false;

          if (cached->buildCommandList.ptr()) {
            // Direct fence check via Vulkan API (matches rtx_scene_manager.cpp:1069-1070)
            VkResult fenceStatus = m_device->vkd()->vkGetFenceStatus(
              m_device->vkd()->device(), cached->buildCommandList->fence());
            isSafeToReuse = (fenceStatus == VK_SUCCESS);
          }

          if (isSafeToReuse) {
            // Inject cluster BLAS
            Rc<PooledBlas> clusterPooledBlas = new PooledBlas();
            clusterPooledBlas->accelStructure = cached->clusterBLAS;
            clusterPooledBlas->accelerationStructureReference = 0;  // Will be GPU-patched
            clusterPooledBlas->frameLastTouched = frameId;
            clusterPooledBlas->isClusterBlas = true;
            clusterPooledBlas->clusterBlasGeometryHash = geometryHash;

            blasEntry->dynamicBlas = clusterPooledBlas;
            preInjectionCount++;

            if (preInjectionCount <= 5) {
              Logger::info(str::format("[TLAS PRE-INJECTION] Injected cluster BLAS: hash=0x",
                                      std::hex, geometryHash, std::dec, ", clusters=", cached->clusterCount));
            }
          }
        }
      }
      Logger::info(str::format("[TLAS PRE-INJECTION] Pre-injected ", preInjectionCount, " cluster BLASes from cache"));
    } else {
      Logger::info("[TLAS PRE-INJECTION] No mega geometry system available");
    }

    // GPU PATCHING: Patch cluster BLAS addresses in TLAS instance descriptors (NVIDIA sample method)
    // This matches fill_instance_descs.hlsl from rtxmg demo (line 1155 in rtxmg_renderer.cpp)
    Logger::info(str::format("[TLAS TIMING] Calling patchClusterBlasAddresses (frame ", frameId, ")..."));
    patchClusterBlasAddresses(ctx);
    Logger::info(str::format("[TLAS TIMING] patchClusterBlasAddresses returned (frame ", frameId, ")"));

    Logger::info(str::format("[TLAS TIMING] Tracking ", m_blasPool.size(), " BLAS resources..."));
    for (auto&& blas : m_blasPool) {
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(blas->accelStructure);
    }
    Logger::info("[TLAS TIMING] BLAS resource tracking complete");

    Logger::info("[TLAS TIMING] Building Opaque TLAS...");
    size_t totalScratchSize = 0;
    internalBuildTlas<Tlas::Opaque>(ctx, totalScratchSize);
    Logger::info("[TLAS TIMING] Opaque TLAS build complete, building Unordered TLAS...");
    internalBuildTlas<Tlas::Unordered>(ctx, totalScratchSize);
    Logger::info("[TLAS TIMING] Unordered TLAS build complete");

    Logger::info("[TLAS TIMING] Emitting post-TLAS barrier (TLAS writes -> ray tracing reads)...");
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    Logger::info("[TLAS TIMING] Post-TLAS barrier complete");

    // Release the scratch memory so it can be reused by rest of the frame.
    m_scratchBuffer = nullptr;

    OpacityMicromapManager* opacityMicromapManager = ctx->getCommonObjects()->getSceneManager().getOpacityMicromapManager();
    if (opacityMicromapManager) {
      opacityMicromapManager->onFinishedBuilding();
    }

    Logger::info("[TLAS TIMING] ========== END TLAS BUILD ==========");
  }

  void AccelManager::patchClusterBlasAddresses(Rc<DxvkContext> ctx) {
    // Get mega geometry system to access blasPtrsBuffer and cluster builder
    RtxMegaGeometry* megaGeometry = getMegaGeometry();
    if (!megaGeometry) {
      Logger::info("[GPU PATCHING] No mega geometry system");
      return;
    }

    // Get blasPtrsBuffer from mega geometry (populated by cluster extension)
    Rc<DxvkBuffer> blasPtrsBuffer = megaGeometry->getBlasAddressesBuffer();
    if (blasPtrsBuffer == nullptr) {
      Logger::info("[GPU PATCHING] No blasPtrsBuffer - cluster BLASes not built yet");
      return;
    }

    // NOTE: Bindless resources are always ready at this point because buildTlas is called
    // after prepareSceneData in the frame flow (see rtx_context.cpp line 563)

    Logger::info(str::format("[GPU PATCHING] blasPtrsBuffer exists, size=", blasPtrsBuffer->info().size));

    // Build patch entries by iterating through all TLAS instances
    // Use hash-based lookup to get correct blasPtrsBuffer indices
    struct PatchEntry {
      uint32_t tlasInstanceIndex;
      uint32_t blasBufferIndex;
    };

    std::vector<PatchEntry> patchEntries;
    uint32_t tlasInstanceIndex = 0;
    uint32_t totalInstances = 0;
    uint32_t clusterBlasInstances = 0;

    // Iterate through all merged instances (same order as TLAS instances)
    for (const auto& instances : m_mergedInstances) {
      totalInstances += instances.size();
      for (const VkAccelerationStructureInstanceKHR& vkInstance : instances) {
        // Get the RtInstance for this TLAS instance via m_reorderedSurfaces
        if (tlasInstanceIndex < m_reorderedSurfaces.size()) {
          RtInstance* rtInstance = m_reorderedSurfaces[tlasInstanceIndex];
          if (rtInstance) {
            BlasEntry* blasEntry = rtInstance->getBlas();
            if (blasEntry) {
              // Check if this instance has a cluster BLAS
              bool hasClusterBlas = (blasEntry->dynamicBlas != nullptr &&
                                    blasEntry->dynamicBlas->isClusterBlas);
              XXH64_hash_t geometryHash = kEmptyHash;
              if (hasClusterBlas) {
                geometryHash = blasEntry->dynamicBlas->clusterBlasGeometryHash;
              }

              if (tlasInstanceIndex < 5) {
                Logger::info(str::format("[GPU PATCHING DEBUG] Instance ", tlasInstanceIndex,
                                        ", BlasEntry=", (void*)blasEntry,
                                        ", hasDynamicBlas=", (blasEntry->dynamicBlas != nullptr),
                                        ", isClusterBlas=", hasClusterBlas,
                                        ", geometryHash=0x", std::hex, geometryHash, std::dec));
              }

              // CRITICAL: Use hash-based lookup to get correct blasPtrsBuffer index
              // The blasPtrsBuffer is populated by cluster builder in geometry hash order,
              // NOT in TLAS instance order, so we must look up the correct index
              if (hasClusterBlas && geometryHash != kEmptyHash) {
                clusterBlasInstances++;
                int32_t blasBufferIndex = megaGeometry->getBlasBufferIndex(geometryHash);
                if (blasBufferIndex >= 0) {
                  patchEntries.push_back({ tlasInstanceIndex, static_cast<uint32_t>(blasBufferIndex) });

                  if (patchEntries.size() <= 5) {
                    Logger::info(str::format("[GPU PATCHING] Found cluster BLAS instance: tlasIdx=", tlasInstanceIndex,
                                            ", hash=0x", std::hex, geometryHash, std::dec,
                                            ", blasBufferIdx=", blasBufferIndex));
                  }
                } else {
                  Logger::warn(str::format("[GPU PATCHING] Hash map lookup failed for hash=0x",
                                          std::hex, geometryHash, std::dec,
                                          " (one-frame delay - will work next frame)"));
                }
              }
            }
          }
        }
        tlasInstanceIndex++;
      }
    }

    Logger::info(str::format("[GPU PATCHING] Scanned ", totalInstances, " instances, found ",
                            clusterBlasInstances, " cluster BLAS instances, ",
                            patchEntries.size(), " patches ready (hash-based indices)"));

    if (patchEntries.empty()) {
      return;  // No cluster BLASes to patch
    }

    Logger::info(str::format("[GPU PATCHING] Patching ", patchEntries.size(),
                            " cluster BLAS instances in TLAS using GPU shader"));

    // Create/update patch entries buffer
    const size_t patchEntriesSize = patchEntries.size() * sizeof(PatchEntry);
    static Rc<DxvkBuffer> patchEntriesBuffer;

    DxvkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    bufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferInfo.size = patchEntriesSize;

    if (patchEntriesBuffer == nullptr || patchEntriesBuffer->info().size < patchEntriesSize) {
      patchEntriesBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                  DxvkMemoryStats::Category::RTXAccelerationStructure,
                                                  "Cluster BLAS Patch Entries");
    }

    // Upload patch entries to GPU
    ctx->writeToBuffer(patchEntriesBuffer, 0, patchEntriesSize, patchEntries.data());

    // Barrier to ensure patch entries are visible to shader
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);

    // CRITICAL: Barrier to ensure cluster extension's BLAS address writes are visible to GPU patching shader!
    // The cluster extension writes BLAS addresses at Frame N-1 end using VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR.
    // Our GPU patching shader (Frame N) reads those addresses using VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT.
    // This barrier ensures Frame N-1's writes complete before Frame N's reads.
    // This matches the NVIDIA RTX Mega Geometry SDK synchronization pattern (they do it implicitly via NVRHI).
    Logger::info("[GPU PATCHING DEBUG] Emitting barrier for blasAddresses buffer (cluster extension writes → compute shader reads)...");
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    Logger::info("[GPU PATCHING DEBUG] Barrier emitted successfully");

    // SDK MATCH: No ring buffering - single-frame buffers with GPU sync
    // NVIDIA sample doesn't use ring buffers, it rebuilds every frame
    // GPU synchronization is handled by the barrier above (lines 1674-1678)
    uint32_t blasAddressesBaseIndex = 0;  // No ring buffer offset

    Logger::info(str::format("[GPU PATCHING] Frame ", m_device->getCurrentFrameId(),
                            " - no ring buffering (SDK-matching)"));

    // Log buffer sizes for validation
    uint32_t totalMergedInstances = m_mergedInstances[Tlas::Opaque].size() + m_mergedInstances[Tlas::Unordered].size();
    Logger::info(str::format("[GPU PATCHING DEBUG] Buffer sizes: patchEntries=", patchEntriesSize,
                            " bytes (", patchEntries.size(), " entries), blasPtrsBuffer=",
                            blasPtrsBuffer->info().size, " bytes, instanceDescsBuffer=",
                            m_vkInstanceBuffer->info().size, " bytes (", totalMergedInstances, " instances)"));

    // SDK MATCH: Create constant buffer for params (binding 0), NOT push constants!
    struct {
      uint32_t numPatchEntries;
      uint32_t blasAddressesBaseIndex;  // Ring buffer offset for this frame
      uint32_t _pad1;
      uint32_t _pad2;
    } params;
    params.numPatchEntries = static_cast<uint32_t>(patchEntries.size());
    params.blasAddressesBaseIndex = blasAddressesBaseIndex;
    params._pad1 = 0;
    params._pad2 = 0;

    Logger::info(str::format("[GPU PATCHING DEBUG] Params: numPatchEntries=", params.numPatchEntries,
                            ", blasAddressesBaseIndex=", params.blasAddressesBaseIndex));

    // Log first few patch entries for debugging
    for (size_t i = 0; i < std::min(size_t(5), patchEntries.size()); i++) {
      Logger::info(str::format("[GPU PATCHING DEBUG] PatchEntry[", i, "]: tlasIdx=",
                              patchEntries[i].tlasInstanceIndex, ", blasBufferIdx=",
                              patchEntries[i].blasBufferIndex, " (ring-adjusted: ",
                              blasAddressesBaseIndex + patchEntries[i].blasBufferIndex, ")"));
    }

    DxvkBufferCreateInfo cbInfo = {};
    cbInfo.size = align(sizeof(params), 256);
    cbInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    cbInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cbInfo.access = VK_ACCESS_UNIFORM_READ_BIT;
    Rc<DxvkBuffer> patchTlasParamsBuffer = m_device->createBuffer(
      cbInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "RTXMG Patch TLAS Params");

    ctx->updateBuffer(patchTlasParamsBuffer, 0, sizeof(params), &params);
    Logger::info(str::format("[GPU PATCHING DEBUG] Constant buffer created and updated"));
    Logger::info(str::format("[GPU PATCHING DEBUG] ========== ABOUT TO DISPATCH GPU PATCHING SHADER =========="));
    Logger::info(str::format("[GPU PATCHING DEBUG] Will patch ", params.numPatchEntries, " TLAS instances with BLAS addresses"));
    Logger::info(str::format("[GPU PATCHING DEBUG] Reading BLAS addresses from blasAddresses[", params.blasAddressesBaseIndex, "...]"));
    Logger::info(str::format("[GPU PATCHING DEBUG] Writing to instance descriptors in TLAS build buffer"));

    // Bind and dispatch shader (accessed via mega geometry to avoid header include issues)
    Rc<DxvkShader> shader = megaGeometry->getPatchTlasInstanceShader();
    if (shader == nullptr) {
      Logger::err("[GPU PATCHING ERROR] GPU patching shader is NULL!");
      return;
    }

    // CRITICAL: Bind shader FIRST
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, shader);
    Logger::info(str::format("[GPU PATCHING DEBUG] Compute shader bound: ", shader->debugName()));

    // SDK MATCH: Bind constant buffer at binding 0
    ctx->bindResourceBuffer(0, DxvkBufferSlice(patchTlasParamsBuffer, 0, sizeof(params)));

    // Bind shader resources - BINDINGS SHIFTED +1: 0→1, 1→2, 2→3
    // Binding 1: patchEntries (TLAS index → blasPtrsBuffer index mapping) - shifted +1
    ctx->bindResourceBuffer(1, DxvkBufferSlice(patchEntriesBuffer, 0, patchEntriesSize));

    // Binding 2: blasAddresses (from cluster extension, ring-buffered) - shifted +1
    ctx->bindResourceBuffer(2, DxvkBufferSlice(blasPtrsBuffer, 0, blasPtrsBuffer->info().size));
    Logger::warn(str::format("[GPU PATCHING BUFFERS] blasPtrsBuffer bound: address=0x", std::hex,
                            blasPtrsBuffer->getDeviceAddress(), std::dec,
                            ", size=", blasPtrsBuffer->info().size, " bytes"));

    // Binding 3: instanceDescsBuffer (TLAS instances to patch) - shifted +1
    ctx->bindResourceBuffer(3, DxvkBufferSlice(m_vkInstanceBuffer, 0, m_vkInstanceBuffer->info().size));
    Logger::warn(str::format("[GPU PATCHING BUFFERS] instanceDescsBuffer bound: address=0x", std::hex,
                            m_vkInstanceBuffer->getDeviceAddress(), std::dec,
                            ", size=", m_vkInstanceBuffer->info().size, " bytes"));

    // Dispatch compute shader (256 threads per group)
    const uint32_t threadsPerGroup = 256;
    const uint32_t numGroups = (params.numPatchEntries + threadsPerGroup - 1) / threadsPerGroup;
    Logger::info(str::format("[GPU PATCHING DEBUG] Dispatching compute shader: ", numGroups,
                            " workgroups × ", threadsPerGroup, " threads = ", numGroups * threadsPerGroup, " total threads"));
    Logger::info(str::format("[GPU PATCHING DEBUG] (Processing ", params.numPatchEntries, " patch entries)"));
    Logger::info(str::format("[GPU PATCHING DISPATCH] vkCmdDispatch(", numGroups, ", 1, 1) - ABOUT TO CALL"));

    ctx->dispatch(numGroups, 1, 1);

    Logger::info(str::format("[GPU PATCHING DISPATCH] vkCmdDispatch RETURNED SUCCESSFULLY"));
    Logger::info(str::format("[GPU PATCHING DEBUG] ========== GPU PATCHING SHADER DISPATCHED =========="));
    Logger::info(str::format("[GPU PATCHING DEBUG] Shader will read ", params.numPatchEntries, " BLAS addresses and patch instance buffer"));

    // Barrier to ensure GPU patching writes complete before TLAS build reads
    // CRITICAL: vkCmdBuildAccelerationStructures reads the instance descriptor buffer
    // - Source: Compute shader writes to buffer (VK_ACCESS_SHADER_WRITE_BIT)
    // - Dest: TLAS build reads instance buffer (VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR)
    // The NVIDIA sample uses this pattern for FillInstanceDescs → buildTopLevelAccelStruct
    Logger::info(str::format("[GPU PATCHING BARRIER] ========== EMITTING POST-PATCH BARRIER =========="));
    Logger::info(str::format("[GPU PATCHING BARRIER] Ensuring compute shader writes complete before TLAS build"));
    Logger::info(str::format("[GPU PATCHING BARRIER] Source: VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT"));
    Logger::info(str::format("[GPU PATCHING BARRIER] Dest: VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR"));
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    Logger::info(str::format("[GPU PATCHING DEBUG] Memory barrier emitted successfully"));

    Logger::info(str::format("[GPU PATCHING] Dispatched shader: ", numGroups, " workgroups, ",
                            patchEntries.size(), " patches"));
    Logger::info("[GPU PATCHING] patchClusterBlasAddresses complete, returning");
  }

  template<Tlas::Type type>
  void AccelManager::internalBuildTlas(Rc<DxvkContext> ctx, size_t& totalScratchSize) {
    static constexpr const char* names[] = { "TLAS_Opaque", "TLAS_NonOpaque" };
    Logger::info(str::format("[TLAS BUILD DEBUG] ========== Starting internalBuildTlas for ", names[type],
                            " (Frame ", m_device->getCurrentFrameId(), ") =========="));
    ScopedGpuProfileZone(ctx, names[type]);
    const VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();

    const auto& vkd = m_device->vkd();

    // Create VkAccelerationStructureGeometryInstancesDataKHR
    // This wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instancesVk.arrayOfPointers = VK_FALSE;
    instancesVk.data.deviceAddress = m_vkInstanceBuffer->getDeviceAddress();
    Logger::info(str::format("[TLAS BUILD DEBUG] Instance buffer device address: 0x", std::hex,
                            instancesVk.data.deviceAddress, std::dec));

    // Rewind address to tlas start
    for (size_t n = 0; n < type; ++n) {
      instancesVk.data.deviceAddress += m_mergedInstances[n].size() * sizeof(VkAccelerationStructureInstanceKHR);
    }

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the
    // instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.flags = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    const uint32_t numInstances = uint32_t(m_mergedInstances[type].size());
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkd->vkGetAccelerationStructureBuildSizesKHR(vkd->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &numInstances, &sizeInfo);

    // Create TLAS
    Tlas& tlas = m_device->getCommon()->getResources().getTLAS(type);

    if (type == Tlas::Opaque)
      std::swap(tlas.accelStructure, tlas.previousAccelStructure);

    if (tlas.accelStructure == nullptr || sizeInfo.accelerationStructureSize > tlas.accelStructure->info().size) {
      ScopedGpuProfileZone(ctx, "buildTLAS_createAccelStructure");
      DxvkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
      info.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      info.size = sizeInfo.accelerationStructureSize;

      tlas.accelStructure = m_device->createAccelStructure(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, names[type]);

      Logger::debug(str::format("DxvkRaytrace: TLAS Realloc"));
    }

    // Allocate the scratch memory, we share the same buffer between all TLAS types, so just ensure we handle the offsetting correctly here.
    const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
    buildInfo.scratchData.deviceAddress = getScratchMemory(totalScratchSize + requiredScratchAllocSize)->getDeviceAddress() + totalScratchSize;
    totalScratchSize += requiredScratchAllocSize;

    // Update build information
    buildInfo.srcAccelerationStructure = nullptr;
    buildInfo.dstAccelerationStructure = tlas.accelStructure->getAccelStructure();

    assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo { numInstances, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    Logger::info(str::format("[TLAS TIMING]   vkCmdBuildAccelerationStructures for ", names[type],
                            ": instances=", numInstances,
                            ", accelSize=", sizeInfo.accelerationStructureSize / 1024, " KB",
                            ", scratchSize=", sizeInfo.buildScratchSize / 1024, " KB"));
    Logger::info(str::format("[TLAS BUILD] ========== ABOUT TO BUILD TLAS: ", names[type], " =========="));
    Logger::info(str::format("[TLAS BUILD] Number of instances: ", numInstances));
    Logger::info(str::format("[TLAS BUILD] Acceleration structure size: ", sizeInfo.accelerationStructureSize, " bytes"));
    Logger::info(str::format("[TLAS BUILD] Scratch size: ", sizeInfo.buildScratchSize, " bytes"));
    Logger::info(str::format("[TLAS BUILD] buildInfo.type: ", buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ? "TOP_LEVEL" : "UNKNOWN"));
    Logger::info(str::format("[TLAS BUILD] buildInfo.flags: 0x", std::hex, buildInfo.flags, std::dec));
    Logger::info(str::format("[TLAS BUILD] buildInfo.mode: ", buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? "BUILD" : "UPDATE"));
    Logger::info(str::format("[TLAS BUILD] buildInfo.srcAccelerationStructure: ", buildInfo.srcAccelerationStructure ? "valid" : "NULL"));
    Logger::info(str::format("[TLAS BUILD] buildInfo.dstAccelerationStructure: 0x", std::hex,
                            (uint64_t)buildInfo.dstAccelerationStructure, std::dec));
    Logger::info(str::format("[TLAS BUILD] buildInfo.scratchData.deviceAddress: 0x", std::hex,
                            buildInfo.scratchData.deviceAddress, std::dec));
    Logger::info(str::format("[TLAS BUILD] instancesVk.data.deviceAddress: 0x", std::hex,
                            topASGeometry.geometry.instances.data.deviceAddress, std::dec));
    Logger::info(str::format("[TLAS BUILD] instancesVk.arrayOfPointers: ", topASGeometry.geometry.instances.arrayOfPointers ? "TRUE" : "FALSE"));
    Logger::info(str::format("[TLAS BUILD] buildOffsetInfo.primitiveCount: ", buildOffsetInfo.primitiveCount));
    Logger::info(str::format("[TLAS BUILD] buildOffsetInfo.primitiveOffset: ", buildOffsetInfo.primitiveOffset));
    Logger::info(str::format("[TLAS BUILD] CALLING vkCmdBuildAccelerationStructuresKHR NOW..."));

    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildOffsetInfo);

    Logger::info(str::format("[TLAS BUILD] ========== vkCmdBuildAccelerationStructuresKHR RETURNED =========="));
    Logger::info(str::format("[TLAS BUILD] TLAS build command for ", names[type], " submitted successfully"));
    Logger::info(str::format("[TLAS TIMING]   ", names[type], " build command submitted"));

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(tlas.accelStructure);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);
  }

  // Check if the existing build geometry info for this blas is compatible with the new one for the purpose of updating rather than rebuilding
  bool AccelManager::validateUpdateMode(const VkAccelerationStructureBuildGeometryInfoKHR& oldInfo, const VkAccelerationStructureBuildGeometryInfoKHR& newInfo) {
    if (!(oldInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)) {
      return false;
    }

    if (oldInfo.type != newInfo.type || oldInfo.flags != newInfo.flags || oldInfo.geometryCount != newInfo.geometryCount) {
      return false;
    }

    for (uint32_t i = 0; i < oldInfo.geometryCount; ++i) {
      const VkAccelerationStructureGeometryKHR* oldGeom = oldInfo.pGeometries ? &oldInfo.pGeometries[i] : oldInfo.ppGeometries[i];
      const VkAccelerationStructureGeometryKHR* newGeom = newInfo.pGeometries ? &newInfo.pGeometries[i] : newInfo.ppGeometries[i];

      if (oldGeom->geometryType != newGeom->geometryType || oldGeom->flags != newGeom->flags) {
        return false;
      }

      // Per validation layers we need to check attributes of geometry types
      switch (oldGeom->geometryType) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      {
        const auto& oldTriangles = oldGeom->geometry.triangles;
        const auto& newTriangles = newGeom->geometry.triangles;
        if (oldTriangles.vertexFormat != newTriangles.vertexFormat ||
            oldTriangles.indexType != newTriangles.indexType ||
            oldTriangles.maxVertex != newTriangles.maxVertex ||
            oldTriangles.vertexStride != newTriangles.vertexStride) {
          return false;
        }
        break;
      }
      case VK_GEOMETRY_TYPE_AABBS_KHR:
      {
        const auto& oldAabbs = oldGeom->geometry.aabbs;
        const auto& newAabbs = newGeom->geometry.aabbs;
        if (oldAabbs.stride != newAabbs.stride) {
          return false;
        }
        break;
      }
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
      {
        const auto& oldInstances = oldGeom->geometry.instances;
        const auto& newInstances = newGeom->geometry.instances;
        if (oldInstances.arrayOfPointers != newInstances.arrayOfPointers) {
          return false;
        }
        break;
      }
      default:
        return false;
      }
    }
    return true;
  }
}  // namespace dxvk
