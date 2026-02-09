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
#include <set>
#include <vector>
#include <assert.h>

// Disable verbose MegaGeo logging in hot path
#define RTXMG_VERBOSE_LOGGING 0
#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_scene_manager.h"
#include "rtx_resources.h"
#include "rtx_accel_manager.h"
#include "rtx_megageo/rtx_megageo_builder.h"
#include "rtx_megageo/nvrhi_adapter/nvrhi_dxvk_buffer.h"

#include "../d3d9/d3d9_state.h"
#include "rtx_matrix_helpers.h"

#include "dxvk_scoped_annotation.h"
#include "rtx_options.h"

#include "rtx/pass/instance_definitions.h"
#include "rtx/concept/billboard.h"

#include "rtx/pass/common_binding_indices.h"

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

    // Initialize RTX Mega Geometry Builder (if VK_NV_cluster_acceleration_structure is supported)
    // Note: Initialization is deferred - will be lazily created when first subdivision surface is encountered
    // This avoids overhead when cluster BLAS is not used
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
      if (instanceMask != geometryInstanceMask) {
        return false;
      }
      if (instanceShaderBindingTableRecordOffset != geometryInstanceShaderBindingTableRecordOffset) {
        return false;
      }
      if (customIndexFlags != geometryCustomIndexFlags) {
        return false;
      }
      if (instanceFlags != geometryInstanceFlags) {
        return false;
      }
      if (usesUnorderedApproximations != geometryUsesUnorderedApproximations) {
        return false;
      }
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
    } else {
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
    if (m_intersectionBlas.ptr()) {
      return;
    }

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

    // RTX Mega Geometry: ClusterBlas uses MegaGeo's own buffers, not the original geometry buffers.
    // Don't track the original buffers for ClusterBlas to avoid keeping destroyed buffers alive.
    if (blasEntry->blasType == BlasEntry::Type::ClusterBlas) {
      return;
    }

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
    // RTX Mega Geometry: Clear cluster instance patches from previous frame
    m_clusterInstancePatches.clear();

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    if (instances.size() > CUSTOM_INDEX_SURFACE_MASK) {
      ONCE(Logger::err("DxvkRaytrace: instances size is greater than max supported custom index value"));
    }

    if (opacityMicromapManager) {
      opacityMicromapManager->onFrameStart(ctx);
    }

    std::vector<std::unique_ptr<BlasBucket>> blasBuckets;
    blasBuckets.reserve(instances.size());

    size_t totalScratchMemory = 0;

    // NOTE: Would like to use the BLAS Linked instances here, but that misses viewmodel and virtual instances
    std::unordered_map<BlasEntry*, std::vector<RtInstance*>> uniqueBlas;

    // RTX Mega Geometry: Process completed surfaces and build cluster BLAS BEFORE the instance loop
    // This ensures surfaceId->instanceIndex mapping is populated before addBlas() is called
    // CRITICAL: buildClusterBlas must run BEFORE addBlas so getInstanceIndexForSurface works

    if (m_megaGeoBuilder != nullptr) {
      Logger::warn("CHECKPOINT: mergeInstances pre-processCompletedSurfaces");
      m_megaGeoBuilder->processCompletedSurfaces();

      // Collect transforms for all cluster BLAS instances FIRST
      // Note: Use getVkInstance().transform directly (row-major) instead of getTransform() (column-major)
      // because MegaGeo's affineToColumnMajor expects row-major input
      std::unordered_map<uint32_t, Matrix4> surfaceTransforms;
      uint32_t clusterInstCount = 0;
      Logger::warn(str::format("CHECKPOINT: mergeInstances collecting transforms, instances=", instances.size()));
      for (RtInstance* inst : instances) {
        if (inst && inst->getBlas() && inst->getBlas()->isClusterBlas()) {
          uint32_t surfaceId = inst->getBlas()->megaGeoSurfaceId;
          // Build Matrix4 directly from VkTransformMatrixKHR (row-major) without transposing
          // This matches what affineToColumnMajor expects
          Matrix4 worldTransform = Matrix4(inst->getVkInstance().transform);
          surfaceTransforms[surfaceId] = worldTransform;
          clusterInstCount++;
        }
      }
      Logger::warn(str::format("CHECKPOINT: mergeInstances transforms collected, cluster=", clusterInstCount, " surfaces=", surfaceTransforms.size()));

      // No transform preservation: ClusterBlas fast-path in DrawCallCache keeps entries alive,
      // and 10x GC lifetime protects them. Preserved transforms waste cluster budget on invisible surfaces.

      // Build cluster BLAS if we have any transforms
      if (!surfaceTransforms.empty()) {
        Rc<RtxContext> rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
        if (rtxCtx != nullptr) {
          const RtCamera& camera = cameraManager.getCamera(CameraType::Main);

          // Get depth buffer for HiZ culling
          Rc<DxvkImageView> depthBuffer = nullptr;
          if (rtxCtx->getCommonObjects()->getResources().isResourceReady()) {
            Resources::RaytracingOutput& rtOutput = rtxCtx->getCommonObjects()->getResources().getRaytracingOutput();
            if (rtOutput.m_primaryDepth.view != nullptr) {
              depthBuffer = rtOutput.m_primaryDepth.view;
            }
          }

          // Build cluster BLAS - this rebuilds instances from surfaceTransforms
          // After this call, getInstanceIndexForSurface will return valid indices
          Logger::warn(str::format("CHECKPOINT: pre-buildClusterBlas surfaces=", surfaceTransforms.size()));
          m_megaGeoBuilder->buildClusterBlas(rtxCtx, depthBuffer, camera, surfaceTransforms);

          // Barrier: RTXMG compute writes → ray tracing shader reads
          // BuildAccel writes to clusterVertexPositions, clusterVertexNormals, clusterShadingData
          // via compute shaders. Ray tracing shaders read these as storage buffers.
          // DXVK's resource tracking doesn't see NVRHI-side writes, so explicit barrier is needed.
          ctx->emitMemoryBarrier(0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_SHADER_READ_BIT);
        }
      }
    }

    Logger::warn("CHECKPOINT: post-buildClusterBlas");

    // RTX MegaGeo: Per-frame geometry pipeline counters
    uint32_t geoCountTotal = 0;           // Total input instances from the game
    uint32_t geoCountHidden = 0;          // Skipped: hidden
    uint32_t geoCountZeroMask = 0;        // Skipped: zero mask
    uint32_t geoCountClusterSubmitted = 0; // Cluster instances submitted to addBlas
    uint32_t geoCountClusterNoBuffers = 0; // Cluster instances skipped: buffers not ready
    uint32_t geoCountStandard = 0;        // Standard BLAS instances
    // Cluster skip counters from addBlas are tracked via m_frameClusterSkipZeroTris / m_frameClusterSkipNoInstance
    m_frameClusterSkipZeroTris = 0;
    m_frameClusterSkipNoInstance = 0;

    for (RtInstance* instance : instances) {
      geoCountTotal++;

      if (instance->isHidden()) {
        geoCountHidden++;
        continue;
      }

      // If the instance has zero mask, do not build BLAS for it: no ray can intersect this instance.
      if (instance->getVkInstance().mask == 0) {
        geoCountZeroMask++;
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

      // RTX Mega Geometry: Cluster-based BLASes are built separately via RtxMegaGeoBuilder
      // They still need to be added to the TLAS, but skip the traditional BLAS building logic
      if (blasEntry->isClusterBlas()) {
        // Check if megaGeoBuilder has valid buffers before adding cluster instances
        // Without valid buffers, the shader would read garbage vertex data causing geometry distortion
        RtxMegaGeoBuilder* megaGeo = getMegaGeoBuilder();
        bool hasValid = megaGeo && megaGeo->hasValidBuffers();

        if (!hasValid) {
          geoCountClusterNoBuffers++;
          continue;
        }

        // Cluster surfaces need a valid surface index for shader lookup.
        // CRITICAL: Must reassign every frame because m_reorderedSurfaces is cleared each frame.
        // Using a stale surfaceIndex from a previous frame would point to the wrong Surface entry,
        // causing the shader to read isClusterSurface=false and take the wrong geometry path.
        instance->setSurfaceIndex(m_reorderedSurfaces.size());
        m_reorderedSurfaces.push_back(instance);
        m_reorderedSurfacesFirstIndexOffset.push_back(0);

        // Add directly to TLAS - the BLAS address will be patched on GPU later
        if (instance->surface.instancesToObject == nullptr) {
          addBlas(instance, blasEntry, nullptr);
          geoCountClusterSubmitted++;
        } else {
          for (auto& instanceToObject : *instance->surface.instancesToObject) {
            addBlas(instance, blasEntry, &instanceToObject);
            geoCountClusterSubmitted++;
          }
        }

        continue; // Skip traditional BLAS building (geometry fill, dynamic/merged decision, etc.)
      }

      geoCountStandard++;
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

      if (requestDynamicBlas && !forceMergedBlas) {
        // Since this loop is iterating over instances, and instances can share BLAS, we will build these later after identifying unique ones.
        uniqueBlas[blasEntry].push_back(instance);
      } else {
        // Make sure we don't double up on blas entries, this should only happen if theres a bug
        // TODO (REMIX-3996) will break the assumptions we make here about all instances in a BlasEntry having the same instancesToObject array
        assert(uniqueBlas.find(blasEntry) == uniqueBlas.end());

        if (blasEntry->dynamicBlas != nullptr) {
          // Move the BLAS used by this geometry to the common pool.
          // This also ensures the dynamic blas resource that's still being used by previous TLAS is properly tracked for the next frame
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
    for (const std::pair<BlasEntry*, std::vector<RtInstance*>> pair : uniqueBlas) {
      BlasEntry* blasEntry = pair.first;
      if (pair.second.size() == 0) {
        continue;
      }
      assert(blasEntry->buildGeometries.size() == 1); // dynamic BLAS should always have this
      assert(blasEntry->buildRanges.size() == 1); // dynamic BLAS should always have this

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
      Rc<PooledBlas>& selectedBlas = blasEntry->dynamicBlas;

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
    if (instanceTransforms.size() > 0) {
      ctx->writeToBuffer(m_transformBuffer, 0, instanceTransforms.size() * sizeof(VkTransformMatrixKHR), instanceTransforms.data());
    }

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

    // RTX MegaGeo: Per-frame cluster instance stability debug
    // Tracks each cluster instance across frames to detect instability
    {
      struct ClusterInstanceInfo {
        uintptr_t instPtr;       // RtInstance pointer (stable identity)
        uintptr_t blasPtr;       // BlasEntry pointer
        uint32_t surfaceId;      // MegaGeo surface ID
        uint32_t surfaceIndex;   // Surface buffer index this frame
        XXH64_hash_t geoHash;    // associatedGeometryHash
        float posX, posY, posZ;  // World position
      };
      static std::vector<ClusterInstanceInfo> s_prevFrameInstances;
      static uint32_t s_debugFrameCount = 0;
      s_debugFrameCount++;

      std::vector<ClusterInstanceInfo> thisFrameInstances;
      for (RtInstance* inst : instances) {
        if (inst->isHidden()) continue;
        if (inst->getVkInstance().mask == 0) continue;
        BlasEntry* blas = inst->getBlas();
        if (!blas || !blas->isClusterBlas()) continue;
        ClusterInstanceInfo info;
        info.instPtr = reinterpret_cast<uintptr_t>(inst);
        info.blasPtr = reinterpret_cast<uintptr_t>(blas);
        info.surfaceId = blas->megaGeoSurfaceId;
        info.surfaceIndex = inst->getSurfaceIndex();
        info.geoHash = inst->surface.associatedGeometryHash;
        Vector3 wp = inst->getWorldPosition();
        info.posX = wp.x; info.posY = wp.y; info.posZ = wp.z;
        thisFrameInstances.push_back(info);
      }

      // Every frame, diff against previous
      if (!thisFrameInstances.empty()) {
        // Build lookup of previous frame by instance pointer
        std::unordered_map<uintptr_t, ClusterInstanceInfo> prevMap;
        for (auto& p : s_prevFrameInstances) prevMap[p.instPtr] = p;

        uint32_t unchanged = 0, hashChanged = 0, blasChanged = 0, surfIdChanged = 0, newInst = 0, gone = 0;
        for (auto& cur : thisFrameInstances) {
          auto it = prevMap.find(cur.instPtr);
          if (it == prevMap.end()) {
            newInst++;
            continue;
          }
          auto& prev = it->second;
          bool anyChange = false;
          if (cur.geoHash != prev.geoHash) { hashChanged++; anyChange = true; }
          if (cur.blasPtr != prev.blasPtr) { blasChanged++; anyChange = true; }
          if (cur.surfaceId != prev.surfaceId) { surfIdChanged++; anyChange = true; }
          if (!anyChange) unchanged++;
          prevMap.erase(it);
        }
        gone = static_cast<uint32_t>(prevMap.size());

        RTXMG_LOG(str::format("RTX MegaGeo STABILITY[", s_debugFrameCount, "]: ",
            "count=", thisFrameInstances.size(),
            " unchanged=", unchanged, " hashChanged=", hashChanged,
            " blasChanged=", blasChanged, " surfIdChanged=", surfIdChanged,
            " new=", newInst, " gone=", gone));
      }
      s_prevFrameInstances = std::move(thisFrameInstances);
    }

    Logger::warn(str::format("CHECKPOINT: post-instanceLoop total=", geoCountTotal, " cluster=", geoCountClusterSubmitted, " std=", geoCountStandard));

    // RTX MegaGeo: Geometry pipeline summary
    // Shows how many game geometries survive to the final TLAS output
    {
      uint32_t tlasOutput = static_cast<uint32_t>(m_mergedInstances[Tlas::Opaque].size() + m_mergedInstances[Tlas::Unordered].size());
      uint32_t clusterInTlas = geoCountClusterSubmitted - m_frameClusterSkipZeroTris - m_frameClusterSkipNoInstance;
      uint32_t totalInvalidated = geoCountHidden + geoCountZeroMask + geoCountClusterNoBuffers + m_frameClusterSkipZeroTris + m_frameClusterSkipNoInstance;
      static uint32_t s_geoPipelineLogCount = 0;
      // Always log when output is 0 but we had cluster input (black frame detection)
      if (tlasOutput == 0 && geoCountTotal > 0) {
        Logger::warn(str::format("RTX MegaGeo GEO PIPELINE: BLACK FRAME - input=", geoCountTotal,
            " | output=0 (standard=", geoCountStandard, " cluster=", clusterInTlas, ")",
            " | invalidated=", totalInvalidated,
            " (hidden=", geoCountHidden, " zeroMask=", geoCountZeroMask,
            " noBuffers=", geoCountClusterNoBuffers,
            " zeroTris=", m_frameClusterSkipZeroTris, " noInstance=", m_frameClusterSkipNoInstance, ")"));
      }
      if ((s_geoPipelineLogCount++ % 60) == 0) {
        RTXMG_LOG(str::format("RTX MegaGeo GEO PIPELINE: input=", geoCountTotal,
            " | output=", tlasOutput, " (standard=", geoCountStandard, " cluster=", clusterInTlas, ")",
            " | invalidated=", totalInvalidated,
            " (hidden=", geoCountHidden, " zeroMask=", geoCountZeroMask,
            " noBuffers=", geoCountClusterNoBuffers,
            " zeroTris=", m_frameClusterSkipZeroTris, " noInstance=", m_frameClusterSkipNoInstance, ")"));
      }
    }

    buildBlases(ctx, execBarriers, cameraManager, opacityMicromapManager, instanceManager,
                textures, instances, blasBuckets, blasToBuild, blasRangesToBuild, totalScratchMemory);
  }

  void AccelManager::addBlas(RtInstance* instance, BlasEntry* blasEntry, const Matrix4* instanceToObject) {
    // Create an instance for this BLAS
    VkAccelerationStructureInstanceKHR blasInstance = instance->getVkInstance();

    // RTX Mega Geometry: For cluster BLASes, the address will be patched on GPU
    // Set to 0 here and record mapping for later GPU-side patching
    bool isClusterBlas = blasEntry->isClusterBlas();
    if (isClusterBlas) {
      RTXMG_LOG(str::format("RTX MegaGeo AccelManager::addBlas: isClusterBlas=", isClusterBlas,
                               " megaGeoBuilder=", (blasEntry->megaGeoBuilder ? "valid" : "NULL"),
                               " surfaceId=", blasEntry->megaGeoSurfaceId));
    }
    if (isClusterBlas && blasEntry->megaGeoBuilder) {
      // If tessellation produced zero triangles, ALL BLAS addresses in blasPtrsBuffer are 0.
      // GPU patching would copy these zeros into the TLAS instance buffer, causing ray tracing
      // to hit null BLAS pointers → GPU hang (TDR/DEVICE_LOST). Skip ALL cluster entries.
      const auto& tessStats = blasEntry->megaGeoBuilder->getStats();
      if (tessStats.numTriangles == 0) {
        m_frameClusterSkipZeroTris++;
        return;
      }

      // Get the RTXMG instance index for this surface
      uint32_t rtxmgIndex = blasEntry->megaGeoBuilder->getInstanceIndexForSurface(blasEntry->megaGeoSurfaceId);
      if (rtxmgIndex == UINT32_MAX) {
        m_frameClusterSkipNoInstance++;
        return;
      }

      // Set placeholder - will be patched by patchClusterBlasAddressesGPU()
      blasInstance.accelerationStructureReference = 0;

      // Determine which TLAS this will go into and calculate the instance index
      Tlas::Type tlasType = (instance->usesUnorderedApproximations() && RtxOptions::enableSeparateUnorderedApproximations())
                            ? Tlas::Unordered : Tlas::Opaque;
      uint32_t remixIndex = static_cast<uint32_t>(m_mergedInstances[tlasType].size());

      // Record the mapping for GPU-side patching
      m_clusterInstancePatches.push_back({remixIndex, rtxmgIndex, tlasType});
    } else {
      // Traditional BLAS - use the address directly
      blasInstance.accelerationStructureReference = blasEntry->getBlasAddress();
    }

    // RTX MegaGeo: For cluster surfaces, use the pre-assigned surface index (set before addBlas call)
    // For non-cluster surfaces, use m_reorderedSurfaces.size() which will be the index after we add it below
    uint32_t surfaceIndexForCustomIndex = isClusterBlas ? instance->getSurfaceIndex() : uint32_t(m_reorderedSurfaces.size());

    blasInstance.instanceCustomIndex =
      (blasInstance.instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
      (surfaceIndexForCustomIndex & uint32_t(CUSTOM_INDEX_SURFACE_MASK));

    // RTX MegaGeo: Set cluster surface flag so shader knows not to add geometryIndex to surfaceIndex
    if (isClusterBlas) {
      blasInstance.instanceCustomIndex |= CUSTOM_INDEX_IS_CLUSTER_SURFACE;
      // Note: Cluster vertices are stored in LOCAL space by fill_clusters.
      // The TLAS instance transform (inherited from the RtInstance) transforms rays to local space.
      // The shader must transform vertex positions by surface.objectToWorld when reading them.

    }

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
    // RTX MegaGeo: Skip for cluster surfaces - they're already added in prepareSceneData before addBlas is called
    if (!isClusterBlas) {
      m_reorderedSurfaces.push_back(instance);
      m_reorderedSurfacesFirstIndexOffset.push_back(0);
    }
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

      // Try to find an existing BLAS that is minimally sufficient to fit this bucket of geometries
      PooledBlas* selectedBlas = nullptr;
      for (const auto& blas : m_blasPool) {
        size_t bufferSize = blas->accelStructure->info().size;
        uint32_t paddedLastTouched = blas->frameLastTouched + 1 + (RtxOptions::enablePreviousTLAS() ? 1u : 0u); /* note: +2 because frameLastTouched is unsigned and init'd with UINT32_MAX, and keep the BLAS'es for one extra frame for previous TLAS access */
        if (bufferSize >= sizeInfo.accelerationStructureSize &&
            (!selectedBlas || bufferSize < selectedBlas->accelStructure->info().size) &&
            paddedLastTouched <= currentFrame) {
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
      assert(selectedBlas);
      selectedBlas->frameLastTouched = currentFrame;

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

      static float identityTransform[3][4] = {
        { 1.f, 0.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f, 0.f },
        { 0.f, 0.f, 1.f, 0.f }
      };

      // Append an instance of this merged BLAS to the merged instance list
      VkAccelerationStructureInstanceKHR instance {};
      instance.accelerationStructureReference = selectedBlas->accelerationStructureReference;
      instance.flags = bucket->instanceFlags;
      instance.instanceShaderBindingTableRecordOffset = bucket->instanceShaderBindingTableRecordOffset;
      instance.mask = bucket->instanceMask;
      instance.instanceCustomIndex =
        (bucket->customIndexFlags & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
        (bucket->reorderedSurfacesOffset & uint32_t(CUSTOM_INDEX_SURFACE_MASK));
      memcpy(static_cast<void*>(&instance.transform.matrix[0][0]), &identityTransform[0][0], sizeof(VkTransformMatrixKHR));

      if (bucket->usesUnorderedApproximations && RtxOptions::enableSeparateUnorderedApproximations()) {
        m_mergedInstances[Tlas::Unordered].push_back(instance);
      } else {
        m_mergedInstances[Tlas::Opaque].push_back(instance);
      }
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

    if (!haveInstances && instanceManager.getBillboards().empty()) {
      return;
    }

    createAndBuildIntersectionBlas(ctx, execBarriers);

    // Prepare billboard data and instances
    std::vector<MemoryBillboard> memoryBillboards;
    uint32_t numActiveBillboards = 0;

    // Check the enablement here - because the instance manager needs to run the billboard analysis all the time
    if (RtxOptions::enableBillboardOrientationCorrection()) {
      memoryBillboards.resize(instanceManager.getBillboards().size());
      uint32_t index = 0;

      for (const auto& billboard : instanceManager.getBillboards()) {
        if (billboard.instanceMask == 0 || !billboard.allowAsIntersectionPrimitive) {
          continue;
        }

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
        if (billboard.isBeam) {
          memory.flags |= billboardFlagIsBeam;
        }
        if (billboard.isCameraFacing) {
          memory.flags |= billboardFlagIsCameraFacing;
        }

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
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;  // RTX MegaGeo: needed for GPU-side cluster BLAS address patching (RWStructuredBuffer)
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

    // RTX Mega Geometry: Patch cluster BLAS addresses directly on GPU
    // This is done AFTER uploading instance data so we can patch in-place
    if (!m_clusterInstancePatches.empty() && m_megaGeoBuilder != nullptr) {
      // Create NVRHI wrapper for the instance buffer
      nvrhi::BufferDesc wrapperDesc;
      wrapperDesc.byteSize = m_vkInstanceBuffer->info().size;
      wrapperDesc.structStride = sizeof(VkAccelerationStructureInstanceKHR);
      wrapperDesc.debugName = "InstanceBufferWrapper";
      wrapperDesc.canHaveUAVs = true;

      nvrhi::BufferHandle instanceBufferWrapper = new NvrhiDxvkBuffer(wrapperDesc, m_vkInstanceBuffer);

      // Prepare mappings - need to account for TLAS offsets
      uint32_t opaqueCount = static_cast<uint32_t>(m_mergedInstances[Tlas::Opaque].size());

      std::vector<RtxMegaGeoBuilder::InstancePatchMapping> mappings;
      mappings.reserve(m_clusterInstancePatches.size());
      std::set<uint32_t> uniqueRtxmgIndices;
      for (const auto& patch : m_clusterInstancePatches) {
        uint32_t globalIndex = patch.remixInstanceIndex;
        if (patch.tlasType == Tlas::Unordered) {
          globalIndex += opaqueCount;
        }
        mappings.push_back({globalIndex, patch.rtxmgInstanceIndex});
        uniqueRtxmgIndices.insert(patch.rtxmgInstanceIndex);
      }

      // Barrier: BLAS build wrote blasPtrsBuffer, transfer wrote instance buffer.
      // GPU PATCH compute shader needs to read blasPtrsBuffer and write instance buffer.
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

      Logger::warn(str::format("CHECKPOINT: pre-patchGPU mappings=", mappings.size()));
      m_megaGeoBuilder->patchClusterBlasAddressesGPU(instanceBufferWrapper.Get(), 0, mappings);
      Logger::warn("CHECKPOINT: post-patchGPU");

      // Barrier: GPU PATCH compute wrote instance buffer, TLAS build needs to read it
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
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

      // Log cluster surface data being uploaded
      if (currentSurface.isClusterSurface) {
        static uint32_t s_clusterSurfUploadLog = 0;
      }

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
    Logger::warn("CHECKPOINT: buildBlases-enter");
    // Upload surfaces before opacity micromap generation which reads the surface data on the GPU
    uploadSurfaceData(ctx);
    Logger::warn("CHECKPOINT: post-uploadSurfaceData");

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

    // RTX Mega Geometry: Build cluster BLAS for subdivision surfaces
    // Check if any instances use cluster BLAS before initializing
    bool hasClusterBlas = false;
    uint32_t clusterBlasCount = 0;
    uint32_t nullBlasCount = 0;
    uint32_t validBlasCount = 0;
    static uint32_t logCount = 0;
    bool shouldLogThisFrame = (logCount < 200); // Log first 200 frames
    logCount++;
    for (const auto& inst : instances) {
      if (inst && inst->getBlas()) {
        validBlasCount++;
        BlasEntry* blas = inst->getBlas();
        if (shouldLogThisFrame && blas->isClusterBlas()) {
          RTXMG_LOG(str::format("RTX MegaGeo buildBlases: FOUND ClusterBlas! inst=", (void*)inst, " blas=", (void*)blas));
        }
        if (blas->isClusterBlas()) {
          hasClusterBlas = true;
          clusterBlasCount++;
        }
      } else if (inst) {
        nullBlasCount++;
      }
    }

    // Log statistics every 60 frames to help debug timing issues
    static uint32_t statsLogCounter = 0;
    if ((statsLogCounter++ % 60) == 0) {
      RTXMG_LOG(str::format("RTX MegaGeo buildBlases: instances=", instances.size(),
                               " validBlas=", validBlasCount, " nullBlas=", nullBlasCount,
                               " clusterBlas=", clusterBlasCount));
    }

    static bool loggedOnce = false;
    if (hasClusterBlas && !loggedOnce) {
      ONCE(Logger::info(str::format("RTX MegaGeo: Found ", clusterBlasCount, " ClusterBlas instances")));
      loggedOnce = true;
    }

    if (hasClusterBlas) {
      // Lazy initialization of RtxMegaGeoBuilder (if not initialized earlier)
      Rc<RtxContext> rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
      if (rtxCtx != nullptr && m_megaGeoBuilder == nullptr) {
        m_megaGeoBuilder = new RtxMegaGeoBuilder(m_device, rtxCtx);
        if (!m_megaGeoBuilder->initialize()) {
          Logger::err("RTX MegaGeo: Failed to initialize RtxMegaGeoBuilder (late init)");
          m_megaGeoBuilder = nullptr;
        }
      }

      // NOTE: buildClusterBlas is now called EARLIER (before instance loop)
      // to ensure getInstanceIndexForSurface returns valid indices in addBlas()
      // This section is kept for lazy initialization only
      if (m_megaGeoBuilder != nullptr) {
        // Already built earlier - just log stats
        RTXMG_LOG("RTX MegaGeo AccelManager: Cluster BLAS already built earlier in frame");
      } else {
        ONCE(Logger::err("RTX MegaGeo AccelManager: Builder is null after late init attempt"));
      }
    } else {
      static bool infoLogged = false;
      if (!infoLogged && clusterBlasCount == 0 && instances.size() > 0) {
        infoLogged = true;
      }
    }
    RTXMG_LOG("RTX MegaGeo AccelManager: buildBlases exiting after cluster BLAS section");
  }

  bool AccelManager::initializeMegaGeoBuilder(Rc<DxvkContext> ctx) {
    // Early exit if already initialized
    if (m_megaGeoBuilder != nullptr) {
      return true;
    }

    Rc<RtxContext> rtxCtx = dynamic_cast<RtxContext*>(ctx.ptr());
    if (rtxCtx == nullptr) {
      Logger::err("RTX MegaGeo: Failed to cast DxvkContext to RtxContext for builder initialization");
      return false;
    }

    m_megaGeoBuilder = new RtxMegaGeoBuilder(m_device, rtxCtx);
    if (!m_megaGeoBuilder->initialize()) {
      Logger::err("RTX MegaGeo: Failed to initialize RtxMegaGeoBuilder");
      m_megaGeoBuilder = nullptr;
      return false;
    }
    return true;
  }

  void AccelManager::buildTlas(Rc<DxvkContext> ctx) {
    Logger::warn("CHECKPOINT: buildTlas-enter");
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas entering");
    if (m_vkInstanceBuffer == nullptr) {
      RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - no instance buffer, returning");
      return;
    }
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - have instance buffer");

    ScopedGpuProfileZone(ctx, "buildTLAS");

    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - emitting memory barrier");
    // Two barriers in one:
    // Accel build bit - to protect from BLAS builds
    // Transfer bit - to protect from updateBuffer in prepareSceneData
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - memory barrier done");

    Logger::warn(str::format("CHECKPOINT: buildTlas-trackBlas pool=", m_blasPool.size()));
    for (auto&& blas : m_blasPool) {
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(blas->accelStructure);
    }
    Logger::warn("CHECKPOINT: buildTlas-trackDone");

    size_t totalScratchSize = 0;
    Logger::warn(str::format("CHECKPOINT: buildTlas-opaque instances=", m_mergedInstances[Tlas::Opaque].size()));
    internalBuildTlas<Tlas::Opaque>(ctx, totalScratchSize);
    Logger::warn(str::format("CHECKPOINT: buildTlas-unordered instances=", m_mergedInstances[Tlas::Unordered].size()));
    internalBuildTlas<Tlas::Unordered>(ctx, totalScratchSize);
    Logger::warn("CHECKPOINT: buildTlas-complete");

    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - emitting final barrier");
    // Include COMPUTE_SHADER as source: cluster compute shaders wrote to clusterShadingData,
    // clusterVertexPositions, clusterVertexNormals buffers during BuildAccel.
    // Include SHADER_READ as dest access: ray tracing closest-hit shader reads these buffers.
    // Without this, the ray tracing shader may read stale/partial cluster data → GPU crash.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - final barrier emitted");

    // Release the scratch memory so it can be reused by rest of the frame.
    m_scratchBuffer = nullptr;
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - scratch buffer released");

    OpacityMicromapManager* opacityMicromapManager = ctx->getCommonObjects()->getSceneManager().getOpacityMicromapManager();
    if (opacityMicromapManager) {
      opacityMicromapManager->onFinishedBuilding();
    }
    RTXMG_LOG("RTX MegaGeo AccelManager: buildTlas - COMPLETE, returning from buildTlas");
  }

  template<Tlas::Type type>
  void AccelManager::internalBuildTlas(Rc<DxvkContext> ctx, size_t& totalScratchSize) {
    static constexpr const char* names[] = { "TLAS_Opaque", "TLAS_NonOpaque" };
    ScopedGpuProfileZone(ctx, names[type]);
    const VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();

    const auto& vkd = m_device->vkd();

    // Create VkAccelerationStructureGeometryInstancesDataKHR
    // This wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instancesVk.arrayOfPointers = VK_FALSE;
    instancesVk.data.deviceAddress = m_vkInstanceBuffer->getDeviceAddress();

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
    Logger::warn(str::format("CHECKPOINT: internalBuildTlas type=", type, " numInstances=", numInstances));
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkd->vkGetAccelerationStructureBuildSizesKHR(vkd->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &numInstances, &sizeInfo);
    Logger::warn(str::format("CHECKPOINT: internalBuildTlas getSizes done asSize=", sizeInfo.accelerationStructureSize, " scratchSize=", sizeInfo.buildScratchSize));

    // Create TLAS
    Tlas& tlas = m_device->getCommon()->getResources().getTLAS(type);

    if (type == Tlas::Opaque) {
      std::swap(tlas.accelStructure, tlas.previousAccelStructure);
    }

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
    Logger::warn(str::format("CHECKPOINT: internalBuildTlas scratch totalSoFar=", totalScratchSize, " required=", requiredScratchAllocSize));
    auto scratchMem = getScratchMemory(totalScratchSize + requiredScratchAllocSize);
    Logger::warn(str::format("CHECKPOINT: internalBuildTlas scratch allocated=", (scratchMem != nullptr)));
    buildInfo.scratchData.deviceAddress = scratchMem->getDeviceAddress() + totalScratchSize;
    totalScratchSize += requiredScratchAllocSize;

    // Update build information
    buildInfo.srcAccelerationStructure = nullptr;
    buildInfo.dstAccelerationStructure = tlas.accelStructure->getAccelStructure();

    assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo { numInstances, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    Logger::warn(str::format("CHECKPOINT: internalBuildTlas pre-vkBuild dstAS=", (buildInfo.dstAccelerationStructure != VK_NULL_HANDLE)));
    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildOffsetInfo);
    Logger::warn("CHECKPOINT: internalBuildTlas post-vkBuild");

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(tlas.accelStructure);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);
    Logger::warn("CHECKPOINT: internalBuildTlas DONE");
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
