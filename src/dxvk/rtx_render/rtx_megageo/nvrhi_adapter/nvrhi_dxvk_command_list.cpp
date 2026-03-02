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
#include "nvrhi_dxvk_command_list.h"
#include "nvrhi_dxvk_buffer.h"
#include "nvrhi_dxvk_texture.h"
#include "nvrhi_dxvk_pipeline.h"
#include "nvrhi_dxvk_sampler.h"
#include "../../../util/log/log.h"
#include "../../rtx_context.h"  // For RtxContext::commitComputeStateForMegaGeo
#include <algorithm>
#include <array>

#include "../rtxmg_log.h"
#if RTXMG_LOG_NVRHI_DXVK_COMMAND_LIST
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace dxvk {

  void NvrhiDxvkCommandList::open() {
    // DXVK contexts are always "open" - no-op
  }

  void NvrhiDxvkCommandList::close() {
    // Flush command list
    m_context->flushCommandList();
  }

  void NvrhiDxvkCommandList::clearState() {
    RTXMG_LOG("RTX MegaGeo: clearState - resetting compute state");
    m_computeState = nvrhi::ComputeState();
    // Reset scratch buffer write pointers at start of frame
    // DXVK ensures GPU is done with previous frame's work
    RTXMG_LOG("RTX MegaGeo: clearState - calling resetWritePointers");
    m_scratchManager->resetWritePointers();

    // Clear HiZ binding state
    m_hasHiZBinding = false;
    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i) {
      m_hiZImageViews[i] = nullptr;
    }
    // Note: Keep m_hiZDescriptorSetLayout and m_hiZPipelineLayout cached for reuse

    // Clear UAV array binding state
    m_hasUAVArrayBinding = false;
    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i) {
      m_uavImageViews[i] = nullptr;
    }

    // Clear pending pre-built descriptor set bindings (these were never dispatched)
    m_pendingDescriptorSets.clear();

    // Note: m_pendingBindingSetsForTracking is NOT cleared here.
    // Binding sets are transferred to DXVK's command list lifetime tracker during
    // dispatch(), and DXVK frees them when the GPU fence signals command completion.

    // Clear automatic barrier tracking state
    m_BufferStates.clear();
    m_PendingBufferBarriers.clear();

    RTXMG_LOG("RTX MegaGeo: clearState - done");
  }

  void NvrhiDxvkCommandList::writeBuffer(
    nvrhi::IBuffer* buffer,
    const void* data,
    size_t size,
    uint64_t offset)
  {
    if (!buffer) {
      Logger::err("RTX MegaGeo: writeBuffer - null buffer");
      return;
    }

    // Verify this is an NvrhiDxvkBuffer before casting
    nvrhi::Object nativeObj = buffer->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    if (nativeObj.pointer == nullptr) {
      Logger::err("RTX MegaGeo: writeBuffer - buffer is not a VK_Buffer (incompatible type), skipping");
      return;
    }

    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
    if (dxvkBuffer == nullptr) {
      Logger::err("RTX MegaGeo: writeBuffer - null DxvkBuffer");
      return;
    }

    size_t bufferSize = nvrhiBuffer->getDesc().byteSize;
    const char* bufferName = nvrhiBuffer->getDesc().debugName ? nvrhiBuffer->getDesc().debugName : "unnamed";

    // Reduced logging - only log buffer name and key sizes
    RTXMG_LOG(str::format("RTX MegaGeo: writeBuffer - buffer='", bufferName,
      "' size=", size, " bufferSize=", bufferSize));

    // CRITICAL: Check for buffer overrun BEFORE any write
    if (offset + size > bufferSize) {
      Logger::err(str::format("RTX MegaGeo: BUFFER OVERRUN DETECTED! buffer='", bufferName,
        "' trying to write ", size, " bytes at offset ", offset,
        " but buffer is only ", bufferSize, " bytes!"));
      // Don't write past buffer end - this causes heap corruption
      size = (offset < bufferSize) ? (bufferSize - offset) : 0;
      if (size == 0) return;
    }

    // Volatile constant buffers: memcpy to host-visible ring buffer (no GPU commands)
    if (nvrhiBuffer->isVolatile()) {
      nvrhiBuffer->writeVolatile(data, size);
      return;
    }

    // Cache data for small constant buffers (used for push constants)
    if (offset == 0 && size <= 128) {
      nvrhiBuffer->setCachedData(data, size);
    }

    // Automatic barrier for non-volatile buffers
    if (m_EnableAutomaticBarriers) {
      requireBufferState(buffer, nvrhi::ResourceStates::CopyDest);
      commitBarriers();
    }

    // =================================================================================
    // BYPASSING DXVK's updateBuffer - calling vkCmdUpdateBuffer directly
    // =================================================================================
    // Same fix as clearBufferUInt: we bypass DXVK's m_context->updateBuffer() because it
    // adds DXVK's own barrier tracking (m_execBarriers at TRANSFER stage) on TOP of our
    // NVRHI m_BufferStates tracking. That dual tracking causes DXVK to later emit spurious
    // barriers that can reorder or invalidate our explicit barrier sequence.
    //
    // Additionally, DXVK's updateBuffer calls tryInvalidateDeviceLocalBuffer() which can
    // REPLACE the VkBuffer allocation, changing its device address mid-operation.
    //
    // By calling vkCmdUpdateBuffer directly through the DxvkCommandList, we:
    //   1. Still get the actual GPU write (vkCmdUpdateBuffer)
    //   2. Still track the resource lifetime (trackResource for fence-based destruction)
    //   3. Do NOT pollute DXVK's m_execBarriers with TRANSFER state
    //   4. Do NOT trigger tryInvalidateDeviceLocalBuffer (no buffer replacement)
    //   5. Our NVRHI m_BufferStates is the SOLE barrier tracker for this buffer
    // =================================================================================
    auto slice = dxvkBuffer->getSliceHandle(offset, size);
    Rc<DxvkCommandList> cmdList = m_context->getCommandList();

    // vkCmdUpdateBuffer has a 64KB limit per call. Chunk large writes.
    static constexpr size_t kMaxUpdateSize = 65536;
    const uint8_t* src = static_cast<const uint8_t*>(data);
    size_t remaining = size;
    VkDeviceSize curOffset = slice.offset;  // Use the slice's VkBuffer offset

    while (remaining > 0) {
      size_t chunkSize = std::min(remaining, kMaxUpdateSize);
      // Vulkan requires vkCmdUpdateBuffer size to be a multiple of 4
      size_t alignedChunk = (chunkSize + 3) & ~3;

      // Clamp if rounding up would overflow buffer
      if ((curOffset - slice.offset) + alignedChunk > size) {
        alignedChunk = chunkSize & ~size_t(3);
        if (alignedChunk == 0) break;
        chunkSize = alignedChunk;
      }

      if (alignedChunk != chunkSize) {
        std::vector<uint8_t> alignedData(alignedChunk, 0);
        std::memcpy(alignedData.data(), src, chunkSize);
        cmdList->cmdUpdateBuffer(DxvkCmdBuffer::ExecBuffer, slice.handle, curOffset, alignedChunk, alignedData.data());
      } else {
        cmdList->cmdUpdateBuffer(DxvkCmdBuffer::ExecBuffer, slice.handle, curOffset, chunkSize, src);
      }

      src += chunkSize;
      curOffset += chunkSize;
      remaining -= chunkSize;
    }

    cmdList->trackResource<DxvkAccess::Write>(dxvkBuffer);
  }

  void NvrhiDxvkCommandList::clearBufferUInt(
    nvrhi::IBuffer* buffer,
    uint32_t value)
  {
    if (!buffer) {
      Logger::err("RTX MegaGeo: clearBufferUInt - null buffer");
      return;
    }

    nvrhi::Object nativeObj = buffer->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    if (nativeObj.pointer == nullptr) {
      Logger::err("RTX MegaGeo: clearBufferUInt - buffer is not a VK_Buffer (incompatible type), skipping");
      return;
    }

    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
    if (dxvkBuffer == nullptr) {
      Logger::err("RTX MegaGeo: clearBufferUInt - null DxvkBuffer");
      return;
    }

    // Automatic barrier: transition buffer to CopyDest (TRANSFER stage) before clearing.
    // This updates our NVRHI m_BufferStates tracking ONLY.
    if (m_EnableAutomaticBarriers) {
      requireBufferState(buffer, nvrhi::ResourceStates::CopyDest);
      commitBarriers();
    }

    // =================================================================================
    // BYPASSING DXVK's clearBuffer - calling vkCmdFillBuffer directly
    // =================================================================================
    // We intentionally do NOT call m_context->clearBuffer() here because it would add
    // DXVK's own barrier tracking (m_execBarriers.accessBuffer at TRANSFER stage) on TOP
    // of our NVRHI m_BufferStates tracking above. That dual tracking was the root cause of
    // GPU hangs after instance buffer resize:
    //
    //   - Our NVRHI tracking: buffer is at CopyDest (TRANSFER) after this clear
    //   - DXVK tracking:      buffer is at TRANSFER (independently tracked)
    //   - Later barriers from our NVRHI adapter wouldn't clear DXVK's stale state
    //   - DXVK would later emit a spurious TRANSFER→X barrier at an unexpected time
    //
    // By calling vkCmdFillBuffer directly through the DxvkCommandList, we:
    //   1. Still get the actual GPU clear operation (vkCmdFillBuffer)
    //   2. Still track the resource lifetime (trackResource for fence-based destruction)
    //   3. Do NOT pollute DXVK's m_execBarriers with TRANSFER state
    //   4. Our NVRHI m_BufferStates is the SOLE barrier tracker for this buffer
    //
    // This eliminates the need for the heavy barrier workaround for the dual-tracking issue,
    // though the heavy barrier may still be needed for other synchronization gaps.
    // =================================================================================
    const nvrhi::BufferDesc& desc = nvrhiBuffer->getDesc();
    auto slice = dxvkBuffer->getSliceHandle(0, desc.byteSize);
    // vkCmdFillBuffer requires size aligned to 4 bytes
    VkDeviceSize fillLength = (slice.length + 3) & ~VkDeviceSize(3);
    Rc<DxvkCommandList> cmdList = m_context->getCommandList();
    cmdList->cmdFillBuffer(slice.handle, slice.offset, fillLength, value);
    cmdList->trackResource<DxvkAccess::Write>(dxvkBuffer);
  }

  void NvrhiDxvkCommandList::copyBuffer(
    nvrhi::IBuffer* dst,
    uint64_t dstOffset,
    nvrhi::IBuffer* src,
    uint64_t srcOffset,
    uint64_t size)
  {
    if (!dst || !src) {
      Logger::err("RTX MegaGeo: copyBuffer - null buffer");
      return;
    }

    nvrhi::Object dstNative = dst->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    nvrhi::Object srcNative = src->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    if (dstNative.pointer == nullptr || srcNative.pointer == nullptr) {
      Logger::err("RTX MegaGeo: copyBuffer - buffer is not a VK_Buffer (incompatible type), skipping");
      return;
    }

    NvrhiDxvkBuffer* nvrhiDst = static_cast<NvrhiDxvkBuffer*>(dst);
    NvrhiDxvkBuffer* nvrhiSrc = static_cast<NvrhiDxvkBuffer*>(src);

    if (nvrhiDst->getDxvkBuffer() == nullptr || nvrhiSrc->getDxvkBuffer() == nullptr) {
      Logger::err("RTX MegaGeo: copyBuffer - null DxvkBuffer");
      return;
    }

    size_t dstSize = nvrhiDst->getDesc().byteSize;
    size_t srcSize = nvrhiSrc->getDesc().byteSize;
    const char* dstName = nvrhiDst->getDesc().debugName ? nvrhiDst->getDesc().debugName : "unnamed";
    const char* srcName = nvrhiSrc->getDesc().debugName ? nvrhiSrc->getDesc().debugName : "unnamed";

    // Check for buffer overruns
    if (dstOffset + size > dstSize) {
      Logger::err(str::format("RTX MegaGeo: copyBuffer DST OVERRUN! dst='", dstName,
        "' dstOffset=", dstOffset, " size=", size, " dstSize=", dstSize));
    }
    if (srcOffset + size > srcSize) {
      Logger::err(str::format("RTX MegaGeo: copyBuffer SRC OVERRUN! src='", srcName,
        "' srcOffset=", srcOffset, " size=", size, " srcSize=", srcSize));
    }

    Rc<DxvkBuffer> dxvkDst = nvrhiDst->getDxvkBuffer();
    Rc<DxvkBuffer> dxvkSrc = nvrhiSrc->getDxvkBuffer();

    // DIAG: Log VkBuffer handles for source buffer to compare with descriptor set binding
    {
      static uint32_t s_copyDiagCount = 0;
      if (s_copyDiagCount < 20 && srcName && strstr(srcName, "vertex pos")) {
        DxvkBufferSliceHandle srcSlice = dxvkSrc->getSliceHandle();
        Logger::warn(str::format("DIAG COPYBUF: src='", srcName,
          "' nvrhiBuf=", (void*)src,
          " VkBuffer=0x", std::hex, (uint64_t)srcSlice.handle, std::dec,
          " sliceOffset=", srcSlice.offset,
          " deviceAddr=0x", std::hex, dxvkSrc->getDeviceAddress(), std::dec,
          " copyOffset=", srcOffset, " copySize=", size));
        s_copyDiagCount++;
      }
    }

    // When the special dispatch path is used (pre-built descriptor sets / raw Vulkan dispatch),
    // DXVK doesn't track compute shader UAV writes. We must insert an explicit barrier
    // to ensure compute writes are visible before the transfer read.
    // Use the NVRHI adapter's own state tracking (m_BufferStates) to detect when a buffer
    // transitions from UAV (compute write) to CopySource (transfer read).
    if (m_EnableAutomaticBarriers) {
      requireBufferState(src, nvrhi::ResourceStates::CopySource);
      requireBufferState(dst, nvrhi::ResourceStates::CopyDest);
      commitBarriers();
    }

    m_context->copyBuffer(dxvkDst, dstOffset, dxvkSrc, srcOffset, size);
  }

  void NvrhiDxvkCommandList::copyTexture(nvrhi::ITexture* dst, nvrhi::ITexture* src) {
    if (!dst || !src) {
      Logger::warn("RTX MegaGeo: copyTexture - null texture");
      return;
    }

    // CRITICAL: Verify textures are NvrhiDxvkTextures before casting
    nvrhi::Object dstNative = dst->getNativeObject(nvrhi::ObjectType::VK_Image);
    nvrhi::Object srcNative = src->getNativeObject(nvrhi::ObjectType::VK_Image);
    if (dstNative.pointer == nullptr || srcNative.pointer == nullptr) {
      Logger::warn("RTX MegaGeo: copyTexture - texture is not a VK_Image (incompatible type), skipping");
      return;
    }

    NvrhiDxvkTexture* dstTex = static_cast<NvrhiDxvkTexture*>(dst);
    NvrhiDxvkTexture* srcTex = static_cast<NvrhiDxvkTexture*>(src);

    const Rc<DxvkImage>& dstImage = dstTex->getDxvkImage();
    const Rc<DxvkImage>& srcImage = srcTex->getDxvkImage();
    if (dstImage == nullptr || srcImage == nullptr) {
      Logger::warn("RTX MegaGeo: copyTexture - null DxvkImage, skipping");
      return;
    }

    const auto& dstDesc = dstTex->getDesc();
    const auto& srcDesc = srcTex->getDesc();

    // Copy entire texture
    VkImageSubresourceLayers subresource;
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount = 1;

    VkExtent3D extent = { srcDesc.width, srcDesc.height, srcDesc.depth };

    m_context->copyImage(
      dstTex->getDxvkImage(), subresource, VkOffset3D{ 0, 0, 0 },
      srcTex->getDxvkImage(), subresource, VkOffset3D{ 0, 0, 0 },
      extent);
  }

  void NvrhiDxvkCommandList::clearTextureFloat(
    nvrhi::ITexture* texture,
    const nvrhi::TextureSubresourceSet& subresources,
    const nvrhi::Color& clearColor)
  {
    if (!texture) {
      RTXMG_LOG("RTX MegaGeo: clearTextureFloat - null texture");
      return;
    }

    // CRITICAL: Verify this is actually an NvrhiDxvkTexture before casting
    // Textures from external sources (like zbuffer->GetHierarchyTexture) may be
    // different types, and static_cast would reinterpret memory incorrectly
    nvrhi::Object nativeObj = texture->getNativeObject(nvrhi::ObjectType::VK_Image);
    if (nativeObj.pointer == nullptr) {
      Logger::warn("RTX MegaGeo: clearTextureFloat - texture is not a VK_Image (incompatible type), skipping");
      return;
    }

    NvrhiDxvkTexture* tex = static_cast<NvrhiDxvkTexture*>(texture);

    const Rc<DxvkImage>& dxvkImage = tex->getDxvkImage();
    if (dxvkImage == nullptr) {
      RTXMG_LOG("RTX MegaGeo: clearTextureFloat - null DxvkImage");
      return;
    }

    VkImageSubresourceRange range;
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = subresources.baseMipLevel;
    range.levelCount = (subresources.numMipLevels == nvrhi::TextureSubresourceSet::AllMipLevels) ?
                       VK_REMAINING_MIP_LEVELS : subresources.numMipLevels;
    range.baseArrayLayer = subresources.baseArraySlice;
    range.layerCount = (subresources.numArraySlices == nvrhi::TextureSubresourceSet::AllArraySlices) ?
                       VK_REMAINING_ARRAY_LAYERS : subresources.numArraySlices;

    // First transition image to GENERAL layout (from UNDEFINED if needed)
    // This is required before vkCmdClearColorImage
    m_context->transformImage(
      dxvkImage,
      range,
      VK_IMAGE_LAYOUT_UNDEFINED,  // oldLayout - assume undefined on first clear
      VK_IMAGE_LAYOUT_GENERAL);   // newLayout - required for clear

    VkClearColorValue vkClearColor;
    vkClearColor.float32[0] = clearColor.r;
    vkClearColor.float32[1] = clearColor.g;
    vkClearColor.float32[2] = clearColor.b;
    vkClearColor.float32[3] = clearColor.a;

    m_context->clearColorImage(
      dxvkImage,
      vkClearColor,
      range);
  }

  void NvrhiDxvkCommandList::setComputeState(const nvrhi::ComputeState& state) {
    // Check if binding sets actually changed - skip redundant barrier/bind work
    bool bindingSetsChanged = false;
    if (state.bindingSets.size() != m_computeState.bindingSets.size()) {
      bindingSetsChanged = true;
    } else {
      for (size_t i = 0; i < state.bindingSets.size(); i++) {
        if (state.bindingSets[i].Get() != m_computeState.bindingSets[i].Get()) {
          bindingSetsChanged = true;
          break;
        }
      }
    }

    bool pipelineChanged = state.pipeline != m_computeState.pipeline;

    // Only issue barriers when binding sets change
    if (bindingSetsChanged && m_EnableAutomaticBarriers) {
      for (const auto& bindingSet : state.bindingSets) {
        setResourceStatesForBindingSet(bindingSet.Get());
      }

      if (state.indirectParamsBuffer) {
        requireBufferState(state.indirectParamsBuffer.Get(), nvrhi::ResourceStates::IndirectArgument);
      }

      commitBarriers();
    }

    m_computeState = state;

    // Only rebind resources when something actually changed
    if (bindingSetsChanged || pipelineChanged) {
      bindComputeResources(state);
    }
  }

  void NvrhiDxvkCommandList::bindHiZDescriptorSet(VkPipelineLayout pipelineLayout) {
    if (!m_hasHiZBinding) {
      return;
    }

    VkDevice vkDevice = m_device->getVkDevice();
    if (vkDevice == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: bindHiZDescriptorSet - null VkDevice");
      return;
    }

    // Get the compute descriptor set that DXVK allocated for set 0
    // After commitComputeState(), this contains all the bound resources
    VkDescriptorSet computeSet = m_context->getComputeDescriptorSet();
    if (computeSet == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: bindHiZDescriptorSet - null compute descriptor set");
      return;
    }

    // HiZ textures are at slot 17, which maps to binding 18 in set 0
    // (CB at binding 0, SRVs 0-16 at bindings 1-17, slot 17 at binding 18)
    // The descriptor layout has count=9 for this binding (array of 9 textures)
    // NOTE: HiZ textures are UAVs that stay in GENERAL layout (written by reduce pass),
    // so we must use GENERAL layout when reading them, not SHADER_READ_ONLY_OPTIMAL.
    std::array<VkDescriptorImageInfo, HIZ_MAX_LODS> imageInfos;

    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i) {
      imageInfos[i] = {};
      imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      if (m_hiZImageViews[i] != nullptr) {
        imageInfos[i].imageView = m_hiZImageViews[i]->handle();
      } else {
        // Use first valid view as fallback for missing levels
        for (uint32_t j = 0; j < HIZ_MAX_LODS; ++j) {
          if (m_hiZImageViews[j] != nullptr) {
            imageInfos[i].imageView = m_hiZImageViews[j]->handle();
            break;
          }
        }
      }
    }

    // Update binding 18 in set 0 with the HiZ texture array
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = computeSet;
    write.dstBinding = 18;  // Binding 18 in set 0 (slot 17 remapped)
    write.dstArrayElement = 0;
    write.descriptorCount = HIZ_MAX_LODS;  // Write all 9 at once
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = imageInfos.data();

    m_device->getDxvkDevice()->vkd()->vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);

    RTXMG_LOG("RTX MegaGeo: Updated HiZ binding 18 in set 0 with 9 textures");

    // Reset HiZ binding state
    m_hasHiZBinding = false;
  }

  void NvrhiDxvkCommandList::bindUAVArrayDescriptorSet() {
    if (!m_hasUAVArrayBinding) {
      return;
    }

    VkDevice vkDevice = m_device->getVkDevice();
    if (vkDevice == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: bindUAVArrayDescriptorSet - null VkDevice");
      return;
    }

    // Get the compute descriptor set that DXVK allocated for set 0
    VkDescriptorSet computeSet = m_context->getComputeDescriptorSet();
    if (computeSet == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: bindUAVArrayDescriptorSet - null compute descriptor set");
      return;
    }

    // UAV array at u0 (slot 200) maps to binding 3 in set 0 for HiZ reduce shaders
    // Layout: binding 0=t0 (zbuffer), 1=t1 (params), 2=s0 (sampler), 3=u0 (UAV array)
    std::array<VkDescriptorImageInfo, HIZ_MAX_LODS> imageInfos;

    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i) {
      imageInfos[i] = {};
      imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // UAVs use GENERAL layout
      if (m_uavImageViews[i] != nullptr) {
        imageInfos[i].imageView = m_uavImageViews[i]->handle();
      } else {
        // Use first valid view as fallback for missing levels
        for (uint32_t j = 0; j < HIZ_MAX_LODS; ++j) {
          if (m_uavImageViews[j] != nullptr) {
            imageInfos[i].imageView = m_uavImageViews[j]->handle();
            break;
          }
        }
      }
    }

    // Update binding 3 in set 0 with the UAV texture array
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = computeSet;
    write.dstBinding = 3;  // Binding 3 in set 0 for UAV array (u0 at slot 200)
    write.dstArrayElement = 0;
    write.descriptorCount = HIZ_MAX_LODS;  // Write all 9 at once
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = imageInfos.data();

    m_device->getDxvkDevice()->vkd()->vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);

    RTXMG_LOG("RTX MegaGeo: Updated UAV binding 3 in set 0 with 9 storage images");

    // Reset UAV binding state
    m_hasUAVArrayBinding = false;
  }

  void NvrhiDxvkCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    // Check if we have pre-built descriptor sets with their own pipeline layout
    // In this case, we use native Vulkan calls instead of DXVK's pipeline
    bool hasNativePipeline = false;
    VkPipeline nativePipeline = VK_NULL_HANDLE;
    VkPipelineLayout nativePipelineLayout = VK_NULL_HANDLE;

    if (!m_pendingDescriptorSets.empty() && m_pendingDescriptorSets[0].pipelineLayout != VK_NULL_HANDLE) {
      // Get the native pipeline from the compute state
      if (m_computeState.pipeline) {
        auto* computePipeline = static_cast<NvrhiDxvkComputePipeline*>(m_computeState.pipeline.Get());
        if (computePipeline && computePipeline->hasVkPipeline()) {
          hasNativePipeline = true;
          nativePipeline = computePipeline->getVkPipeline();
          nativePipelineLayout = computePipeline->getPipelineLayout();
        }
      }
    }

    // For shaders with HiZ/UAV array bindings or pre-built descriptor sets, we need special handling.
    if (m_hasHiZBinding || m_hasUAVArrayBinding || !m_pendingDescriptorSets.empty()) {
      VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

      // Diagnostic: log dispatch state when we have multiple pending descriptor sets (tiling shader)
    {
      static uint32_t s_tilingDispatchLogCount = 0;
      // Tiling shader has 2 pending sets (main + HiZ), other shaders have 1
      bool isTilingDispatch = m_pendingDescriptorSets.size() >= 2;
      // Also log first dispatch of any kind
      static bool s_loggedFirstDispatch = false;
      if (!s_loggedFirstDispatch || (isTilingDispatch && s_tilingDispatchLogCount < 3)) {
        if (isTilingDispatch) s_tilingDispatchLogCount++;
        s_loggedFirstDispatch = true;
        Logger::warn(str::format("DISPATCH_DBG: hasNativePipeline=", hasNativePipeline ? "YES" : "NO",
          " nativePipeline=", (void*)nativePipeline,
          " nativePipelineLayout=", (void*)nativePipelineLayout,
          " pendingDescSets=", m_pendingDescriptorSets.size(),
          " dispatch(", x, ",", y, ",", z, ")",
          isTilingDispatch ? " [TILING]" : ""));
        for (size_t di = 0; di < m_pendingDescriptorSets.size(); ++di) {
          const auto& p = m_pendingDescriptorSets[di];
          Logger::warn(str::format("DISPATCH_DBG:   pendingSet[", di, "] descSet=", (void*)p.descriptorSet,
            " setIndex=", p.setIndex,
            " pipelineLayout=", (void*)p.pipelineLayout));
        }
      }
    }

    if (hasNativePipeline) {
        // Use native Vulkan pipeline - completely bypass DXVK's pipeline
        m_device->getDxvkDevice()->vkd()->vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, nativePipeline);
        RTXMG_LOG(str::format("RTX MegaGeo: bound native VkPipeline ", (void*)nativePipeline));
      } else {
        // Commit DXVK compute state - this binds the pipeline and set 0 resources
        if (!m_pendingDescriptorSets.empty()) {
          Logger::err("RTX MegaGeo: CRITICAL WARNING - DXVK fallback path with pre-built descriptor sets! "
            "Descriptor set layout will NOT match DXVK's pipeline layout. Native VkPipeline creation may have failed.");
        }
        RtxContext* rtxContext = static_cast<RtxContext*>(m_context.ptr());
        if (!rtxContext->commitComputeStateForMegaGeo()) {
          Logger::err("RTX MegaGeo: Failed to commit compute state for MegaGeo dispatch");
          return;
        }
      }

      // Bind pre-built descriptor sets with vkCmdBindDescriptorSets
      // Also handle push constants for constant buffers
      if (!m_pendingDescriptorSets.empty()) {
        for (const auto& pending : m_pendingDescriptorSets) {
          VkPipelineLayout layoutToUse = pending.pipelineLayout != VK_NULL_HANDLE
            ? pending.pipelineLayout
            : (hasNativePipeline ? nativePipelineLayout : m_context->getComputePipelineLayout());

          RTXMG_LOG(str::format("RTX MegaGeo: binding pre-built descriptor set at index ", pending.setIndex,
            " with pipelineLayout=", (void*)layoutToUse));

          m_device->getDxvkDevice()->vkd()->vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            layoutToUse,
            pending.setIndex,
            1,
            &pending.descriptorSet,
            0,
            nullptr);

          // Push constants for constant buffers in this binding set
          // The shader may have been compiled with push constants for small CBs
          if (pending.bindingSetRef) {
            auto* nvrhiBindingSet = static_cast<NvrhiDxvkBindingSet*>(pending.bindingSetRef.Get());
            if (nvrhiBindingSet) {
              const auto& desc = nvrhiBindingSet->getDesc();
              for (const auto& item : desc.bindings) {
                if (item.type == nvrhi::BindingSetItem::Type::ConstantBuffer && item.resourceHandle) {
                  auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
                  if (nvrhiBuffer->hasCachedData()) {
                    uint32_t pushSize = static_cast<uint32_t>(nvrhiBuffer->getCachedDataSize());
                    RTXMG_LOG(str::format("RTX MegaGeo: vkCmdPushConstants size=", pushSize, " (b", item.slot, ")"));
                    m_device->getDxvkDevice()->vkd()->vkCmdPushConstants(
                      cmdBuffer,
                      layoutToUse,
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0,
                      pushSize,
                      nvrhiBuffer->getCachedData());
                  }
                }
              }
            }
          }

          // Keep the binding set alive until command buffer execution completes
          m_pendingBindingSetsForTracking.push_back(pending.bindingSetRef);
        }
        m_pendingDescriptorSets.clear();
      }

      // Update HiZ texture array at binding 18 in set 0 (if bound)
      if (m_hasHiZBinding) {
        VkPipelineLayout pipelineLayout = hasNativePipeline ? nativePipelineLayout : m_context->getComputePipelineLayout();
        bindHiZDescriptorSet(pipelineLayout);
      }

      // Update UAV texture array at binding 3 in set 0 (if bound)
      if (m_hasUAVArrayBinding) {
        bindUAVArrayDescriptorSet();
      }

      // Commit any pending barriers right before dispatch.
      // 1) DXVK barrier flush: updateBuffer() tracks writes in DXVK's m_execBarriers,
      //    but the native Vulkan dispatch path bypasses DXVK's dispatch() which would normally
      //    flush them. Emit a full memory barrier to make all prior writes visible to compute.
      m_context->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT);
      // 2) Flush NVRHI-tracked barriers (m_PendingBufferBarriers) for bufferBarrier() calls.
      commitBarriers();

      // Call vkCmdDispatch directly
      m_device->getDxvkDevice()->vkd()->vkCmdDispatch(cmdBuffer, x, y, z);
      RTXMG_LOG(str::format("RTX MegaGeo: dispatch(", x, ", ", y, ", ", z, ") with special bindings"));
    } else {
      // Normal dispatch path through DXVK
      // Automatic barriers already committed in setComputeState
      m_context->dispatch(x, y, z);
    }

    // Transfer pending binding sets to DXVK's command list lifetime tracker.
    // This ensures descriptor sets stay alive until the GPU finishes the command buffer.
    trackPendingBindingSets();
  }

  void NvrhiDxvkCommandList::dispatchIndirect(nvrhi::IBuffer* buffer, uint64_t offset) {
    if (!buffer) {
      Logger::err("RTX MegaGeo: dispatchIndirect - null buffer");
      return;
    }

    nvrhi::Object nativeObj = buffer->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    if (nativeObj.pointer == nullptr) {
      Logger::err("RTX MegaGeo: dispatchIndirect - buffer is not a VK_Buffer (incompatible type), skipping");
      return;
    }

    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
    if (dxvkBuffer == nullptr) {
      Logger::err("RTX MegaGeo: dispatchIndirect - null DxvkBuffer");
      return;
    }

    // Bind indirect argument buffer
    DxvkBufferSlice argBufferSlice(dxvkBuffer, offset, sizeof(VkDispatchIndirectCommand));
    m_context->bindDrawBuffers(argBufferSlice, DxvkBufferSlice());

    // Check if we have pre-built descriptor sets with their own pipeline layout
    bool hasNativePipeline = false;
    VkPipeline nativePipeline = VK_NULL_HANDLE;
    VkPipelineLayout nativePipelineLayout = VK_NULL_HANDLE;

    if (!m_pendingDescriptorSets.empty() && m_pendingDescriptorSets[0].pipelineLayout != VK_NULL_HANDLE) {
      if (m_computeState.pipeline) {
        auto* computePipeline = static_cast<NvrhiDxvkComputePipeline*>(m_computeState.pipeline.Get());
        if (computePipeline && computePipeline->hasVkPipeline()) {
          hasNativePipeline = true;
          nativePipeline = computePipeline->getVkPipeline();
          nativePipelineLayout = computePipeline->getPipelineLayout();
        }
      }
    }

    // For shaders with HiZ/UAV array bindings or pre-built descriptor sets, we need special handling
    if (m_hasHiZBinding || m_hasUAVArrayBinding || !m_pendingDescriptorSets.empty()) {
      VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

      if (hasNativePipeline) {
        // Use native Vulkan pipeline - completely bypass DXVK's pipeline
        m_device->getDxvkDevice()->vkd()->vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, nativePipeline);
        RTXMG_LOG(str::format("RTX MegaGeo: bound native VkPipeline for dispatchIndirect ", (void*)nativePipeline));
      } else {
        RtxContext* rtxContext = static_cast<RtxContext*>(m_context.ptr());
        if (!rtxContext->commitComputeStateForMegaGeo()) {
          Logger::err("RTX MegaGeo: Failed to commit compute state for MegaGeo dispatchIndirect");
          return;
        }
      }

      // Bind pre-built descriptor sets with vkCmdBindDescriptorSets
      // Also handle push constants for constant buffers (matching dispatch() behavior)
      if (!m_pendingDescriptorSets.empty()) {
        for (const auto& pending : m_pendingDescriptorSets) {
          VkPipelineLayout layoutToUse = pending.pipelineLayout != VK_NULL_HANDLE
            ? pending.pipelineLayout
            : (hasNativePipeline ? nativePipelineLayout : m_context->getComputePipelineLayout());

          RTXMG_LOG(str::format("RTX MegaGeo: binding pre-built descriptor set at index ", pending.setIndex,
            " with pipelineLayout=", (void*)layoutToUse));

          // Always log VkDescriptorSet handle being bound (dispatchIndirect path)
          m_device->getDxvkDevice()->vkd()->vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            layoutToUse,
            pending.setIndex,
            1,
            &pending.descriptorSet,
            0,
            nullptr);

          // Push constants for constant buffers in this binding set
          if (pending.bindingSetRef) {
            auto* nvrhiBindingSet = static_cast<NvrhiDxvkBindingSet*>(pending.bindingSetRef.Get());
            if (nvrhiBindingSet) {
              const auto& desc = nvrhiBindingSet->getDesc();
              for (const auto& item : desc.bindings) {
                if (item.type == nvrhi::BindingSetItem::Type::ConstantBuffer && item.resourceHandle) {
                  auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
                  if (nvrhiBuffer->hasCachedData()) {
                    uint32_t pushSize = static_cast<uint32_t>(nvrhiBuffer->getCachedDataSize());
                    RTXMG_LOG(str::format("RTX MegaGeo: dispatchIndirect vkCmdPushConstants size=", pushSize, " (b", item.slot, ")"));
                    m_device->getDxvkDevice()->vkd()->vkCmdPushConstants(
                      cmdBuffer,
                      layoutToUse,
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0,
                      pushSize,
                      nvrhiBuffer->getCachedData());
                  }
                }
              }
            }
          }

          m_pendingBindingSetsForTracking.push_back(pending.bindingSetRef);
        }
        m_pendingDescriptorSets.clear();
      }

      // Update HiZ texture array at binding 18 in set 0 (if bound)
      if (m_hasHiZBinding) {
        VkPipelineLayout pipelineLayout = hasNativePipeline ? nativePipelineLayout : m_context->getComputePipelineLayout();
        bindHiZDescriptorSet(pipelineLayout);
      }

      // Update UAV texture array at binding 3 in set 0 (if bound)
      if (m_hasUAVArrayBinding) {
        bindUAVArrayDescriptorSet();
      }

      // Commit any pending barriers right before dispatch.
      // 1) DXVK barrier flush (same rationale as dispatch() above)
      m_context->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT);
      // 2) Flush NVRHI-tracked barriers
      commitBarriers();

      // Call vkCmdDispatchIndirect directly
      auto bufferSlice = argBufferSlice.getSliceHandle(0, sizeof(VkDispatchIndirectCommand));
      m_device->getDxvkDevice()->vkd()->vkCmdDispatchIndirect(cmdBuffer, bufferSlice.handle, bufferSlice.offset);
      RTXMG_LOG("RTX MegaGeo: dispatchIndirect with special bindings");
    } else {
      // Automatic barriers already committed in setComputeState
      // Note: Pass 0 here because the buffer slice (argBufferSlice) was already created
      // with the offset at line 527. DXVK's dispatchIndirect adds this offset to the bound
      // buffer, so passing the offset again would double it.
      m_context->dispatchIndirect(0);
    }

    // Transfer pending binding sets to DXVK's command list lifetime tracker.
    trackPendingBindingSets();
  }

  void NvrhiDxvkCommandList::trackPendingBindingSets() {
    if (m_pendingBindingSetsForTracking.empty())
      return;

    Rc<DxvkCommandList> cmdList = m_context->getCommandList();
    for (auto& bindingSet : m_pendingBindingSetsForTracking) {
      Rc<DxvkResource> holder = new DxvkBindingSetHolder(std::move(bindingSet));
      cmdList->trackResource<DxvkAccess::Read>(std::move(holder));
    }
    m_pendingBindingSetsForTracking.clear();
  }

  void NvrhiDxvkCommandList::dispatchIndirect(uint64_t offset) {
    // Use the indirect params buffer from the current compute state
    if (!m_computeState.indirectParamsBuffer) {
      Logger::err("dispatchIndirect(offset) called without setting indirect params buffer in compute state");
      return;
    }
    dispatchIndirect(m_computeState.indirectParamsBuffer.Get(), offset);
  }

  void NvrhiDxvkCommandList::executeMultiIndirectClusterOperation(
    const nvrhi::rt::cluster::OperationDesc& nvrhiDesc)
  {
    RTXMG_LOG("RTX MegaGeo: executeMultiIndirectClusterOperation - Entry");

    const nvrhi::rt::cluster::OperationDesc& desc = nvrhiDesc;

    // Automatic barrier tracking for cluster operation buffers
    // NOTE: The sample uses ShaderResource for indirect args (not IndirectArgument) and
    // UnorderedAccess for addresses buffer (not AccelStructBuildInput). Match the sample exactly.
    if (m_EnableAutomaticBarriers) {
      if (desc.inIndirectArgsBuffer) {
        requireBufferState(desc.inIndirectArgsBuffer, nvrhi::ResourceStates::ShaderResource);
      }
      if (desc.inIndirectArgCountBuffer) {
        requireBufferState(desc.inIndirectArgCountBuffer, nvrhi::ResourceStates::ShaderResource);
      }
      if (desc.inOutAddressesBuffer) {
        // Match sample: use UnorderedAccess (COMPUTE_SHADER stage + SHADER_READ|WRITE access).
        // The driver's vkCmdBuildClusterAccelerationStructureIndirectNV internally dispatches a
        // compute shader that reads AND writes the addresses buffer. Using AccelStructBuildInput
        // (ACCEL_STRUCT_BUILD stage) does NOT synchronize with the internal compute dispatch,
        // causing stale data reads on reallocated buffers (page fault on freed VA).
        requireBufferState(desc.inOutAddressesBuffer, nvrhi::ResourceStates::UnorderedAccess);
      }
      if (desc.outSizesBuffer) {
        requireBufferState(desc.outSizesBuffer, nvrhi::ResourceStates::UnorderedAccess);
      }
      if (desc.outAccelerationStructuresBuffer) {
        // Match sample: AccelStructWrite only.
        // clearBufferUInt now bypasses DXVK's barrier tracking (calls vkCmdFillBuffer directly),
        // so our m_BufferStates is the sole tracker. The state chain is:
        //   AccelStructWrite (post-build) → CopyDest (clear) → AccelStructWrite (pre-build)
        // No dual-tracking conflict, no need for the combined UnorderedAccess|AccelStructWrite
        // workaround that was previously required.
        requireBufferState(desc.outAccelerationStructuresBuffer,
            nvrhi::ResourceStates::AccelStructWrite);
      }
      commitBarriers();
    }

    // Diagnostic logging for crash investigation
    RTXMG_LOG(str::format("RTX MegaGeo: executeMultiIndirectClusterOperation - scratchSize=", desc.scratchSizeInBytes));
    if (desc.inIndirectArgsBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo:   inIndirectArgsBuffer addr=", desc.inIndirectArgsBuffer->getGpuVirtualAddress()));
    }
    if (desc.inOutAddressesBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo:   inOutAddressesBuffer addr=", desc.inOutAddressesBuffer->getGpuVirtualAddress()));
    }
    if (desc.outSizesBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo:   outSizesBuffer addr=", desc.outSizesBuffer->getGpuVirtualAddress()));
    }
    if (desc.outAccelerationStructuresBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo:   outAccelStructBuffer addr=", desc.outAccelerationStructuresBuffer->getGpuVirtualAddress()));
    }

    // Get direct Vulkan command buffer from DXVK
    VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
    RTXMG_LOG(str::format("RTX MegaGeo: Got command buffer: ", (void*)cmdBuffer));

    // Translate NVRHI descriptor to Vulkan cluster commands
    // blasInput/clusterInput live here so pointers stored in vkCmds.input.opInput remain valid
    // until after vkCmdBuildClusterAccelerationStructureIndirectNV is called.
    VkClusterAccelerationStructureCommandsInfoNV vkCmds = {};
    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {};
    VkClusterAccelerationStructureTriangleClusterInputNV clusterInput = {};
    vkCmds.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    RTXMG_LOG("RTX MegaGeo: About to translate cluster operation");
    translateClusterOperation(desc, vkCmds, blasInput, clusterInput);
    RTXMG_LOG("RTX MegaGeo: Cluster operation translated");

    // Load cluster AS extension function pointer manually
    // Note: This function is not in DXVK's InstanceFn table yet, so we load it dynamically
    typedef void (VKAPI_PTR *PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)(
      VkCommandBuffer commandBuffer,
      const VkClusterAccelerationStructureCommandsInfoNV* pInfo);

    static PFN_vkCmdBuildClusterAccelerationStructureIndirectNV vkCmdBuildClusterAS = nullptr;
    if (!vkCmdBuildClusterAS) {
      RTXMG_LOG("RTX MegaGeo: Loading cluster AS extension function");
      // Load via device loader
      vkCmdBuildClusterAS = reinterpret_cast<PFN_vkCmdBuildClusterAccelerationStructureIndirectNV>(
        m_device->getDxvkDevice()->vkd()->sym("vkCmdBuildClusterAccelerationStructureIndirectNV"));

      if (vkCmdBuildClusterAS) {
        RTXMG_LOG(str::format("RTX MegaGeo: Cluster AS function loaded at: ", (void*)vkCmdBuildClusterAS));
      }
    }

    if (vkCmdBuildClusterAS) {
      // Always log key values for crash diagnosis
      RTXMG_LOG(str::format("RTX MegaGeo: vkCmdBuildClusterAS scratchData=0x", std::hex, vkCmds.scratchData,
                               " dstImplicit=0x", vkCmds.dstImplicitData,
                               " srcInfos=0x", vkCmds.srcInfosArray.deviceAddress,
                               " count=0x", vkCmds.srcInfosCount));

      // Alignment checking - cluster operations require specific alignments
      constexpr uint64_t kAS_ALIGNMENT = 256;
      constexpr uint64_t kBUFFER_ALIGNMENT = 16;

      RTXMG_LOG(str::format("RTX MegaGeo: ALIGNMENT CHECK:"));
      RTXMG_LOG(str::format("  scratchData (0x", std::hex, vkCmds.scratchData,
                               ") aligned(256)=", (vkCmds.scratchData % kAS_ALIGNMENT == 0)));
      RTXMG_LOG(str::format("  dstImplicitData (0x", std::hex, vkCmds.dstImplicitData,
                               ") aligned(256)=", (vkCmds.dstImplicitData == 0 || vkCmds.dstImplicitData % kAS_ALIGNMENT == 0)));
      RTXMG_LOG(str::format("  srcInfosArray.deviceAddress (0x", std::hex, vkCmds.srcInfosArray.deviceAddress,
                               ") aligned(16)=", (vkCmds.srcInfosArray.deviceAddress % kBUFFER_ALIGNMENT == 0)));
      RTXMG_LOG(str::format("  dstAddressesArray.deviceAddress (0x", std::hex, vkCmds.dstAddressesArray.deviceAddress,
                               ") aligned(16)=", (vkCmds.dstAddressesArray.deviceAddress == 0 || vkCmds.dstAddressesArray.deviceAddress % kBUFFER_ALIGNMENT == 0)));
      RTXMG_LOG(str::format("  dstSizesArray.deviceAddress (0x", std::hex, vkCmds.dstSizesArray.deviceAddress,
                               ") aligned(16)=", (vkCmds.dstSizesArray.deviceAddress == 0 || vkCmds.dstSizesArray.deviceAddress % kBUFFER_ALIGNMENT == 0)));

      RTXMG_LOG("RTX MegaGeo: About to call vkCmdBuildClusterAccelerationStructureIndirectNV");
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.sType=", vkCmds.sType, " vkCmds.pNext=", (void*)vkCmds.pNext));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.scratchData=", vkCmds.scratchData,
                               " vkCmds.dstImplicitData=", vkCmds.dstImplicitData,
                               " vkCmds.srcInfosCount=", vkCmds.srcInfosCount,
                               " vkCmds.addressResolutionFlags=", vkCmds.addressResolutionFlags));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.srcInfosArray.deviceAddress=", vkCmds.srcInfosArray.deviceAddress,
                               " stride=", vkCmds.srcInfosArray.stride,
                               " size=", vkCmds.srcInfosArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.dstAddressesArray.deviceAddress=", vkCmds.dstAddressesArray.deviceAddress,
                               " stride=", vkCmds.dstAddressesArray.stride,
                               " size=", vkCmds.dstAddressesArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.dstSizesArray.deviceAddress=", vkCmds.dstSizesArray.deviceAddress,
                               " stride=", vkCmds.dstSizesArray.stride,
                               " size=", vkCmds.dstSizesArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.input.sType=", vkCmds.input.sType,
                               " maxAccelStructCount=", vkCmds.input.maxAccelerationStructureCount,
                               " opType=", (uint32_t)vkCmds.input.opType,
                               " opMode=", (uint32_t)vkCmds.input.opMode));
      // CRITICAL: Log the opInput union pointers to verify they're set
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.input.opInput.pTriangleClusters=",
                               (void*)vkCmds.input.opInput.pTriangleClusters,
                               " pClustersBottomLevel=", (void*)vkCmds.input.opInput.pClustersBottomLevel,
                               " pMoveObjects=", (void*)vkCmds.input.opInput.pMoveObjects));
      // Always-visible logging of operation type for crash diagnosis
      const char* opTypeStr = "Unknown";
      switch ((uint32_t)vkCmds.input.opType) {
        case 0: opTypeStr = "MoveObjects"; break;
        case 1: opTypeStr = "BuildClustersBottomLevel"; break;
        case 2: opTypeStr = "BuildTriangleCluster"; break;
        case 3: opTypeStr = "BuildTriangleClusterTemplate"; break;
        case 4: opTypeStr = "InstantiateTriangleCluster"; break;
      }

      // Heavy barrier no longer needed. The root cause was bufferBarrier() not updating
      // m_BufferStates, causing requireBufferState() to emit wrong barriers (e.g. TRANSFER→X
      // instead of the correct post-barrier state→X). Now that bufferBarrier() updates
      // m_BufferStates and emits buffer-specific VkBufferMemoryBarrier, the automatic
      // barrier system in executeMultiIndirectClusterOperation sees correct states.

      // ALWAYS-ON: Log key values for CLAS build/instantiate debugging
      {
        const char* opTypeStr = "?";
        const char* opModeStr = "?";
        switch (vkCmds.input.opType) {
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV: opTypeStr = "BuildTemplate"; break;
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV: opTypeStr = "BuildTemplate"; break;
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV: opTypeStr = "Instantiate"; break;
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV: opTypeStr = "BlasBuild"; break;
        }
        switch (vkCmds.input.opMode) {
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV: opModeStr = "GetSizes"; break;
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV: opModeStr = "ExplicitDest"; break;
          case VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV: opModeStr = "ImplicitDest"; break;
        }
        Logger::warn(str::format("DIAG VK-CLAS-CALL: type=", opTypeStr, " mode=", opModeStr,
            " maxAS=", vkCmds.input.maxAccelerationStructureCount,
            " srcInfos={0x", std::hex, vkCmds.srcInfosArray.deviceAddress,
            " stride=", std::dec, vkCmds.srcInfosArray.stride,
            " size=", vkCmds.srcInfosArray.size, "}",
            " dstAddrs={0x", std::hex, vkCmds.dstAddressesArray.deviceAddress,
            " stride=", std::dec, vkCmds.dstAddressesArray.stride,
            " size=", vkCmds.dstAddressesArray.size, "}",
            " srcInfosCount=0x", std::hex, vkCmds.srcInfosCount,
            " scratch=0x", vkCmds.scratchData, std::dec));

        // For CLAS template/instantiate ops, log the triangle cluster input params
        // Only read pTriangleClusters for non-BlasBuild ops to avoid union aliasing
        if (vkCmds.input.opType != VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV
            && vkCmds.input.opInput.pTriangleClusters) {
          auto* tc = vkCmds.input.opInput.pTriangleClusters;
          Logger::warn(str::format("DIAG VK-CLAS-INPUT: vertexFormat=", (uint32_t)tc->vertexFormat,
              " maxGeomIdx=", tc->maxGeometryIndexValue,
              " maxUniqueGeom=", tc->maxClusterUniqueGeometryCount,
              " maxTri=", tc->maxClusterTriangleCount, " maxVert=", tc->maxClusterVertexCount,
              " totalTri=", tc->maxTotalTriangleCount, " totalVert=", tc->maxTotalVertexCount,
              " truncBits=", tc->minPositionTruncateBitCount));
        }
      }

      RTXMG_LOG("RTX MegaGeo: Calling vkCmdBuildClusterAccelerationStructureIndirectNV");
      vkCmdBuildClusterAS(cmdBuffer, &vkCmds);
      RTXMG_LOG("RTX MegaGeo: vkCmdBuildClusterAccelerationStructureIndirectNV completed");
    } else {
      Logger::err("VK_NV_cluster_acceleration_structure extension function not available");
      return;
    }

    // Track all cluster operation buffers on the DXVK command list so they stay alive
    // until the GPU fence signals completion. This replaces the broken CPU frame-counting
    // deferred destruction that couldn't account for GPU lag.
    {
      Rc<DxvkCommandList> cmdList = m_context->getCommandList();
      auto trackBuffer = [&](nvrhi::IBuffer* buffer) {
        if (!buffer) return;
        auto* nvrhiBuf = static_cast<NvrhiDxvkBuffer*>(buffer);
        cmdList->trackResource<DxvkAccess::Read>(nvrhiBuf->getDxvkBuffer());
      };
      trackBuffer(desc.inIndirectArgsBuffer);
      trackBuffer(desc.inIndirectArgCountBuffer);
      trackBuffer(desc.inOutAddressesBuffer);
      trackBuffer(desc.outSizesBuffer);
      trackBuffer(desc.outAccelerationStructuresBuffer);
      // outScratchBuffer is used via raw device address (vkCmds.scratchData) by the
      // cluster AS build, not through a descriptor set. Must be explicitly tracked
      // or the buffer can be freed while the GPU is still using it (page fault crash).
      trackBuffer(desc.outScratchBuffer);
    }

    // Update buffer states AFTER the cluster operation. The AS build command formally
    // operates at VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR per Vulkan spec.
    // Post-build tracking ensures subsequent barriers use the correct source stage.
    if (desc.inOutAddressesBuffer) {
      m_BufferStates[desc.inOutAddressesBuffer] = nvrhi::ResourceStates::AccelStructWrite;
    }
    if (desc.outSizesBuffer) {
      m_BufferStates[desc.outSizesBuffer] = nvrhi::ResourceStates::AccelStructWrite;
    }
    if (desc.outAccelerationStructuresBuffer) {
      // Post-build: AccelStructWrite (matches pre-build barrier and sample).
      // Next frame's clearBufferUInt will transition AccelStructWrite → CopyDest.
      m_BufferStates[desc.outAccelerationStructuresBuffer] =
          nvrhi::ResourceStates::AccelStructWrite;
    }

    RTXMG_LOG("RTX MegaGeo: executeMultiIndirectClusterOperation - Complete");
  }

  // Automatic barrier system implementation (matching sample's NVRHI behavior)
  void NvrhiDxvkCommandList::requireBufferState(nvrhi::IBuffer* buffer, nvrhi::ResourceStates state) {
    if (!buffer) return;

    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    const nvrhi::BufferDesc& desc = nvrhiBuffer->getDesc();

    // CPU-visible buffers can't change state (matching sample's NVRHI behavior)
    if (desc.cpuAccess != nvrhi::CpuAccessMode::None) {
      return;
    }

    // Get current state
    nvrhi::ResourceStates currentState;
    auto it = m_BufferStates.find(buffer);
    if (it != m_BufferStates.end()) {
      currentState = it->second;
    } else {
      // First time seeing this buffer - use its initial state from the desc
      currentState = desc.initialState;
    }

    // Record a barrier if state changes, OR if both states include UAV
    // (UAV → UAV needs an execution+memory barrier to ensure write visibility between dispatches)
    bool needsBarrier = (currentState != state);
    if (!needsBarrier && static_cast<uint32_t>(state & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      needsBarrier = true; // UAV → UAV barrier for write visibility
    }
    if (needsBarrier) {
      RTXMG_LOG(str::format("RTX MegaGeo: AUTO BARRIER buffer ", (void*)buffer,
        " from ", (uint32_t)currentState, " to ", (uint32_t)state));
      m_PendingBufferBarriers.push_back({buffer, currentState, state});
      m_BufferStates[buffer] = state;
    }
  }

  void NvrhiDxvkCommandList::setResourceStatesForBindingSet(nvrhi::IBindingSet* bindingSet) {
    if (!bindingSet) return;

    // Cast to NvrhiDxvkBindingSet to access getDesc()
    NvrhiDxvkBindingSet* nvrhiBindingSet = static_cast<NvrhiDxvkBindingSet*>(bindingSet);
    const nvrhi::BindingSetDesc& desc = nvrhiBindingSet->getDesc();

    for (const auto& binding : desc.bindings) {
      switch (binding.type) {
        case nvrhi::BindingSetItem::Type::TypedBuffer_SRV:
        case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
        case nvrhi::BindingSetItem::Type::RawBuffer_SRV:
          if (binding.resourceHandle) {
            requireBufferState(static_cast<nvrhi::IBuffer*>(binding.resourceHandle), nvrhi::ResourceStates::ShaderResource);
          }
          break;

        case nvrhi::BindingSetItem::Type::TypedBuffer_UAV:
        case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
        case nvrhi::BindingSetItem::Type::RawBuffer_UAV:
          if (binding.resourceHandle) {
            requireBufferState(static_cast<nvrhi::IBuffer*>(binding.resourceHandle), nvrhi::ResourceStates::UnorderedAccess);
          }
          break;

        case nvrhi::BindingSetItem::Type::ConstantBuffer:
          if (binding.resourceHandle) {
            requireBufferState(static_cast<nvrhi::IBuffer*>(binding.resourceHandle), nvrhi::ResourceStates::ConstantBuffer);
          }
          break;

        default:
          // Textures, samplers, etc. - not handling here
          break;
      }
    }
  }

  void NvrhiDxvkCommandList::commitBarriers() {
    if (m_PendingBufferBarriers.empty()) return;

    VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    std::vector<VkBufferMemoryBarrier> bufferBarriers;
    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;

    for (const auto& barrier : m_PendingBufferBarriers) {
      nvrhi::Object nativeObj = barrier.buffer->getNativeObject(nvrhi::ObjectType::VK_Buffer);
      if (nativeObj.pointer == nullptr) continue;

      NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(barrier.buffer);
      Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
      if (dxvkBuffer == nullptr) continue;

      VkBufferMemoryBarrier vkBarrier = {};
      vkBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      vkBarrier.buffer = dxvkBuffer->getSliceHandle().handle;
      vkBarrier.offset = 0;
      vkBarrier.size = VK_WHOLE_SIZE;
      vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      // Translate source state
      // CRITICAL: Use VK_PIPELINE_STAGE_ALL_COMMANDS_BIT for ShaderResource/UnorderedAccess
      // (matching sample's NVRHI Vulkan backend convertResourceState table).
      // Using only COMPUTE_SHADER_BIT misses ACCELERATION_STRUCTURE_BUILD_BIT_KHR and
      // CLUSTER_ACCELERATION_STRUCTURE_BIT_NV, causing the barrier to not synchronize
      // with vkCmdBuildClusterAccelerationStructureIndirectNV reads/writes.
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::UnorderedAccess) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::ShaderResource) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_SHADER_READ_BIT;
        srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::ConstantBuffer) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_UNIFORM_READ_BIT;
        srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::CopySource) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_TRANSFER_READ_BIT;
        srcStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::CopyDest) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        srcStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      }
      if (static_cast<uint32_t>(barrier.stateBefore & nvrhi::ResourceStates::AccelStructWrite) != 0) {
        vkBarrier.srcAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        srcStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      }

      // Translate destination state
      // CRITICAL: Use VK_PIPELINE_STAGE_ALL_COMMANDS_BIT for ShaderResource/UnorderedAccess
      // (matching sample's NVRHI Vulkan backend convertResourceState table).
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::UnorderedAccess) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::ShaderResource) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT;
        dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::ConstantBuffer) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_UNIFORM_READ_BIT;
        dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::IndirectArgument) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        dstStages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::CopySource) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_TRANSFER_READ_BIT;
        dstStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::CopyDest) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
        dstStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        dstStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      }
      if (static_cast<uint32_t>(barrier.stateAfter & nvrhi::ResourceStates::AccelStructWrite) != 0) {
        vkBarrier.dstAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        dstStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      }

      bufferBarriers.push_back(vkBarrier);
    }

    if (!bufferBarriers.empty()) {
      // Default stages if none were set
      if (srcStages == 0) srcStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      if (dstStages == 0) dstStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

      RTXMG_LOG(str::format("RTX MegaGeo: COMMITTING ", bufferBarriers.size(), " buffer barriers, srcStages=0x",
        std::hex, srcStages, " dstStages=0x", dstStages));

      m_device->getDxvkDevice()->vkd()->vkCmdPipelineBarrier(
        cmdBuffer,
        srcStages,
        dstStages,
        0,  // dependencyFlags
        0, nullptr,  // memory barriers
        static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),  // buffer barriers
        0, nullptr);  // image barriers
    }

    m_PendingBufferBarriers.clear();
  }

  void NvrhiDxvkCommandList::bufferBarrier(
    nvrhi::IBuffer* buffer,
    nvrhi::ResourceStates stateBefore,
    nvrhi::ResourceStates stateAfter)
  {
    if (!buffer) {
      Logger::warn("RTX MegaGeo: bufferBarrier - null buffer, skipping");
      return;
    }

    nvrhi::Object nativeObj = buffer->getNativeObject(nvrhi::ObjectType::VK_Buffer);
    if (nativeObj.pointer == nullptr) {
      Logger::warn("RTX MegaGeo: bufferBarrier - buffer is not a VK_Buffer (incompatible type), skipping");
      return;
    }

    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
    if (dxvkBuffer == nullptr) {
      Logger::warn("RTX MegaGeo: bufferBarrier - null DxvkBuffer, skipping");
      return;
    }

    // Translate NVRHI resource states to Vulkan stages/access
    // IMPORTANT: Use VkAccessFlags directly, NOT DxvkAccessFlags.
    // DxvkAccessFlags::raw() returns bit indices (Read=1, Write=2) which are NOT
    // the same as VkAccessFlags values (e.g. VK_ACCESS_SHADER_READ_BIT=0x20).
    // Passing DxvkAccessFlags::raw() to emitMemoryBarrier (which expects VkAccessFlags)
    // causes Vulkan validation errors due to the value mismatch.
    VkAccessFlags srcAccess = 0;
    VkAccessFlags dstAccess = 0;
    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;

    // Source state - map NVRHI states to proper Vulkan access/stage flags
    // CRITICAL: Use ALL_COMMANDS_BIT for ShaderResource/UnorderedAccess to match
    // the reference NVRHI Vulkan backend's convertResourceState table. Using narrow
    // stages (COMPUTE_SHADER_BIT only) misses cluster AS build/instantiation stages.
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      srcAccess |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::IndirectArgument) != 0) {
      srcAccess |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      srcStages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
      srcAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      srcStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::ShaderResource) != 0) {
      srcAccess |= VK_ACCESS_SHADER_READ_BIT;
      srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::CopySource) != 0) {
      srcAccess |= VK_ACCESS_TRANSFER_READ_BIT;
      srcStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::CopyDest) != 0) {
      srcAccess |= VK_ACCESS_TRANSFER_WRITE_BIT;
      srcStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::AccelStructWrite) != 0) {
      srcAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      srcStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::AccelStructRead) != 0) {
      srcAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      srcStages |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::ConstantBuffer) != 0) {
      srcAccess |= VK_ACCESS_UNIFORM_READ_BIT;
      srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (stateBefore == nvrhi::ResourceStates::Common) {
      srcStages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    // Destination state
    // CRITICAL: Use ALL_COMMANDS_BIT for ShaderResource/UnorderedAccess to match sample.
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      dstAccess |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::IndirectArgument) != 0) {
      dstAccess |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      dstStages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
      dstAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      dstStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::ShaderResource) != 0) {
      dstAccess |= VK_ACCESS_SHADER_READ_BIT;
      dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::CopySource) != 0) {
      dstAccess |= VK_ACCESS_TRANSFER_READ_BIT;
      dstStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::CopyDest) != 0) {
      dstAccess |= VK_ACCESS_TRANSFER_WRITE_BIT;
      dstStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::AccelStructWrite) != 0) {
      dstAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      dstStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::AccelStructRead) != 0) {
      dstAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      dstStages |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::ConstantBuffer) != 0) {
      dstAccess |= VK_ACCESS_UNIFORM_READ_BIT;
      dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    if (stateAfter == nvrhi::ResourceStates::Common) {
      dstStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    // Fallback: if no stages were set, use ALL_COMMANDS to be safe
    if (srcStages == 0) srcStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (dstStages == 0) dstStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    // Emit buffer-specific barrier (not global memory barrier) for precise synchronization.
    // This matches what commitBarriers() does for automatic barriers.
    VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
    auto slice = dxvkBuffer->getSliceHandle();

    VkBufferMemoryBarrier vkBarrier = {};
    vkBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    vkBarrier.srcAccessMask = srcAccess;
    vkBarrier.dstAccessMask = dstAccess;
    vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkBarrier.buffer = slice.handle;
    vkBarrier.offset = 0;
    vkBarrier.size = VK_WHOLE_SIZE;

    m_device->getDxvkDevice()->vkd()->vkCmdPipelineBarrier(
      cmdBuffer,
      srcStages,
      dstStages,
      0,  // dependencyFlags
      0, nullptr,  // memory barriers
      1, &vkBarrier,  // buffer barriers
      0, nullptr);  // image barriers

    // CRITICAL: Update m_BufferStates so subsequent requireBufferState() calls see the
    // correct post-barrier state. Without this, the automatic barrier system uses stale
    // state from before the explicit barrier, emitting wrong transitions (e.g. TRANSFER→X
    // instead of the correct post-barrier state→X). This was the root cause of Issue A:
    // dual barrier tracking between explicit bufferBarrier calls and the automatic system.
    m_BufferStates[buffer] = stateAfter;
  }

  void NvrhiDxvkCommandList::textureBarrier(
    nvrhi::ITexture* texture,
    nvrhi::ResourceStates stateBefore,
    nvrhi::ResourceStates stateAfter)
  {
    if (!texture) {
      RTXMG_LOG("RTX MegaGeo: textureBarrier - null texture, skipping");
      return;
    }

    // CRITICAL: Verify this is an NvrhiDxvkTexture before casting
    nvrhi::Object nativeObj = texture->getNativeObject(nvrhi::ObjectType::VK_Image);
    if (nativeObj.pointer == nullptr) {
      Logger::warn("RTX MegaGeo: textureBarrier - texture is not a VK_Image (incompatible type), skipping");
      return;
    }

    NvrhiDxvkTexture* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(texture);

    const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
    if (dxvkImage == nullptr) {
      RTXMG_LOG("RTX MegaGeo: textureBarrier - null DxvkImage, skipping");
      return;
    }

    // Determine Vulkan image layouts based on resource states
    VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Source state
    if (stateBefore == nvrhi::ResourceStates::Common) {
      oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Common/Unknown means UNDEFINED in Vulkan
    }
    else if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    else if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::ShaderResource) != 0) {
      oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Destination state
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::ShaderResource) != 0) {
      newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: textureBarrier - transitioning from ", oldLayout, " to ", newLayout));

    // Emit image layout transition through DXVK context
    VkImageSubresourceRange subresourceRange = {
      VK_IMAGE_ASPECT_COLOR_BIT,
      0, VK_REMAINING_MIP_LEVELS,
      0, VK_REMAINING_ARRAY_LAYERS
    };

    m_context->transformImage(
      dxvkImage,
      subresourceRange,
      oldLayout,
      newLayout);
  }

  void NvrhiDxvkCommandList::globalBarrier(
    nvrhi::ResourceStates stateBefore,
    nvrhi::ResourceStates stateAfter)
  {
    // Similar translation as bufferBarrier, but for global memory barrier
    VkPipelineStageFlags srcStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkAccessFlags srcAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkAccessFlags dstAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    m_context->emitMemoryBarrier(0, srcStages, srcAccess, dstStages, dstAccess);
  }

  void NvrhiDxvkCommandList::bindComputeResources(const nvrhi::ComputeState& state) {
    // Reset HiZ binding state for this new shader - only set true if we actually bind HiZ textures
    // NOTE: Do NOT clear m_hiZImageViews here - the views may still be referenced by in-flight
    // descriptor sets. The Rc<> ref counting will keep them alive, and they'll be overwritten
    // when new HiZ textures are bound.
    m_hasHiZBinding = false;

    // Bind the compute shader from the pipeline
    if (state.pipeline) {
      auto* computePipeline = static_cast<NvrhiDxvkComputePipeline*>(state.pipeline.Get());
      const Rc<DxvkShader>& shader = computePipeline->getShader();
      if (shader != nullptr) {
        m_context->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, shader);
        RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bound shader"));
      }
    }

    // Bind resources from binding sets
    // DXVK slot mapping (from Slang/HLSL register to Vulkan binding):
    //   t0-t99 (SRVs)     -> slots 0-99
    //   s0-s99 (Samplers) -> slots 100-199
    //   u0-u99 (UAVs)     -> slots 200-299
    //   b0-b99 (CBVs)     -> slots 300-399
    uint32_t bindingSetIndex = 0;
    for (const auto& bindingSet : state.bindingSets) {
      if (!bindingSet) {
        RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bindingSet[", bindingSetIndex, "] is null"));
        bindingSetIndex++;
        continue;
      }

      auto* nvrhiBindingSet = static_cast<NvrhiDxvkBindingSet*>(bindingSet.Get());
      const nvrhi::BindingSetDesc& desc = nvrhiBindingSet->getDesc();
      RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bindingSet[", bindingSetIndex, "] has ", desc.bindings.size(), " bindings"));

      // NVRHI-Vulkan Native Approach: If this binding set has a pre-built VkDescriptorSet,
      // queue it for binding after commitComputeState() instead of using per-binding calls.
      // This matches the sample's approach of pre-building descriptor sets in createBindingSet()
      // and binding them with a single vkCmdBindDescriptorSets call.
      if (nvrhiBindingSet->hasDescriptorSet()) {
        auto* bindingLayout = nvrhiBindingSet->getLayout();
        auto* nvrhiLayout = static_cast<NvrhiDxvkBindingLayout*>(bindingLayout);
        uint32_t setIndex = nvrhiLayout ? nvrhiLayout->getDesc().registerSpace : bindingSetIndex;

        // Get the pipeline layout from the MegaGeo compute pipeline (not DXVK's layout)
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        if (state.pipeline) {
          auto* computePipeline = static_cast<NvrhiDxvkComputePipeline*>(state.pipeline.Get());
          if (computePipeline && computePipeline->hasVkPipeline()) {
            pipelineLayout = computePipeline->getPipelineLayout();
          }
        }

        RTXMG_LOG(str::format("RTX MegaGeo: bindingSet[", bindingSetIndex,
          "] has pre-built descriptor set, queuing for bind at set index ", setIndex,
          " with pipelineLayout=", (void*)pipelineLayout));

        // Store the binding set handle to keep the descriptor set alive until command buffer completes
        m_pendingDescriptorSets.push_back({nvrhiBindingSet->getDescriptorSet(), setIndex, pipelineLayout, bindingSet});

        // Even though we skip DXVK per-binding calls, update m_BufferStates so the automatic
        // barrier system knows what state each buffer is in after this dispatch.
        // Without this, subsequent copyBuffer/bufferBarrier calls would see stale states
        // and might not insert necessary barriers (e.g. compute UAV write → transfer read).
        if (m_EnableAutomaticBarriers) {
          setResourceStatesForBindingSet(bindingSet.Get());
        }

        bindingSetIndex++;
        continue;  // Skip per-binding approach for this set
      }

      // For binding set 0 (main descriptor set), use DXVK convention slot numbers
      // that match the pre-compiled SPIRV layout:
      //   SRVs (t0, t1, ...) → slots 0, 1, 2, ...
      //   Samplers (s0, s1, ...) → slots 100, 101, ...
      //   UAVs (u0, u1, ...) → slots 200, 201, ...
      //   CBs (b0, b1, ...) → slots 300, 301, ...
      // The SPIRV bindings use these DXVK convention values, and the slot mapping
      // remaps them to sequential Vulkan binding indices.

      for (const auto& item : desc.bindings) {
        uint32_t dxvkSlot = item.slot;  // Default for non-set-0 bindings

        switch (item.type) {
          case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
            // SRV: DXVK convention slot = register number (t0→0, t1→1, etc.)
            if (bindingSetIndex == 0) {
              dxvkSlot = item.slot;  // SRV slots are at register number
            }
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t bufferSize = dxvkBuffer->info().size;
              // Resolve range to fit buffer (matching NVRHI's BufferRange::resolve)
              uint64_t offset = std::min(item.range.byteOffset, bufferSize);
              uint64_t size = (item.range.byteSize > 0)
                ? std::min(item.range.byteSize, bufferSize - offset)
                : bufferSize - offset;

              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              RTXMG_LOG(str::format("RTX MegaGeo: bind SRV slot=", dxvkSlot, " offset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            } else {
              Logger::err(str::format("RTX MegaGeo: SRV slot=", dxvkSlot, " has NULL resourceHandle!"));
            }
            break;

          case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
            // UAV: DXVK convention slot = 200 + register number (u0→200, u1→201, etc.)
            if (bindingSetIndex == 0) {
              dxvkSlot = 200 + item.slot;
            }
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t bufferSize = dxvkBuffer->info().size;
              // Resolve range to fit buffer (matching NVRHI's BufferRange::resolve)
              uint64_t offset = std::min(item.range.byteOffset, bufferSize);
              uint64_t size = (item.range.byteSize > 0)
                ? std::min(item.range.byteSize, bufferSize - offset)
                : bufferSize - offset;

              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              RTXMG_LOG(str::format("RTX MegaGeo: bind UAV slot=", dxvkSlot, " (u", item.slot, ") offset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            }
            break;

          case nvrhi::BindingSetItem::Type::ConstantBuffer:
            // CB: DXVK convention slot = 300 + register number (b0→300, b1→301, etc.)
            if (bindingSetIndex == 0) {
              dxvkSlot = 300 + item.slot;
            }
            // Also push constants if data is cached (for Slang push constant compilation)
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);

              // Push constants for Slang-compiled shaders that use push constants
              if (nvrhiBuffer->hasCachedData()) {
                uint32_t pushOffset = 0;
                uint32_t pushSize = static_cast<uint32_t>(nvrhiBuffer->getCachedDataSize());
                RTXMG_LOG(str::format("RTX MegaGeo: pushConstants size=", pushSize, " (b", item.slot, ")"));
                m_context->pushConstants(pushOffset, pushSize, nvrhiBuffer->getCachedData());
              }

              // Bind as uniform buffer - for volatile buffers, use current version's offset
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t offset;
              uint64_t size;
              if (nvrhiBuffer->isVolatile()) {
                offset = nvrhiBuffer->getCurrentVersionOffset();
                size = nvrhiBuffer->getPerVersionSize();
              } else {
                uint64_t bufferSize = dxvkBuffer->info().size;
                offset = std::min(item.range.byteOffset, bufferSize);
                size = (item.range.byteSize > 0)
                  ? std::min(item.range.byteSize, bufferSize - offset)
                  : bufferSize - offset;
              }

              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              RTXMG_LOG(str::format("RTX MegaGeo: bind CBV slot=", dxvkSlot, " (b", item.slot, ") offset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            }
            break;

          case nvrhi::BindingSetItem::Type::Sampler:
            // Sampler: DXVK convention slot = 100 + register number (s0→100, s1→101, etc.)
            if (bindingSetIndex == 0) {
              dxvkSlot = 100 + item.slot;
            }
            if (item.resourceHandle) {
              auto* nvrhiSampler = static_cast<NvrhiDxvkSampler*>(item.resourceHandle);
              Rc<DxvkSampler> dxvkSampler = nvrhiSampler->getDxvkSampler();
              RTXMG_LOG(str::format("RTX MegaGeo: bind Sampler slot=", dxvkSlot, " (s", item.slot, ")"));
              m_context->bindResourceSampler(dxvkSlot, dxvkSampler);
            }
            break;

          case nvrhi::BindingSetItem::Type::Texture_SRV:
            // Bind textures in DXVK's flat slot model
            // For HiZ textures (binding set 1), they need special handling because the shader
            // expects them as an ARRAY at binding 17 (9 elements), not separate bindings 17-25.
            // DXVK's bindResourceView() doesn't support arrays, so we store the views and
            // update the descriptor set manually in dispatch() after commitComputeState().
            {
              auto* bindingLayout = nvrhiBindingSet->getLayout();
              auto* nvrhiLayout = static_cast<NvrhiDxvkBindingLayout*>(bindingLayout);
              bool isHiZBinding = nvrhiLayout && nvrhiLayout->getDesc().registerSpaceIsDescriptorSet;

              if (isHiZBinding) {
                // HiZ textures: DON'T use DXVK's bindResourceView() because it doesn't support arrays.
                // The shader expects binding 17 with array count 9, not separate bindings.
                // Store the image views and update the descriptor set array in dispatch().
                RTXMG_LOG(str::format("RTX MegaGeo: storing HiZ Texture_SRV array[", item.arrayElement, "]"));

                if (item.resourceHandle) {
                  nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
                  if (nativeObj.pointer == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: HiZ Texture_SRV array[", item.arrayElement,
                      "] - resource is not a VK_Image (incompatible texture), skipping"));
                    break;
                  }

                  auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
                  const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
                  if (dxvkImage == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: HiZ Texture_SRV array[", item.arrayElement,
                      "] - null DxvkImage, skipping"));
                    break;
                  }

                  // Create image view for HiZ texture
                  DxvkImageViewCreateInfo viewInfo;
                  viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
                  viewInfo.format = dxvkImage->info().format;
                  viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
                  // HiZ uses depth aspect since it's a depth buffer hierarchy
                  VkFormat fmt = dxvkImage->info().format;
                  if (fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D16_UNORM ||
                      fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                      fmt == VK_FORMAT_R32_SFLOAT) {
                    // Depth or R32 format - use appropriate aspect
                    viewInfo.aspect = (fmt == VK_FORMAT_R32_SFLOAT) ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
                  } else {
                    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                  }
                  viewInfo.minLevel = 0;
                  viewInfo.numLevels = dxvkImage->info().mipLevels;
                  viewInfo.minLayer = 0;
                  viewInfo.numLayers = 1;

                  Rc<DxvkImageView> imageView = m_device->getDxvkDevice()->createImageView(dxvkImage, viewInfo);
                  RTXMG_LOG(str::format("RTX MegaGeo: created HiZ image view for array[", item.arrayElement,
                    "] format=", (uint32_t)fmt, " aspect=", (uint32_t)viewInfo.aspect));

                  // Track both the image AND the image view to keep them alive until command buffer execution completes
                  // CRITICAL: The image view must be tracked, not just the image, otherwise the VkImageView
                  // can be destroyed while still referenced by pending commands
                  m_context->getCommandList()->trackResource<DxvkAccess::Read>(dxvkImage);
                  m_context->getCommandList()->trackResource<DxvkAccess::Read>(imageView);

                  // Store in m_hiZImageViews for array update in dispatch()
                  if (item.arrayElement < HIZ_MAX_LODS) {
                    m_hiZImageViews[item.arrayElement] = imageView;
                    m_hasHiZBinding = true;
                  }
                } else {
                  Logger::err(str::format("RTX MegaGeo: HiZ Texture_SRV array[", item.arrayElement, "] has NULL resourceHandle!"));
                }
              } else {
                // Texture_SRV: DXVK convention slot = register number (t0→0, t1→1, etc.)
                if (bindingSetIndex == 0) {
                  dxvkSlot = item.slot;  // SRV slots are at register number
                }
                uint32_t effectiveSlot = dxvkSlot + item.arrayElement;
                if (item.resourceHandle) {
                  nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
                  if (nativeObj.pointer == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: Texture_SRV slot=", effectiveSlot,
                      " - resource is not a VK_Image (incompatible texture), skipping bind"));
                    break;
                  }

                  auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
                  const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
                  if (dxvkImage == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: Texture_SRV slot=", effectiveSlot,
                      " - null DxvkImage, skipping bind"));
                    break;
                  }

                  DxvkImageViewCreateInfo viewInfo;
                  viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
                  viewInfo.format = dxvkImage->info().format;
                  viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
                  viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                  viewInfo.minLevel = 0;
                  viewInfo.numLevels = dxvkImage->info().mipLevels;
                  viewInfo.minLayer = 0;
                  viewInfo.numLayers = 1;

                  Rc<DxvkImageView> imageView = m_device->getDxvkDevice()->createImageView(dxvkImage, viewInfo);
                  RTXMG_LOG(str::format("RTX MegaGeo: bind Texture_SRV slot=", effectiveSlot, " (t", item.slot, " array=", item.arrayElement, ")"));
                  m_context->bindResourceView(effectiveSlot, imageView, nullptr);
                } else {
                  Logger::err(str::format("RTX MegaGeo: Texture_SRV slot=", effectiveSlot, " has NULL resourceHandle!"));
                }
              }
            }
            break;

          case nvrhi::BindingSetItem::Type::Texture_UAV:
            // Texture_UAV: DXVK convention slot = 200 + register number (u0→200, u1→201, etc.)
            // For UAV arrays (like HiZ reduce u_output[HIZ_MAX_LODS]), we need special handling
            // because DXVK's bindResourceView() doesn't support descriptor arrays.
            {
              if (bindingSetIndex == 0) {
                dxvkSlot = 200 + item.slot;
              }

              // Check if this is a UAV array binding (slot 0 with array elements for HiZ reduce)
              // The binding layout for UAV arrays has slot 0 with HIZ_MAX_LODS elements
              bool isUAVArrayBinding = (item.slot == 0 && item.arrayElement < HIZ_MAX_LODS);

              if (isUAVArrayBinding) {
                // UAV array: DON'T use DXVK's bindResourceView() because it doesn't support arrays.
                // Store the image views and update the descriptor set array in dispatch().
                RTXMG_LOG(str::format("RTX MegaGeo: storing UAV array Texture_UAV[", item.arrayElement, "]"));

                if (item.resourceHandle) {
                  nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
                  if (nativeObj.pointer == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: UAV array Texture_UAV[", item.arrayElement,
                      "] - resource is not a VK_Image, skipping"));
                    break;
                  }

                  auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
                  const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
                  if (dxvkImage == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: UAV array Texture_UAV[", item.arrayElement,
                      "] - null DxvkImage, skipping"));
                    break;
                  }

                  // Verify the image supports storage usage
                  VkImageUsageFlags imageUsage = dxvkImage->info().usage;
                  if (!(imageUsage & VK_IMAGE_USAGE_STORAGE_BIT)) {
                    Logger::warn(str::format("RTX MegaGeo: UAV array Texture_UAV[", item.arrayElement,
                      "] - image doesn't have STORAGE_BIT, skipping"));
                    break;
                  }

                  // Create image view for UAV (storage image)
                  DxvkImageViewCreateInfo viewInfo;
                  viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
                  viewInfo.format = dxvkImage->info().format;
                  viewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                  viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                  viewInfo.minLevel = 0;
                  viewInfo.numLevels = 1;  // UAVs bind single mip level
                  viewInfo.minLayer = 0;
                  viewInfo.numLayers = 1;

                  Rc<DxvkImageView> imageView = m_device->getDxvkDevice()->createImageView(dxvkImage, viewInfo);
                  if (imageView != nullptr) {
                    RTXMG_LOG(str::format("RTX MegaGeo: created UAV image view for array[", item.arrayElement,
                      "] format=", (uint32_t)dxvkImage->info().format));

                    // Track the image and view for command buffer lifetime
                    m_context->getCommandList()->trackResource<DxvkAccess::Write>(dxvkImage);
                    m_context->getCommandList()->trackResource<DxvkAccess::Write>(imageView);

                    // Store in m_uavImageViews for array update in dispatch()
                    m_uavImageViews[item.arrayElement] = imageView;
                    m_hasUAVArrayBinding = true;
                  } else {
                    Logger::err(str::format("RTX MegaGeo: UAV array Texture_UAV[", item.arrayElement,
                      "] - failed to create image view"));
                  }
                } else {
                  Logger::err(str::format("RTX MegaGeo: UAV array Texture_UAV[", item.arrayElement, "] has NULL resourceHandle!"));
                }
              } else {
                // Non-array UAV: use standard binding
                uint32_t effectiveSlot = dxvkSlot + item.arrayElement;
                if (item.resourceHandle) {
                  // Try to get native object to verify this is actually a texture
                  nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
                  if (nativeObj.pointer == nullptr) {
                    // Not a texture - might be a buffer passed incorrectly, skip binding
                    Logger::warn(str::format("RTX MegaGeo: Texture_UAV slot=", effectiveSlot,
                      " - resource is not a VK_Image, skipping bind"));
                    break;
                  }

                  auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
                  const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
                  if (dxvkImage == nullptr) {
                    Logger::warn(str::format("RTX MegaGeo: Texture_UAV slot=", effectiveSlot,
                      " - null DxvkImage, skipping bind"));
                    break;
                  }

                  // Verify the image supports storage usage
                  VkImageUsageFlags imageUsage = dxvkImage->info().usage;
                  if (!(imageUsage & VK_IMAGE_USAGE_STORAGE_BIT)) {
                    Logger::warn(str::format("RTX MegaGeo: Texture_UAV slot=", effectiveSlot,
                      " - image doesn't have STORAGE_BIT, skipping bind"));
                    break;
                  }

                  // Create image view for UAV (storage image)
                  DxvkImageViewCreateInfo viewInfo;
                  viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
                  viewInfo.format = dxvkImage->info().format;
                  viewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                  viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                  viewInfo.minLevel = 0;
                  viewInfo.numLevels = 1;  // UAVs typically bind single mip level
                  viewInfo.minLayer = 0;
                  viewInfo.numLayers = 1;

                  Rc<DxvkImageView> imageView = m_device->getDxvkDevice()->createImageView(dxvkImage, viewInfo);
                  if (imageView != nullptr) {
                    RTXMG_LOG(str::format("RTX MegaGeo: bind Texture_UAV slot=", effectiveSlot, " (u", item.slot, " array=", item.arrayElement, ")"));
                    m_context->bindResourceView(effectiveSlot, imageView, nullptr);
                  } else {
                    Logger::err(str::format("RTX MegaGeo: Texture_UAV slot=", effectiveSlot, " - failed to create image view"));
                  }
                } else {
                  Logger::err(str::format("RTX MegaGeo: Texture_UAV slot=", effectiveSlot, " has NULL resourceHandle!"));
                }
              }
            }
            break;

          default:
            Logger::warn(str::format("bindComputeResources: Unhandled binding type ", (int)item.type));
            break;
        }
      }
      bindingSetIndex++;
    }
  }

  void NvrhiDxvkCommandList::translateClusterOperation(
    const nvrhi::rt::cluster::OperationDesc& desc,
    VkClusterAccelerationStructureCommandsInfoNV& vkCmds,
    VkClusterAccelerationStructureClustersBottomLevelInputNV& blasInput,
    VkClusterAccelerationStructureTriangleClusterInputNV& clusterInput)
  {
    // NOTE: The Vulkan cluster AS API has changed significantly from the original SDK version.
    // The actual implementation needs adaptation based on RTX Remix's usage patterns.

    memset(&vkCmds, 0, sizeof(vkCmds));
    vkCmds.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    vkCmds.pNext = nullptr;  // Explicitly set pNext

    // Set up operation input info
    VkClusterAccelerationStructureInputInfoNV& input = vkCmds.input;
    memset(&input, 0, sizeof(input));  // Zero-initialize input structure
    input.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    input.pNext = nullptr;  // Explicitly set pNext
    input.maxAccelerationStructureCount = desc.params.maxArgCount;

    // Map operation type (NOTE: enum values have changed in the new API)
    const char* opTypeName = "Unknown";
    switch (desc.params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        opTypeName = "ClasBuildTemplates";
        break;
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        opTypeName = "ClasInstantiateTemplates";
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        opTypeName = "BlasBuild";
        break;
    }

    // Map operation mode (NOTE: enum values have changed in the new API)
    const char* opModeName = "Unknown";
    switch (desc.params.mode) {
      case nvrhi::rt::cluster::OperationMode::GetSizes:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        opModeName = "GetSizes";
        break;
      case nvrhi::rt::cluster::OperationMode::ExplicitDestinations:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        opModeName = "ExplicitDestinations";
        break;
      case nvrhi::rt::cluster::OperationMode::ImplicitDestinations:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        opModeName = "ImplicitDestinations";
        break;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Operation type=", opTypeName, " mode=", opModeName,
                             " maxArgCount=", input.maxAccelerationStructureCount));
    RTXMG_LOG(str::format("RTX MegaGeo: input.sType=", input.sType, " input.pNext=", (void*)input.pNext,
                             " input.opType=", (uint32_t)input.opType, " input.opMode=", (uint32_t)input.opMode));

    // Set up opInput based on operation type
    // Note: blasInput/clusterInput are allocated in the caller's stack frame so they
    // persist until after vkCmdBuildClusterAS is called.

    switch (desc.params.type) {
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        blasInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
        blasInput.maxTotalClusterCount = desc.params.blas.maxTotalClasCount;
        blasInput.maxClusterCountPerAccelerationStructure = desc.params.blas.maxClasPerBlasCount;
        input.opInput.pClustersBottomLevel = &blasInput;
        // Always log BlasBuild parameters for crash debugging
        RTXMG_LOG(str::format("RTX MegaGeo: BlasBuild maxTotalClusterCount=", blasInput.maxTotalClusterCount,
                                 " maxClusterCountPerAS=", blasInput.maxClusterCountPerAccelerationStructure,
                                 " maxArgCount=", input.maxAccelerationStructureCount));
        break;

      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        clusterInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
        clusterInput.vertexFormat = desc.params.clas.vertexFormat;
        clusterInput.maxGeometryIndexValue = desc.params.clas.maxGeometryIndex;
        clusterInput.maxClusterUniqueGeometryCount = desc.params.clas.maxUniqueGeometryCount;
        clusterInput.maxClusterTriangleCount = desc.params.clas.maxTriangleCount;
        clusterInput.maxClusterVertexCount = desc.params.clas.maxVertexCount;
        clusterInput.maxTotalTriangleCount = desc.params.clas.maxTotalTriangleCount;
        clusterInput.maxTotalVertexCount = desc.params.clas.maxTotalVertexCount;
        clusterInput.minPositionTruncateBitCount = desc.params.clas.minPositionTruncateBitCount;
        input.opInput.pTriangleClusters = &clusterInput;
        RTXMG_LOG(str::format("RTX MegaGeo: ClusterOp maxTriCount=", clusterInput.maxTotalTriangleCount,
                                 " maxVertCount=", clusterInput.maxTotalVertexCount));
        RTXMG_LOG(str::format("RTX MegaGeo: clusterInput.sType=", clusterInput.sType, " clusterInput.pNext=", (void*)clusterInput.pNext,
                                 " vertexFormat=", (uint32_t)clusterInput.vertexFormat,
                                 " maxGeometryIndexValue=", clusterInput.maxGeometryIndexValue,
                                 " maxClusterUniqueGeometryCount=", clusterInput.maxClusterUniqueGeometryCount,
                                 " opInput.pTriangleClusters=", (void*)input.opInput.pTriangleClusters));
        RTXMG_LOG(str::format("RTX MegaGeo: clusterInput sizeof=", sizeof(clusterInput),
                                 " maxClusterTriangleCount=", clusterInput.maxClusterTriangleCount,
                                 " maxClusterVertexCount=", clusterInput.maxClusterVertexCount,
                                 " minPositionTruncateBitCount=", clusterInput.minPositionTruncateBitCount));
        break;
    }

    // Set up srcInfosArray (indirect args buffer)
    // Match sample: use buffer->getDesc().byteSize - offset for strided array .size
    // The sample uses full buffer extent, not maxArgCount * stride.
    // srcInfosCount (indirect count buffer) limits actual processing count.
    if (desc.inIndirectArgsBuffer) {
      NvrhiDxvkBuffer* argsBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inIndirectArgsBuffer);
      vkCmds.srcInfosArray.deviceAddress = argsBuffer->getGpuVirtualAddress() + desc.inIndirectArgsOffsetInBytes;
      vkCmds.srcInfosArray.stride = argsBuffer->getDesc().structStride;
      vkCmds.srcInfosArray.size = argsBuffer->getDesc().byteSize - desc.inIndirectArgsOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: ", opTypeName, " srcInfosArray addr=0x", std::hex, vkCmds.srcInfosArray.deviceAddress,
                               " stride=", std::dec, vkCmds.srcInfosArray.stride, " size=", vkCmds.srcInfosArray.size,
                               " (maxArgCount=", input.maxAccelerationStructureCount, " bufByteSize=", argsBuffer->getDesc().byteSize, ")"));
    } else {
      Logger::warn("RTX MegaGeo: inIndirectArgsBuffer is NULL");
    }

    // Set up inOutAddressesBuffer (BLAS/CLAS addresses)
    if (desc.inOutAddressesBuffer) {
      NvrhiDxvkBuffer* addressesBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inOutAddressesBuffer);
      vkCmds.dstAddressesArray.deviceAddress = addressesBuffer->getGpuVirtualAddress() + desc.inOutAddressesOffsetInBytes;
      vkCmds.dstAddressesArray.stride = addressesBuffer->getDesc().structStride;
      vkCmds.dstAddressesArray.size = addressesBuffer->getDesc().byteSize - desc.inOutAddressesOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: ", opTypeName, " dstAddressesArray addr=0x", std::hex, vkCmds.dstAddressesArray.deviceAddress,
                               " stride=", std::dec, vkCmds.dstAddressesArray.stride, " size=", vkCmds.dstAddressesArray.size,
                               " (bufByteSize=", addressesBuffer->getDesc().byteSize, ")"));
    } else {
      // Explicitly zero the array for GetSizes mode
      vkCmds.dstAddressesArray.deviceAddress = 0;
      vkCmds.dstAddressesArray.stride = 0;
      vkCmds.dstAddressesArray.size = 0;
      RTXMG_LOG("RTX MegaGeo: inOutAddressesBuffer is NULL (expected for GetSizes mode)");
    }

    // Set up output acceleration structures buffer
    // For GetSizes mode, this should be NULL/zero
    if (desc.outAccelerationStructuresBuffer) {
      NvrhiDxvkBuffer* accelBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outAccelerationStructuresBuffer);
      vkCmds.dstImplicitData = accelBuffer->getGpuVirtualAddress() + desc.outAccelerationStructuresOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: dstImplicitData addr=", vkCmds.dstImplicitData));
    } else {
      // Explicitly zero for GetSizes mode
      vkCmds.dstImplicitData = 0;
      RTXMG_LOG("RTX MegaGeo: outAccelerationStructuresBuffer is NULL (expected for GetSizes mode)");
    }

    // Set up output sizes buffer
    // For GetSizes operations (outAccelerationStructuresBuffer is NULL), outSizesBuffer should be non-NULL
    // For Build operations (outAccelerationStructuresBuffer is non-NULL), outSizesBuffer can be NULL
    if (desc.outSizesBuffer) {
      NvrhiDxvkBuffer* sizesBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outSizesBuffer);
      vkCmds.dstSizesArray.deviceAddress = sizesBuffer->getGpuVirtualAddress() + desc.outSizesOffsetInBytes;
      vkCmds.dstSizesArray.stride = sizesBuffer->getDesc().structStride;
      vkCmds.dstSizesArray.size = sizesBuffer->getDesc().byteSize - desc.outSizesOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: dstSizesArray addr=", vkCmds.dstSizesArray.deviceAddress,
                               " stride=", vkCmds.dstSizesArray.stride, " size=", vkCmds.dstSizesArray.size,
                               " (bufByteSize=", sizesBuffer->getDesc().byteSize, ")"));
    } else if (!desc.outAccelerationStructuresBuffer && !desc.inOutAddressesBuffer) {
      // GetSizes mode but no sizes output - this is unexpected
      Logger::warn("RTX MegaGeo: outSizesBuffer is NULL in GetSizes mode");
    } else {
      // Build mode - outSizesBuffer being NULL is expected
      RTXMG_LOG("RTX MegaGeo: outSizesBuffer is NULL (expected for Build mode)");
    }

    // Set up scratch data
    if (desc.outScratchBuffer) {
      NvrhiDxvkBuffer* scratchBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outScratchBuffer);
      vkCmds.scratchData = scratchBuffer->getGpuVirtualAddress();
      RTXMG_LOG(str::format("RTX MegaGeo: scratchData addr=", vkCmds.scratchData,
                               " size=", desc.scratchSizeInBytes));
    } else if (desc.scratchSizeInBytes > 0) {
      // Suballocate scratch buffer from pool (matching NVRHI's scratch manager behavior)
      Rc<DxvkBuffer> scratchBuffer;
      uint64_t scratchOffset = 0;
      constexpr uint32_t kClusterScratchAlignment = 256;

      if (!m_scratchManager->suballocateBuffer(desc.scratchSizeInBytes, &scratchBuffer, &scratchOffset,
                                                kClusterScratchAlignment)) {
        Logger::err(str::format("RTX MegaGeo: Failed to suballocate scratch buffer, size=", desc.scratchSizeInBytes));
        return;
      }

      vkCmds.scratchData = scratchBuffer->getDeviceAddress() + scratchOffset;
      RTXMG_LOG(str::format("RTX MegaGeo: suballocated scratch buffer, addr=", std::hex, vkCmds.scratchData,
                               " offset=", std::dec, scratchOffset,
                               " (requested=", desc.scratchSizeInBytes, ")"));

      // NOTE: Scratch zeroing REMOVED — sample does NOT zero scratch buffers.
      // The driver's internal compute shader should handle scratch contents correctly
      // without requiring pre-zeroed memory.

      // Track the buffer for command list lifetime
      m_context->getCommandList()->trackResource<DxvkAccess::Write>(scratchBuffer);
    } else {
      vkCmds.scratchData = 0;
      RTXMG_LOG("RTX MegaGeo: scratchSizeInBytes is 0, no scratch needed");
    }

    // Set srcInfosCount if indirect arg count buffer is provided
    if (desc.inIndirectArgCountBuffer) {
      NvrhiDxvkBuffer* countBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inIndirectArgCountBuffer);
      vkCmds.srcInfosCount = countBuffer->getGpuVirtualAddress() + desc.inIndirectArgCountOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: INDIRECT COUNT BUFFER SET - baseAddr=0x", std::hex, countBuffer->getGpuVirtualAddress(),
                               " offset=", std::dec, desc.inIndirectArgCountOffsetInBytes,
                               " srcInfosCount=0x", std::hex, vkCmds.srcInfosCount));
    } else {
      vkCmds.srcInfosCount = 0;
      RTXMG_LOG("RTX MegaGeo: NO INDIRECT COUNT BUFFER - srcInfosCount=0, will use srcInfosArray.size/stride as count");
    }

    // Set address resolution flags to none for now
    vkCmds.addressResolutionFlags = 0;
  }

} // namespace dxvk
