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
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "nvrhi_types.h"
#include "nvrhi_dxvk_device.h"
#include "nvrhi_scratch_manager.h"

// HiZ constants
#include "../hiz/hiz_buffer_constants.h"

namespace dxvk {

  // Buffer barrier tracking for automatic barrier system
  struct BufferBarrier {
    nvrhi::IBuffer* buffer;
    nvrhi::ResourceStates stateBefore;
    nvrhi::ResourceStates stateAfter;
  };

  // Wrapper to hold NVRHI binding sets inside DXVK's lifetime tracker.
  // When attached to a command list via trackResource(), DXVK keeps the reference
  // alive until the GPU fence signals that the command buffer has completed.
  // This matches the sample's referencedResources pattern.
  class DxvkBindingSetHolder : public DxvkResource {
  public:
    DxvkBindingSetHolder(nvrhi::BindingSetHandle bindingSet)
      : m_bindingSet(std::move(bindingSet)) {}
    ~DxvkBindingSetHolder() {
    }
  private:
    nvrhi::BindingSetHandle m_bindingSet;
  };

  // NVRHI ICommandList implementation using DxvkContext
  class NvrhiDxvkCommandList : public nvrhi::ICommandList {
  public:
    // Default scratch chunk size (256MB - large enough for most cluster operations)
    static constexpr uint64_t kDefaultScratchChunkSize = 256 * 1024 * 1024;

    NvrhiDxvkCommandList(
      NvrhiDxvkDevice* device,
      const Rc<DxvkContext>& context)
      : m_device(device)
      , m_context(context)
      , m_scratchManager(std::make_unique<ScratchManager>(device->getDxvkDevice().ptr(), kDefaultScratchChunkSize))
    {
    }

    ~NvrhiDxvkCommandList() {
      // Clean up HiZ pipeline layout if created
      if (m_hiZPipelineLayout != VK_NULL_HANDLE) {
        VkDevice vkDevice = m_device->getVkDevice();
        vkDestroyPipelineLayout(vkDevice, m_hiZPipelineLayout, nullptr);
        m_hiZPipelineLayout = VK_NULL_HANDLE;
      }
    }

    // ICommandList interface - Lifecycle
    void open();
    void close();
    void clearState();

    // ICommandList interface - Buffer operations
    void writeBuffer(
      nvrhi::IBuffer* buffer,
      const void* data,
      size_t size,
      uint64_t offset = 0);

    void clearBufferUInt(
      nvrhi::IBuffer* buffer,
      uint32_t value);

    void copyBuffer(
      nvrhi::IBuffer* dst,
      uint64_t dstOffset,
      nvrhi::IBuffer* src,
      uint64_t srcOffset,
      uint64_t size);

    // ICommandList interface - Texture operations
    void copyTexture(nvrhi::ITexture* dst, nvrhi::ITexture* src);
    void clearTextureFloat(nvrhi::ITexture* texture, const nvrhi::TextureSubresourceSet& subresources, const nvrhi::Color& clearColor);

    // ICommandList interface - Compute
    void setComputeState(const nvrhi::ComputeState& state);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void dispatchIndirect(nvrhi::IBuffer* buffer, uint64_t offset);
    void dispatchIndirect(uint64_t offset);  // Uses buffer from current compute state

    // ICommandList interface - Cluster operations (KEY FEATURE)
    void executeMultiIndirectClusterOperation(
      const nvrhi::rt::cluster::OperationDesc& desc);

    // ICommandList interface - Barriers
    void bufferBarrier(
      nvrhi::IBuffer* buffer,
      nvrhi::ResourceStates stateBefore,
      nvrhi::ResourceStates stateAfter);

    void textureBarrier(
      nvrhi::ITexture* texture,
      nvrhi::ResourceStates stateBefore,
      nvrhi::ResourceStates stateAfter);

    void globalBarrier(
      nvrhi::ResourceStates stateBefore,
      nvrhi::ResourceStates stateAfter);

    // Device access
    nvrhi::IDevice* getDevice() override {
      return m_device;
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      if (type == nvrhi::ObjectType::VK_CommandBuffer) {
        nvrhi::Object obj;
        obj.pointer = (void*)(uintptr_t)m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
        obj.type = nvrhi::ObjectType::VK_CommandBuffer;
        return obj;
      }
      return nvrhi::Object();
    }

  private:
    NvrhiDxvkDevice* m_device;
    Rc<DxvkContext> m_context;
    nvrhi::ComputeState m_computeState;
    std::unique_ptr<ScratchManager> m_scratchManager;

    // HiZ descriptor set binding for compute_cluster_tiling shader
    // This handles VK_BINDING(0, 1) Texture2D<float> t_HiZBuffer[HIZ_MAX_LODS]: register(t0, space1)
    bool m_hasHiZBinding = false;
    VkDescriptorSetLayout m_hiZDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_hiZPipelineLayout = VK_NULL_HANDLE;
    Rc<DxvkImageView> m_hiZImageViews[HIZ_MAX_LODS] = {};

    // UAV texture array binding for HiZ reduce shader
    // This handles RWTexture2D<float> u_output[HIZ_MAX_LODS]: register(u0)
    bool m_hasUAVArrayBinding = false;
    uint32_t m_uavArrayBinding = 0;  // Vulkan binding index for UAV array
    Rc<DxvkImageView> m_uavImageViews[HIZ_MAX_LODS] = {};

    // Pre-built descriptor sets to bind after commitComputeState()
    // This matches the NVRHI-Vulkan native approach: pre-build VkDescriptorSets in createBindingSet(),
    // then bind them with a single vkCmdBindDescriptorSets call at dispatch time.
    // We hold a reference to the binding set to keep the descriptor set alive until dispatch completes.
    struct PendingDescriptorSetBinding {
      VkDescriptorSet descriptorSet;
      uint32_t setIndex;
      VkPipelineLayout pipelineLayout;  // Pipeline layout to use when binding (from MegaGeo pipeline)
      nvrhi::BindingSetHandle bindingSetRef;  // Ref-counted handle to keep descriptor set alive
    };
    std::vector<PendingDescriptorSetBinding> m_pendingDescriptorSets;

    // Binding sets whose descriptor sets must stay alive until the GPU finishes
    // the command buffer that references them. We wrap each in a DxvkResource and
    // attach it to DXVK's command list lifetime tracker, which frees them when
    // the GPU fence signals completion - matching the sample's referencedResources pattern.
    std::vector<nvrhi::BindingSetHandle> m_pendingBindingSetsForTracking;

    void bindComputeResources(const nvrhi::ComputeState& state);
    void bindHiZDescriptorSet(VkPipelineLayout pipelineLayout);  // Binds HiZ descriptor set before dispatch
    void bindUAVArrayDescriptorSet();  // Binds UAV texture array before dispatch
    void trackPendingBindingSets();  // Transfer binding sets to DXVK's lifetime tracker
    void translateClusterOperation(
      const nvrhi::rt::cluster::OperationDesc& nvrhiDesc,
      VkClusterAccelerationStructureCommandsInfoNV& vkCmds,
      VkClusterAccelerationStructureClustersBottomLevelInputNV& blasInput,
      VkClusterAccelerationStructureTriangleClusterInputNV& clusterInput);

    // Automatic barrier system (matching sample's NVRHI behavior)
    bool m_EnableAutomaticBarriers = true;
    std::unordered_map<nvrhi::IBuffer*, nvrhi::ResourceStates> m_BufferStates;
    std::vector<BufferBarrier> m_PendingBufferBarriers;

    // Automatic barrier methods
    void requireBufferState(nvrhi::IBuffer* buffer, nvrhi::ResourceStates state);
    void setResourceStatesForBindingSet(nvrhi::IBindingSet* bindingSet);
    void commitBarriers();
  };

} // namespace dxvk
