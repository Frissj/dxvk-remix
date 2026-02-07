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

#include "nvrhi_types.h"
#include "../../dxvk_compute.h"
#include "../../dxvk_shader.h"

namespace dxvk {

  // NVRHI IBindingLayout implementation storing binding descriptions
  // When registerSpaceIsDescriptorSet is true, also holds a VkDescriptorSetLayout
  class NvrhiDxvkBindingLayout : public nvrhi::IBindingLayout {
  public:
    NvrhiDxvkBindingLayout(const nvrhi::BindingLayoutDesc& desc)
      : m_desc(desc)
      , m_vkDescriptorSetLayout(VK_NULL_HANDLE)
      , m_vkDevice(VK_NULL_HANDLE)
    {
    }

    ~NvrhiDxvkBindingLayout() {
      // Destroy the VkDescriptorSetLayout if we created one
      if (m_vkDescriptorSetLayout != VK_NULL_HANDLE && m_vkDevice != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayout, nullptr);
        m_vkDescriptorSetLayout = VK_NULL_HANDLE;
      }
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      // VkDescriptorSetLayout doesn't have a dedicated ObjectType, use getVkDescriptorSetLayout() instead
      return nvrhi::Object();
    }

    const nvrhi::BindingLayoutDesc& getDesc() const { return m_desc; }

    // Set the VkDescriptorSetLayout (called by NvrhiDxvkDevice when registerSpaceIsDescriptorSet is true)
    void setVkDescriptorSetLayout(VkDescriptorSetLayout layout, VkDevice device) {
      m_vkDescriptorSetLayout = layout;
      m_vkDevice = device;
    }

    VkDescriptorSetLayout getVkDescriptorSetLayout() const { return m_vkDescriptorSetLayout; }
    bool hasVkDescriptorSetLayout() const { return m_vkDescriptorSetLayout != VK_NULL_HANDLE; }

  private:
    nvrhi::BindingLayoutDesc m_desc;
    VkDescriptorSetLayout m_vkDescriptorSetLayout;
    VkDevice m_vkDevice;  // Needed for cleanup
  };

  // NVRHI IBindingSet implementation with pre-built VkDescriptorSet
  // Matches the native NVRHI-Vulkan approach: descriptor set is built once at creation,
  // then bound with a single vkCmdBindDescriptorSets call at dispatch time.
  class NvrhiDxvkBindingSet : public nvrhi::IBindingSet {
  public:
    NvrhiDxvkBindingSet(const nvrhi::BindingSetDesc& desc, nvrhi::IBindingLayout* layout)
      : m_desc(desc)
      , m_layout(layout)
      , m_descriptorPool(VK_NULL_HANDLE)
      , m_descriptorSet(VK_NULL_HANDLE)
      , m_vkDevice(VK_NULL_HANDLE)
      , m_ownsPool(false)
    {
    }

    ~NvrhiDxvkBindingSet() {
      if (m_vkDevice != VK_NULL_HANDLE && m_descriptorSet != VK_NULL_HANDLE) {
        if (m_ownsPool) {
          // If we own the pool, destroy it (which implicitly frees all sets)
          if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_vkDevice, m_descriptorPool, nullptr);
          }
        } else {
          // If using shared pool, just free the descriptor set back to the pool
          if (m_descriptorPool != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(m_vkDevice, m_descriptorPool, 1, &m_descriptorSet);
          }
        }
        m_descriptorPool = VK_NULL_HANDLE;
        m_descriptorSet = VK_NULL_HANDLE;
      }
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      return nvrhi::Object();
    }

    const nvrhi::BindingSetDesc& getDesc() const { return m_desc; }
    nvrhi::IBindingLayout* getLayout() const { return m_layout; }

    // Set the pre-built descriptor set (called by device during createBindingSet)
    // ownsPool: if true, this binding set owns the pool and will destroy it on destruction
    //           if false, pool is shared and we only free the descriptor set
    void setDescriptorSet(VkDescriptorPool pool, VkDescriptorSet set, VkDevice device, bool ownsPool = false) {
      m_descriptorPool = pool;
      m_descriptorSet = set;
      m_vkDevice = device;
      m_ownsPool = ownsPool;
    }

    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }
    bool hasDescriptorSet() const { return m_descriptorSet != VK_NULL_HANDLE; }

    // Track DxvkBuffers used in this binding set's descriptor set.
    // Ensures buffers stay alive until the GPU finishes using this binding set,
    // preventing use-after-free when ClusterAccelBuilder's deferred destruction
    // frees buffers before the GPU is done.
    void addTrackedBuffer(const Rc<DxvkBuffer>& buffer) {
      m_trackedBuffers.push_back(buffer);
    }

  private:
    nvrhi::BindingSetDesc m_desc;
    nvrhi::IBindingLayout* m_layout;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_descriptorSet;
    VkDevice m_vkDevice;  // Needed for cleanup
    bool m_ownsPool;      // Whether we own the pool or it's shared
    std::vector<Rc<DxvkBuffer>> m_trackedBuffers;  // Keep buffers alive for GPU lifetime
  };

  // NVRHI IComputePipeline implementation wrapping DXVK compute pipeline
  class NvrhiDxvkComputePipeline : public nvrhi::IComputePipeline {
  public:
    NvrhiDxvkComputePipeline(
      const Rc<DxvkShader>& shader,
      const nvrhi::ComputePipelineDesc& desc)
      : m_shader(shader)
      , m_desc(desc)
      , m_pipeline(VK_NULL_HANDLE)
      , m_pipelineLayout(VK_NULL_HANDLE)
      , m_descriptorSetLayout(VK_NULL_HANDLE)
      , m_vkDevice(VK_NULL_HANDLE)
    {
    }

    ~NvrhiDxvkComputePipeline() {
      if (m_vkDevice != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
          vkDestroyPipeline(m_vkDevice, m_pipeline, nullptr);
          m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
          vkDestroyPipelineLayout(m_vkDevice, m_pipelineLayout, nullptr);
          m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
          vkDestroyDescriptorSetLayout(m_vkDevice, m_descriptorSetLayout, nullptr);
          m_descriptorSetLayout = VK_NULL_HANDLE;
        }
      }
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      return nvrhi::Object();
    }

    const nvrhi::ComputePipelineDesc& getDesc() const { return m_desc; }
    const Rc<DxvkShader>& getShader() const { return m_shader; }

    // Native Vulkan pipeline and layout for MegaGeo shaders
    void setVkPipeline(VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSetLayout dsLayout, VkDevice device) {
      m_pipeline = pipeline;
      m_pipelineLayout = layout;
      m_descriptorSetLayout = dsLayout;
      m_vkDevice = device;
    }
    VkPipeline getVkPipeline() const { return m_pipeline; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }
    bool hasVkPipeline() const { return m_pipeline != VK_NULL_HANDLE; }

  private:
    Rc<DxvkShader> m_shader;
    nvrhi::ComputePipelineDesc m_desc;
    VkPipeline m_pipeline;
    VkPipelineLayout m_pipelineLayout;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDevice m_vkDevice;
  };

} // namespace dxvk
