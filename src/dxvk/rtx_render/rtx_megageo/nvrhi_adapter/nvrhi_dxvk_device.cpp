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
#include "nvrhi_dxvk_device.h"
#include "nvrhi_dxvk_command_list.h"
#include "nvrhi_dxvk_sampler.h"
#include "nvrhi_dxvk_texture.h"
#include "nvrhi_dxvk_pipeline.h"
#include "nvrhi_dxvk_shader.h"
#include "../../../util/log/log.h"
#include <unordered_map>

#include "../rtxmg_log.h"
#if RTXMG_LOG_NVRHI_DXVK_DEVICE
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace dxvk {

  nvrhi::BufferHandle NvrhiDxvkDevice::createBuffer(const nvrhi::BufferDesc& desc) {
    // Translate NVRHI buffer descriptor to DXVK buffer create info
    DxvkBufferCreateInfo dxvkInfo;
    dxvkInfo.usage = 0;
    dxvkInfo.stages = 0;
    dxvkInfo.access = 0;

    // For volatile constant buffers: allocate byteSize * maxVersions with alignment
    uint64_t perVersionSize = desc.byteSize;
    uint32_t maxVersions = desc.maxVersions;
    if (desc.isVolatile && maxVersions > 0) {
      // Align per-version size to 256 bytes (minUniformBufferOffsetAlignment is typically 256)
      perVersionSize = (desc.byteSize + 255) & ~255ull;
      dxvkInfo.size = perVersionSize * maxVersions;
    } else {
      dxvkInfo.size = desc.byteSize;
    }

    // Translate usage flags
    if (desc.isConstantBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dxvkInfo.access |= VK_ACCESS_UNIFORM_READ_BIT;
    }

    if (desc.canHaveUAVs || desc.canHaveRawViews || desc.canHaveTypedViews) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dxvkInfo.access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (desc.isVertexBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }

    if (desc.isIndexBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }

    if (desc.isDrawIndirectArgs) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      dxvkInfo.access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }

    if (desc.isAccelStructBuildInput) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      dxvkInfo.access |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    }

    if (desc.isAccelStructStorage) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      dxvkInfo.access |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    }

    // All buffers used in RTX MG need device address for cluster operations
    dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    dxvkInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Default stages if none set
    if (dxvkInfo.stages == 0) {
      dxvkInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }

    // Default access if none set
    if (dxvkInfo.access == 0) {
      dxvkInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }

    // Create DXVK buffer (GPU local memory by default)
    // Note: Debug name is set during buffer creation, not after
    const char* debugName = desc.debugName ? desc.debugName : "NVRHI Buffer";

    // DEBUG: Log buffer creation details
    RTXMG_LOG(str::format("RTX MegaGeo: Creating buffer '", debugName, "' size=", desc.byteSize,
                             " usage=0x", std::hex, dxvkInfo.usage, std::dec));

    // Determine memory type based on buffer purpose
    VkMemoryPropertyFlags memoryFlags;

    if (desc.isVolatile && desc.maxVersions > 0) {
      // Volatile constant buffers: host-visible for CPU memcpy writes (no GPU commands)
      memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else if (desc.cpuAccess == nvrhi::CpuAccessMode::Read ||
        desc.cpuAccess == nvrhi::CpuAccessMode::Write ||
        desc.cpuAccess == nvrhi::CpuAccessMode::ReadWrite ||
        desc.initialState == nvrhi::ResourceStates::CopyDest) {
      memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
      memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    Rc<DxvkBuffer> dxvkBuffer = m_device->createBuffer(
      dxvkInfo,
      memoryFlags,
      DxvkMemoryStats::Category::RTXAccelerationStructure,
      debugName);

    if (dxvkBuffer == nullptr) {
      Logger::err(str::format("Failed to create buffer of size ", desc.byteSize));
      return nullptr;
    }

    // DEBUG: Log device address and slice offset after creation
    VkDeviceAddress addr = dxvkBuffer->getDeviceAddress();
    DxvkBufferSliceHandle sliceHandle = dxvkBuffer->getSliceHandle();
    RTXMG_LOG(str::format("RTX MegaGeo: Created buffer '", debugName, "' with device address: ", addr));

    // Check alignment of the physical slice offset
    if (sliceHandle.offset % 16 != 0) {
      Logger::err(str::format("RTX MegaGeo: WARNING - Buffer '", debugName,
        "' created with non-16-byte-aligned slice offset: ", sliceHandle.offset,
        " (mod 16 = ", sliceHandle.offset % 16, ")"));
    }

    NvrhiDxvkBuffer* result = new NvrhiDxvkBuffer(desc, dxvkBuffer);

    if (desc.isVolatile && maxVersions > 0) {
      result->setVolatile(maxVersions, perVersionSize);
      ONCE(RTXMG_LOG(str::format("RTX MegaGeo: Created volatile CB '",
        desc.debugName ? desc.debugName : "unnamed",
        "' perVersion=", perVersionSize, " maxVersions=", maxVersions,
        " totalSize=", perVersionSize * maxVersions)));
    }

    return result;
  }

  void* NvrhiDxvkDevice::mapBuffer(nvrhi::IBuffer* buffer, nvrhi::CpuAccessMode access) {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Map the entire buffer
    DxvkBufferSliceHandle slice = dxvkBuffer->getSliceHandle();
    return dxvkBuffer->mapPtr(0);
  }

  void NvrhiDxvkDevice::unmapBuffer(nvrhi::IBuffer* buffer) {
    // DXVK handles unmapping automatically via DxvkStagingDataAlloc
    // No explicit unmap needed
  }

  nvrhi::TextureHandle NvrhiDxvkDevice::createTexture(const nvrhi::TextureDesc& desc) {
    // Translate NVRHI texture descriptor to DXVK image create info
    DxvkImageCreateInfo dxvkInfo;

    dxvkInfo.type = VK_IMAGE_TYPE_2D;
    dxvkInfo.format = static_cast<VkFormat>(desc.format);
    dxvkInfo.flags = 0;
    dxvkInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    dxvkInfo.extent = { desc.width, desc.height, desc.depth };
    dxvkInfo.numLayers = desc.arraySize;
    dxvkInfo.mipLevels = desc.mipLevels;
    dxvkInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dxvkInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dxvkInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    dxvkInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

    if (desc.isUAV) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      dxvkInfo.access |= VK_ACCESS_SHADER_WRITE_BIT;
      // UAV images need GENERAL layout for compute storage access
      dxvkInfo.layout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
      // Non-UAV images use SHADER_READ_ONLY for sampled access
      // This is the "stable" layout DXVK will use for layout tracking
      dxvkInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (desc.isRenderTarget) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      dxvkInfo.access |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }

    // Check format to determine if depth/stencil
    bool isDepth = (dxvkInfo.format >= VK_FORMAT_D16_UNORM && dxvkInfo.format <= VK_FORMAT_D32_SFLOAT_S8_UINT);
    if (isDepth) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      dxvkInfo.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }

    // Create DXVK image
    Rc<DxvkImage> dxvkImage = m_device->createImage(
      dxvkInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXMaterialTexture,
      desc.debugName ? desc.debugName : "NVRHI Texture");

    if (dxvkImage == nullptr) {
      Logger::err(str::format("Failed to create texture: ", desc.debugName ? desc.debugName : "unnamed"));
      return nullptr;
    }

    return new NvrhiDxvkTexture(desc, dxvkImage);
  }

  nvrhi::SamplerHandle NvrhiDxvkDevice::createSampler(const nvrhi::SamplerDesc& desc) {
    // Translate NVRHI sampler descriptor to DXVK sampler create info
    DxvkSamplerCreateInfo dxvkInfo;

    // Filter modes
    dxvkInfo.minFilter = desc.minFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    dxvkInfo.magFilter = desc.magFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    dxvkInfo.mipmapMode = desc.mipFilter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;

    // Address modes
    auto translateAddressMode = [](nvrhi::SamplerAddressMode mode) -> VkSamplerAddressMode {
      switch (mode) {
        case nvrhi::SamplerAddressMode::Clamp:       return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case nvrhi::SamplerAddressMode::Wrap:        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case nvrhi::SamplerAddressMode::Border:      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case nvrhi::SamplerAddressMode::Mirror:      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case nvrhi::SamplerAddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case nvrhi::SamplerAddressMode::MirrorOnce:  return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        default:                                      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      }
    };

    dxvkInfo.addressModeU = translateAddressMode(desc.addressU);
    dxvkInfo.addressModeV = translateAddressMode(desc.addressV);
    dxvkInfo.addressModeW = translateAddressMode(desc.addressW);

    // Mip LOD bias and range
    dxvkInfo.mipmapLodBias = desc.mipBias;
    dxvkInfo.mipmapLodMin = 0.0f;
    dxvkInfo.mipmapLodMax = VK_LOD_CLAMP_NONE;

    // Anisotropy
    dxvkInfo.maxAnisotropy = desc.maxAnisotropy;
    dxvkInfo.useAnisotropy = (desc.maxAnisotropy > 1.0f) ? VK_TRUE : VK_FALSE;

    // Comparison/reduction type
    // Note: DXVK doesn't support sampler reduction modes, only comparison for depth
    switch (desc.reductionType) {
      case nvrhi::SamplerReductionType::Comparison:
        dxvkInfo.compareToDepth = VK_TRUE;
        dxvkInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        break;
      case nvrhi::SamplerReductionType::Minimum:
      case nvrhi::SamplerReductionType::Maximum:
      case nvrhi::SamplerReductionType::Standard:
      default:
        dxvkInfo.compareToDepth = VK_FALSE;
        dxvkInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        break;
    }

    // Border color (stored as VkClearColorValue union)
    if (desc.borderColor) {
      dxvkInfo.borderColor.float32[0] = 1.0f;
      dxvkInfo.borderColor.float32[1] = 1.0f;
      dxvkInfo.borderColor.float32[2] = 1.0f;
      dxvkInfo.borderColor.float32[3] = 1.0f;
    } else {
      dxvkInfo.borderColor.float32[0] = 0.0f;
      dxvkInfo.borderColor.float32[1] = 0.0f;
      dxvkInfo.borderColor.float32[2] = 0.0f;
      dxvkInfo.borderColor.float32[3] = 0.0f;
    }

    // Unnormalized coordinates - always use normalized for standard samplers
    dxvkInfo.usePixelCoord = VK_FALSE;

    // Create DXVK sampler
    Rc<DxvkSampler> dxvkSampler = m_device->createSampler(dxvkInfo);

    if (dxvkSampler == nullptr) {
      Logger::err("Failed to create sampler");
      return nullptr;
    }

    return new NvrhiDxvkSampler(desc, dxvkSampler);
  }

  nvrhi::ShaderHandle NvrhiDxvkDevice::createShader(
    const nvrhi::ShaderDesc& desc,
    const void* binary,
    size_t size)
  {
    // RTX MG shaders will be pre-compiled to SPIR-V and loaded via RTX Remix's shader system
    // This is a stub - actual shader loading happens in ClusterAccelBuilder via RtxShaderManager
    Logger::warn("NvrhiDxvkDevice::createShader() called - shaders should be loaded via RtxShaderManager");
    return nullptr;
  }

  nvrhi::ComputePipelineHandle NvrhiDxvkDevice::createComputePipeline(
    const nvrhi::ComputePipelineDesc& desc)
  {
    // Extract the DXVK shader from the NVRHI shader wrapper
    if (!desc.computeShader) {
      Logger::err("createComputePipeline: compute shader is null");
      return nullptr;
    }

    auto* shaderWrapper = static_cast<NvrhiDxvkShader*>(desc.computeShader.Get());
    if (!shaderWrapper) {
      Logger::err("createComputePipeline: failed to cast shader");
      return nullptr;
    }

    const Rc<DxvkShader>& dxvkShader = shaderWrapper->getDxvkShader();
    if (dxvkShader == nullptr) {
      Logger::err("createComputePipeline: DXVK shader is null");
      return nullptr;
    }

    // Create compute pipeline wrapper
    auto* pipeline = new NvrhiDxvkComputePipeline(dxvkShader, desc);

    // Create native VkPipeline with prebuilt descriptor set layouts (matching the sample)
    // This is the key optimization: instead of using DXVK's per-binding approach, we create
    // a pipeline with descriptor set layouts that match our prebuilt descriptor sets.
    // At dispatch time, we bind the pipeline and descriptor sets with single Vulkan calls.

    // Collect VkDescriptorSetLayouts from binding layouts
    std::vector<VkDescriptorSetLayout> setLayouts;
    for (const auto& bindingLayout : desc.bindingLayouts) {
      if (!bindingLayout) continue;

      auto* nvrhiLayout = static_cast<NvrhiDxvkBindingLayout*>(bindingLayout.Get());
      if (nvrhiLayout && nvrhiLayout->hasVkDescriptorSetLayout()) {
        setLayouts.push_back(nvrhiLayout->getVkDescriptorSetLayout());
      }
    }

    if (!setLayouts.empty()) {
      // Create pipeline layout from descriptor set layouts
      VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
      pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
      pipelineLayoutInfo.pSetLayouts = setLayouts.data();

      // Check for push constants in binding layouts
      // Also check for constant buffers - Slang may compile small CBs to push constants
      VkPushConstantRange pushConstantRange = {};
      bool hasPushConstants = false;
      bool hasConstantBuffer = false;
      for (const auto& bindingLayout : desc.bindingLayouts) {
        if (!bindingLayout) continue;
        auto* nvrhiLayout = static_cast<NvrhiDxvkBindingLayout*>(bindingLayout.Get());
        if (!nvrhiLayout) continue;

        for (const auto& item : nvrhiLayout->getDesc().bindings) {
          if (item.resourceType == nvrhi::BindingLayoutItem::ResourceType::PushConstants) {
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = item.size;
            hasPushConstants = true;
            break;
          }
          if (item.resourceType == nvrhi::BindingLayoutItem::ResourceType::ConstantBuffer) {
            hasConstantBuffer = true;
          }
        }
      }

      // If there's a constant buffer but no explicit push constants, add a default push constant range
      // This handles the case where Slang compiles the CB to push constants in SPIR-V
      if (!hasPushConstants && hasConstantBuffer) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = 128;  // Default size to cover most small CBs
        hasPushConstants = true;
      }

      if (hasPushConstants) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
      }

      VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
      VkResult result = vkCreatePipelineLayout(m_vkDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout);
      if (result != VK_SUCCESS) {
        Logger::err(str::format("RTX MegaGeo: createComputePipeline - failed to create pipeline layout, result=", (int)result));
        return pipeline;  // Return pipeline without native VkPipeline, will use DXVK fallback
      }

      // Get SPIRV from the DXVK shader
      SpirvCodeBuffer spirvCode = dxvkShader->getCode();
      if (spirvCode.size() == 0) {
        Logger::err("RTX MegaGeo: createComputePipeline - shader has no SPIRV code");
        vkDestroyPipelineLayout(m_vkDevice, pipelineLayout, nullptr);
        return pipeline;
      }

      // Create shader module
      VkShaderModuleCreateInfo shaderModuleInfo = {};
      shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shaderModuleInfo.codeSize = spirvCode.size();
      shaderModuleInfo.pCode = spirvCode.data();

      VkShaderModule shaderModule = VK_NULL_HANDLE;
      result = vkCreateShaderModule(m_vkDevice, &shaderModuleInfo, nullptr, &shaderModule);
      if (result != VK_SUCCESS) {
        Logger::err(str::format("RTX MegaGeo: createComputePipeline - failed to create shader module, result=", (int)result));
        vkDestroyPipelineLayout(m_vkDevice, pipelineLayout, nullptr);
        return pipeline;
      }

      // Create compute pipeline
      VkComputePipelineCreateInfo pipelineInfo = {};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipelineInfo.stage.module = shaderModule;
      pipelineInfo.stage.pName = "main";
      pipelineInfo.layout = pipelineLayout;

      VkPipeline vkPipeline = VK_NULL_HANDLE;
      result = vkCreateComputePipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vkPipeline);

      // Shader module can be destroyed after pipeline creation
      vkDestroyShaderModule(m_vkDevice, shaderModule, nullptr);

      if (result != VK_SUCCESS) {
        Logger::err(str::format("RTX MegaGeo: createComputePipeline - failed to create pipeline, result=", (int)result));
        vkDestroyPipelineLayout(m_vkDevice, pipelineLayout, nullptr);
        return pipeline;
      }

      // Store in pipeline wrapper - use first layout for descriptor set layout reference
      pipeline->setVkPipeline(vkPipeline, pipelineLayout, setLayouts[0], m_vkDevice);

      RTXMG_LOG(str::format("RTX MegaGeo: Created native VkPipeline ", (void*)vkPipeline,
        " with ", setLayouts.size(), " descriptor set layouts"));
    }

    return pipeline;
  }

  // DXVK binding offset convention for HLSL register types:
  // - SRV (t registers): offset 0
  // - Sampler (s registers): offset 100
  // - UAV (u registers): offset 200
  // - CB (b registers): offset 300
  static constexpr uint32_t kSrvBindingOffset = 0;
  static constexpr uint32_t kSamplerBindingOffset = 100;
  static constexpr uint32_t kUavBindingOffset = 200;
  static constexpr uint32_t kCbBindingOffset = 300;

  static uint32_t getBindingOffsetForLayoutType(nvrhi::BindingLayoutItem::ResourceType type) {
    switch (type) {
      case nvrhi::BindingLayoutItem::ResourceType::StructuredBuffer_SRV:
      case nvrhi::BindingLayoutItem::ResourceType::RawBuffer_SRV:
      case nvrhi::BindingLayoutItem::ResourceType::TypedBuffer_SRV:
      case nvrhi::BindingLayoutItem::ResourceType::Texture_SRV:
        return kSrvBindingOffset;
      case nvrhi::BindingLayoutItem::ResourceType::Sampler:
        return kSamplerBindingOffset;
      case nvrhi::BindingLayoutItem::ResourceType::StructuredBuffer_UAV:
      case nvrhi::BindingLayoutItem::ResourceType::RawBuffer_UAV:
      case nvrhi::BindingLayoutItem::ResourceType::TypedBuffer_UAV:
      case nvrhi::BindingLayoutItem::ResourceType::Texture_UAV:
        return kUavBindingOffset;
      case nvrhi::BindingLayoutItem::ResourceType::ConstantBuffer:
        return kCbBindingOffset;
      default:
        return 0;
    }
  }

  static uint32_t getBindingOffsetForSetType(nvrhi::BindingSetItem::Type type) {
    switch (type) {
      case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
      case nvrhi::BindingSetItem::Type::RawBuffer_SRV:
      case nvrhi::BindingSetItem::Type::TypedBuffer_SRV:
      case nvrhi::BindingSetItem::Type::Texture_SRV:
        return kSrvBindingOffset;
      case nvrhi::BindingSetItem::Type::Sampler:
        return kSamplerBindingOffset;
      case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
      case nvrhi::BindingSetItem::Type::RawBuffer_UAV:
      case nvrhi::BindingSetItem::Type::TypedBuffer_UAV:
      case nvrhi::BindingSetItem::Type::Texture_UAV:
        return kUavBindingOffset;
      case nvrhi::BindingSetItem::Type::ConstantBuffer:
        return kCbBindingOffset;
      default:
        return 0;
    }
  }

  nvrhi::BindingLayoutHandle NvrhiDxvkDevice::createBindingLayout(
    const nvrhi::BindingLayoutDesc& desc)
  {
    RTXMG_LOG(str::format("RTX MegaGeo: createBindingLayout space=", desc.registerSpace,
      " registerSpaceIsDescriptorSet=", desc.registerSpaceIsDescriptorSet ? "true" : "false",
      " bindings=", desc.bindings.size()));

    // Create binding layout wrapper
    auto* layout = new NvrhiDxvkBindingLayout(desc);

    // ALWAYS create a VkDescriptorSetLayout for prebuilt descriptor sets (matching the sample)
    // This is a major optimization - instead of per-binding calls, we create descriptor sets once
    // and bind them with a single vkCmdBindDescriptorSets call.
    //
    // Binding number assignment uses DXVK's binding offset convention:
    // - SRV (t registers): binding = slot
    // - Sampler (s registers): binding = 100 + slot
    // - UAV (u registers): binding = 200 + slot
    // - CB (b registers): binding = 300 + slot
    // For registerSpaceIsDescriptorSet layouts: use slot directly (for HiZ textures in space 1)

    std::vector<VkDescriptorSetLayoutBinding> bindings;

    for (const auto& item : desc.bindings) {
      VkDescriptorSetLayoutBinding binding = {};

      // For registerSpaceIsDescriptorSet, use the slot directly
      // Otherwise, use binding offset based on resource type
      if (desc.registerSpaceIsDescriptorSet) {
        binding.binding = item.slot;
      } else {
        binding.binding = getBindingOffsetForLayoutType(item.resourceType) + item.slot;
      }
      binding.descriptorCount = item.size > 0 ? item.size : 1;
      binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      switch (item.resourceType) {
        case nvrhi::BindingLayoutItem::ResourceType::StructuredBuffer_SRV:
        case nvrhi::BindingLayoutItem::ResourceType::RawBuffer_SRV:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::StructuredBuffer_UAV:
        case nvrhi::BindingLayoutItem::ResourceType::RawBuffer_UAV:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::Texture_SRV:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::Texture_UAV:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::Sampler:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::ConstantBuffer:
          binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
          break;
        case nvrhi::BindingLayoutItem::ResourceType::PushConstants:
          // Push constants don't need descriptor bindings
          continue;
        default:
          Logger::warn(str::format("RTX MegaGeo: Unknown binding type ", (int)item.resourceType));
          continue;
      }

      bindings.push_back(binding);
      RTXMG_LOG(str::format("RTX MegaGeo:   binding=", binding.binding, " type=", binding.descriptorType,
        " count=", binding.descriptorCount, " (slot=", item.slot, ")"));
    }

    if (!bindings.empty()) {
      VkDescriptorSetLayoutCreateInfo layoutInfo = {};
      layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
      layoutInfo.pBindings = bindings.data();

      VkDescriptorSetLayout vkLayout = VK_NULL_HANDLE;
      VkResult result = vkCreateDescriptorSetLayout(m_vkDevice, &layoutInfo, nullptr, &vkLayout);
      if (result == VK_SUCCESS && vkLayout != VK_NULL_HANDLE) {
        layout->setVkDescriptorSetLayout(vkLayout, m_vkDevice);
        RTXMG_LOG(str::format("RTX MegaGeo: Created VkDescriptorSetLayout ", (void*)vkLayout,
          " for space ", desc.registerSpace, " with ", bindings.size(), " bindings"));
      } else {
        Logger::err(str::format("RTX MegaGeo: Failed to create VkDescriptorSetLayout, result=", (int)result));
      }
    }

    return layout;
  }

  nvrhi::BindlessLayoutHandle NvrhiDxvkDevice::createBindlessLayout(
    const nvrhi::BindlessLayoutDesc& desc)
  {
    // Create a binding layout that represents the bindless descriptor space.
    // In RTX Remix, bindless resources are managed by the runtime, but we still
    // need a valid layout object so pipelines can be created correctly.
    // The actual bindless descriptors will be provided via createDescriptorTable().
    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = desc.visibility;
    // No explicit bindings - this is a bindless layout
    return new NvrhiDxvkBindingLayout(layoutDesc);
  }

  nvrhi::BindingSetHandle NvrhiDxvkDevice::createDescriptorTable(
    nvrhi::IBindingLayout* layout)
  {
    // Create an empty binding set for the bindless layout.
    // This serves as a placeholder descriptor table that satisfies the pipeline's
    // binding requirements. When displacement maps are enabled, this would need
    // to be populated with actual texture descriptors.
    nvrhi::BindingSetDesc emptyDesc;
    return new NvrhiDxvkBindingSet(emptyDesc, layout);
  }

  nvrhi::BindingSetHandle NvrhiDxvkDevice::createBindingSet(
    const nvrhi::BindingSetDesc& desc,
    nvrhi::IBindingLayout* layout)
  {
    // Create binding set wrapper
    auto* bindingSet = new NvrhiDxvkBindingSet(desc, layout);

    // PREBUILT DESCRIPTOR SET APPROACH (matching the sample)
    // Create a VkDescriptorSet at binding set creation time, then at dispatch we just call
    // vkCmdBindDescriptorSets - this is much faster than per-binding calls.
    //
    // The layout must have a VkDescriptorSetLayout (created in createBindingLayout).
    // The pipeline must also be created with matching descriptor set layouts (in createComputePipeline).
    auto* nvrhiLayout = static_cast<NvrhiDxvkBindingLayout*>(layout);
    if (!nvrhiLayout) {
      RTXMG_LOG("RTX MegaGeo: createBindingSet - null layout, returning without descriptor set");
      return bindingSet;
    }

    const nvrhi::BindingLayoutDesc& layoutDesc = nvrhiLayout->getDesc();

    // Only create prebuilt descriptor sets if the layout has a VkDescriptorSetLayout
    if (!nvrhiLayout->hasVkDescriptorSetLayout()) {
      RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - layout space=", layoutDesc.registerSpace,
        " has no VkDescriptorSetLayout, skipping prebuilt descriptor set"));
      return bindingSet;
    }

    VkDescriptorSetLayout descriptorSetLayout = nvrhiLayout->getVkDescriptorSetLayout();

    RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - creating prebuilt descriptor set for space ",
      layoutDesc.registerSpace, " with ", desc.bindings.size(), " bindings"));

    // Count descriptors by type for pool creation (from binding SET, not layout)
    std::unordered_map<VkDescriptorType, uint32_t> poolSizeMap;
    for (const auto& item : desc.bindings) {
      VkDescriptorType descType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      switch (item.type) {
        case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
        case nvrhi::BindingSetItem::Type::RawBuffer_SRV:
        case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
        case nvrhi::BindingSetItem::Type::RawBuffer_UAV:
          descType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          break;
        case nvrhi::BindingSetItem::Type::TypedBuffer_SRV:
          descType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
          break;
        case nvrhi::BindingSetItem::Type::TypedBuffer_UAV:
          descType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
          break;
        case nvrhi::BindingSetItem::Type::Texture_SRV:
          descType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          break;
        case nvrhi::BindingSetItem::Type::Texture_UAV:
          descType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          break;
        case nvrhi::BindingSetItem::Type::Sampler:
          descType = VK_DESCRIPTOR_TYPE_SAMPLER;
          break;
        case nvrhi::BindingSetItem::Type::ConstantBuffer:
          descType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
          break;
        default:
          continue;
      }
      poolSizeMap[descType]++;
    }

    if (poolSizeMap.empty()) {
      RTXMG_LOG("RTX MegaGeo: createBindingSet - no valid bindings, returning without descriptor set");
      return bindingSet;
    }

    // Create a per-binding-set descriptor pool for thread-safe destruction.
    // DXVK's lifetime tracker destroys binding sets on its retirement thread,
    // so sharing a pool would cause threading errors when the recording thread
    // allocates while the retirement thread frees from the same pool.
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto& [type, count] : poolSizeMap) {
      poolSizes.push_back({ type, count });
    }

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkResult poolResult = vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &descriptorPool);
    if (poolResult != VK_SUCCESS || descriptorPool == VK_NULL_HANDLE) {
      Logger::err(str::format("RTX MegaGeo: createBindingSet - failed to create descriptor pool, result=", (int)poolResult));
      return bindingSet;
    }

    // Allocate descriptor set from the layout's VkDescriptorSetLayout
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkResult result = vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &descriptorSet);
    if (result != VK_SUCCESS) {
      // Don't destroy shared pool on failure - just log and return
      Logger::err(str::format("RTX MegaGeo: createBindingSet - failed to allocate descriptor set, result=", (int)result));
      return bindingSet;
    }

    // Write descriptors - use binding numbers that match how the layout was created:
    // - registerSpaceIsDescriptorSet: use slot directly (matches shader)
    // Write descriptors - binding numbers must match the layout:
    // - For registerSpaceIsDescriptorSet: use slot directly
    // - Otherwise: use binding offset based on resource type (matching createBindingLayout)
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<Rc<DxvkImageView>> imageViews;  // Keep image views alive until vkUpdateDescriptorSets

    bufferInfos.reserve(desc.bindings.size());
    imageInfos.reserve(desc.bindings.size());
    writes.reserve(desc.bindings.size());
    imageViews.reserve(desc.bindings.size());

    for (const auto& item : desc.bindings) {
      if (item.resourceHandle == nullptr) continue;

      VkWriteDescriptorSet write = {};
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.dstSet = descriptorSet;

      // Calculate binding number using the same logic as createBindingLayout
      if (layoutDesc.registerSpaceIsDescriptorSet) {
        write.dstBinding = item.slot;
      } else {
        write.dstBinding = getBindingOffsetForSetType(item.type) + item.slot;
      }

      write.dstArrayElement = item.arrayElement;
      write.descriptorCount = 1;

      switch (item.type) {
        case nvrhi::BindingSetItem::Type::ConstantBuffer:
        case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
        case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
        {
          auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
          Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

          uint64_t bufferSize = dxvkBuffer->info().size;
          uint64_t offset = std::min(item.range.byteOffset, bufferSize);
          uint64_t size = (item.range.byteSize > 0)
            ? std::min(item.range.byteSize, bufferSize - offset)
            : bufferSize - offset;

          DxvkBufferSliceHandle sliceHandle = dxvkBuffer->getSliceHandle();

          VkDescriptorBufferInfo bufferInfo = {};
          bufferInfo.buffer = sliceHandle.handle;
          bufferInfo.offset = sliceHandle.offset + offset;
          bufferInfo.range = size;
          bufferInfos.push_back(bufferInfo);

          // CB → UNIFORM_BUFFER; SRV/UAV → STORAGE_BUFFER (must match createBindingLayout)
          write.descriptorType = (item.type == nvrhi::BindingSetItem::Type::ConstantBuffer)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          write.pBufferInfo = &bufferInfos.back();
          writes.push_back(write);

          bindingSet->addTrackedBuffer(dxvkBuffer);

          RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - buffer binding=", item.slot,
            " type=", (int)write.descriptorType, " size=", size,
            " deviceAddr=0x", std::hex, dxvkBuffer->getDeviceAddress(), std::dec,
            " VkBuffer=0x", std::hex, (uint64_t)sliceHandle.handle, std::dec));
          break;
        }
        case nvrhi::BindingSetItem::Type::Sampler:
        {
          auto* nvrhiSampler = static_cast<NvrhiDxvkSampler*>(item.resourceHandle);
          Rc<DxvkSampler> dxvkSampler = nvrhiSampler->getDxvkSampler();

          VkDescriptorImageInfo imageInfo = {};
          imageInfo.sampler = dxvkSampler->handle();
          imageInfos.push_back(imageInfo);

          write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
          write.pImageInfo = &imageInfos.back();
          writes.push_back(write);

          RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - sampler binding=", item.slot));
          break;
        }
        case nvrhi::BindingSetItem::Type::Texture_SRV:
        {
          // Create image view and write descriptor for texture SRV
          nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
          if (nativeObj.pointer == nullptr) {
            Logger::warn(str::format("RTX MegaGeo: createBindingSet - Texture_SRV slot=", item.slot,
              " is not a VK_Image, skipping"));
            break;
          }

          auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
          const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
          if (dxvkImage == nullptr) {
            Logger::warn(str::format("RTX MegaGeo: createBindingSet - Texture_SRV slot=", item.slot,
              " has null DxvkImage, skipping"));
            break;
          }

          // Create image view
          DxvkImageViewCreateInfo viewInfo;
          viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
          viewInfo.format = dxvkImage->info().format;
          viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

          // Handle depth formats
          VkFormat fmt = dxvkImage->info().format;
          if (fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D16_UNORM ||
              fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            viewInfo.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
          } else {
            viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
          }
          viewInfo.minLevel = 0;
          viewInfo.numLevels = dxvkImage->info().mipLevels;
          viewInfo.minLayer = 0;
          viewInfo.numLayers = 1;

          Rc<DxvkImageView> imageView = m_device->createImageView(dxvkImage, viewInfo);
          if (imageView != nullptr) {
            imageViews.push_back(imageView);  // Keep alive until vkUpdateDescriptorSets

            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageView = imageView->handle();
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos.push_back(imageInfo);

            write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.pImageInfo = &imageInfos.back();
            writes.push_back(write);

            RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - texture SRV binding=", item.slot,
              " array=", item.arrayElement));
          }
          break;
        }
        case nvrhi::BindingSetItem::Type::Texture_UAV:
        {
          // Create image view and write descriptor for texture UAV
          nvrhi::Object nativeObj = item.resourceHandle->getNativeObject(nvrhi::ObjectType::VK_Image);
          if (nativeObj.pointer == nullptr) {
            Logger::warn(str::format("RTX MegaGeo: createBindingSet - Texture_UAV slot=", item.slot,
              " is not a VK_Image, skipping"));
            break;
          }

          auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
          const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();
          if (dxvkImage == nullptr) {
            Logger::warn(str::format("RTX MegaGeo: createBindingSet - Texture_UAV slot=", item.slot,
              " has null DxvkImage, skipping"));
            break;
          }

          // Create image view for storage
          DxvkImageViewCreateInfo viewInfo;
          viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
          viewInfo.format = dxvkImage->info().format;
          viewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
          viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
          viewInfo.minLevel = 0;
          viewInfo.numLevels = 1;
          viewInfo.minLayer = 0;
          viewInfo.numLayers = 1;

          Rc<DxvkImageView> imageView = m_device->createImageView(dxvkImage, viewInfo);
          if (imageView != nullptr) {
            imageViews.push_back(imageView);  // Keep alive until vkUpdateDescriptorSets

            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageView = imageView->handle();
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageInfos.push_back(imageInfo);

            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfos.back();
            writes.push_back(write);

            RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - texture UAV binding=", item.slot,
              " array=", item.arrayElement));
          }
          break;
        }
        default:
          break;
      }
    }

    // Update descriptor set with all writes
    if (!writes.empty()) {
      vkUpdateDescriptorSets(m_vkDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // Store in binding set (owns pool - pool is destroyed when binding set is destroyed, thread-safe)
    bindingSet->setDescriptorSet(descriptorPool, descriptorSet, m_vkDevice, true);

    RTXMG_LOG(str::format("RTX MegaGeo: createBindingSet - created pre-built descriptor set with ",
      writes.size(), " descriptors for space ", layoutDesc.registerSpace));

    return bindingSet;
  }

  nvrhi::rt::cluster::OperationSizeInfo NvrhiDxvkDevice::getClusterOperationSizeInfo(
    const nvrhi::rt::cluster::OperationParams& params)
  {
    nvrhi::rt::cluster::OperationSizeInfo result = {};

    // Load the Vulkan function pointer for querying cluster AS sizes
    typedef void (VKAPI_PTR *PFN_vkGetClusterAccelerationStructureBuildSizesNV)(
      VkDevice device,
      const VkClusterAccelerationStructureInputInfoNV* pInfo,
      VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);

    static PFN_vkGetClusterAccelerationStructureBuildSizesNV vkGetClusterASSizes = nullptr;
    if (!vkGetClusterASSizes) {
      vkGetClusterASSizes = reinterpret_cast<PFN_vkGetClusterAccelerationStructureBuildSizesNV>(
        m_device->vkd()->sym("vkGetClusterAccelerationStructureBuildSizesNV"));
      if (vkGetClusterASSizes) {
        RTXMG_LOG("RTX MegaGeo: Loaded vkGetClusterAccelerationStructureBuildSizesNV");
      }
    }

    if (!vkGetClusterASSizes) {
      Logger::err("RTX MegaGeo: vkGetClusterAccelerationStructureBuildSizesNV not available - using fallback estimates");
      // Fallback to conservative estimates
      result.resultMaxSizeInBytes = 256 * 1024 * 1024; // 256 MB
      result.scratchSizeInBytes = 256 * 1024 * 1024;   // 256 MB
      return result;
    }

    // Build the input info structure
    VkClusterAccelerationStructureInputInfoNV inputInfo = {};
    inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    inputInfo.maxAccelerationStructureCount = params.maxArgCount;

    // Map operation type
    switch (params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        break;
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        break;
      default:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
        break;
    }

    // Map operation mode
    switch (params.mode) {
      case nvrhi::rt::cluster::OperationMode::GetSizes:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        break;
      case nvrhi::rt::cluster::OperationMode::ExplicitDestinations:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        break;
      case nvrhi::rt::cluster::OperationMode::ImplicitDestinations:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        break;
    }

    // Set up operation-specific input structures
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput = {};
    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {};

    switch (params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        triangleInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
        triangleInput.vertexFormat = static_cast<VkFormat>(params.clas.vertexFormat);
        triangleInput.maxGeometryIndexValue = params.clas.maxGeometryIndex;
        triangleInput.maxClusterUniqueGeometryCount = params.clas.maxUniqueGeometryCount;
        triangleInput.maxClusterTriangleCount = params.clas.maxTriangleCount;
        triangleInput.maxClusterVertexCount = params.clas.maxVertexCount;
        triangleInput.maxTotalTriangleCount = params.clas.maxTotalTriangleCount;
        triangleInput.maxTotalVertexCount = params.clas.maxTotalVertexCount;
        triangleInput.minPositionTruncateBitCount = params.clas.minPositionTruncateBitCount;
        inputInfo.opInput.pTriangleClusters = &triangleInput;
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        blasInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
        blasInput.maxClusterCountPerAccelerationStructure = params.blas.maxClasPerBlasCount;
        blasInput.maxTotalClusterCount = params.blas.maxTotalClasCount;
        inputInfo.opInput.pClustersBottomLevel = &blasInput;
        break;
      default:
        break;
    }

    // Query the actual sizes from the driver
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetClusterASSizes(m_device->handle(), &inputInfo, &sizeInfo);

    result.resultMaxSizeInBytes = sizeInfo.accelerationStructureSize;
    result.scratchSizeInBytes = sizeInfo.buildScratchSize;

    RTXMG_LOG(str::format("RTX MegaGeo: getClusterOperationSizeInfo - type=", (int)params.type,
      " resultMaxSize=", result.resultMaxSizeInBytes, " scratchSize=", result.scratchSizeInBytes));

    return result;
  }

  nvrhi::CommandListHandle NvrhiDxvkDevice::createCommandList(const nvrhi::CommandListParameters& params) {
    // Create a command list wrapper around the existing DxvkContext
    // In RTX Remix, we use a single shared context for most operations
    return new NvrhiDxvkCommandList(this, m_context);
  }

  uint64_t NvrhiDxvkDevice::executeCommandList(nvrhi::ICommandList* commandList) {
    // In RTX Remix/DXVK, command lists are executed immediately through DxvkContext
    // The command list wrapper (NvrhiDxvkCommandList) already submits commands to DxvkContext
    //
    // IMPORTANT: flushCommandList() just prepares a new command buffer, but doesn't
    // actually submit existing commands to GPU. The commands stay in DXVK's internal
    // queue until the frame is presented.
    //
    // For BuildAccel's synchronous readbacks to work, we need to actually submit
    // the commands to GPU and wait.

    if (m_context != nullptr) {
      // Flush pending commands to DXVK's internal queue
      // Note: We do NOT call waitForIdle() here because:
      // 1. DXVK's mapPtr() on staging buffers handles synchronization internally
      // 2. Calling waitForIdle() every frame kills performance
      m_context->flushCommandList();
    }

    // Return a dummy fence value (RTX Remix doesn't use explicit fences in the same way)
    return 1;
  }

  void NvrhiDxvkDevice::waitForIdle() {
    // Wait for all GPU work to complete.
    // IMPORTANT: Only call this when absolutely necessary (e.g., before destroying resources)
    // as it stalls the entire GPU pipeline and kills performance.
    m_device->waitForIdle();
  }

  VkDescriptorPool NvrhiDxvkDevice::getSharedDescriptorPool() {
    if (m_sharedDescriptorPool != VK_NULL_HANDLE) {
      return m_sharedDescriptorPool;
    }

    // Create a large shared descriptor pool with FREE_DESCRIPTOR_SET_BIT
    // This allows individual descriptor sets to be freed back to the pool.
    // Pool sizes are generous to handle all descriptor types used by RTX MG.
    std::vector<VkDescriptorPoolSize> poolSizes = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_SAMPLER, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, kSharedPoolDescriptorCount },
      { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, kSharedPoolDescriptorCount },
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = kSharedPoolMaxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkResult result = vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &m_sharedDescriptorPool);
    if (result != VK_SUCCESS) {
      Logger::err(str::format("RTX MegaGeo: Failed to create shared descriptor pool, result=", (int)result));
      return VK_NULL_HANDLE;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Created shared descriptor pool with maxSets=", kSharedPoolMaxSets,
      " and ", kSharedPoolDescriptorCount, " descriptors per type"));

    return m_sharedDescriptorPool;
  }

} // namespace dxvk
