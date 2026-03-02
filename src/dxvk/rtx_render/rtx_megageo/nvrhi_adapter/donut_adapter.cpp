/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// Donut adapter implementation - provides compatibility layer between donut sample code and RTX Remix

#include "nvrhi_types.h"
#include "nvrhi_dxvk_device.h"
#include "nvrhi_dxvk_shader.h"
#include "../../rtx_context.h"
#include "../../rtx_shader_manager.h"
#include "../../../util/log/log.h"
#include "../../../spirv/spirv_code_buffer.h"

// Pre-compiled RTX MG shaders
#include <rtx_shaders/copy_cluster_offset.h>
#include <rtx_shaders/compute_cluster_tiling.h>
#include <rtx_shaders/fill_clusters.h>
#include <rtx_shaders/fill_clusters_texcoords.h>
#include <rtx_shaders/fill_blas_from_clas_args.h>
#include <rtx_shaders/fill_instantiate_template_args.h>
#include <rtx_shaders/patch_cluster_blas_addresses.h>
#include <rtx_shaders/fill_meshlet_clusters.h>

// HiZ constants
#include "../hiz/hiz_buffer_constants.h"

#include <cstring>

#include "../rtxmg_log.h"
#if RTXMG_LOG_DONUT_ADAPTER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace donut {
namespace engine {

// Resource slots for copy_cluster_offset shader
// SPIRV uses DXVK convention: SRV at 0, UAV at 200+, CB at 300
static const dxvk::DxvkResourceSlot s_copyClusterOffsetSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // cb (b0)
  // SRV: t0 → slot 0
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // InTessellationCounters (t0)
  // UAVs: u0-u1 → slots 200-201
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // InOutClusterOffsetCounts (u0)
  { 201, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // InOutFillClustersIndirectArgs (u1)
};
static const uint32_t s_copyClusterOffsetSlotsCount = sizeof(s_copyClusterOffsetSlots) / sizeof(s_copyClusterOffsetSlots[0]);

// Resource slots for fill_clusters shader
// SPIRV uses DXVK convention: SRV at 0-20, Sampler at 100, UAV at 200+, CB at 300
static const dxvk::DxvkResourceSlot s_fillClustersSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // g_TessParams (b0)
  // SRVs: t0-t20 → slots 0-20
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_GridSamplers (t0)
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_ClusterOffsetCounts (t1)
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_Clusters (t2)
  { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexControlPoints (t3)
  { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexSurfaceDescriptors (t4)
  { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexControlPointIndices (t5)
  { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexPatchPointsOffsets (t6)
  { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_Plans (t7)
  { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_SubpatchTrees (t8)
  { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_PatchPointIndices (t9)
  { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_StencilMatrix (t10)
  { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_VertexPatchPoints (t11)
  { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_GeometryData (t12)
  { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_MaterialConstants (t13)
  { 14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_SurfaceToGeometryIndex (t14)
  { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordSurfaceDescriptors (t15)
  { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordControlPointIndices (t16)
  { 17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordPatchPointsOffsets (t17)
  { 18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordPatchPoints (t18)
  { 19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoords (t19)
  // bindlessTextures array: t20 → slot 20
  { 20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE },   // bindlessTextures (t20)
  // Sampler: s0 → slot 100
  { 100, VK_DESCRIPTOR_TYPE_SAMPLER },         // s_DisplacementSampler (s0)
  // UAVs: u0-u3 → slots 200-203
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterVertexPositions (u0)
  { 201, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterShadingData (u1)
  { 202, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_Debug (u2)
  { 203, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterVertexNormals (u3)
};
static const uint32_t s_fillClustersSlotsCount = sizeof(s_fillClustersSlots) / sizeof(s_fillClustersSlots[0]);

// Resource slots for compute_cluster_tiling shader
// SPIRV uses DXVK convention: SRV at 0-17, Sampler at 100-101, UAV at 200-208, CB at 300
// t_HiZBuffer (t17) is an array of 9 SAMPLED_IMAGEs at slot 17
static const dxvk::DxvkResourceSlot s_computeClusterTilingSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // g_Params (b0)
  // SRVs: t0-t16 → slots 0-16
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexControlPoints (t0)
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_GeometryData (t1)
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_MaterialConstants (t2)
  { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_SurfaceToGeometryIndex (t3)
  { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexSurfaceDescriptors (t4)
  { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexControlPointIndices (t5)
  { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_VertexPatchPointsOffsets (t6)
  { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_Plans (t7)
  { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_SubpatchTrees (t8)
  { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_PatchPointIndices (t9)
  { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_StencilMatrix (t10)
  { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_ClasInstantiationBytes (t11)
  { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TemplateAddresses (t12)
  { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordSurfaceDescriptors (t13)
  { 14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordControlPointIndices (t14)
  { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoordPatchPointsOffsets (t15)
  { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },  // t_TexCoords (t16)
  // t_HiZBuffer (t17) - array of 9 SAMPLED_IMAGEs for hierarchical Z-buffer
  // Constructor: slot, type, view, access, count, flags
  { 17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_NONE_KHR, 9 },   // t_HiZBuffer (t17) - array[9]
  // Samplers: s0-s1 → slots 100-101
  { 100, VK_DESCRIPTOR_TYPE_SAMPLER },         // s_DisplacementSampler (s0)
  { 101, VK_DESCRIPTOR_TYPE_SAMPLER },         // s_HizSampler (s1)
  // UAVs: u0-u8 → slots 200-208
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_GridSamplers (u0)
  { 201, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_TessellationCounters (u1)
  { 202, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_Clusters (u2)
  { 203, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterShadingData (u3)
  { 204, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_IndirectArgData (u4)
  { 205, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClasAddresses (u5)
  { 206, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_VertexPatchPoints (u6)
  { 207, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_TexCoordPatchPoints (u7)
  { 208, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_Debug (u8)
};
static const uint32_t s_computeClusterTilingSlotsCount = sizeof(s_computeClusterTilingSlots) / sizeof(s_computeClusterTilingSlots[0]);

// Resource slots for fill_blas_from_clas_args shader
// SPIRV uses DXVK convention: SRV at 0, UAV at 200, CB at 300
static const dxvk::DxvkResourceSlot s_fillBlasFromClasArgsSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // g_Params (b0)
  // SRV: t0 → slot 0
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_ClusterOffsetCounts (t0)
  // UAV: u0 → slot 200
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_BlasFromClasArgs (u0)
};
static const uint32_t s_fillBlasFromClasArgsSlotsCount = sizeof(s_fillBlasFromClasArgsSlots) / sizeof(s_fillBlasFromClasArgsSlots[0]);

// Resource slots for fill_instantiate_template_args shader
// SPIRV uses DXVK convention: SRV at 0, UAV at 200, CB at 300
static const dxvk::DxvkResourceSlot s_fillInstantiateTemplateArgsSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // cb (b0)
  // SRV: t0 → slot 0
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // InTemplateAddresses (t0)
  // UAV: u0 → slot 200
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // OutArgs (u0)
};
static const uint32_t s_fillInstantiateTemplateArgsSlotsCount = sizeof(s_fillInstantiateTemplateArgsSlots) / sizeof(s_fillInstantiateTemplateArgsSlots[0]);

// Resource slots for patch_cluster_blas_addresses shader
// SPIRV uses DXVK convention: SRV at 0-1, UAV at 200, CB at 300
static const dxvk::DxvkResourceSlot s_patchClusterBlasAddressesSlots[] = {
  // Constant buffer: b0 → slot 300
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // cb (b0)
  // SRVs: t0-t1 → slots 0-1
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // InMappings (t0)
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // InBlasAddresses (t1)
  // UAV: u0 → slot 200
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // InOutInstanceBuffer (u0)
};
static const uint32_t s_patchClusterBlasAddressesSlotsCount = sizeof(s_patchClusterBlasAddressesSlots) / sizeof(s_patchClusterBlasAddressesSlots[0]);

// Resource slots for fill_meshlet_clusters shader
// CB at b0 (push constant), SRVs t0-t3, UAVs u0-u4
static const dxvk::DxvkResourceSlot s_fillMeshletClustersSlots[] = {
  // Constant buffer: b0 → slot 300 (push constant)
  { 300, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },   // g_Params (b0)
  // SRVs: t0-t3 → slots 0-3
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_ClusterInfos (t0)
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_PersistentExpandedVertices (t1)
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_PersistentExpandedNormals (t2)
  { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },   // t_PersistentUniqueVertices (t3)
  // UAVs: u0-u4 → slots 200-204
  { 200, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterVertexPositions (u0)
  { 201, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterVertexNormals (u1)
  { 202, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_IndirectArgData (u2)
  { 203, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClasAddresses (u3)
  { 204, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_WRITE_BIT },  // u_ClusterShadingData (u4)
};
static const uint32_t s_fillMeshletClustersSlotsCount = sizeof(s_fillMeshletClustersSlots) / sizeof(s_fillMeshletClustersSlots[0]);

ShaderFactory::ShaderFactory(dxvk::RtxContext* ctx)
  : m_rtxContext(ctx)
{
  // Get VkDevice for creating descriptor set layouts
  if (m_rtxContext) {
    m_vkDevice = m_rtxContext->getDevice()->handle();
  }
  RTXMG_LOG("ShaderFactory: Constructor called");
}

ShaderFactory::~ShaderFactory() {
  // Destroy the HiZ descriptor set layout if we created one
  if (m_hiZDescriptorSetLayout != VK_NULL_HANDLE && m_vkDevice != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(m_vkDevice, m_hiZDescriptorSetLayout, nullptr);
    m_hiZDescriptorSetLayout = VK_NULL_HANDLE;
    RTXMG_LOG("ShaderFactory: Destroyed HiZ descriptor set layout");
  }
}

void ShaderFactory::ensureHiZDescriptorSetLayout() {
  if (m_hiZDescriptorSetLayout != VK_NULL_HANDLE) {
    return;  // Already created
  }

  if (m_vkDevice == VK_NULL_HANDLE) {
    dxvk::Logger::err("ShaderFactory: Cannot create HiZ descriptor set layout - no VkDevice");
    return;
  }

  // Create HiZ descriptor set layout for set 1
  // This matches the shader's VK_BINDING(0, 1) Texture2D<float> t_HiZBuffer[HIZ_MAX_LODS]: register(t0, space1);
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;  // Binding 0 in set 1
  binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  binding.descriptorCount = HIZ_MAX_LODS;  // 9 textures
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &binding;

  VkResult result = vkCreateDescriptorSetLayout(m_vkDevice, &layoutInfo, nullptr, &m_hiZDescriptorSetLayout);
  if (result == VK_SUCCESS && m_hiZDescriptorSetLayout != VK_NULL_HANDLE) {
    RTXMG_LOG(dxvk::str::format("ShaderFactory: Created HiZ descriptor set layout ", (void*)m_hiZDescriptorSetLayout,
      " with ", HIZ_MAX_LODS, " SAMPLED_IMAGE bindings at binding 0"));
  } else {
    dxvk::Logger::err(dxvk::str::format("ShaderFactory: Failed to create HiZ descriptor set layout, result=", (int)result));
  }
}

nvrhi::ShaderHandle ShaderFactory::CreateShader(
    const char* path,
    const char* entryPoint,
    const std::vector<ShaderMacro>* macros,
    nvrhi::ShaderType shaderType)
{
  // Map shader path to pre-compiled SPIR-V bytecode
  const uint32_t* spirvCode = nullptr;
  size_t spirvSize = 0;
  const dxvk::DxvkResourceSlot* resourceSlots = nullptr;
  uint32_t slotCount = 0;

  std::string pathStr(path);

  if (pathStr.find("copy_cluster_offset") != std::string::npos) {
    spirvCode = copy_cluster_offset;
    spirvSize = sizeof(copy_cluster_offset);
    resourceSlots = s_copyClusterOffsetSlots;
    slotCount = s_copyClusterOffsetSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled copy_cluster_offset shader");
  }
  else if (pathStr.find("compute_cluster_tiling") != std::string::npos) {
    spirvCode = compute_cluster_tiling;
    spirvSize = sizeof(compute_cluster_tiling);
    resourceSlots = s_computeClusterTilingSlots;
    slotCount = s_computeClusterTilingSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled compute_cluster_tiling shader");
  }
  else if (pathStr.find("fill_clusters") != std::string::npos) {
    // Check entry point to distinguish between FillClustersMain and FillClustersTexcoordsMain
    std::string entryStr(entryPoint ? entryPoint : "");
    if (entryStr.find("Texcoords") != std::string::npos) {
      spirvCode = fill_clusters_texcoords;
      spirvSize = sizeof(fill_clusters_texcoords);
      resourceSlots = s_fillClustersSlots;
      slotCount = s_fillClustersSlotsCount;
      RTXMG_LOG("ShaderFactory: Loading pre-compiled fill_clusters_texcoords shader");
    } else {
      spirvCode = fill_clusters;
      spirvSize = sizeof(fill_clusters);
      resourceSlots = s_fillClustersSlots;
      slotCount = s_fillClustersSlotsCount;
      RTXMG_LOG("ShaderFactory: Loading pre-compiled fill_clusters shader");
    }
  }
  else if (pathStr.find("fill_blas_from_clas_args") != std::string::npos) {
    spirvCode = fill_blas_from_clas_args;
    spirvSize = sizeof(fill_blas_from_clas_args);
    resourceSlots = s_fillBlasFromClasArgsSlots;
    slotCount = s_fillBlasFromClasArgsSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled fill_blas_from_clas_args shader");
  }
  else if (pathStr.find("fill_instantiate_template_args") != std::string::npos) {
    spirvCode = fill_instantiate_template_args;
    spirvSize = sizeof(fill_instantiate_template_args);
    resourceSlots = s_fillInstantiateTemplateArgsSlots;
    slotCount = s_fillInstantiateTemplateArgsSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled fill_instantiate_template_args shader");
  }
  else if (pathStr.find("patch_cluster_blas_addresses") != std::string::npos) {
    spirvCode = patch_cluster_blas_addresses;
    spirvSize = sizeof(patch_cluster_blas_addresses);
    resourceSlots = s_patchClusterBlasAddressesSlots;
    slotCount = s_patchClusterBlasAddressesSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled patch_cluster_blas_addresses shader");
  }
  else if (pathStr.find("fill_meshlet_clusters") != std::string::npos) {
    spirvCode = fill_meshlet_clusters;
    spirvSize = sizeof(fill_meshlet_clusters);
    resourceSlots = s_fillMeshletClustersSlots;
    slotCount = s_fillMeshletClustersSlotsCount;
    RTXMG_LOG("ShaderFactory: Loading pre-compiled fill_meshlet_clusters shader");
  }
  else {
    std::string msg = std::string("ShaderFactory: Unknown shader path: ") + path;
    dxvk::Logger::warn(msg);
    return nullptr;
  }

  if (!spirvCode || spirvSize == 0) {
    dxvk::Logger::err("ShaderFactory: Failed to get SPIR-V bytecode");
    return nullptr;
  }

  // Create DxvkShader from SPIR-V bytecode
  dxvk::SpirvCodeBuffer codeBuffer(spirvSize / sizeof(uint32_t), spirvCode);

  // Map nvrhi shader type to VkShaderStageFlagBits
  VkShaderStageFlagBits vkStage = VK_SHADER_STAGE_COMPUTE_BIT;
  if (shaderType == nvrhi::ShaderType::Vertex)
    vkStage = VK_SHADER_STAGE_VERTEX_BIT;
  else if (shaderType == nvrhi::ShaderType::Pixel)
    vkStage = VK_SHADER_STAGE_FRAGMENT_BIT;

  // Create DxvkShader with resource slots for binding remapping
  // CRITICAL: Set push constant range if the shader uses constant buffers
  // The Slang compiler may compile small CBs to push constants in SPIR-V
  dxvk::DxvkInterfaceSlots iface = {};

  // Check if this shader uses a constant buffer - if so, configure push constants
  // DXVK needs this to set VK_SHADER_STAGE_COMPUTE_BIT in the push constant range
  for (uint32_t i = 0; i < slotCount; i++) {
    if (resourceSlots[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
      // Use a generous push constant size (128 bytes) to cover all our CBs
      // This allows the Slang-compiled SPIR-V to use push constants if it wants
      iface.pushConstOffset = 0;
      iface.pushConstSize = 128;
      break;
    }
  }

  // Create shader options - add HiZ descriptor set layout for compute_cluster_tiling
  // This shader uses VK_BINDING(0, 1) for HiZ textures (descriptor set 1)
  dxvk::DxvkShaderOptions shaderOptions;
  bool isClusterTilingShader = (pathStr.find("compute_cluster_tiling") != std::string::npos);
  if (isClusterTilingShader) {
    ensureHiZDescriptorSetLayout();
    if (m_hiZDescriptorSetLayout != VK_NULL_HANDLE) {
      shaderOptions.extraLayouts.push_back(m_hiZDescriptorSetLayout);
      RTXMG_LOG(dxvk::str::format("ShaderFactory: Adding HiZ descriptor set layout to compute_cluster_tiling shader"));
    }
  }

  dxvk::Rc<dxvk::DxvkShader> dxvkShader = new dxvk::DxvkShader(
    vkStage,
    slotCount, resourceSlots,
    iface,
    codeBuffer,
    shaderOptions,
    dxvk::DxvkShaderConstData()
  );

  if (dxvkShader == nullptr) {
    dxvk::Logger::err("ShaderFactory: Failed to create DxvkShader");
    return nullptr;
  }

  dxvkShader->setDebugName(path);
  RTXMG_LOG(dxvk::str::format("ShaderFactory: Created shader '", path, "' with ", slotCount, " resource slots"));

  return new dxvk::NvrhiDxvkShader(dxvkShader);
}

} // namespace engine
} // namespace donut
