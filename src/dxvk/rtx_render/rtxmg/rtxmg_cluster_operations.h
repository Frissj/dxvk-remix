/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Adapted from NVIDIA RTX Mega Geometry SDK for RTX Remix integration
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*/

#pragma once

#include "rtxmg_math_types.h"
#include <vulkan/vulkan.h>
#include <cstdint>

namespace dxvk {

// Forward declarations
class DxvkContext;

// Cluster operation types
// These map to VK_NV_cluster_acceleration_structure operations
enum class ClusterOperationType : uint32_t {
  // Create a cluster template (pre-generated mesh topology)
  // Input: Vertex positions + indices
  // Output: CLAS template that can be instantiated
  CREATE_CLUSTER_TEMPLATE = 0,

  // Get size requirements for creating a cluster template
  GET_CLUSTER_TEMPLATE_SIZE = 1,

  // Instantiate a cluster template with per-instance vertex data
  // Input: Template + per-instance vertex buffer
  // Output: Fully tessellated CLAS ready for ray tracing
  INSTANTIATE_CLUSTER_TEMPLATE = 2,

  // Get size requirements for instantiating a cluster template
  GET_INSTANTIATE_SIZE = 3,

  // Build BLAS from CLAS
  // Input: CLAS
  // Output: Traditional BLAS that RTX Remix can ray trace
  BUILD_BLAS_FROM_CLAS = 4,

  COUNT
};

// Cluster template creation parameters
struct CreateClusterTemplateParams {
  VkDeviceAddress templateBuffer;     // Output: CLAS template buffer
  VkDeviceAddress vertexBuffer;       // Input: Template vertex positions (normalized UV)
  VkDeviceAddress indexBuffer;        // Input: Template triangle indices
  uint32_t vertexCount;               // Number of vertices in template
  uint32_t indexCount;                // Number of indices in template
  uint32_t vertexStride;              // Stride between vertices in bytes
  VkFormat vertexFormat;              // Vertex position format (usually R32G32_SFLOAT)
  VkIndexType indexType;              // Index type (VK_INDEX_TYPE_UINT8_EXT for templates)
};

// Cluster template instantiation parameters
struct InstantiateClusterTemplateParams {
  VkDeviceAddress clasBuffer;         // Output: Instantiated CLAS buffer
  VkDeviceAddress templateBuffer;     // Input: CLAS template to instantiate
  VkDeviceAddress vertexBuffer;       // Input: Per-instance vertex data (3D positions)
  VkDeviceAddress normalBuffer;       // Input: Per-instance normal data (optional)
  VkDeviceAddress texcoordBuffer;     // Input: Per-instance texcoord data
  uint32_t vertexCount;               // Number of vertices
  uint32_t vertexStride;              // Stride between vertices in bytes
  uint32_t normalStride;              // Stride between normals in bytes (0 if no normals)
  uint32_t texcoordStride;            // Stride between texcoords in bytes
  VkFormat vertexFormat;              // Vertex format (VK_FORMAT_R32G32B32_SFLOAT)
  VkFormat normalFormat;              // Normal format (VK_FORMAT_R32G32B32_SFLOAT or UNDEFINED)
  VkFormat texcoordFormat;            // Texcoord format (VK_FORMAT_R32G32_SFLOAT)
};

// BLAS from CLAS build parameters
struct BuildBlasFromClasParams {
  VkDeviceAddress blasBuffer;         // Output: BLAS buffer
  VkDeviceAddress clasBuffer;         // Input: CLAS buffer
  VkDeviceAddress scratchBuffer;      // Scratch memory for build
  VkBuildAccelerationStructureFlagsKHR flags;  // Build flags
};

// Indirect cluster operation arguments
// Used for multi-indirect cluster operations (batch processing)
struct IndirectClusterArgs {
  ClusterOperationType operationType;
  union {
    CreateClusterTemplateParams createTemplate;
    InstantiateClusterTemplateParams instantiateTemplate;
    BuildBlasFromClasParams buildBlas;
  };
};

// Size query results
struct ClusterOperationSizeInfo {
  size_t clusterSize;     // Size of CLAS in bytes
  size_t scratchSize;     // Size of scratch memory needed
  size_t updateSize;      // Size needed for updates (not used currently)
};

// ============================================================================
// SDK-MATCHING INTERFACE - Direct wrappers for VK extension commands
// ============================================================================

// NOTE: ClusterOperationDesc and executeMultiIndirectClusterOperation are now defined in rtxmg_accel.h
// to avoid circular dependencies

} // namespace dxvk
