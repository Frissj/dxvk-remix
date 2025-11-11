/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* VK_NV_cluster_acceleration_structure extension supplemental definitions
*
* The SDK provides the core extension. This header only adds missing helper types.
*/

#pragma once

#include <vulkan/vulkan.h>

// SDK provides the extension, we only need to ensure these helpers exist
// These are NOT in the SDK but may be needed for template-based CLAS building

// Add missing Vulkan extension enum values (from VK_NV_cluster_acceleration_structure)
#ifndef VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV
#define VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV \
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV
#endif

#ifndef VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_INPUT_NV
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_INPUT_NV ((VkStructureType)1000557000)
#endif

#ifndef VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_NV
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_NV ((VkStructureType)1000557001)
#endif

// Add missing structure definitions
#ifndef VkClusterAccelerationStructureClustersInputNV_DEFINED
#define VkClusterAccelerationStructureClustersInputNV_DEFINED
typedef struct VkClusterAccelerationStructureClustersInputNV {
  VkStructureType sType;
  const void* pNext;
  uint32_t maxClusterCount;
} VkClusterAccelerationStructureClustersInputNV;
#endif

#ifndef VkClusterAccelerationStructureBuildSizesInfoNV_DEFINED
#define VkClusterAccelerationStructureBuildSizesInfoNV_DEFINED
typedef struct VkClusterAccelerationStructureBuildSizesInfoNV {
  VkStructureType sType;
  const void* pNext;
  VkDeviceSize resultSizeInBytes;
  VkDeviceSize scratchSizeInBytes;
} VkClusterAccelerationStructureBuildSizesInfoNV;
#endif

#ifndef VK_NV_cluster_template_helper
// Simplified cluster template descriptor for template grids
// (SDK uses more complex VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV)
typedef struct VkClusterTemplateNV {
  uint32_t xEdges;   // 1-11
  uint32_t yEdges;   // 1-11
  uint32_t pattern;  // 0=regular, 1=slanted
  uint32_t reserved;
} VkClusterTemplateNV;
#endif

#ifndef VK_NV_cluster_instance_helper
// Simplified cluster instance data for BLAS building
// (SDK uses VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV)
typedef struct VkClusterInstanceDataNV {
  VkDeviceAddress vertexBufferAddress;
  uint32_t vertexBufferStrideInBytes;
  uint32_t clusterIdOffset;
  uint32_t geometryIndexOffset;
  VkDeviceAddress clusterTemplateAddress;
} VkClusterInstanceDataNV;
#endif
