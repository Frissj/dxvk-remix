/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* VK_NV_cluster_acceleration_structure extension helper
*
* This header provides a helper class for loading the cluster acceleration
* structure extension function pointers from the SDK.
*/

#pragma once

#include "vk_nv_cluster_acceleration_structure.h"

namespace dxvk {

// DXVK wrapper for cluster acceleration structure extension
// Uses SDK-provided function pointer types
class VkClusterAccelExtension {
public:
  // Function pointers (loaded via vkGetDeviceProcAddr)
  // These match the SDK's VK_NV_cluster_acceleration_structure extension
  PFN_vkGetClusterAccelerationStructureBuildSizesNV vkGetClusterAccelerationStructureBuildSizesNV = nullptr;
  PFN_vkCmdBuildClusterAccelerationStructureIndirectNV vkCmdBuildClusterAccelerationStructureIndirectNV = nullptr;

  // Initialize extension (load function pointers)
  bool init(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr) {
    if (!device || !getDeviceProcAddr) {
      return false;
    }

    vkGetClusterAccelerationStructureBuildSizesNV =
      reinterpret_cast<PFN_vkGetClusterAccelerationStructureBuildSizesNV>(
        getDeviceProcAddr(device, "vkGetClusterAccelerationStructureBuildSizesNV"));

    vkCmdBuildClusterAccelerationStructureIndirectNV =
      reinterpret_cast<PFN_vkCmdBuildClusterAccelerationStructureIndirectNV>(
        getDeviceProcAddr(device, "vkCmdBuildClusterAccelerationStructureIndirectNV"));

    return isValid();
  }

  // Check if extension is available
  bool isValid() const {
    return vkGetClusterAccelerationStructureBuildSizesNV != nullptr &&
           vkCmdBuildClusterAccelerationStructureIndirectNV != nullptr;
  }
};

} // namespace dxvk
