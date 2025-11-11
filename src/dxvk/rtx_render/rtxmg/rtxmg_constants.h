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

#include <cstdint>
#include "rtxmg_math_types.h"
#include "../rtx_mg_cluster.h"  // Use constants from existing header

namespace dxvk {

// Use existing constants from rtx_mg_cluster.h:
// - kMaxApiClusterCount
// - kMaxClusterEdgeSegments
// - kClusterTemplateCount

// Alias for compatibility
static constexpr uint32_t kNumClusterTemplates = kClusterTemplateCount;

// Calculate template index from cluster size
// Templates are arranged in an 11x11 grid indexed by (sizeX-1, sizeY-1)
inline uint32_t GetTemplateIndex(uint16_t2 clusterSize) {
  return (clusterSize.y - 1) * kMaxClusterEdgeSegments + (clusterSize.x - 1);
}

} // namespace dxvk
