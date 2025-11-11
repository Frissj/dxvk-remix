/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* RTX Mega Geometry shader definitions
*/

#pragma once

#include "../rtx_shader_manager.h"

// Include compiled shaders
#include <rtx_shaders/compute_cluster_tiling.h>
#include <rtx_shaders/fill_clusters.h>
#include <rtx_shaders/copy_cluster_offset.h>
#include <rtx_shaders/fill_blas_from_clas_args.h>
#include <rtx_shaders/hiz_pyramid_generate.h>
#include <rtx_shaders/patch_tlas_instance_blas_addresses.h>

namespace dxvk {

// Cluster tiling compute shader
class ClusterTilingShader : public ManagedShader {
  SHADER_SOURCE(ClusterTilingShader, VK_SHADER_STAGE_COMPUTE_BIT, compute_cluster_tiling)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Inputs (t0-t8)
    STRUCTURED_BUFFER(0)       // inputPositions
    STRUCTURED_BUFFER(1)       // inputNormals
    STRUCTURED_BUFFER(2)       // inputTexcoords
    STRUCTURED_BUFFER(3)       // inputIndices
    STRUCTURED_BUFFER(4)       // surfaceInfo
    STRUCTURED_BUFFER(5)       // templateAddresses
    STRUCTURED_BUFFER(6)       // clasInstantiationBytes
    SAMPLER2DARRAY(7)          // hizBuffer (optional, for HiZ culling)
    SAMPLER(8)                 // hizSampler

    // Outputs (u9-u14)
    RW_STRUCTURED_BUFFER(9)    // gridSamplersOut
    RW_STRUCTURED_BUFFER(10)   // clustersOut
    RW_STRUCTURED_BUFFER(11)   // clusterShadingDataOut
    RW_STRUCTURED_BUFFER(12)   // clusterIndirectArgsOut
    RW_STRUCTURED_BUFFER(13)   // clasAddressesOut
    RW_STRUCTURED_BUFFER(14)   // tessCountersOut
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(ClusterTilingShader);

// Cluster filling compute shader
class FillClustersShader : public ManagedShader {
  SHADER_SOURCE(FillClustersShader, VK_SHADER_STAGE_COMPUTE_BIT, fill_clusters)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Constant buffer (SDK MATCH: binding 0)
    CONSTANT_BUFFER(0)         // FillClustersParams

    // Inputs (t1-t8) - shifted +1 for constant buffer
    STRUCTURED_BUFFER(1)       // inputPositions
    STRUCTURED_BUFFER(2)       // inputNormals
    STRUCTURED_BUFFER(3)       // inputTexcoords
    STRUCTURED_BUFFER(4)       // inputIndices
    STRUCTURED_BUFFER(5)       // surfaceInfo
    STRUCTURED_BUFFER(6)       // gridSamplers
    STRUCTURED_BUFFER(7)       // clusters
    STRUCTURED_BUFFER(8)       // clusterShadingData

    // Outputs (u9-u11) - shifted +1
    RW_STRUCTURED_BUFFER(9)    // clusterVertexPositionsOut
    RW_STRUCTURED_BUFFER(10)   // clusterVertexNormalsOut
    RW_STRUCTURED_BUFFER(11)   // clusterShadingDataOut
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(FillClustersShader);

// Copy cluster offset compute shader
class CopyClusterOffsetShader : public ManagedShader {
  SHADER_SOURCE(CopyClusterOffsetShader, VK_SHADER_STAGE_COMPUTE_BIT, copy_cluster_offset)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Constant buffer (SDK MATCH: binding 0)
    CONSTANT_BUFFER(0)         // CopyClusterOffsetParams

    // Inputs (t1) - shifted +1
    STRUCTURED_BUFFER(1)       // tessCountersIn

    // Outputs (u2) - shifted +1
    RW_STRUCTURED_BUFFER(2)    // clusterOffsetCountsOut
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(CopyClusterOffsetShader);

// Fill BLAS from CLAS args compute shader
class FillBlasFromClasArgsShader : public ManagedShader {
  SHADER_SOURCE(FillBlasFromClasArgsShader, VK_SHADER_STAGE_COMPUTE_BIT, fill_blas_from_clas_args)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Constant buffer (SDK MATCH: binding 0)
    CONSTANT_BUFFER(0)         // FillBlasFromClasArgsParams

    // Inputs (t1) - shifted +1
    STRUCTURED_BUFFER(1)       // clusterOffsetCounts

    // Outputs (u2) - shifted +1
    RW_STRUCTURED_BUFFER(2)    // blasFromClasArgsOut
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(FillBlasFromClasArgsShader);

// HiZ pyramid generation shader
class HiZPyramidGenerateShader : public ManagedShader {
  SHADER_SOURCE(HiZPyramidGenerateShader, VK_SHADER_STAGE_COMPUTE_BIT, hiz_pyramid_generate)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Constant buffer (SDK MATCH: binding 0)
    CONSTANT_BUFFER(0)         // HiZGenerateParams

    // Inputs (shifted +1)
    TEXTURE2D(1)               // srcDepth
    SAMPLER(2)                 // depthSampler

    // Outputs (shifted +1)
    RW_TEXTURE2DARRAY(3)       // hizPyramid
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(HiZPyramidGenerateShader);

// Patch TLAS instance BLAS addresses compute shader
class PatchTlasInstanceBlasAddressesShader : public ManagedShader {
  SHADER_SOURCE(PatchTlasInstanceBlasAddressesShader, VK_SHADER_STAGE_COMPUTE_BIT, patch_tlas_instance_blas_addresses)

public:
  BINDLESS_ENABLED()

  BEGIN_PARAMETER()
    // Constant buffer (SDK MATCH: binding 0)
    CONSTANT_BUFFER(0)         // PatchTlasInstanceParams

    // Inputs (shifted +1)
    STRUCTURED_BUFFER(1)       // patchEntries (TLAS index → blasPtrsBuffer index mapping)
    STRUCTURED_BUFFER(2)       // blasAddresses (from cluster extension)

    // Outputs (shifted +1)
    RW_STRUCTURED_BUFFER(3)    // instanceDescsBuffer (TLAS instances to patch)
  END_PARAMETER()
};

PREWARM_SHADER_PIPELINE(PatchTlasInstanceBlasAddressesShader);

} // namespace dxvk
