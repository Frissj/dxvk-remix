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

#include "rtxmg_buffer.h"
#include "rtxmg_cluster.h"
#include "rtxmg_math_types.h"
#include "vk_nv_cluster_acceleration_structure.h"
#include "vk_cluster_accel_helper.h"
#include <vulkan/vulkan.h>  // Includes VK_NV_cluster_acceleration_structure extension

namespace dxvk {

// Global cluster acceleration structure extension instance
extern VkClusterAccelExtension g_clusterAccelExt;

// Forward declarations
class DxvkDevice;
class RtxContext;
struct RtxmgConfig;
struct ClusterOutputGeometry;

// Forward declare DxvkAccelStructure to avoid header dependency
class DxvkAccelStructure;
template<typename T> class Rc;

// Cluster acceleration structures
// Container for all GPU buffers needed for cluster-based ray tracing
struct ClusterAccels {
  // Bottom-level acceleration structure
  // NV-DXVK: Production implementation uses DxvkAccelStructure for clean integration
  Rc<DxvkAccelStructure> blasAccelStructure;

  // Legacy buffer reference (kept for size queries during transition)
  RtxmgBuffer<uint8_t> blasBuffer;

  // BLAS scratch buffer (allocated ONCE at max size, reused every frame)
  // Reference: cluster_accel_builder.cpp:1081 uses m_createBlasSizeInfo.scratchSizeInBytes
  RtxmgBuffer<uint8_t> blasScratchBuffer;

  // Cluster acceleration structure buffer (CLAS)
  // VK_NV_cluster_acceleration_structure extension
  RtxmgBuffer<uint8_t> clasBuffer;

  // Array of CLAS pointers (device addresses)
  // One address per CLAS header in clasBuffer
  RtxmgBuffer<VkDeviceAddress> clasPtrsBuffer;

  // PERSISTENT cluster instance references (not in ring buffer!)
  // Must live as long as BLAS lives - contains VkDeviceAddress values that BLAS references
  // SDK MATCH: Sample stores CLAS instance addresses that BLAS references
  RtxmgBuffer<VkDeviceAddress> clusterReferencesBuffer;

  // PERSISTENT cluster instance data (not in ring buffer!)
  // The addresses in clusterReferencesBuffer POINT TO this data
  RtxmgBuffer<uint8_t> persistentInstanceBuffer;

  // Array of BLAS pointers (device addresses)
  RtxmgBuffer<VkDeviceAddress> blasPtrsBuffer;

  // Array of BLAS sizes in bytes
  RtxmgBuffer<uint32_t> blasSizesBuffer;

  // Cluster shading data for each cluster
  // Contains texture coordinates, surface IDs, and cluster layout info
  RtxmgBuffer<RtxmgClusterShadingData> clusterShadingDataBuffer;

  // Cluster vertex positions
  // Staged buffer before creating CLASes
  RtxmgBuffer<float3> clusterVertexPositionsBuffer;

  // Cluster vertex normals (optional)
  // Only allocated when vertex normals are enabled in config
  RtxmgBuffer<float3> clusterVertexNormalsBuffer;

  // Helper to check if buffers are allocated
  bool isValid() const {
    const bool blasReady = blasAccelStructure.ptr() != nullptr || blasBuffer.isValid();
    return blasReady && clasBuffer.isValid();
  }

  // Release all buffers
  void release() {
    blasBuffer.release();
    blasAccelStructure = nullptr;
    blasScratchBuffer.release();
    clasBuffer.release();
    clasPtrsBuffer.release();
    clusterReferencesBuffer.release();
    persistentInstanceBuffer.release();
    blasPtrsBuffer.release();
    blasSizesBuffer.release();
    clusterShadingDataBuffer.release();
    clusterVertexPositionsBuffer.release();
    clusterVertexNormalsBuffer.release();
  }
};

// Cluster template grid descriptor
// Describes a pre-generated cluster template mesh
struct TemplateGridDesc {
  uint32_t xEdges = 0;        // Number of edges in X direction
  uint32_t yEdges = 0;        // Number of edges in Y direction
  uint32_t indexOffset = 0;   // Offset into index buffer
  uint32_t vertexOffset = 0;  // Offset into vertex buffer

  uint32_t getXVerts() const { return xEdges + 1; }
  uint32_t getYVerts() const { return yEdges + 1; }
  uint32_t getNumTriangles() const { return xEdges * yEdges * 2; }
  uint32_t getNumVerts() const { return getXVerts() * getYVerts(); }
};

// Cluster template grids
// Pre-generated meshes for 121 cluster templates (11x11 grid of sizes)
struct TemplateGrids {
  using IndexType = uint8_t;  // Cluster vertices fit in uint8_t (max 12x12 = 144)

  std::vector<TemplateGridDesc> descs;      // Template descriptors
  std::vector<IndexType> indices;           // Triangle indices
  std::vector<float> vertices;              // Vertex positions (normalized UV)

  uint32_t maxVertices = 0;     // Maximum vertices in any template
  uint32_t maxTriangles = 0;    // Maximum triangles in any template
  uint32_t totalVertices = 0;   // Total vertices across all templates
  uint32_t totalTriangles = 0;  // Total triangles across all templates
};

// Cluster instance data for Phase 2 (GPU-generated, used for cluster instantiation)
// This structure is written by cluster_tessellation.comp and read by instantiation
// NOTE: Renamed to avoid conflict with legacy ClusterInstanceData in rtxmg_cluster.h
struct RtxmgClusterInstantiationData {
  uint32_t templateIndex;       // Index into template addresses array (0-120)
  uint32_t geometryIndex;       // Geometry ID for this cluster
  uint32_t vertexBufferOffset;  // Offset in cluster vertex buffer (in vertices, not bytes)
  uint32_t vertexCount;         // Actual vertex count for this cluster
  // Note: AABBs are computed automatically by cluster extension from vertex data
};

//===========================================================================
// Cluster acceleration structure functions
//===========================================================================

// Initialize VK_NV_cluster_acceleration_structure extension
// Must be called once during device initialization
bool initClusterAccelerationExtension(DxvkDevice* device);

// Check if cluster acceleration extension is available
bool isClusterAccelerationExtensionAvailable();

// Generate 121 cluster template grids (11x11 grid of sizes from 1x1 to 11x11 edges)
// This creates the index and vertex data for all template meshes
// Returns TemplateGrids structure containing all template descriptors and geometry
TemplateGrids generateTemplateGrids();

// Get required CLAS size for a cluster template
VkDeviceSize getClusterAccelerationStructureSize(
  DxvkDevice* device,
  uint32_t xEdges,
  uint32_t yEdges,
  uint32_t pattern = 0);  // 0 = regular pattern

// Build cluster acceleration structures (CLAS) for all 121 templates
// This creates a buffer containing CLAS headers that can be instantiated
// Returns true on success, false on failure
// If outTotalClasSize is provided, it will be set to the total CLAS buffer size in bytes
bool buildTemplateClusterAccelerationStructures(
  DxvkDevice* device,
  RtxContext* ctx,
  const TemplateGrids& templateGrids,
  RtxmgBuffer<uint8_t>& clasBuffer,
  std::vector<VkDeviceAddress>& templateAddresses,
  std::vector<uint32_t>& clasInstantiationBytes,
  size_t* outTotalClasSize = nullptr);

// Instantiate clusters from templates using GPU-generated instance data
// This creates actual cluster instances by applying vertex data to templates
// Returns true on success, false on failure
// If outTotalInstanceSize is provided, it will be set to the total instance buffer size
bool instantiateClusterInstances(
  DxvkDevice* device,
  RtxContext* ctx,
  uint32_t numClusters,
  const std::vector<VkDeviceAddress>& templateAddresses,
  const RtxmgBuffer<RtxmgClusterInstantiationData>& instanceDataBuffer,
  const RtxmgBuffer<float3>& clusterVertexBuffer,
  RtxmgBuffer<uint8_t>& instanceBuffer,
  std::vector<VkDeviceAddress>& instanceAddresses,
  size_t* outTotalInstanceSize = nullptr);

// Direct instantiation - uses GPU-written VkClusterAccelerationStructureInstantiateClusterInfoNV
// Bypasses CPU conversion for maximum performance
// Persistent buffers passed in to avoid lifetime issues (buffers must outlive GPU commands)
// SDK MATCH: Can use GPU counter buffer for fully GPU-driven instantiation
bool instantiateClusterInstancesDirect(
  DxvkDevice* device,
  RtxContext* ctx,
  uint32_t numClusters,
  const Rc<DxvkBuffer>& instantiateInfosBuffer,
  std::vector<VkDeviceAddress>& instanceAddresses,
  RtxmgBuffer<uint8_t>& persistentScratchBuffer,
  RtxmgBuffer<uint32_t>& persistentCountBuffer,
  RtxmgBuffer<VkDeviceAddress>& persistentAddressesBuffer,
  RtxmgBuffer<uint8_t>& persistentInstanceBuffer,
  size_t* outTotalInstanceSize = nullptr,
  VkDeviceSize bufferOffsetBytes = 0,
  VkDeviceSize instanceBufferOffsetBytes = 0,
  const Rc<DxvkBuffer>& gpuCounterBuffer = nullptr,  // SDK MATCH: GPU-written cluster count
  VkDeviceSize gpuCounterOffset = 0);                 // Offset to clusters field in TessellationCounters

// Build sizes for BLAS construction
struct BlasBuildSizes {
  size_t blasSize = 0;
  size_t blasScratchSize = 0;
};

// Build bottom-level acceleration structure (BLAS) from cluster geometry
// Uses cluster instances to reference template CLAS structures
// Returns true on success, false on failure
// If outBuildSizes is provided, it will be set to the BLAS and scratch sizes
bool buildClusterGeometryBLAS(
  DxvkDevice* device,
  RtxContext* ctx,
  const ClusterOutputGeometry& geometry,
  const RtxmgConfig& config,
  const std::vector<VkDeviceAddress>& templateAddresses,
  const RtxmgBuffer<RtxmgCluster>& clustersBuffer,
  const RtxmgBuffer<RtxmgClusterShadingData>& clusterShadingDataBuffer,
  const RtxmgBuffer<float3>& clusterVertexPositionsBuffer,
  ClusterAccels& accels,
  BlasBuildSizes* outBuildSizes = nullptr);

// GPU-optimized overload that builds BLAS from instantiated cluster addresses
// Takes instance addresses from Phase 2 (instantiateClusterInstances) and builds BLAS
// Uses VK_NV_cluster_acceleration_structure extension for efficient BLAS building
bool buildClusterGeometryBLAS(
  DxvkDevice* device,
  RtxContext* ctx,
  uint32_t numClusters,
  const std::vector<VkDeviceAddress>& instanceAddresses,  // From Phase 2 instantiation
  ClusterAccels& accels,
  BlasBuildSizes* outBuildSizes = nullptr);

//===========================================================================
// Cluster Operations - SDK-matching interface
//===========================================================================

// SDK Match: cluster::OperationDesc structure (cluster_accel_builder.cpp:602-616, 1078-1092)
// Contains all parameters for executeMultiIndirectClusterOperation
struct ClusterOperationDesc {
  // Operation parameters (CLAS instantiation or BLAS building)
  VkClusterAccelerationStructureInputInfoNV params;

  // Scratch memory (CRITICAL: Use scratchData for the GPU address, NOT scratchSizeInBytes)
  VkDeviceAddress scratchData;     // GPU address of scratch buffer (what VkClusterAccelerationStructureCommandsInfoNV.scratchData expects)
  size_t scratchSizeInBytes;        // Size of scratch buffer (for validation/debugging)

  // GPU counter buffer (SDK line 606: inIndirectArgCountBuffer)
  VkDeviceAddress inIndirectArgCountBuffer;
  size_t inIndirectArgCountOffsetInBytes;

  // Indirect arguments buffer (SDK line 608/1084: inIndirectArgsBuffer)
  VkDeviceAddress inIndirectArgsBuffer;
  size_t inIndirectArgsOffsetInBytes;

  // Output addresses buffer (SDK line 610/1086: inOutAddressesBuffer)
  VkDeviceAddress inOutAddressesBuffer;
  size_t inOutAddressesOffsetInBytes;

  // Output sizes buffer (SDK line 1088: outSizesBuffer)
  VkDeviceAddress outSizesBuffer;
  size_t outSizesOffsetInBytes;

  // Output acceleration structures buffer (SDK line 1090: outAccelerationStructuresBuffer)
  VkDeviceAddress outAccelerationStructuresBuffer;
  size_t outAccelerationStructuresOffsetInBytes;
};

// Execute multiple cluster operations using indirect buffer
// SDK Match: commandList->executeMultiIndirectClusterOperation (lines 618, 1093)
void executeMultiIndirectClusterOperation(
  DxvkContext* ctx,
  const ClusterOperationDesc& desc);

} // namespace dxvk
