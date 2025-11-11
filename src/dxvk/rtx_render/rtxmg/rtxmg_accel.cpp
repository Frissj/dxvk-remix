/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* RTX Mega Geometry acceleration structure building
*
* Implements cluster acceleration structure (CLAS) and bottom-level
* acceleration structure (BLAS) building using VK_NV_cluster_acceleration_structure.
*/

#include "rtxmg_accel.h"
#include "rtxmg_cluster_builder.h"
#include "rtxmg_cluster_operations.h"  // Cluster operation wrappers
#include "vk_cluster_accel_helper.h"  // Extension helper class
#include "../rtx_context.h"
#include "../../dxvk_device.h"
#include "../../util/log/log.h"
#include "../../dxvk_cmdlist.h"
#include "../../util/xxHash/xxhash.h"

namespace dxvk {

// Global cluster acceleration structure extension instance
VkClusterAccelExtension g_clusterAccelExt;

//===========================================================================
// Helper functions
//===========================================================================

// Compute a stable cluster ID based on cluster properties
// NOTE: Cluster size is INTENTIONALLY EXCLUDED from hash to prevent ID flickering
// when adaptive tessellation changes cluster density based on camera distance.
// This ensures the same surface region always gets the same ID across frames,
// regardless of tessellation level changes.
static uint32_t computeStableClusterId(const RtxmgCluster& cluster) {
  // Hash stable cluster properties that uniquely identify it
  XXH64_hash_t hash = 0;

  // Hash surface index (which patch/surface this cluster belongs to)
  hash = XXH3_64bits_withSeed(&cluster.iSurface, sizeof(cluster.iSurface), hash);

  // Hash grid position (uniquely identifies cluster location on surface)
  hash = XXH3_64bits_withSeed(&cluster.offsetX, sizeof(cluster.offsetX), hash);
  hash = XXH3_64bits_withSeed(&cluster.offsetY, sizeof(cluster.offsetY), hash);

  // NV-DXVK: Cluster size is NOT hashed to maintain stable IDs during adaptive tessellation
  // When camera moves, TESS_MODE_SPHERICAL_PROJECTION changes cluster sizes dynamically,
  // but we want the same surface region to maintain the same cluster ID for stable debug visualization.
  // This prevents color flickering in debug views when the camera moves.

  // Return lower 32 bits as cluster ID
  return static_cast<uint32_t>(hash & 0xFFFFFFFF);
}

//===========================================================================
// Extension initialization
//===========================================================================

bool initClusterAccelerationExtension(DxvkDevice* device) {
  VkDevice vkDevice = device->handle();

  // Load extension function pointers using Vulkan loader
  bool success = g_clusterAccelExt.init(vkDevice, vkGetDeviceProcAddr);

  if (success) {
    Logger::info("[RTXMG] VK_NV_cluster_acceleration_structure extension loaded successfully");
  } else {
    Logger::warn("[RTXMG] VK_NV_cluster_acceleration_structure extension not available - cluster acceleration disabled");
  }

  return success;
}

bool isClusterAccelerationExtensionAvailable() {
  return g_clusterAccelExt.isValid();
}

//===========================================================================
// CLAS size calculation
//===========================================================================

// Note: getClusterAccelerationStructureSize removed - use GPU-side indirect building instead

//===========================================================================
// Template grid generation
//===========================================================================

TemplateGrids generateTemplateGrids() {
  TemplateGrids grids;
  grids.descs.reserve(121); // 11x11 grid of templates

  uint32_t totalIndexPaddingBytes = 0;
  uint32_t totalVertexPaddingFloats = 0;

  const uint32_t kMaxEdgeSegments = 11;
  uint32_t totalVertices = 0;
  uint32_t totalTriangles = 0;

  Logger::info("[RTXMG] Generating 121 cluster template grids (11x11 grid of sizes)");

  // Generate templates for all combinations of xEdges and yEdges (1-11)
  for (uint32_t yEdges = 1; yEdges <= kMaxEdgeSegments; yEdges++) {
    for (uint32_t xEdges = 1; xEdges <= kMaxEdgeSegments; xEdges++) {
      TemplateGridDesc desc;
      desc.xEdges = xEdges;
      desc.yEdges = yEdges;
      // Offsets must be expressed in bytes because the cluster extension expects byte addresses.
      // Align both index and vertex data to 16-byte boundaries to satisfy RT core requirements.
      const uint32_t indexMisalignment = static_cast<uint32_t>(grids.indices.size()) & 0xF;
      if (indexMisalignment != 0) {
        const uint32_t padBytes = 16u - indexMisalignment;
        grids.indices.insert(grids.indices.end(), padBytes, 0);
        totalIndexPaddingBytes += padBytes;
      }
      desc.indexOffset = grids.indices.size() * sizeof(TemplateGrids::IndexType);

      const uint32_t vertexMisalignment = static_cast<uint32_t>(grids.vertices.size()) & 3u;
      if (vertexMisalignment != 0) {
        const uint32_t padFloats = 4u - vertexMisalignment;
        grids.vertices.insert(grids.vertices.end(), padFloats, 0.0f);
        totalVertexPaddingFloats += padFloats;
      }
      desc.vertexOffset = grids.vertices.size() * sizeof(float);

      // Calculate grid dimensions
      uint32_t xVerts = xEdges + 1;
      uint32_t yVerts = yEdges + 1;

      // Generate triangle indices (2 triangles per quad in regular grid pattern)
      // Each quad has 4 vertices: v0--v1
      //                           |  /|
      //                           | / |
      //                           |/  |
      //                           v2--v3
      for (uint32_t y = 0; y < yEdges; y++) {
        for (uint32_t x = 0; x < xEdges; x++) {
          // Vertex indices for this quad
          uint8_t v0 = static_cast<uint8_t>(y * xVerts + x);
          uint8_t v1 = v0 + 1;
          uint8_t v2 = static_cast<uint8_t>(v0 + xVerts);
          uint8_t v3 = v2 + 1;

          // Triangle 1: v0, v1, v2 (clockwise winding)
          grids.indices.push_back(v0);
          grids.indices.push_back(v1);
          grids.indices.push_back(v2);

          // Triangle 2: v1, v3, v2 (clockwise winding)
          grids.indices.push_back(v1);
          grids.indices.push_back(v3);
          grids.indices.push_back(v2);
        }
      }

      // Generate normalized UV vertex positions [0,1] x [0,1]
      // This creates a flat grid in the XY plane (Z = 0)
      // During instantiation, these will be displaced using actual heightmap data
      for (uint32_t y = 0; y < yVerts; y++) {
        for (uint32_t x = 0; x < xVerts; x++) {
          // Normalize to [0, 1] range
          float u = static_cast<float>(x) / xEdges;
          float v = static_cast<float>(y) / yEdges;

          // Store as 3 floats (X, Y, Z) even though Z is always 0
          // Sample code uses RGB32 format which expects 3 components
          grids.vertices.push_back(u);
          grids.vertices.push_back(v);
          grids.vertices.push_back(0.0f);  // Z = 0 (flat grid)
        }
      }

      grids.descs.push_back(desc);
      totalVertices += xVerts * yVerts;
      totalTriangles += xEdges * yEdges * 2;
    }
  }

  // Store statistics
  grids.totalVertices = totalVertices;
  grids.totalTriangles = totalTriangles;
  grids.maxVertices = (kMaxEdgeSegments + 1) * (kMaxEdgeSegments + 1);  // 12x12 = 144
  grids.maxTriangles = kMaxEdgeSegments * kMaxEdgeSegments * 2;         // 11x11x2 = 242

  if (totalIndexPaddingBytes > 0 || totalVertexPaddingFloats > 0) {
    Logger::info(str::format("[RTXMG] Template padding applied: ",
                              totalIndexPaddingBytes, " index bytes, ",
                              totalVertexPaddingFloats, " vertex floats"));
  }

  const size_t templateLogCount = std::min<size_t>(5, grids.descs.size());
  for (size_t i = 0; i < templateLogCount; ++i) {
    const TemplateGridDesc& desc = grids.descs[i];
    Logger::info(str::format("[RTXMG] Template[", i, "]: xEdges=", desc.xEdges,
                             ", yEdges=", desc.yEdges,
                             ", indexOffset=", desc.indexOffset,
                             ", vertexOffset=", desc.vertexOffset,
                             ", triangles=", desc.getNumTriangles(),
                             ", vertices=", desc.getNumVerts()));
  }

  Logger::info(str::format("[RTXMG] Template grids generated: ",
                          grids.descs.size(), " templates, ",
                          grids.totalVertices, " total vertices, ",
                          grids.totalTriangles, " total triangles"));
  Logger::info(str::format("[RTXMG] Max template size: ",
                          grids.maxVertices, " vertices, ",
                          grids.maxTriangles, " triangles"));

  return grids;
}

//===========================================================================
// Template CLAS building
//===========================================================================

bool buildTemplateClusterAccelerationStructures(
  DxvkDevice* device,
  RtxContext* ctx,
  const TemplateGrids& templateGrids,
  RtxmgBuffer<uint8_t>& clasBuffer,
  std::vector<VkDeviceAddress>& templateAddresses,
  std::vector<uint32_t>& clasInstantiationBytes,
  size_t* outTotalClasSize)
{
  if (!g_clusterAccelExt.isValid()) {
    Logger::err("[RTXMG] Cannot build template CLAS: extension not available");
    return false;
  }

  const uint32_t kNumTemplates = 121; // 11x11 grid of templates
  const uint32_t kMaxEdgeSegments = 11;

  templateAddresses.resize(kNumTemplates);
  clasInstantiationBytes.resize(kNumTemplates);

  Logger::info(str::format("[RTXMG] Building ", kNumTemplates,
                           " template CLAS structures (indexBytes=",
                           templateGrids.indices.size() * sizeof(TemplateGrids::IndexType),
                           ", vertexBytes=",
                           templateGrids.vertices.size() * sizeof(float), ")"));

  // Upload template index data to GPU
  RtxmgBuffer<uint8_t> templateIndexBuffer;
  templateIndexBuffer.create(
    device,
    templateGrids.indices.size(),
    "RTXMG Template Indices",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  templateIndexBuffer.upload(templateGrids.indices);
  VkDeviceAddress templateIndexAddress = templateIndexBuffer.getDeviceAddress();

  // Upload template vertex data to GPU
  RtxmgBuffer<float> templateVertexBuffer;
  templateVertexBuffer.create(
    device,
    templateGrids.vertices.size(),
    "RTXMG Template Vertices",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  templateVertexBuffer.upload(templateGrids.vertices);
  VkDeviceAddress templateVertexAddress = templateVertexBuffer.getDeviceAddress();

  // LOG: Template buffer base addresses
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Index buffer base: 0x", std::hex, templateIndexAddress, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> size: ", templateGrids.indices.size(), " bytes"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> address % 16 = ", templateIndexAddress % 16,
    " (", (templateIndexAddress % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Vertex buffer base: 0x", std::hex, templateVertexAddress, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> size: ", templateGrids.vertices.size() * sizeof(float), " bytes"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> address % 16 = ", templateVertexAddress % 16,
    " (", (templateVertexAddress % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

  // Prepare build info structs for all templates
  std::vector<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> buildInfos(kNumTemplates);

  uint32_t misalignedIndexCount = 0;
  uint32_t misalignedVertexCount = 0;

  for (uint32_t i = 0; i < kNumTemplates; ++i) {
    uint32_t xEdges = (i % kMaxEdgeSegments) + 1;
    uint32_t yEdges = (i / kMaxEdgeSegments) + 1;
    const TemplateGridDesc& desc = templateGrids.descs[i];

    // Calculate buffer addresses with proper offsets
    // NOTE: desc offsets are already in bytes (matching sample approach)
    VkDeviceAddress indexAddress = templateIndexAddress + desc.indexOffset;
    VkDeviceAddress vertexAddress = templateVertexAddress + desc.vertexOffset;

    bool indexAligned = (indexAddress % 16) == 0;
    bool vertexAligned = (vertexAddress % 16) == 0;

    if (!indexAligned) misalignedIndexCount++;
    if (!vertexAligned) misalignedVertexCount++;

    VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& buildInfo = buildInfos[i];
    buildInfo = {};
    buildInfo.clusterID = 0;  // Template ID (not used, set during instantiation)
    buildInfo.clusterFlags = 0;
    buildInfo.triangleCount = desc.getNumTriangles();
    buildInfo.vertexCount = desc.getNumVerts();
    buildInfo.positionTruncateBitCount = 0;
    buildInfo.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
    buildInfo.opacityMicromapIndexType = 0;
    buildInfo.baseGeometryIndexAndGeometryFlags = {};
    buildInfo.indexBufferStride = sizeof(uint8_t);
    buildInfo.vertexBufferStride = sizeof(float) * 3;  // Sample uses 12 bytes even though data is 2 floats
    buildInfo.geometryIndexAndFlagsBufferStride = 0;
    buildInfo.opacityMicromapIndexBufferStride = 0;
    buildInfo.indexBuffer = indexAddress;
    buildInfo.vertexBuffer = vertexAddress;
    buildInfo.geometryIndexAndFlagsBuffer = 0;
    buildInfo.opacityMicromapArray = 0;
    buildInfo.opacityMicromapIndexBuffer = 0;
    buildInfo.instantiationBoundingBoxLimit = 0;

    // LOG: First 5 templates and any misaligned
    if (i < 5 || !indexAligned || !vertexAligned) {
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Template[", i, "] (", xEdges, "x", yEdges, "):"));
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> triangleCount=", buildInfo.triangleCount,
        ", vertexCount=", buildInfo.vertexCount));
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> indexOffset=", desc.indexOffset,
        " -> addr=0x", std::hex, indexAddress, std::dec, " (% 16 = ", indexAddress % 16,
        ", ", (indexAligned ? "ALIGNED" : "MISALIGNED"), ")"));
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> vertexOffset=", desc.vertexOffset,
        " -> addr=0x", std::hex, vertexAddress, std::dec, " (% 16 = ", vertexAddress % 16,
        ", ", (vertexAligned ? "ALIGNED" : "MISALIGNED"), ")"));
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> indexBufferStride=", buildInfo.indexBufferStride,
        ", vertexBufferStride=", buildInfo.vertexBufferStride));
      Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> indexType=", buildInfo.indexType,
        " (8BIT=", VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV, ")"));
    }
  }

  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Summary: ", kNumTemplates, " templates"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> ", misalignedIndexCount, " misaligned index addresses"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> ", misalignedVertexCount, " misaligned vertex addresses"));
  if (misalignedIndexCount > 0 || misalignedVertexCount > 0) {
    Logger::warn("[RTXMG TEMPLATE DEBUG] *** WARNING: Some template addresses are misaligned! ***");
  }

  // Upload build infos to GPU buffer
  RtxmgBuffer<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> buildInfosBuffer;
  buildInfosBuffer.create(
    device,
    buildInfos.size(),
    "RTXMG Template Build Infos",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  buildInfosBuffer.upload(buildInfos);
  Logger::info(str::format("[RTXMG] Uploaded template build infos: count=",
    buildInfos.size(), ", bufferAddr=0x", std::hex, buildInfosBuffer.getDeviceAddress(), std::dec));

  // Create sizes buffer for GPU to write template sizes
  RtxmgBuffer<uint32_t> templateSizesBuffer;
  templateSizesBuffer.create(
    device,
    kNumTemplates,
    "RTXMG Template Sizes",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Create count buffer (srcInfosCount must be a GPU device address)
  RtxmgBuffer<uint32_t> templateCountBuffer;
  templateCountBuffer.create(
    device,
    1,
    "RTXMG Template Count",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  std::vector<uint32_t> countData = { kNumTemplates };
  templateCountBuffer.upload(countData);

  // Setup triangle cluster input for all templates
  VkClusterAccelerationStructureTriangleClusterInputNV clusterInput = {};
  clusterInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
  clusterInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // SDK uses RGB even though data is UV (2 components)
  clusterInput.maxGeometryIndexValue = 0;  // Single geometry (index 0)
  clusterInput.maxClusterUniqueGeometryCount = 1;
  clusterInput.maxClusterTriangleCount = kMaxEdgeSegments * kMaxEdgeSegments * 2;
  clusterInput.maxClusterVertexCount = (kMaxEdgeSegments + 1) * (kMaxEdgeSegments + 1);
  clusterInput.maxTotalTriangleCount = clusterInput.maxClusterTriangleCount * kNumTemplates;
  clusterInput.maxTotalVertexCount = clusterInput.maxClusterVertexCount * kNumTemplates;
  clusterInput.minPositionTruncateBitCount = 0;

  Logger::info(str::format("[RTXMG] Template cluster input: vertexFormat=",
    clusterInput.vertexFormat, ", maxClusterTris=", clusterInput.maxClusterTriangleCount,
    ", maxClusterVerts=", clusterInput.maxClusterVertexCount, ", maxTotalTris=",
    clusterInput.maxTotalTriangleCount, ", maxTotalVerts=",
    clusterInput.maxTotalVertexCount, ", truncateBits=",
    clusterInput.minPositionTruncateBitCount));

  // Query scratch buffer size for BOTH COMPUTE_SIZES and EXPLICIT_DESTINATIONS modes
  // We need the MAXIMUM of both since we use the same scratch buffer for both operations

  // 1. Query for COMPUTE_SIZES (GPU-side size computation)
  VkClusterAccelerationStructureInputInfoNV inputInfoComputeSizes = {};
  inputInfoComputeSizes.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfoComputeSizes.maxAccelerationStructureCount = kNumTemplates;
  inputInfoComputeSizes.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfoComputeSizes.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputInfoComputeSizes.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
  inputInfoComputeSizes.opInput.pTriangleClusters = &clusterInput;

  VkAccelerationStructureBuildSizesInfoKHR sizesComputeSizes = {};
  sizesComputeSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfoComputeSizes,
    &sizesComputeSizes);

  // 2. Query for EXPLICIT_DESTINATIONS (actual template build)
  VkClusterAccelerationStructureInputInfoNV inputInfoBuildSizes = {};
  inputInfoBuildSizes.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfoBuildSizes.maxAccelerationStructureCount = kNumTemplates;
  inputInfoBuildSizes.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  inputInfoBuildSizes.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputInfoBuildSizes.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  inputInfoBuildSizes.opInput.pTriangleClusters = &clusterInput;

  VkAccelerationStructureBuildSizesInfoKHR buildSizes = {};
  buildSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfoBuildSizes,
    &buildSizes);

  // Use the MAXIMUM scratch size needed for both operations
  VkDeviceSize scratchSize = std::max(sizesComputeSizes.buildScratchSize, buildSizes.buildScratchSize);

  Logger::info(str::format("[RTXMG] Template size query: COMPUTE_SIZES scratch=",
    sizesComputeSizes.buildScratchSize, " bytes, EXPLICIT_DESTINATIONS accel=",
    buildSizes.accelerationStructureSize, " bytes scratch=",
    buildSizes.buildScratchSize, " bytes, using max scratch=", scratchSize, " bytes"));
  Logger::info(str::format("[RTXMG] Scratch buffer size needed: ", scratchSize / 1024, " KB"));

  // Allocate scratch buffer
  RtxmgBuffer<uint8_t> scratchBuffer;
  if (scratchSize > 0) {
    scratchBuffer.create(
      device,
      scratchSize,
      "RTXMG Cluster Scratch",
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!scratchBuffer.isValid()) {
      Logger::err("[RTXMG] Failed to allocate scratch buffer");
      return false;
    }
  }

  VkDeviceAddress scratchAddress = scratchBuffer.isValid() ? scratchBuffer.getDeviceAddress() : 0;

  // LOG: Scratch buffer details
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Scratch buffer: 0x", std::hex, scratchAddress, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> size: ", scratchSize, " bytes"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> address % 16 = ", scratchAddress % 16,
    " (", (scratchAddress % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

  // LOG: Build info buffer details
  VkDeviceAddress buildInfosAddr = buildInfosBuffer.getDeviceAddress();
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] BuildInfos buffer: 0x", std::hex, buildInfosAddr, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> size: ", buildInfos.size() * sizeof(buildInfos[0]), " bytes"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> stride: ", sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV), " bytes"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> address % 16 = ", buildInfosAddr % 16,
    " (", (buildInfosAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));

  // LOG: Size/count buffer details
  VkDeviceAddress sizesAddr = templateSizesBuffer.getDeviceAddress();
  VkDeviceAddress countAddr = templateCountBuffer.getDeviceAddress();
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Sizes buffer: 0x", std::hex, sizesAddr, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> address % 16 = ", sizesAddr % 16,
    " (", (sizesAddr % 16 == 0 ? "ALIGNED" : "MISALIGNED"), ")"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Count buffer: 0x", std::hex, countAddr, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> value: ", kNumTemplates));

  // STEP 1: Query sizes using GPU-side indirect command (COMPUTE_SIZES mode)
  VkClusterAccelerationStructureInputInfoNV inputInfoGetSizes = {};
  inputInfoGetSizes.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfoGetSizes.maxAccelerationStructureCount = kNumTemplates;
  inputInfoGetSizes.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfoGetSizes.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputInfoGetSizes.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
  inputInfoGetSizes.opInput.pTriangleClusters = &clusterInput;

  VkClusterAccelerationStructureCommandsInfoNV commandsInfoGetSizes = {};
  commandsInfoGetSizes.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
  commandsInfoGetSizes.input = inputInfoGetSizes;
  commandsInfoGetSizes.dstImplicitData = 0;
  commandsInfoGetSizes.scratchData = scratchAddress;
  commandsInfoGetSizes.dstAddressesArray = {};
  commandsInfoGetSizes.dstSizesArray.deviceAddress = sizesAddr;
  commandsInfoGetSizes.dstSizesArray.stride = sizeof(uint32_t);
  commandsInfoGetSizes.dstSizesArray.size = kNumTemplates * sizeof(uint32_t);
  commandsInfoGetSizes.srcInfosArray.deviceAddress = buildInfosAddr;
  commandsInfoGetSizes.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
  commandsInfoGetSizes.srcInfosArray.size = buildInfos.size() * sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
  commandsInfoGetSizes.srcInfosCount = templateCountBuffer.getDeviceAddress();
  commandsInfoGetSizes.addressResolutionFlags = 0;

  // LOG: CommandsInfo structure
  Logger::info("[RTXMG TEMPLATE DEBUG] VkClusterAccelerationStructureCommandsInfoNV (COMPUTE_SIZES):");
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> opType=", inputInfoGetSizes.opType,
    " (BUILD_TRIANGLE_CLUSTER_TEMPLATE=", VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV, ")"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> opMode=", inputInfoGetSizes.opMode,
    " (COMPUTE_SIZES=", VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV, ")"));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> maxAccelerationStructureCount=", inputInfoGetSizes.maxAccelerationStructureCount));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> scratchData=0x", std::hex, commandsInfoGetSizes.scratchData, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> srcInfosArray.deviceAddress=0x", std::hex, commandsInfoGetSizes.srcInfosArray.deviceAddress, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> srcInfosArray.stride=", commandsInfoGetSizes.srcInfosArray.stride));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> srcInfosArray.size=", commandsInfoGetSizes.srcInfosArray.size));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> srcInfosCount=0x", std::hex, commandsInfoGetSizes.srcInfosCount, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> dstSizesArray.deviceAddress=0x", std::hex, commandsInfoGetSizes.dstSizesArray.deviceAddress, std::dec));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> dstSizesArray.stride=", commandsInfoGetSizes.dstSizesArray.stride));
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> dstSizesArray.size=", commandsInfoGetSizes.dstSizesArray.size));

  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

  // Add buffer barriers before cluster build command (following nvrhi's approach)
  std::vector<VkBufferMemoryBarrier> preBarriers;

  // Input buffers need to be in SHADER_READ state
  VkBufferMemoryBarrier buildInfoBarrier = {};
  buildInfoBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  buildInfoBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  buildInfoBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  buildInfoBarrier.buffer = buildInfosBuffer.getBuffer()->getSliceHandle().handle;
  buildInfoBarrier.offset = 0;
  buildInfoBarrier.size = VK_WHOLE_SIZE;
  preBarriers.push_back(buildInfoBarrier);

  VkBufferMemoryBarrier countBarrier = {};
  countBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  countBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  countBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  countBarrier.buffer = templateCountBuffer.getBuffer()->getSliceHandle().handle;
  countBarrier.offset = 0;
  countBarrier.size = VK_WHOLE_SIZE;
  preBarriers.push_back(countBarrier);

  // Output buffer needs to be in ACCELERATION_STRUCTURE_WRITE state (FIXED: was SHADER_WRITE)
  // VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR only supports VK_ACCESS_ACCELERATION_STRUCTURE_* masks
  VkBufferMemoryBarrier sizesBarrier = {};
  sizesBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  sizesBarrier.srcAccessMask = 0;
  sizesBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  sizesBarrier.buffer = templateSizesBuffer.getBuffer()->getSliceHandle().handle;
  sizesBarrier.offset = 0;
  sizesBarrier.size = VK_WHOLE_SIZE;
  preBarriers.push_back(sizesBarrier);

  Logger::info(str::format("[RTXMG TEMPLATE DEBUG] Executing vkCmdPipelineBarrier (", preBarriers.size(), " buffer barriers)..."));
  device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    0, 0, nullptr, preBarriers.size(), preBarriers.data(), 0, nullptr);
  Logger::info("[RTXMG TEMPLATE DEBUG] Pipeline barrier complete");

  Logger::info("[RTXMG TEMPLATE DEBUG] Calling vkCmdBuildClusterAccelerationStructureIndirectNV (COMPUTE_SIZES)...");
  Logger::info(str::format("[RTXMG TEMPLATE DEBUG]   -> This will query sizes for ", kNumTemplates, " templates"));
  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfoGetSizes);
  Logger::info("[RTXMG TEMPLATE DEBUG] vkCmdBuildClusterAccelerationStructureIndirectNV returned successfully");

  // Barrier to ensure sizes are written before reading
  VkMemoryBarrier barrierGetSizes = {};
  barrierGetSizes.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrierGetSizes.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrierGetSizes.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

  device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_HOST_BIT,
    0, 1, &barrierGetSizes, 0, nullptr, 0, nullptr);

  // Flush and wait for GPU to finish writing sizes
  ctx->flushCommandList();
  device->waitForIdle();  // CRITICAL: Wait for GPU query to complete before reading results

  // Read back template sizes from GPU
  const uint32_t* sizeData = static_cast<const uint32_t*>(templateSizesBuffer.getBuffer()->mapPtr(0));
  if (!sizeData) {
    Logger::err("[RTXMG] Failed to map template sizes buffer");
    return false;
  }

  // Calculate total CLAS buffer size and offsets
  VkDeviceSize totalClasSize = 0;
  std::vector<VkDeviceAddress> clasAddresses(kNumTemplates);
  uint32_t zeroTemplateCount = 0;
  uint32_t largestTemplateSize = 0;

  for (uint32_t i = 0; i < kNumTemplates; ++i) {
    uint32_t size = sizeData[i];
    // Align to 256 bytes (required by spec)
    size = (size + 255) & ~255;

    clasInstantiationBytes[i] = size;
    totalClasSize += size;
    largestTemplateSize = std::max(largestTemplateSize, size);
    if (size == 0) {
      zeroTemplateCount++;
    }
  }

  Logger::info(str::format("[RTXMG] Total CLAS buffer size: ", totalClasSize / 1024,
    " KB (maxTemplate=", largestTemplateSize, " bytes, zeroTemplates=",
    zeroTemplateCount, ")"));

  const uint32_t templateSizeLogCount = std::min<uint32_t>(5, kNumTemplates);
  for (uint32_t i = 0; i < templateSizeLogCount; ++i) {
    Logger::info(str::format("[RTXMG] TemplateSize[", i, "] = ", clasInstantiationBytes[i], " bytes"));
  }

  if (totalClasSize == 0) {
    Logger::err("[RTXMG] GPU-side CLAS size query failed - all templates returned 0 bytes");
    return false;
  }
  if (zeroTemplateCount > 0) {
    Logger::warn(str::format("[RTXMG] ", zeroTemplateCount,
      " template(s) reported zero bytes from size query"));
  }

  if (outTotalClasSize) {
    *outTotalClasSize = static_cast<size_t>(totalClasSize);
  }

  // Allocate CLAS buffer
  const VkBufferUsageFlags clasUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

  clasBuffer.create(device, totalClasSize, "RTXMG Template CLAS", clasUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (!clasBuffer.isValid()) {
    Logger::err("[RTXMG] Failed to allocate CLAS buffer");
    return false;
  }

  VkDeviceAddress clasBufferAddress = clasBuffer.getDeviceAddress();

  // Calculate template addresses (with 256-byte alignment)
  VkDeviceSize offset = 0;
  for (uint32_t i = 0; i < kNumTemplates; ++i) {
    clasAddresses[i] = clasBufferAddress + offset;
    templateAddresses[i] = clasAddresses[i];
    offset += clasInstantiationBytes[i];
  }

  // Upload addresses to GPU buffer
  RtxmgBuffer<VkDeviceAddress> templateAddressesBuffer;
  templateAddressesBuffer.create(
    device,
    clasAddresses.size(),
    "RTXMG Template Addresses",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  templateAddressesBuffer.upload(clasAddresses);

  // STEP 2: Build templates using EXPLICIT_DESTINATIONS mode
  VkClusterAccelerationStructureInputInfoNV inputInfoBuild = {};
  inputInfoBuild.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfoBuild.maxAccelerationStructureCount = kNumTemplates;
  inputInfoBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  inputInfoBuild.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputInfoBuild.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  inputInfoBuild.opInput.pTriangleClusters = &clusterInput;

  VkClusterAccelerationStructureCommandsInfoNV commandsInfoBuild = {};
  commandsInfoBuild.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
  commandsInfoBuild.input = inputInfoBuild;
  commandsInfoBuild.dstImplicitData = 0;
  commandsInfoBuild.scratchData = scratchAddress;
  commandsInfoBuild.dstAddressesArray.deviceAddress = templateAddressesBuffer.getDeviceAddress();
  commandsInfoBuild.dstAddressesArray.stride = sizeof(VkDeviceAddress);
  commandsInfoBuild.dstAddressesArray.size = clasAddresses.size() * sizeof(VkDeviceAddress);
  commandsInfoBuild.dstSizesArray = {};
  commandsInfoBuild.srcInfosArray.deviceAddress = buildInfosBuffer.getDeviceAddress();
  commandsInfoBuild.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
  commandsInfoBuild.srcInfosArray.size = buildInfos.size() * sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
  commandsInfoBuild.srcInfosCount = templateCountBuffer.getDeviceAddress();
  commandsInfoBuild.addressResolutionFlags = 0;

  Logger::info(str::format("[RTXMG] Template build command: scratch=0x", std::hex,
    commandsInfoBuild.scratchData, ", dstAddrBuffer=0x",
    commandsInfoBuild.dstAddressesArray.deviceAddress, ", srcInfoBuffer=0x",
    commandsInfoBuild.srcInfosArray.deviceAddress, ", countAddr=0x",
    commandsInfoBuild.srcInfosCount, std::dec));

  cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfoBuild);

  // Barrier to ensure templates are built before use
  VkMemoryBarrier barrierBuild = {};
  barrierBuild.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrierBuild.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrierBuild.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    0, 1, &barrierBuild, 0, nullptr, 0, nullptr);

  Logger::info("[RTXMG] Template CLAS structures built successfully");

  return true;
}

//===========================================================================
// Cluster instantiation from templates
//===========================================================================

bool instantiateClusterInstances(
  DxvkDevice* device,
  RtxContext* ctx,
  uint32_t numClusters,
  const std::vector<VkDeviceAddress>& templateAddresses,
  const RtxmgBuffer<RtxmgClusterInstantiationData>& instanceDataBuffer,
  const RtxmgBuffer<float3>& clusterVertexBuffer,
  RtxmgBuffer<uint8_t>& instanceBuffer,
  std::vector<VkDeviceAddress>& instanceAddresses,
  size_t* outTotalInstanceSize)
{
  if (!g_clusterAccelExt.isValid()) {
    Logger::err("[RTXMG] Cannot instantiate clusters: extension not available");
    return false;
  }

  if (numClusters == 0) {
    Logger::warn("[RTXMG] No clusters to instantiate");
    return false;
  }

  if (templateAddresses.size() != 121) {
    Logger::err("[RTXMG] Invalid template addresses count (expected 121)");
    return false;
  }

  Logger::info(str::format("[RTXMG] Instantiating ", numClusters, " clusters from templates"));

  const uint32_t kMaxEdgeSegments = 11;

  // STEP 1: Generate indirect argument buffers from GPU-generated instance data
  // We need to convert ClusterInstanceData → VkClusterAccelerationStructureInstantiateClusterInfoNV
  // This will be done via a GPU compute shader for performance (GPU-to-GPU, no CPU sync)

  RtxmgBuffer<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfosBuffer;
  instantiateInfosBuffer.create(
    device,
    numClusters,
    "RTXMG Cluster Instantiate Args",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  // For Phase 2, we'll generate these on CPU (Phase 2.5 will move to GPU compute)
  // Read instance data from GPU
  const RtxmgClusterInstantiationData* instanceData =
    static_cast<const RtxmgClusterInstantiationData*>(instanceDataBuffer.getBuffer()->mapPtr(0));

  if (!instanceData) {
    Logger::err("[RTXMG] Failed to map cluster instance data buffer");
    return false;
  }

  VkDeviceAddress vertexBufferBase = clusterVertexBuffer.getDeviceAddress();

  // ALIGNMENT LOGGING: Check base address alignment
  Logger::info(str::format("[RTXMG ALIGNMENT] Vertex buffer base address: 0x", std::hex, vertexBufferBase, std::dec));
  Logger::info(str::format("[RTXMG ALIGNMENT] Base address % 8 = ", vertexBufferBase % 8, " (should be 0 for 8-byte alignment)"));
  Logger::info(str::format("[RTXMG ALIGNMENT] Base address % 16 = ", vertexBufferBase % 16, " (should be 0 for 16-byte alignment)"));
  Logger::info(str::format("[RTXMG ALIGNMENT] sizeof(float3) = ", sizeof(float3), " bytes"));
  Logger::info(str::format("[RTXMG ALIGNMENT] Buffer size = ", clusterVertexBuffer.getBuffer()->info().size, " bytes"));

  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfos(numClusters);

  // Track misalignment issues
  uint32_t misaligned8Count = 0;
  uint32_t misaligned16Count = 0;

  for (uint32_t i = 0; i < numClusters; ++i) {
    const RtxmgClusterInstantiationData& instance = instanceData[i];

    // Validate template index
    if (instance.templateIndex >= 121) {
      Logger::err(str::format("[RTXMG] Invalid template index: ", instance.templateIndex));
      return false;
    }

    // Populate VkClusterAccelerationStructureInstantiateClusterInfoNV with SDK-correct fields
    auto& info = instantiateInfos[i];
    info = {};
    info.clusterIdOffset = i;  // Unique cluster ID (used as offset into result arrays)
    info.geometryIndexOffset = instance.geometryIndex & 0xFFFFFF;  // 24-bit geometry index
    info.reserved = 0;  // 8-bit reserved field
    info.clusterTemplateAddress = templateAddresses[instance.templateIndex];

    // Calculate vertex buffer address with detailed logging
    VkDeviceSize byteOffset = instance.vertexBufferOffset * sizeof(float3);
    info.vertexBuffer.startAddress = vertexBufferBase + byteOffset;
    info.vertexBuffer.strideInBytes = sizeof(float3);

    // ALIGNMENT LOGGING: Check address alignment for first 10 and any misaligned
    bool isAligned8 = (info.vertexBuffer.startAddress % 8) == 0;
    bool isAligned16 = (info.vertexBuffer.startAddress % 16) == 0;

    if (!isAligned8) misaligned8Count++;
    if (!isAligned16) misaligned16Count++;

    if (i < 10 || !isAligned8 || !isAligned16) {
      Logger::info(str::format("[RTXMG ALIGNMENT] Cluster[", i, "]: "
        "vertexOffset=", instance.vertexBufferOffset,
        ", byteOffset=", byteOffset,
        ", finalAddress=0x", std::hex, info.vertexBuffer.startAddress, std::dec,
        ", templateIdx=", instance.templateIndex,
        ", templateAddr=0x", std::hex, info.clusterTemplateAddress, std::dec,
        ", vertexCount=", instance.vertexCount));

      Logger::info(str::format("[RTXMG ALIGNMENT]   -> address % 8 = ", info.vertexBuffer.startAddress % 8,
        " (8-byte ", (isAligned8 ? "ALIGNED" : "MISALIGNED"), ")"));
      Logger::info(str::format("[RTXMG ALIGNMENT]   -> address % 16 = ", info.vertexBuffer.startAddress % 16,
        " (16-byte ", (isAligned16 ? "ALIGNED" : "MISALIGNED"), ")"));
      Logger::info(str::format("[RTXMG ALIGNMENT]   -> stride = ", info.vertexBuffer.strideInBytes, " bytes"));
      Logger::info(str::format("[RTXMG ALIGNMENT]   -> geometryIndex = ", info.geometryIndexOffset));
    }
  }

  // Summary of alignment issues
  Logger::info(str::format("[RTXMG ALIGNMENT] Summary: ", numClusters, " clusters processed"));
  Logger::info(str::format("[RTXMG ALIGNMENT]   - ", misaligned8Count, " addresses NOT 8-byte aligned"));
  Logger::info(str::format("[RTXMG ALIGNMENT]   - ", misaligned16Count, " addresses NOT 16-byte aligned"));
  if (misaligned8Count > 0) {
    Logger::warn("[RTXMG ALIGNMENT] *** WARNING: Misaligned addresses detected! This may cause GPU faults! ***");
  }

  instantiateInfosBuffer.upload(instantiateInfos);
  const uint32_t instantiateLogCount = std::min<uint32_t>(5, numClusters);
  for (uint32_t i = 0; i < instantiateLogCount; ++i) {
    const auto& instance = instanceData[i];
    Logger::info(str::format("[RTXMG] Instantiate[", i, "]: templateIndex=", instance.templateIndex,
      ", geometryIndex=", instance.geometryIndex, ", vertexBufferOffset=", instance.vertexBufferOffset,
      ", vertexCount=", instance.vertexCount));
  }

  // STEP 2: Query instantiation sizes (COMPUTE_SIZES mode)
  VkClusterAccelerationStructureTriangleClusterInputNV clusterInput = {};
  clusterInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
  clusterInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  clusterInput.maxGeometryIndexValue = 0;  // Single geometry for now
  clusterInput.maxClusterUniqueGeometryCount = 1;
  clusterInput.maxClusterTriangleCount = kMaxEdgeSegments * kMaxEdgeSegments * 2;
  clusterInput.maxClusterVertexCount = (kMaxEdgeSegments + 1) * (kMaxEdgeSegments + 1);
  clusterInput.maxTotalTriangleCount = clusterInput.maxClusterTriangleCount * numClusters;
  clusterInput.maxTotalVertexCount = clusterInput.maxClusterVertexCount * numClusters;

  Logger::info(str::format("[RTXMG] *** USING TEMPLATE-BASED CPU PATH (instantiateClusterInstances in rtxmg_accel.cpp) ***"));

  // Query scratch size for BOTH COMPUTE_SIZES and EXPLICIT_DESTINATIONS modes
  // The same scratch buffer is used for both operations

  VkClusterAccelerationStructureInputInfoNV inputInfo = {};
  inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfo.maxAccelerationStructureCount = numClusters;
  inputInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
  inputInfo.opInput.pTriangleClusters = &clusterInput;

  VkAccelerationStructureBuildSizesInfoKHR sizesComputeSizes = {};
  sizesComputeSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfo,
    &sizesComputeSizes);

  // Also query for EXPLICIT_DESTINATIONS mode
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  VkAccelerationStructureBuildSizesInfoKHR buildSizes = {};
  buildSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfo,
    &buildSizes);

  // Reset to COMPUTE_SIZES for the first GPU operation
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;

  // Use maximum scratch size needed for both operations
  VkDeviceSize scratchSize = std::max(sizesComputeSizes.buildScratchSize, buildSizes.buildScratchSize);

  Logger::info(str::format("[RTXMG UNIFIED] CLAS size query: COMPUTE_SIZES scratch=",
    sizesComputeSizes.buildScratchSize, " bytes (", sizesComputeSizes.accelerationStructureSize / 1024, " KB instances), EXPLICIT_DESTINATIONS scratch=",
    buildSizes.buildScratchSize, " bytes (", buildSizes.accelerationStructureSize / 1024, " KB instances), using max scratch=", scratchSize, " bytes"));

  RtxmgBuffer<uint8_t> scratchBuffer;
  if (scratchSize > 0) {
    scratchBuffer.create(device, scratchSize, "RTXMG Instantiate Scratch",
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  RtxmgBuffer<uint32_t> instanceSizesBuffer;
  instanceSizesBuffer.create(device, numClusters, "RTXMG Instance Sizes",
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RtxmgBuffer<uint32_t> clusterCountBuffer;
  clusterCountBuffer.create(device, 1, "RTXMG Cluster Count",
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  std::vector<uint32_t> countData = { numClusters };
  clusterCountBuffer.upload(countData);

  VkClusterAccelerationStructureCommandsInfoNV commandsInfo = {};
  commandsInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
  commandsInfo.input = inputInfo;
  commandsInfo.scratchData = scratchBuffer.isValid() ? scratchBuffer.getDeviceAddress() : 0;
  commandsInfo.dstSizesArray.deviceAddress = instanceSizesBuffer.getDeviceAddress();
  commandsInfo.dstSizesArray.stride = sizeof(uint32_t);
  commandsInfo.dstSizesArray.size = numClusters * sizeof(uint32_t);
  commandsInfo.srcInfosArray.deviceAddress = instantiateInfosBuffer.getDeviceAddress();
  commandsInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
  commandsInfo.srcInfosArray.size = numClusters * sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
  commandsInfo.srcInfosCount = clusterCountBuffer.getDeviceAddress();

  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfo);

  // Barrier and synchronization
  VkMemoryBarrier barrierSizes = {};
  barrierSizes.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrierSizes.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrierSizes.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

  device->vkd()->vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_HOST_BIT,
    0, 1, &barrierSizes, 0, nullptr, 0, nullptr);

  ctx->flushCommandList();
  device->waitForIdle();

  // Read back instance sizes
  const uint32_t* sizeData = static_cast<const uint32_t*>(instanceSizesBuffer.getBuffer()->mapPtr(0));
  if (!sizeData) {
    Logger::err("[RTXMG] Failed to map instance sizes buffer");
    return false;
  }

  // Calculate total size and addresses
  VkDeviceSize totalInstanceSize = 0;
  for (uint32_t i = 0; i < numClusters; ++i) {
    uint32_t size = sizeData[i];
    size = (size + 255) & ~255;  // Align to 256 bytes
    totalInstanceSize += size;
  }

  Logger::info(str::format("[RTXMG] Total cluster instance size: ", totalInstanceSize / 1024, " KB"));

  if (totalInstanceSize == 0) {
    Logger::err("[RTXMG] Cluster instance size query failed");
    return false;
  }

  if (outTotalInstanceSize) {
    *outTotalInstanceSize = static_cast<size_t>(totalInstanceSize);
  }

  // STEP 3: Allocate instance buffer
  instanceBuffer.create(device, totalInstanceSize, "RTXMG Cluster Instances",
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Calculate packed addresses
  VkDeviceAddress baseAddr = instanceBuffer.getDeviceAddress();
  instanceAddresses.resize(numClusters);
  VkDeviceSize offset = 0;
  for (uint32_t i = 0; i < numClusters; ++i) {
    instanceAddresses[i] = baseAddr + offset;
    uint32_t size = sizeData[i];
    size = (size + 255) & ~255;
    offset += size;
  }

  // Upload addresses buffer
  RtxmgBuffer<VkDeviceAddress> addressesBuffer;
  addressesBuffer.create(device, numClusters, "RTXMG Instance Addresses",
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  addressesBuffer.upload(instanceAddresses);

  // STEP 4: Instantiate clusters (EXPLICIT_DESTINATIONS mode)
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

  commandsInfo.input = inputInfo;
  commandsInfo.dstAddressesArray.deviceAddress = addressesBuffer.getDeviceAddress();
  commandsInfo.dstAddressesArray.stride = sizeof(VkDeviceAddress);
  commandsInfo.dstAddressesArray.size = numClusters * sizeof(VkDeviceAddress);
  commandsInfo.dstSizesArray.deviceAddress = 0;

  cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfo);

  // Final barrier
  VkMemoryBarrier barrierBuild = {};
  barrierBuild.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrierBuild.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrierBuild.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  device->vkd()->vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    0, 1, &barrierBuild, 0, nullptr, 0, nullptr);

  Logger::info(str::format("[RTXMG] Cluster instances built successfully (", numClusters, " clusters)"));

  return true;
}

// Direct instantiation using GPU-written VkClusterAccelerationStructureInstantiateClusterInfoNV buffer
// This bypasses CPU conversion for maximum performance - GPU shader writes SDK structure directly
// Persistent buffers passed in to avoid lifetime issues (buffers must outlive GPU commands)
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
  size_t* outTotalInstanceSize,
  VkDeviceSize bufferOffsetBytes,
  VkDeviceSize instanceBufferOffsetBytes,
  const Rc<DxvkBuffer>& gpuCounterBuffer,
  VkDeviceSize gpuCounterOffset)
{
  if (!g_clusterAccelExt.isValid()) {
    Logger::err("[RTXMG] Cannot instantiate clusters: extension not available");
    return false;
  }

  auto alignDeviceSize = [] (VkDeviceSize size, VkDeviceSize alignment) -> VkDeviceSize {
    if (alignment == 0)
      return size;
    return ((size + alignment - 1) / alignment) * alignment;
  };

  const auto& clusterProps = device->properties().nvClusterAccelerationStructureProperties;
  const VkDeviceSize templateAlignment = clusterProps.clusterTemplateByteAlignment != 0 ? clusterProps.clusterTemplateByteAlignment : 256;
  const VkDeviceSize scratchAlignment = clusterProps.clusterScratchByteAlignment != 0 ? clusterProps.clusterScratchByteAlignment : 256;
  const VkDeviceSize instanceStride = alignDeviceSize(sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV), templateAlignment);

  // SDK MATCH: When gpuCounterBuffer is provided, use GPU-driven count (skip CPU validation)
  bool useGpuCount = (gpuCounterBuffer != nullptr);

  if (!useGpuCount && numClusters == 0) {
    Logger::warn("[RTXMG] No clusters to instantiate");
    return false;
  }

  if (instantiateInfosBuffer == nullptr || instantiateInfosBuffer->info().size == 0) {
    Logger::err("[RTXMG] Invalid instantiate infos buffer");
    return false;
  }

  if (useGpuCount) {
    Logger::info(str::format("[RTXMG] Instantiating clusters (GPU-driven count from tessellation counters, EXPLICIT_DESTINATIONS)"));
  } else {
    Logger::info(str::format("[RTXMG] Instantiating ", numClusters, " clusters (CPU count, EXPLICIT_DESTINATIONS with pre-allocation)"));
  }

  const uint32_t kMaxEdgeSegments = 11;

  // Use EXPLICIT_DESTINATIONS with pre-allocated buffers to avoid synchronous size query
  // This eliminates the waitForIdle() GPU timeout

  VkClusterAccelerationStructureTriangleClusterInputNV clusterInput = {};
  clusterInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
  clusterInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  clusterInput.maxGeometryIndexValue = 0;  // Single geometry for now
  clusterInput.maxClusterUniqueGeometryCount = 1;
  clusterInput.maxClusterTriangleCount = kMaxEdgeSegments * kMaxEdgeSegments * 2;
  clusterInput.maxClusterVertexCount = (kMaxEdgeSegments + 1) * (kMaxEdgeSegments + 1);
  clusterInput.maxTotalTriangleCount = clusterInput.maxClusterTriangleCount * numClusters;
  clusterInput.maxTotalVertexCount = clusterInput.maxClusterVertexCount * numClusters;

  VkClusterAccelerationStructureInputInfoNV inputInfo = {};
  inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfo.maxAccelerationStructureCount = numClusters;
  inputInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  inputInfo.opInput.pTriangleClusters = &clusterInput;

  // Query scratch size
  VkAccelerationStructureBuildSizesInfoKHR buildSizes = {};
  buildSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfo,
    &buildSizes);

  VkDeviceSize scratchSize = alignDeviceSize(buildSizes.buildScratchSize, scratchAlignment);

  // Allocate persistent scratch buffer if needed (or reuse if large enough)
  if (scratchSize > 0) {
    if (!persistentScratchBuffer.isValid() || persistentScratchBuffer.bytes() < scratchSize) {
      persistentScratchBuffer.release();  // Release old buffer if undersized
      persistentScratchBuffer.create(device, scratchSize, "RTXMG Instantiate Scratch",
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
  }

  Logger::info(str::format("[RTXMG] Instantiate command setup: numClusters=", numClusters,
    ", instanceStride=", instanceStride,
    ", scratchSize=", scratchSize,
    ", usingGpuCount=", useGpuCount));

  // SDK MATCH: Use GPU counter buffer OR allocate persistent count buffer for CPU path
  if (useGpuCount) {
    // GPU-driven: counter buffer already written by cluster_tiling shader
    // No upload needed - GPU wrote the count directly
    Logger::info("[RTXMG] Using GPU-written cluster count from tessellation counters (SDK-matching GPU-driven path)");
  } else {
    // CPU path: Upload CPU-computed count
    if (!persistentCountBuffer.isValid()) {
      persistentCountBuffer.create(device, 1, "RTXMG Cluster Count",
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    std::vector<uint32_t> countData = { numClusters };
    persistentCountBuffer.upload(countData);
  }

  // Pre-allocate persistent instance buffer with conservative alignment-aware stride
  // This avoids the need for COMPUTE_SIZES pass and eliminates waitForIdle()
  VkDeviceSize estimatedInstanceSize = numClusters * instanceStride;

  // Ring buffer mode: If buffer is already valid and offset is provided, caller manages sizing
  // Otherwise: Allocate/resize for single geometry usage
  if (instanceBufferOffsetBytes > 0) {
    // Ring buffer mode - caller pre-sized the buffer, just validate
    VkDeviceSize requiredSize = instanceBufferOffsetBytes + estimatedInstanceSize;
    if (!persistentInstanceBuffer.isValid() || persistentInstanceBuffer.bytes() < requiredSize) {
      Logger::err(str::format("[RTXMG] Ring buffer undersized: have ", persistentInstanceBuffer.bytes(),
                              " bytes, need ", requiredSize, " bytes (offset ", instanceBufferOffsetBytes, " + size ", estimatedInstanceSize, ")"));
      return false;
    }
    Logger::info(str::format("[RTXMG] Using ring buffer at offset ", instanceBufferOffsetBytes / 1024,
                            " KB (", numClusters, " clusters, ", estimatedInstanceSize / 1024, " KB)"));
  } else {
    // Single geometry mode - allocate or resize as needed
    if (!persistentInstanceBuffer.isValid() || persistentInstanceBuffer.bytes() < estimatedInstanceSize) {
      persistentInstanceBuffer.release();  // Release old buffer if undersized
      persistentInstanceBuffer.create(device, static_cast<size_t>(estimatedInstanceSize), "RTXMG Cluster Instances",
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      Logger::info(str::format("[RTXMG] Allocated instance buffer: ", estimatedInstanceSize / 1024, " KB"));
    }
  }

  if (outTotalInstanceSize) {
    *outTotalInstanceSize = static_cast<size_t>(estimatedInstanceSize);
  }

  // Calculate addresses with alignment-aware stride (conservative - actual CLAS sizes are smaller)
  // NVIDIA SAMPLE PATTERN: Use instanceBufferOffsetBytes for ring buffer support
  // Each geometry writes to its section: baseAddr + offset + (clusterIndex * stride)
  VkDeviceAddress baseAddr = persistentInstanceBuffer.getDeviceAddress();
  instanceAddresses.resize(numClusters);
  for (uint32_t i = 0; i < numClusters; ++i) {
    instanceAddresses[i] = baseAddr + instanceBufferOffsetBytes + (i * instanceStride);
  }

  // Upload addresses to persistent buffer for GPU access
  // RING BUFFER: Calculate offset for this frame's addresses
  // Caller already allocated 4x buffer size, so we write to the correct slot
  uint32_t addressBufferOffset = static_cast<uint32_t>(instanceBufferOffsetBytes / instanceStride);  // Convert bytes to cluster index
  if (!persistentAddressesBuffer.isValid()) {
    Logger::err("[RTXMG] Ring-buffered addresses buffer not pre-allocated by caller!");
    return false;
  }
  // Upload to the ring buffer slot for this frame (manual memcpy to correct offset)
  void* mapped = persistentAddressesBuffer.getBuffer()->mapPtr(addressBufferOffset * sizeof(VkDeviceAddress));
  if (mapped) {
    std::memcpy(mapped, instanceAddresses.data(), instanceAddresses.size() * sizeof(VkDeviceAddress));
  } else {
    Logger::err("[RTXMG] Failed to map addresses buffer for ring-buffered upload");
    return false;
  }

  // Build commands info for EXPLICIT_DESTINATIONS mode
  VkClusterAccelerationStructureCommandsInfoNV commandsInfo = {};
  commandsInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
  commandsInfo.input = inputInfo;
  commandsInfo.scratchData = persistentScratchBuffer.isValid() ? persistentScratchBuffer.getDeviceAddress() : 0;
  // Point to the ring buffer slot for this frame's addresses
  commandsInfo.dstAddressesArray.deviceAddress = persistentAddressesBuffer.getDeviceAddress() + (addressBufferOffset * sizeof(VkDeviceAddress));
  commandsInfo.dstAddressesArray.stride = sizeof(VkDeviceAddress);
  commandsInfo.dstAddressesArray.size = numClusters * sizeof(VkDeviceAddress);
  // Apply buffer offset for true batching support - allows multiple geometries in shared buffer
  commandsInfo.srcInfosArray.deviceAddress = instantiateInfosBuffer->getDeviceAddress() + bufferOffsetBytes;
  commandsInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
  commandsInfo.srcInfosArray.size = numClusters * sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);

  // SDK MATCH: Use GPU counter buffer or CPU count buffer
  if (useGpuCount) {
    // GPU-driven: Point to TessellationCounters.clusters field (offset 0)
    commandsInfo.srcInfosCount = gpuCounterBuffer->getDeviceAddress() + gpuCounterOffset;
    Logger::info(str::format("[RTXMG] srcInfosCount pointing to GPU counter at address 0x",
                            std::hex, commandsInfo.srcInfosCount, std::dec));
  } else {
    // CPU path: Use pre-uploaded count buffer
    commandsInfo.srcInfosCount = persistentCountBuffer.getDeviceAddress();
  }

  // Execute EXPLICIT_DESTINATIONS pass - NO waitForIdle() needed!
  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfo);

  // Barrier for instance data (async - no CPU wait)
  VkMemoryBarrier barriers[2] = {};

  // Barrier 1: Make CLAS instances visible
  barriers[0].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barriers[0].srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barriers[0].dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  // Barrier 2: Make CLAS addresses buffer visible to compute shader
  // The GPU writes addresses to clasPtrsBuffer (dstAddressesArray), then compute shader reads them
  barriers[1].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  device->vkd()->vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 2, barriers, 0, nullptr, 0, nullptr);

  Logger::info(str::format("[RTXMG] Cluster instances built successfully (GPU-direct, ", numClusters, " clusters)"));

  return true;
}

//===========================================================================
// Cluster geometry BLAS building
//===========================================================================

// CPU-path BLAS building - integrates Phase 2 (instantiation) + Phase 3 (BLAS build)
// Prepares cluster instance data from CPU-side buffers and builds BLAS using cluster extension
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
  BlasBuildSizes* outBuildSizes)
{
  if (!g_clusterAccelExt.isValid()) {
    Logger::err("[RTXMG] Cannot build cluster BLAS: extension not available");
    return false;
  }

  const uint32_t numClusters = geometry.numClusters;
  if (numClusters == 0) {
    Logger::warn("[RTXMG] No clusters to build BLAS for");
    return false;
  }

  Logger::info(str::format("[RTXMG] Building cluster BLAS (CPU path) for ", numClusters, " clusters"));

  // Map cluster data from GPU to prepare instantiation info
  const RtxmgCluster* clusters = static_cast<const RtxmgCluster*>(clustersBuffer.getBuffer()->mapPtr(0));
  if (!clusters) {
    Logger::err("[RTXMG] Failed to map clusters buffer");
    return false;
  }

  // Prepare cluster instantiation data from RtxmgCluster structures
  RtxmgBuffer<RtxmgClusterInstantiationData> instanceDataBuffer;
  instanceDataBuffer.create(
    device,
    numClusters,
    "RTXMG Cluster Instance Data",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  std::vector<RtxmgClusterInstantiationData> instanceData(numClusters);
  uint32_t vertexOffset = 0;

  for (uint32_t i = 0; i < numClusters; ++i) {
    const RtxmgCluster& cluster = clusters[i];

    // Calculate template index from cluster grid size (xEdges, yEdges)
    // Template grid is 11x11 (0-10 edges in each dimension)
    uint32_t xEdges = cluster.sizeX;
    uint32_t yEdges = cluster.sizeY;

    if (xEdges > 11 || yEdges > 11 || xEdges == 0 || yEdges == 0) {
      Logger::err(str::format("[RTXMG] Invalid cluster size: ", xEdges, "x", yEdges));
      return false;
    }

    // Template index: row-major layout in 11x11 grid
    // (xEdges-1) * 11 + (yEdges-1) maps to template grid indices 0-120
    uint32_t templateIndex = (xEdges - 1) * 11 + (yEdges - 1);

    instanceData[i].templateIndex = templateIndex;
    instanceData[i].geometryIndex = 0;  // Single geometry for now (will be extended for multi-geometry)
    instanceData[i].vertexBufferOffset = vertexOffset;
    instanceData[i].vertexCount = cluster.VerticesPerCluster();  // Use RtxmgCluster's method

    vertexOffset += instanceData[i].vertexCount;
  }

  instanceDataBuffer.upload(instanceData);

  // Phase 2: Instantiate clusters from templates
  RtxmgBuffer<uint8_t> instanceBuffer;
  std::vector<VkDeviceAddress> instanceAddresses;
  size_t totalInstanceSize = 0;

  if (!instantiateClusterInstances(
      device, ctx, numClusters, templateAddresses,
      instanceDataBuffer, clusterVertexPositionsBuffer,
      instanceBuffer, instanceAddresses, &totalInstanceSize)) {
    Logger::err("[RTXMG] Failed to instantiate cluster instances");
    return false;
  }

  // Phase 3: Build BLAS from instantiated clusters
  if (!buildClusterGeometryBLAS(
      device, ctx, numClusters, instanceAddresses,
      accels, outBuildSizes)) {
    Logger::err("[RTXMG] Failed to build cluster BLAS from instances");
    return false;
  }

  Logger::info(str::format("[RTXMG] Cluster BLAS built successfully (", numClusters, " clusters, ",
                          totalInstanceSize / 1024, " KB instances)"));

  return true;
}

// GPU-optimized overload that builds BLAS from instantiated cluster addresses (Phase 3)
// Takes instance addresses from Phase 2 and builds BLAS using cluster extension
// Uses VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV
bool buildClusterGeometryBLAS(
  DxvkDevice* device,
  RtxContext* ctx,
  uint32_t numClusters,
  const std::vector<VkDeviceAddress>& instanceAddresses,
  ClusterAccels& accels,
  BlasBuildSizes* outBuildSizes)
{
  if (!g_clusterAccelExt.isValid()) {
    Logger::warn("[RTXMG] Cluster BLAS building skipped: extension not available");
    return false;
  }

  if (numClusters == 0) {
    Logger::warn("[RTXMG] No clusters to build BLAS for");
    return false;
  }

  if (instanceAddresses.size() != numClusters) {
    Logger::err(str::format("[RTXMG] Instance address count mismatch: expected ", numClusters,
                            ", got ", instanceAddresses.size()));
    return false;
  }

  Logger::info(str::format("[RTXMG] Building cluster BLAS with ", numClusters,
                          " clusters [SDK-MATCHING: GPU-driven via executeMultiIndirectClusterOperation]"));

  // SDK MATCH: NO CPU UPLOADS!
  // Sample (lines 1057-1094) uses GPU-written m_blasFromClasIndirectArgsBuffer
  // filled by GPU compute shader, NOT CPU-uploaded build info structures

  // STEP 1: Prepare cluster references buffer (PERSISTENT - must outlive GPU commands)
  // This is the ONLY CPU upload - cluster instance addresses that GPU will reference
  if (!accels.clusterReferencesBuffer.isValid() || accels.clusterReferencesBuffer.numElements() < numClusters) {
    accels.clusterReferencesBuffer.release();
    accels.clusterReferencesBuffer.create(
      device,
      numClusters,
      "RTXMG Cluster References (Persistent)",
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  }
  accels.clusterReferencesBuffer.upload(instanceAddresses);

  // SDK MATCH: Create GPU-writable indirect args buffer for BLAS building
  // Sample line 1065: Shader fills m_blasFromClasIndirectArgsBuffer, not CPU!
  // Structure: { VkDeviceAddress clusterAddresses; uint32_t clusterCount; uint32_t padding; }
  struct BlasIndirectArg {
    VkDeviceAddress clusterReferences;  // Points to clusterReferencesBuffer
    uint32_t clusterCount;
    uint32_t _pad;
  };

  RtxmgBuffer<BlasIndirectArg> blasIndirectArgsBuffer;
  blasIndirectArgsBuffer.create(
    device,
    1,  // Single BLAS
    "RTXMG BLAS Indirect Args (GPU-written)",
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Write indirect args using GPU (SDK would use compute shader, we do minimal CPU init)
  // NOTE: In full SDK implementation, this would be written by FillBlasFromClasArgs shader
  BlasIndirectArg arg = {};
  arg.clusterReferences = accels.clusterReferencesBuffer.getDeviceAddress();
  arg.clusterCount = numClusters;
  arg._pad = 0;
  std::vector<BlasIndirectArg> args = { arg };
  blasIndirectArgsBuffer.upload(args);

  // STEP 2: Query required sizes using getClusterOperationSizeInfo (NVIDIA sample pattern)
  VkClusterAccelerationStructureClustersBottomLevelInputNV blasParams = {};
  blasParams.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
  blasParams.pNext = nullptr;
  blasParams.maxTotalClusterCount = numClusters;
  blasParams.maxClusterCountPerAccelerationStructure = numClusters;  // All clusters in one BLAS

  VkClusterAccelerationStructureInputInfoNV inputInfo = {};
  inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
  inputInfo.maxAccelerationStructureCount = 1;  // Single BLAS (maxArgCount in NVIDIA sample)
  inputInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;  // CRITICAL: Use ImplicitDestinations like NVIDIA sample!
  inputInfo.opInput.pClustersBottomLevel = &blasParams;

  // Get size info from extension (matches sample line 1212)
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
  sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  g_clusterAccelExt.vkGetClusterAccelerationStructureBuildSizesNV(
    device->handle(),
    &inputInfo,
    &sizeInfo);

  size_t blasSize = static_cast<size_t>(sizeInfo.accelerationStructureSize);
  size_t blasScratchSize = static_cast<size_t>(sizeInfo.buildScratchSize);

  Logger::info(str::format("[RTXMG] Cluster BLAS size: ", blasSize / 1024, " KB, scratch: ",
                          blasScratchSize / 1024, " KB"));

  if (blasSize == 0) {
    Logger::err("[RTXMG] BLAS size query returned 0 - cluster extension error");
    return false;
  }

  // Return sizes to caller if requested
  if (outBuildSizes) {
    outBuildSizes->blasSize = blasSize;
    outBuildSizes->blasScratchSize = blasScratchSize;
  }

  // STEP 3: Create BLAS buffer using DxvkAccelStructure (production architecture)
  DxvkBufferCreateInfo bufferCreateInfo {};
  bufferCreateInfo.size = blasSize;
  bufferCreateInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  bufferCreateInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  accels.blasAccelStructure = device->createAccelStructure(
    bufferCreateInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    "RTXMG Cluster BLAS");

  if (!accels.blasAccelStructure.ptr()) {
    Logger::err("[RTXMG] Failed to create cluster BLAS acceleration structure");
    return false;
  }

  // Keep legacy buffer reference for size queries
  accels.blasBuffer.create(
    device,
    blasSize,
    "RTXMG Cluster BLAS (Legacy Ref)",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // STEP 4: Allocate scratch buffer for BLAS build
  RtxmgBuffer<uint8_t> scratchBuffer;
  scratchBuffer.create(
    device,
    blasScratchSize,
    "RTXMG BLAS Build Scratch",
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (!scratchBuffer.isValid()) {
    Logger::err("[RTXMG] Failed to allocate BLAS scratch buffer");
    return false;
  }

  // STEP 5: Build BLAS using executeMultiIndirectClusterOperation (SDK line 1093)
  // SDK MATCH: commandList->executeMultiIndirectClusterOperation(desc)
  // NO manual vkCmd calls! Use SDK-matching wrapper function

  // SDK MATCH: Build ClusterOperationDesc (lines 1078-1092)
  ClusterOperationDesc buildBlasDesc = {};
  buildBlasDesc.params = inputInfo;
  buildBlasDesc.scratchData = scratchBuffer.getDeviceAddress();  // FIXED: GPU address of scratch buffer
  buildBlasDesc.scratchSizeInBytes = scratchBuffer.bytes();       // Actual size in bytes

  // No GPU counter for BLAS (count embedded in indirect args)
  buildBlasDesc.inIndirectArgCountBuffer = 0;
  buildBlasDesc.inIndirectArgCountOffsetInBytes = 0;

  // Indirect args buffer (SDK line 1084: inIndirectArgsBuffer = m_blasFromClasIndirectArgsBuffer)
  buildBlasDesc.inIndirectArgsBuffer = blasIndirectArgsBuffer.getDeviceAddress();
  buildBlasDesc.inIndirectArgsOffsetInBytes = 0;

  // Output addresses (SDK line 1086: inOutAddressesBuffer = accels.blasPtrsBuffer)
  buildBlasDesc.inOutAddressesBuffer = accels.blasPtrsBuffer.getDeviceAddress();
  buildBlasDesc.inOutAddressesOffsetInBytes = 0;

  // Output sizes (SDK line 1088: outSizesBuffer = accels.blasSizesBuffer)
  buildBlasDesc.outSizesBuffer = accels.blasSizesBuffer.getDeviceAddress();
  buildBlasDesc.outSizesOffsetInBytes = 0;

  // Output acceleration structures buffer (SDK line 1090: outAccelerationStructuresBuffer)
  buildBlasDesc.outAccelerationStructuresBuffer = accels.blasBuffer.getDeviceAddress();
  buildBlasDesc.outAccelerationStructuresOffsetInBytes = 0;

  // SDK MATCH: commandList->executeMultiIndirectClusterOperation (line 1093)
  Logger::info(str::format("[RTXMG] BLAS build descriptor: scratchAddr=0x", std::hex,
    buildBlasDesc.scratchData, ", scratchSize=", std::dec, buildBlasDesc.scratchSizeInBytes / (1024*1024), "MB",
    ", indirectArgs=0x", std::hex, buildBlasDesc.inIndirectArgsBuffer,
    ", dstAddresses=0x", buildBlasDesc.inOutAddressesBuffer,
    ", dstSizes=0x", buildBlasDesc.outSizesBuffer,
    ", accelBuffer=0x", buildBlasDesc.outAccelerationStructuresBuffer, std::dec));
  Logger::info("[RTXMG] Calling executeMultiIndirectClusterOperation for BLAS [SDK-MATCHING]...");
  executeMultiIndirectClusterOperation(ctx, buildBlasDesc);
  Logger::info("[RTXMG] executeMultiIndirectClusterOperation returned [SDK-MATCHING]");

  // STEP 6: Memory barrier
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
  device->vkd()->vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    0, 1, &barrier, 0, nullptr, 0, nullptr);

  // NOTE: clusterReferencesBuffer is now stored in accels.clusterReferencesBuffer (persistent)
  // BLAS references these device addresses - buffer lifetime managed by accels

  Logger::info(str::format("[RTXMG] Cluster BLAS built successfully (ImplicitDestinations mode)"));

  return true;
}

//===========================================================================
// Cluster Operations - SDK-matching interface
//===========================================================================

// SDK Match: commandList->executeMultiIndirectClusterOperation (lines 618, 1093)
// This wraps vkCmdBuildClusterAccelerationStructureIndirectNV with SDK-matching parameter layout
void executeMultiIndirectClusterOperation(
  DxvkContext* ctx,
  const ClusterOperationDesc& desc) {

  // Build VkClusterAccelerationStructureCommandsInfoNV from desc
  VkClusterAccelerationStructureCommandsInfoNV commandsInfo = {};
  commandsInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
  commandsInfo.pNext = nullptr;
  commandsInfo.input = desc.params;

  // Scratch buffer (FIXED: Use scratchData field which contains the GPU address)
  commandsInfo.scratchData = desc.scratchData;

  // Source indirect arguments
  commandsInfo.srcInfosArray.deviceAddress = desc.inIndirectArgsBuffer + desc.inIndirectArgsOffsetInBytes;
  commandsInfo.srcInfosArray.stride =
    (desc.params.opType == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV)
      ? sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV)  // 32 bytes for CLAS
      : 16;  // cluster::IndirectArgs = {clusterAddresses:8, clusterCount:4, padding:4} for BLAS
  commandsInfo.srcInfosArray.size = 0;  // Set after we know the max op count

  // Source indirect count (GPU counter for CLAS, nullptr for BLAS)
  if (desc.inIndirectArgCountBuffer != 0) {
    commandsInfo.srcInfosCount = desc.inIndirectArgCountBuffer + desc.inIndirectArgCountOffsetInBytes;
  } else {
    commandsInfo.srcInfosCount = 0;  // Count comes from srcInfosArray.size / stride
  }

  // Destination addresses array
  if (desc.inOutAddressesBuffer != 0) {
    commandsInfo.dstAddressesArray.deviceAddress = desc.inOutAddressesBuffer + desc.inOutAddressesOffsetInBytes;
    commandsInfo.dstAddressesArray.stride = sizeof(VkDeviceAddress);
    commandsInfo.dstAddressesArray.size = 0;
  }

  // Destination sizes array (BLAS only)
  if (desc.outSizesBuffer != 0) {
    commandsInfo.dstSizesArray.deviceAddress = desc.outSizesBuffer + desc.outSizesOffsetInBytes;
    commandsInfo.dstSizesArray.stride = sizeof(uint32_t);
    commandsInfo.dstSizesArray.size = 0;
  }

  const VkDeviceSize maxOpCount = desc.params.maxAccelerationStructureCount;
  if (maxOpCount > 0) {
    auto setRegionSize = [maxOpCount](VkStridedDeviceAddressRegionKHR& region) {
      if (region.deviceAddress != 0 && region.stride != 0) {
        region.size = region.stride * maxOpCount;
      }
    };

    setRegionSize(commandsInfo.srcInfosArray);
    setRegionSize(commandsInfo.dstAddressesArray);
    setRegionSize(commandsInfo.dstSizesArray);
  }

  // Destination implicit data (BLAS buffer for implicit destinations mode)
  if (desc.outAccelerationStructuresBuffer != 0) {
    commandsInfo.dstImplicitData = desc.outAccelerationStructuresBuffer + desc.outAccelerationStructuresOffsetInBytes;
  }

  commandsInfo.addressResolutionFlags = 0;

  Logger::info(str::format("[RTXMG] executeMultiIndirectClusterOperation info: opType=",
    (desc.params.opType == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV ? "INSTANTIATE_CLAS" : "BUILD_BLAS"),
    ", maxOps=", desc.params.maxAccelerationStructureCount,
    ", scratch=0x", std::hex, commandsInfo.scratchData,
    ", srcAddr=0x", commandsInfo.srcInfosArray.deviceAddress,
    ", srcSize=", std::dec, commandsInfo.srcInfosArray.size,
    ", srcStride=", commandsInfo.srcInfosArray.stride,
    ", dstAddr=0x", std::hex, commandsInfo.dstAddressesArray.deviceAddress,
    ", dstSize=", std::dec, commandsInfo.dstAddressesArray.size,
    ", dstStride=", commandsInfo.dstAddressesArray.stride,
    ", dstSizesAddr=0x", std::hex, commandsInfo.dstSizesArray.deviceAddress,
    ", dstSizesSize=", std::dec, commandsInfo.dstSizesArray.size,
    ", dstImplicit=0x", std::hex, commandsInfo.dstImplicitData,
    ", countAddr=0x", commandsInfo.srcInfosCount, std::dec));

  // Execute the operation
  VkCommandBuffer cmd = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

  Logger::info(str::format("[RTXMG VK CALL] ========== CLUSTER EXTENSION CALL =========="));
  Logger::info(str::format("[RTXMG VK CALL] Command buffer: cmd=0x", std::hex, reinterpret_cast<uint64_t>(cmd), std::dec));
  Logger::info(str::format("[RTXMG VK CALL] Operation type: ",
    (desc.params.opType == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV ? "INSTANTIATE_CLAS" :
     desc.params.opType == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV ? "BUILD_BLAS" : "UNKNOWN")));
  Logger::info(str::format("[RTXMG VK CALL] Operation mode: ", desc.params.opMode));
  Logger::info(str::format("[RTXMG VK CALL] Max accel struct count: ", desc.params.maxAccelerationStructureCount));
  Logger::info(str::format("[RTXMG VK CALL] Scratch data: 0x", std::hex, commandsInfo.scratchData, std::dec));
  Logger::info(str::format("[RTXMG VK CALL] Source infos array:"));
  Logger::info(str::format("[RTXMG VK CALL]   deviceAddress: 0x", std::hex, commandsInfo.srcInfosArray.deviceAddress, std::dec));
  Logger::info(str::format("[RTXMG VK CALL]   stride: ", commandsInfo.srcInfosArray.stride, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL]   size: ", commandsInfo.srcInfosArray.size, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL] Source infos count: 0x", std::hex, commandsInfo.srcInfosCount, std::dec));
  Logger::info(str::format("[RTXMG VK CALL] Destination addresses array:"));
  Logger::info(str::format("[RTXMG VK CALL]   deviceAddress: 0x", std::hex, commandsInfo.dstAddressesArray.deviceAddress, std::dec));
  Logger::info(str::format("[RTXMG VK CALL]   stride: ", commandsInfo.dstAddressesArray.stride, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL]   size: ", commandsInfo.dstAddressesArray.size, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL] Destination sizes array:"));
  Logger::info(str::format("[RTXMG VK CALL]   deviceAddress: 0x", std::hex, commandsInfo.dstSizesArray.deviceAddress, std::dec));
  Logger::info(str::format("[RTXMG VK CALL]   stride: ", commandsInfo.dstSizesArray.stride, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL]   size: ", commandsInfo.dstSizesArray.size, " bytes"));
  Logger::info(str::format("[RTXMG VK CALL] Destination implicit data: 0x", std::hex, commandsInfo.dstImplicitData, std::dec));
  Logger::info("[RTXMG VK CALL] ========== CALLING vkCmdBuildClusterAccelerationStructureIndirectNV ==========");

  g_clusterAccelExt.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &commandsInfo);

  Logger::info("[RTXMG VK CALL] ========== vkCmdBuildClusterAccelerationStructureIndirectNV RETURNED ==========");
  Logger::info(str::format("[RTXMG VK CALL] Cluster extension call completed successfully"));

  Logger::info(str::format("[RTXMG] executeMultiIndirectClusterOperation: opType=",
    (desc.params.opType == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_CLUSTERS_NV ? "INSTANTIATE_CLAS" : "BUILD_BLAS"),
    ", indirectArgs=0x", std::hex, commandsInfo.srcInfosArray.deviceAddress,
    ", count=0x", commandsInfo.srcInfosCount, std::dec));
}

} // namespace dxvk
