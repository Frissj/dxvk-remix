/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Fill Clusters - Vertex generation shader
*
* This compute shader fills cluster vertices by sampling the input
* triangle mesh at UV positions determined by the cluster layout.
*
* Simplified for triangle mesh tessellation (no subdivision surfaces).
*/

#include "rtx/pass/rtxmg/rtxmg_bindings.slangh"
#include "rtx/utility/common.slangh"

// Constants
static const uint kThreadsPerWave = 32;
static const uint kWavesPerGroup = 4;
static const uint kThreadsPerGroup = kThreadsPerWave * kWavesPerGroup;

// Fill clusters parameters
struct FillClustersParams {
  uint surfaceStart;
  uint surfaceEnd;
  uint enableVertexNormals;  // Generate vertex normals
  uint clusterPattern;       // REGULAR or SLANTED

  float4x4 localToWorld;     // Local to world transform (for normal transform)

  // Phase 4: GPU batching support
  uint enableBatching;       // Enable GPU batching mode (0 = sequential, 1 = batched)
  uint instanceCount;        // Number of instances in batch
  uint _pad0;
  uint _pad1;

  int debugSurfaceIndex;
  int debugClusterIndex;
  int debugLaneIndex;
  uint _pad2;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0, 0)]] ConstantBuffer<FillClustersParams> cb;

// Input buffers (shifted +1 to accommodate constant buffer at binding 0)
[[vk::binding(1, 0)]] StructuredBuffer<float3> inputPositions;
[[vk::binding(2, 0)]] StructuredBuffer<float3> inputNormals;
// Use ByteAddressBuffer for potentially misaligned game buffers (only requires 4-byte alignment)
[[vk::binding(3, 0)]] ByteAddressBuffer inputTexcoords;
[[vk::binding(4, 0)]] ByteAddressBuffer inputIndices;
[[vk::binding(5, 0)]] StructuredBuffer<SurfaceInfo> surfaceInfo;

// Cluster data
[[vk::binding(6, 0)]] StructuredBuffer<GridSampler> gridSamplers;
[[vk::binding(7, 0)]] StructuredBuffer<Cluster> clusters;
[[vk::binding(8, 0)]] StructuredBuffer<ClusterShadingData> clusterShadingData;

// Output buffers
[[vk::binding(9, 0)]] RWStructuredBuffer<float3> clusterVertexPositionsOut;
[[vk::binding(10, 0)]] RWStructuredBuffer<float3> clusterVertexNormalsOut;
[[vk::binding(11, 0)]] RWStructuredBuffer<ClusterShadingData> clusterShadingDataOut;

// Phase 4: GPU batching support - Instance data buffer (optional)
[[vk::binding(12, 0)]] StructuredBuffer<InstanceData> instanceDataBuffer;

// Calculate UV position within surface for a cluster vertex
float2 calculateVertexUV(
  Cluster cluster,
  GridSampler sampler,
  uint2 vertexIdx2D)
{
  // Total grid size
  uint gridSizeX = max(sampler.edgeSegments[0], sampler.edgeSegments[2]);
  uint gridSizeY = max(sampler.edgeSegments[1], sampler.edgeSegments[3]);

  // Vertex position in global grid
  uint globalX = cluster.offsetX + vertexIdx2D.x;
  uint globalY = cluster.offsetY + vertexIdx2D.y;

  // Normalize to [0,1]
  float u = float(globalX) / float(gridSizeX);
  float v = float(globalY) / float(gridSizeY);

  return float2(u, v);
}

// Sample triangle mesh at UV position
// Simple barycentric interpolation within triangle
void sampleTriangleMesh(
  StructuredBuffer<float3> positions,
  StructuredBuffer<float3> normals,
  ByteAddressBuffer texcoords,
  ByteAddressBuffer indices,
  SurfaceInfo surface,
  float2 uv,
  out float3 outPosition,
  out float3 outNormal,
  out float2 outTexcoord)
{
  // For simplicity, assume the surface is a quad made of 2 triangles
  // UV [0,1] maps to the quad

  // Find which triangle we're in
  uint triIndex;
  float3 bary;

  if (uv.x + uv.y <= 1.0) {
    // Lower-left triangle (0,1,2)
    triIndex = 0;
    bary = float3(1.0 - uv.x - uv.y, uv.x, uv.y);
  } else {
    // Upper-right triangle (2,1,3)
    triIndex = 1;
    bary = float3(uv.x + uv.y - 1.0, 1.0 - uv.y, 1.0 - uv.x);
  }

  // Get triangle indices (ByteAddressBuffer requires byte offset)
  uint i0 = indices.Load((surface.firstIndex + triIndex * 3 + 0) * 4);
  uint i1 = indices.Load((surface.firstIndex + triIndex * 3 + 1) * 4);
  uint i2 = indices.Load((surface.firstIndex + triIndex * 3 + 2) * 4);

  // Interpolate position
  float3 p0 = positions[i0];
  float3 p1 = positions[i1];
  float3 p2 = positions[i2];
  outPosition = p0 * bary.x + p1 * bary.y + p2 * bary.z;

  // Interpolate normal
  float3 n0 = normals[i0];
  float3 n1 = normals[i1];
  float3 n2 = normals[i2];
  outNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

  // Interpolate texcoord (ByteAddressBuffer - load 2 floats = 8 bytes)
  float2 t0 = asfloat(texcoords.Load2(i0 * 8));
  float2 t1 = asfloat(texcoords.Load2(i1 * 8));
  float2 t2 = asfloat(texcoords.Load2(i2 * 8));
  outTexcoord = t0 * bary.x + t1 * bary.y + t2 * bary.z;
}

// Process a single cluster
void processCluster(uint clusterIndex, uint laneId) {
  Cluster cluster = clusters[clusterIndex];
  GridSampler sampler = gridSamplers[cluster.iSurface];
  SurfaceInfo surface = surfaceInfo[cluster.iSurface];

  // Calculate vertices for this cluster
  uint numVerticesX = cluster.sizeX + 1;
  uint numVerticesY = cluster.sizeY + 1;
  uint numVertices = numVerticesX * numVerticesY;

  // Process vertices in parallel across lanes
  for (uint vertexIdx = laneId; vertexIdx < numVertices; vertexIdx += kThreadsPerWave) {
    // Convert linear index to 2D
    uint vertexX = vertexIdx % numVerticesX;
    uint vertexY = vertexIdx / numVerticesX;
    uint2 vertexIdx2D = uint2(vertexX, vertexY);

    // Calculate UV position
    float2 uv = calculateVertexUV(cluster, sampler, vertexIdx2D);

    // Sample input mesh
    float3 position, normal;
    float2 texcoord;

    sampleTriangleMesh(
      inputPositions,
      inputNormals,
      inputTexcoords,
      inputIndices,
      surface,
      uv,
      position,
      normal,
      texcoord
    );

    // Write output
    uint outputVertexIndex = cluster.nVertexOffset + vertexIdx;
    clusterVertexPositionsOut[outputVertexIndex] = position;

    if (cb.enableVertexNormals) {
      // Transform normal to world space
      float3 normalWorld = mul((float3x3)cb.localToWorld, normal);
      clusterVertexNormalsOut[outputVertexIndex] = normalize(normalWorld);
    }
  }

  // Write corner texcoords (lane 0 only)
  if (laneId == 0) {
    ClusterShadingData shadingData = clusterShadingData[clusterIndex];

    // Sample corner UVs
    float2 uv00 = calculateVertexUV(cluster, sampler, uint2(0, 0));
    float2 uv10 = calculateVertexUV(cluster, sampler, uint2(cluster.sizeX, 0));
    float2 uv01 = calculateVertexUV(cluster, sampler, uint2(0, cluster.sizeY));
    float2 uv11 = calculateVertexUV(cluster, sampler, uint2(cluster.sizeX, cluster.sizeY));

    float3 pos00, pos10, pos01, pos11;
    float3 norm00, norm10, norm01, norm11;
    float2 tc00, tc10, tc01, tc11;

    sampleTriangleMesh(inputPositions, inputNormals, inputTexcoords, inputIndices, surface, uv00, pos00, norm00, tc00);
    sampleTriangleMesh(inputPositions, inputNormals, inputTexcoords, inputIndices, surface, uv10, pos10, norm10, tc10);
    sampleTriangleMesh(inputPositions, inputNormals, inputTexcoords, inputIndices, surface, uv01, pos01, norm01, tc01);
    sampleTriangleMesh(inputPositions, inputNormals, inputTexcoords, inputIndices, surface, uv11, pos11, norm11, tc11);

    shadingData.texcoords[0] = tc00;
    shadingData.texcoords[1] = tc10;
    shadingData.texcoords[2] = tc11;
    shadingData.texcoords[3] = tc01;

    clusterShadingDataOut[clusterIndex] = shadingData;
  }
}

// Main entry point
[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(
  uint3 threadIdx : SV_GroupThreadID,
  uint3 groupIdx : SV_GroupID,
  uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint laneId = threadIdx.x % kThreadsPerWave;
  uint waveId = threadIdx.x / kThreadsPerWave;

  // Phase 4: Check if batching mode is enabled
  if (cb.enableBatching != 0) {
    // GPU Batching Mode: Process clusters from all instances
    // Each thread processes one cluster across all instances
    uint clusterIndex = dispatchThreadId.x;

    if (clusterIndex < cb.surfaceEnd) {
      // In batched mode, cluster indices are global across all instances
      // The instance data is already encoded in the cluster data from the tiling phase
      processCluster(clusterIndex, laneId);
    }
  }
  else {
    // Sequential Mode: Process clusters in range
    uint clusterIndex = groupIdx.x * kWavesPerGroup + waveId + cb.surfaceStart;

    // Early out if beyond cluster range
    if (clusterIndex >= cb.surfaceEnd) {
      return;
    }

    // Process this cluster
    processCluster(clusterIndex, laneId);
  }
}
