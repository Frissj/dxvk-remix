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

// RTX Mega Geometry - Cluster Tiling Compute Shader
// Simplified version that generates uniform cluster metadata for each triangle
// Mimics SDK's compute_cluster_tiling.hlsl but without Catmull-Clark subdivision

#include "cluster_tess_bindings.slangh"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
  uint triangleID = dispatchThreadID.x;

  if (triangleID >= cb.numTriangles) return;

  // For uniform tessellation: each triangle becomes one cluster
  // with uniform size determined by gridSize parameter

  uint gridSize = cb.gridSize;
  uint clusterEdges = gridSize - 1;  // gridSize=5 means 5x5 verts = 4x4 edges
  uint verticesPerCluster = gridSize * gridSize;
  uint vertexOffset = triangleID * verticesPerCluster;

  // Write cluster metadata (matches SDK pattern)
  Cluster cluster;
  cluster.iSurface = 0;                          // Single surface for now
  cluster.nVertexOffset = vertexOffset;
  cluster.offsetX = 0;                           // No grid layout in uniform tessellation
  cluster.offsetY = 0;
  cluster.sizeX = uint8_t(clusterEdges);         // Cluster size in edges (not vertices)
  cluster.sizeY = uint8_t(clusterEdges);
  cluster.pad0 = 0;
  cluster.pad1 = 0;

  // Write to output buffer (matches SDK's u_Clusters[clusterIndex] = cluster)
  outputClustersBuffer[triangleID] = cluster;

  // Calculate template index from cluster size
  uint kMaxClusterEdgeSegments = 11;  // Must match C++ constant
  uint templateIdx = ((clusterEdges - 1) * kMaxClusterEdgeSegments) + (clusterEdges - 1);

  // Write cluster instance data directly on GPU (avoids CPU-GPU sync)
  // This matches VkClusterAccelerationStructureInstantiateClusterInfoNV structure
  ClusterInstanceData instance;
  instance.clusterIdOffset = triangleID;  // Use triangle ID as unique cluster ID
  instance.geometryIndexOffset = 0;  // Single geometry for now (24-bit field)
  instance.clusterTemplateAddress = inputTemplateAddresses[templateIdx];  // Template device address
  instance.vertexBufferAddress = cb.vertexBufferAddress + (vertexOffset * 12);  // float3 = 12 bytes
  instance.vertexBufferStrideInBytes = 12;  // sizeof(float3)
  instance.pad0 = 0;  // Padding for alignment

  outputClusterInstancesBuffer[triangleID] = instance;
}
