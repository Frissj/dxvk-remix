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

// RTX Mega Geometry - Cluster Tessellation Compute Shader
// Tessellates triangles into NxN grids on GPU

#include "cluster_tess_bindings.slangh"

// Helper: Write interleaved vertex data
void writeVertex(uint vertexIndex, float3 position, float3 normal, float2 texcoord) {
  uint baseOffset = vertexIndex * 32; // 32 bytes per vertex

  // Position (12 bytes)
  outputVertexBuffer.Store<float>(baseOffset, position.x);
  outputVertexBuffer.Store<float>(baseOffset + 4, position.y);
  outputVertexBuffer.Store<float>(baseOffset + 8, position.z);

  // Normal (12 bytes)
  outputVertexBuffer.Store<float>(baseOffset + 12, normal.x);
  outputVertexBuffer.Store<float>(baseOffset + 16, normal.y);
  outputVertexBuffer.Store<float>(baseOffset + 20, normal.z);

  // Texcoord (8 bytes)
  outputVertexBuffer.Store<float>(baseOffset + 24, texcoord.x);
  outputVertexBuffer.Store<float>(baseOffset + 28, texcoord.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
  uint triangleID = dispatchThreadID.x;

  if (triangleID >= cb.numTriangles) return;

  // Read triangle indices
  uint idx0 = inputIndices[triangleID * 3 + 0];
  uint idx1 = inputIndices[triangleID * 3 + 1];
  uint idx2 = inputIndices[triangleID * 3 + 2];

  // Read triangle vertices
  float3 pos0 = inputPositions[idx0];
  float3 pos1 = inputPositions[idx1];
  float3 pos2 = inputPositions[idx2];

  float3 norm0 = cb.hasNormals ? inputNormals[idx0] : float3(0.0, 0.0, 1.0);
  float3 norm1 = cb.hasNormals ? inputNormals[idx1] : float3(0.0, 0.0, 1.0);
  float3 norm2 = cb.hasNormals ? inputNormals[idx2] : float3(0.0, 0.0, 1.0);

  float2 uv0 = cb.hasTexcoords ? inputTexcoords[idx0] : float2(0.0, 0.0);
  float2 uv1 = cb.hasTexcoords ? inputTexcoords[idx1] : float2(0.0, 0.0);
  float2 uv2 = cb.hasTexcoords ? inputTexcoords[idx2] : float2(0.0, 0.0);

  uint gridSize = cb.gridSize;
  uint verticesPerTriangle = gridSize * gridSize;
  uint baseVertexIndex = triangleID * verticesPerTriangle;

  // Generate tessellated grid vertices
  // For a triangle, we only generate vertices where v + u <= 1 (inside triangle)
  for (uint y = 0; y < gridSize; y++) {
    for (uint x = 0; x < gridSize; x++) {
      // Normalized grid coordinates (0 to 1)
      float u = float(x) / float(gridSize - 1);
      float v = float(y) / float(gridSize - 1);

      // For triangular tessellation: only generate vertices inside triangle (u + v <= 1)
      // But we still allocate full grid for simplicity - fill outside with edge vertices
      float actualU = u;
      float actualV = v;

      // Clamp to triangle boundary
      if (u + v > 1.0) {
        // Outside triangle - clamp to opposite edge
        float sum = u + v;
        actualU = u / sum;
        actualV = v / sum;
      }

      float w = 1.0 - actualU - actualV;

      // Interpolate position using barycentric coords
      float3 position = pos0 * w + pos1 * actualU + pos2 * actualV;

      // Interpolate normal and normalize
      float3 normal = norm0 * w + norm1 * actualU + norm2 * actualV;
      normal = normalize(normal);

      // Interpolate UV
      float2 texcoord = uv0 * w + uv1 * actualU + uv2 * actualV;

      // Write vertex
      uint vertexIndex = baseVertexIndex + y * gridSize + x;
      writeVertex(vertexIndex, position, normal, texcoord);
    }
  }

  // Generate indices for tessellated grid
  // Each quad in grid = 2 triangles = 6 indices
  uint baseIndexIndex = triangleID * (gridSize - 1) * (gridSize - 1) * 6;
  uint indexOffset = 0;

  for (uint y = 0; y < gridSize - 1; y++) {
    for (uint x = 0; x < gridSize - 1; x++) {
      uint v0 = baseVertexIndex + y * gridSize + x;
      uint v1 = baseVertexIndex + y * gridSize + (x + 1);
      uint v2 = baseVertexIndex + (y + 1) * gridSize + x;
      uint v3 = baseVertexIndex + (y + 1) * gridSize + (x + 1);

      // Triangle 1: v0, v1, v2
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v0;
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v1;
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v2;

      // Triangle 2: v1, v3, v2
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v1;
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v3;
      outputIndexBuffer[baseIndexIndex + indexOffset++] = v2;
    }
  }
}
