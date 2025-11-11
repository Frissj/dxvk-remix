/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

// RTX Mega Geometry: Fill Cluster Data
// Tessellates surfaces into cluster vertices

#include "rtx/utility/common.slangh"

struct FillConstants {
  uint clusterCount;
  uint vertexCount;
  uint indexCount;
  uint pad;
};

struct ClusterTilingInfo {
  uint surfaceIndex;
  uint gridSizeX;
  uint gridSizeY;
  uint clusterOffset;
};

struct ClusterVertex {
  float3 position;
  float3 normal;
  float2 texCoord;
  uint clusterId;
};

layout(binding = 0)
ConstantBuffer<FillConstants> cb;

layout(binding = 1)
StructuredBuffer<ClusterTilingInfo> TilingInput;

layout(binding = 2)
StructuredBuffer<float3> VertexPositions;

layout(binding = 3)
StructuredBuffer<float3> VertexNormals;

layout(binding = 4)
StructuredBuffer<float2> VertexTexCoords;

layout(binding = 5)
StructuredBuffer<uint> IndexBuffer;

layout(binding = 6)
RWStructuredBuffer<ClusterVertex> VertexOutput;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  uint vertexIdx = threadId.x;

  if (vertexIdx >= cb.vertexCount)
    return;

  // Compute which cluster this vertex belongs to
  uint clusterId = vertexIdx / 144; // 144 verts per cluster (12x12 grid)
  uint localVertIdx = vertexIdx % 144;

  // Get source vertex data
  float3 position = VertexPositions[vertexIdx];
  float3 normal = VertexNormals[vertexIdx];
  float2 texCoord = VertexTexCoords[vertexIdx];

  // Write cluster vertex
  ClusterVertex outVertex;
  outVertex.position = position;
  outVertex.normal = normalize(normal);
  outVertex.texCoord = texCoord;
  outVertex.clusterId = clusterId;

  VertexOutput[vertexIdx] = outVertex;
}
