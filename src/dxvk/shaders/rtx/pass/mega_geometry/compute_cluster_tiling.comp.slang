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

// RTX Mega Geometry: Compute Cluster Tiling
// Determines cluster layout for subdivision surfaces

#include "rtx/utility/common.slangh"

struct TilingConstants {
  uint surfaceCount;
  float tessellationDensity;
  uint2 viewportSize;
};

struct SurfaceInfo {
  float4 boundingBox;
  uint vertexOffset;
  uint vertexCount;
  uint materialId;
  uint pad;
};

struct ClusterTilingInfo {
  uint surfaceIndex;
  uint gridSizeX;
  uint gridSizeY;
  uint clusterOffset;
};

layout(binding = 0)
ConstantBuffer<TilingConstants> cb;

layout(binding = 1)
Sampler2D<float> HiZBuffer;

layout(binding = 2)
StructuredBuffer<SurfaceInfo> SurfaceData;

layout(binding = 3)
RWStructuredBuffer<ClusterTilingInfo> TilingOutput;

layout(binding = 4)
RWStructuredBuffer<uint> Counters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  uint surfaceIdx = threadId.x;

  if (surfaceIdx >= cb.surfaceCount)
    return;

  SurfaceInfo surface = SurfaceData[surfaceIdx];

  // Placeholder: simple grid assignment
  ClusterTilingInfo tiling;
  tiling.surfaceIndex = surfaceIdx;
  tiling.gridSizeX = uint(cb.tessellationDensity * 11.0);
  tiling.gridSizeY = uint(cb.tessellationDensity * 11.0);
  tiling.clusterOffset = surfaceIdx * 121; // Max clusters per surface

  TilingOutput[surfaceIdx] = tiling;
}
