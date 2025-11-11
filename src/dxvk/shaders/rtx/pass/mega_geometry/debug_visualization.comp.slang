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

// RTX Mega Geometry: Debug Visualization
// Renders various debug views for cluster tessellation

#include "rtx/utility/common.slangh"
#include "rtx/utility/color.slangh"

struct DebugConstants {
  uint debugMode;
  uint2 outputSize;
  uint pad;
};

struct ClusterStatData {
  uint totalClusters;
  uint visibleClusters;
  uint culledByHiZ;
  uint culledByFrustum;
};

layout(binding = 0)
ConstantBuffer<DebugConstants> cb;

layout(binding = 1)
Texture2D<float> DepthBuffer;

layout(binding = 2)
Sampler2D<float> HiZBuffer;

layout(binding = 3)
StructuredBuffer<ClusterStatData> ClusterStatistics;

layout(rgba16f, binding = 4)
RWTexture2D<float4> OutputImage;

float3 clusterIdToColor(uint clusterId) {
  uint hash = clusterId * 2654435761u;
  float r = float((hash >> 0) & 0xFF) / 255.0;
  float g = float((hash >> 8) & 0xFF) / 255.0;
  float b = float((hash >> 16) & 0xFF) / 255.0;
  return float3(r, g, b);
}

float3 heatMapColor(float t) {
  if (t < 0.5)
    return lerp(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 0.0), t * 2.0);
  else
    return lerp(float3(0.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  uint2 pixelPos = threadId.xy;

  if (pixelPos.x >= cb.outputSize.x || pixelPos.y >= cb.outputSize.y)
    return;

  float2 uv = (float2(pixelPos) + 0.5) / float2(cb.outputSize);

  // Read the existing rendered output so we can preserve the scene geometry
  float4 existingColor = OutputImage[pixelPos];

  // TODO: Mega geometry debug visualization needs access to:
  // - Cluster ID buffer (to color by cluster)
  // - Surface index buffer (to color by surface)
  // - Tessellation factor buffer (to show density as heat map)
  // - Geometry buffers to compute proper normals and UVs
  //
  // Without this data, we can only show placeholder visualizations.
  // Proper implementation requires integrating debug coloring into
  // the main ray tracing pass (closest hit shaders) where cluster
  // data is available, similar to how the RTXMG demo does it.

  // For now, show a warning overlay that makes it clear this is not implemented
  float3 debugColor = existingColor.rgb;

  // Add a colored overlay to indicate debug mode is active but not functional
  float overlayStrength = 0.3;
  float3 modeColor;

  if (cb.debugMode == 900) { // Cluster ID (would need cluster ID buffer)
    modeColor = float3(1.0, 0.5, 0.5); // Red tint
  }
  else if (cb.debugMode == 901) { // Tessellation Density (would need tess factor buffer)
    modeColor = float3(0.5, 1.0, 0.5); // Green tint
  }
  else if (cb.debugMode == 902) { // Surface UV (would need UV buffer from G-buffer)
    modeColor = float3(0.5, 0.5, 1.0); // Blue tint
  }
  else if (cb.debugMode == 903) { // Surface Index (would need surface index buffer)
    modeColor = float3(1.0, 1.0, 0.5); // Yellow tint
  }
  else if (cb.debugMode == 904) { // Vertex Normals (would need normal buffer)
    modeColor = float3(1.0, 0.5, 1.0); // Magenta tint
  }
  else if (cb.debugMode == 905) { // Wireframe Overlay (would need edge detection or geometry shader)
    modeColor = float3(0.5, 1.0, 1.0); // Cyan tint
  }
  else if (cb.debugMode == 906) { // HiZ Visualization
    float depth = HiZBuffer.SampleLevel(uv, 0);
    debugColor = lerp(debugColor, float3(depth, depth, depth), 0.7);
    modeColor = float3(1.0, 1.0, 1.0);
    overlayStrength = 0.0; // Already blended above
  }
  else if (cb.debugMode == 907) { // Cluster Bounds (would need cluster bounds buffer)
    modeColor = float3(1.0, 0.7, 0.3); // Orange tint
  }
  else {
    modeColor = float3(1.0, 1.0, 1.0);
  }

  // Apply colored overlay to show which debug mode is active
  // This preserves the rendered geometry while indicating the debug mode
  debugColor = lerp(debugColor, modeColor, overlayStrength);

  // Add a grid overlay so user can see SOMETHING is happening
  // This makes it clear the debug view is active, just not fully implemented
  uint2 gridCell = pixelPos / 64;
  bool isGridLine = ((pixelPos.x % 64) == 0 || (pixelPos.y % 64) == 0);
  if (isGridLine) {
    debugColor = lerp(debugColor, modeColor, 0.5);
  }

  OutputImage[pixelPos] = float4(debugColor, 1.0);
}
