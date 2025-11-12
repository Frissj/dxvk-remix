/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* HiZ Pyramid Generation - Hierarchical Z-buffer reduction shader
*
* This compute shader generates a mipmap pyramid from the depth buffer
* for fast occlusion culling in cluster tessellation.
*
* Each mip level is half the resolution of the previous level.
* We use MAX reduction to store furthest (conservative) depth.
*/

#include "rtx/utility/common.slangh"

// Constants
static const uint kThreadsPerGroup = 64;  // 8×8 thread group

// HiZ generation parameters
struct HiZGenerateParams {
  uint2 srcSize;          // Source mip level size
  uint2 dstSize;          // Destination mip level size
  uint srcMipLevel;       // Source mip level
  uint dstMipLevel;       // Destination mip level
  float nearPlane;        // Camera near plane
  float farPlane;         // Camera far plane
  uint2 _pad;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0, 0)]] ConstantBuffer<HiZGenerateParams> cb;

// Input: Previous mip level (or depth buffer for mip 0) - shifted +1
[[vk::binding(1, 0)]] Texture2D<float> srcDepth;
[[vk::binding(2, 0)]] SamplerState depthSampler;

// Output: Next mip level - shifted +1
[[vk::binding(3, 0)]] RWTexture2DArray<float> hizPyramid;

// Convert depth buffer value to linear depth
float linearizeDepth(float depth, float near, float far) {
  // Reverse-Z depth buffer: 1.0 at near plane, 0.0 at far plane
  return near / (far - depth * (far - near));
}

// Main entry point
[shader("compute")]
[numthreads(8, 8, 1)]
void main(
  uint3 threadIdx : SV_DispatchThreadID)
{
  uint2 dstPixel = threadIdx.xy;

  // Early out if beyond destination size
  if (any(dstPixel >= cb.dstSize)) {
    return;
  }

  // Source coordinates (2×2 block in source mip)
  uint2 srcPixel = dstPixel * 2;

  // Sample 4 depth values from source mip (2×2 block)
  // We use MAX to get furthest (most conservative) depth
  float depth00 = srcDepth[srcPixel + uint2(0, 0)];
  float depth10 = srcDepth[srcPixel + uint2(1, 0)];
  float depth01 = srcDepth[srcPixel + uint2(0, 1)];
  float depth11 = srcDepth[srcPixel + uint2(1, 1)];

  // For mip 0, linearize depth values from hardware depth buffer
  if (cb.srcMipLevel == 0) {
    depth00 = linearizeDepth(depth00, cb.nearPlane, cb.farPlane);
    depth10 = linearizeDepth(depth10, cb.nearPlane, cb.farPlane);
    depth01 = linearizeDepth(depth01, cb.nearPlane, cb.farPlane);
    depth11 = linearizeDepth(depth11, cb.nearPlane, cb.farPlane);
  }

  // MAX reduction (furthest depth for conservative culling)
  float maxDepth = max(max(depth00, depth10), max(depth01, depth11));

  // Write to HiZ pyramid
  hizPyramid[uint3(dstPixel, cb.dstMipLevel)] = maxDepth;
}
