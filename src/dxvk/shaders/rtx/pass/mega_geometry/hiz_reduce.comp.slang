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

// RTX Mega Geometry: Hierarchical Z-Buffer Reduction
// Generates mipmap chain for conservative occlusion culling

#include "rtx/utility/common.slangh"

struct HiZConstants {
  uint2 inputSize;
  uint mipLevel;
  uint pad;
};

layout(binding = 0)
ConstantBuffer<HiZConstants> cb;

layout(binding = 1)
Texture2D<float> InputDepth;

layout(r32f, binding = 2)
RWTexture2D<float> OutputHiZ;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  uint2 pixelPos = threadId.xy;

  if (pixelPos.x >= cb.inputSize.x || pixelPos.y >= cb.inputSize.y)
    return;

  uint2 sourcePos = pixelPos * 2;

  // Sample 2x2 quad and take maximum depth (conservative)
  float d0 = InputDepth.Load(int3(sourcePos + uint2(0, 0), 0));
  float d1 = InputDepth.Load(int3(sourcePos + uint2(1, 0), 0));
  float d2 = InputDepth.Load(int3(sourcePos + uint2(0, 1), 0));
  float d3 = InputDepth.Load(int3(sourcePos + uint2(1, 1), 0));

  float maxDepth = max(max(d0, d1), max(d2, d3));

  OutputHiZ[pixelPos] = maxDepth;
}
