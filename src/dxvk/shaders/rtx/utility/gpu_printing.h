/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

static const int16_t kInvalidThreadIndex = 32767; // ~ int16_t max

// NV-DXVK: [PathAProbe] append region. The legacy single-slot GPU-print ring occupies the first
// GPU_PRINT_RING_SLOTS elements (== kMaxFramesInFlight). After it, each frame-in-flight owns a
// region of PATHA_PROBE_CAP hash-indexed slots that Path A probes scatter into (different hits ->
// different slots by a geometry hash, so no single-slot starvation and no atomics needed). Total
// buffer length = GPU_PRINT_RING_SLOTS * (1 + PATHA_PROBE_CAP). The CPU scans the oldest ring's
// region each frame and aggregates. Sentinel threadIndex.y == kPathAProbeSentinel marks a record.
#define GPU_PRINT_RING_SLOTS 4        // MUST equal kMaxFramesInFlight (static_assert in rtx_resources.cpp)
#define PATHA_PROBE_CAP 8192
#define kPathAProbeSentinel 0xC120u

// Note: ensure alignment for C++ and Slang to match
struct GpuPrintBufferElement
{  
  float4 writtenData;

  u16vec2 threadIndex;    // Thread index of the written data
  uint frameIndex;        // Frame index when the data was written
  uint2 pad;

#ifndef __cplusplus
  [mutating]
#endif
  void invalidate()
  {
    threadIndex.x = kInvalidThreadIndex;
  }

  bool isValid() { return threadIndex.x != kInvalidThreadIndex; }
};
