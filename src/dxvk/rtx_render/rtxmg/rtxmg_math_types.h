/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Adapted from NVIDIA RTX Mega Geometry SDK for RTX Remix integration
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*/

#pragma once

#include <cstdint>

// HLSL-style vector types for RTXMG compatibility
// Used in both C++ code and compute shaders (when included in Slang)

#ifdef __cplusplus
namespace dxvk {

// 2-component unsigned 16-bit integer vector
struct uint16_t2 {
  uint16_t x, y;

  uint16_t2() : x(0), y(0) {}
  uint16_t2(uint16_t x_, uint16_t y_) : x(x_), y(y_) {}

  uint16_t2 operator*(const uint16_t2& o) const {
    return uint16_t2(x * o.x, y * o.y);
  }

  uint16_t2 operator+(const uint16_t2& o) const {
    return uint16_t2(x + o.x, y + o.y);
  }

  bool operator==(const uint16_t2& o) const {
    return x == o.x && y == o.y;
  }
};

// 2-component unsigned 32-bit integer vector
struct uint2 {
  uint32_t x, y;

  uint2() : x(0), y(0) {}
  uint2(uint32_t x_, uint32_t y_) : x(x_), y(y_) {}

  uint2 operator*(const uint2& o) const {
    return uint2(x * o.x, y * o.y);
  }

  uint2 operator+(const uint2& o) const {
    return uint2(x + o.x, y + o.y);
  }

  bool operator==(const uint2& o) const {
    return x == o.x && y == o.y;
  }
};

// 4-component unsigned 32-bit integer vector
struct uint4 {
  uint32_t x, y, z, w;

  uint4() : x(0), y(0), z(0), w(0) {}
  uint4(uint32_t x_, uint32_t y_, uint32_t z_, uint32_t w_)
    : x(x_), y(y_), z(z_), w(w_) {}

  bool operator==(const uint4& o) const {
    return x == o.x && y == o.y && z == o.z && w == o.w;
  }
};

// 2-component float vector (alias to Vector2 when needed)
struct float2 {
  float x, y;

  float2() : x(0.0f), y(0.0f) {}
  float2(float x_, float y_) : x(x_), y(y_) {}

  float2 operator+(const float2& o) const {
    return float2(x + o.x, y + o.y);
  }

  float2 operator-(const float2& o) const {
    return float2(x - o.x, y - o.y);
  }

  float2 operator*(float s) const {
    return float2(x * s, y * s);
  }

  bool operator==(const float2& o) const {
    return x == o.x && y == o.y;
  }
};

// 3-component float vector (alias to Vector3 when needed)
struct float3 {
  float x, y, z;

  float3() : x(0.0f), y(0.0f), z(0.0f) {}
  float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  float3 operator+(const float3& o) const {
    return float3(x + o.x, y + o.y, z + o.z);
  }

  float3 operator-(const float3& o) const {
    return float3(x - o.x, y - o.y, z - o.z);
  }

  float3 operator*(float s) const {
    return float3(x * s, y * s, z * s);
  }

  bool operator==(const float3& o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

// 4-component float vector
struct float4 {
  float x, y, z, w;

  float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
  float4(float x_, float y_, float z_, float w_)
    : x(x_), y(y_), z(z_), w(w_) {}

  bool operator==(const float4& o) const {
    return x == o.x && y == o.y && z == o.z && w == o.w;
  }
};

} // namespace dxvk
#endif // __cplusplus
