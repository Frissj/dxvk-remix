/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

// NV-DXVK: C++ host mirror of the vk_animated_clusters shader interface
// (shaders/rtx/pass/lodclusters/shaderio_animated.h) - only the structs the
// RTX Remix Path B integration consumes. The originals live in the global
// `shaderio` namespace, which in this codebase already holds vk_lod_clusters'
// (different) types; to avoid any ODR hazard the animated mirrors live in
// animatedclusters::shaderio. Layouts are byte-identical to the GLSL scalar
// layout (static asserts below); the GLSL side is untouched.

#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// must match shaderio_animated.h
#define CLUSTER_BLAS_WORKGROUP_SIZE 128

namespace animatedclusters {
namespace shaderio {

using namespace glm;

struct BBox
{
  vec3 lo;
  vec3 hi;
};

// one clusterized meshlet of the (bind-pose) topology
struct Cluster
{
  uint16_t numVertices;
  uint16_t numTriangles;
  uint32_t firstTriangle;
  uint32_t firstLocalVertex;
  uint32_t firstLocalTriangle;
};

// element layout of the RenderInstances_in buffer reference consumed by
// cluster_blas_instances.comp; Remix only feeds the geometryID field (the
// kernel's static branch reads it as the BLAS index for a TLAS slot).
struct RenderInstance
{
  mat4 worldMatrix;

  uint32_t numTriangles;
  uint32_t numVertices;
  uint32_t numClusters;
  uint32_t geometryID;

  // animated
  uint64_t positions;
  uint64_t normals;

  // original
  uint64_t triangles;
  uint64_t clusters;
  uint64_t clusterLocalVertices;
  uint64_t clusterLocalTriangles;
  uint64_t clusterBboxes;
  uint64_t originalPositions;
};

// push constants of cluster_blas_instances.comp (BUFFER_REF == uint64_t on
// the host side, exactly like the sample's __cplusplus branch)
struct ClusterBlasConstants
{
  uint32_t instanceCount;
  uint32_t sumCount;
  uint32_t animated;
  uint32_t _pad;

  uint64_t instances;      // RenderInstances_in
  uint64_t rayInstances;   // RayTracingInstances_inout (VkAccelerationStructureInstanceKHR)

  uint64_t blasAddresses;  // uint64s_inout

  uint64_t sizes;          // uints_in
  uint64_t sum;            // uint64s_inout
};

// push constants of anim_gather_positions.comp (per-pose live-position
// reorder into cluster-local layout; see the razor-triangle fix in
// renderer_raytrace_clusters.cpp)
struct GatherConstants
{
  uint64_t srcPositions;    // live positions base (global vertex order)
  uint64_t localVertexMap;  // uint32 per local slot -> global vertex index
  uint64_t dstPositions;    // gathered tightly-packed vec3 out
  uint32_t vertexCount;     // local slots to gather (numClusterVertices)
  uint32_t srcStrideBytes;  // live buffer position stride
};

static_assert(sizeof(BBox) == 24, "animated shaderio::BBox layout mismatch");
static_assert(sizeof(Cluster) == 16, "animated shaderio::Cluster layout mismatch");
static_assert(sizeof(RenderInstance) == 144, "animated shaderio::RenderInstance layout mismatch");
static_assert(sizeof(ClusterBlasConstants) == 56, "animated shaderio::ClusterBlasConstants layout mismatch");
static_assert(sizeof(GatherConstants) == 32, "animated shaderio::GatherConstants layout mismatch");
static_assert(sizeof(GatherConstants) <= sizeof(ClusterBlasConstants),
              "GatherConstants must fit the shared compute pipeline layout's push range");

}  // namespace shaderio
}  // namespace animatedclusters
