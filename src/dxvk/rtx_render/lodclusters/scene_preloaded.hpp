/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include "scene.hpp"
#include "resources.hpp"

namespace lodclusters {

// With this class we pre-load all lod levels of the rendered scene.
// It is much more memory intensive.
class ScenePreloaded
{
public:
  struct Config
  {
    VkBuildAccelerationStructureFlagsKHR clasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t                             clasPositionTruncateBits = 0;

    // NV-DXVK P2.5: number of geometry slots the shaderio::Geometry table is
    // created with (0 = exactly the scene's geometry count, sample behavior).
    // A capacity larger than the initial count lets appendGeometries add
    // geometries without recreating the table buffer - its device address and
    // descriptor stay valid for in-flight frames.
    uint32_t geometryCapacity = 0;
  };

  static bool canPreload(VkDeviceSize, const Scene* scene);

  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene, const Config& config);

  // NV-DXVK P2.5: uploads scene geometries [firstGeometry, firstGeometry +
  // geometryCount) that were appended to the Scene after init (see
  // Scene::appendCachedGeometries). Only the new geometries' buffers are
  // created and uploaded and only their table entries are written - resident
  // geometry is untouched. If CLAS are active, the new geometries' CLAS and
  // low-detail BLAS are built as well. Fails (false) when the geometry table
  // capacity would be exceeded or the preload memory estimate fails; the
  // caller then falls back to a full generation rebuild.
  bool appendGeometries(size_t firstGeometry, size_t geometryCount);

  // run prior the renderer starts referencing resources
  // if true CLAS for all clusters will be built
  bool updateClasRequired(bool state);

  // tear down, safe to call without init
  void deinit();

  // renderers need to access this buffer
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }

  // NV-DXVK: CPU copy of the geometry table; Remix pre-fills each TlasInstance's
  // blasReference with its geometry's lowDetailBlasAddress (the safe default the
  // instance_assign_blas kernel expects when a BLAS build was skipped).
  const std::vector<shaderio::Geometry>& getShaderGeometries() const { return m_shaderGeometries; }

  // device memory usage
  size_t getClasSize() const { return m_clasSize; }
  size_t getBlasSize() const { return m_blasSize; };
  size_t getGeometrySize() const { return m_geometrySize; }
  size_t getOperationsSize() const { return m_operationsSize + m_clasOperationsSize; }

private:
  struct Geometry
  {
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<shaderio::Node>     lodNodes;
    nvvk::BufferTyped<shaderio::BBox>     lodNodeBboxes;

    nvvk::Buffer                groupData;
    nvvk::BufferTyped<uint64_t> groupAddresses;
    nvvk::BufferTyped<uint64_t> clusterAddresses;

    // for ray tracing
    nvvk::BufferTyped<uint64_t> clusterClasAddresses;
    nvvk::BufferTyped<uint32_t> clusterClasSizes;
    nvvk::Buffer                clasData;
  };

  Config       m_config;
  bool         m_hasClas   = false;
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  size_t m_clasSize           = 0;
  size_t m_blasSize           = 0;
  size_t m_clasOperationsSize = 0;
  size_t m_geometrySize       = 0;
  size_t m_operationsSize     = 0;

  std::vector<ScenePreloaded::Geometry> m_geometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;

  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;

  // NV-DXVK P2.5: one implicit-destination low-detail BLAS buffer per
  // init/append batch (the sample's single buffer becomes entry [0])
  std::vector<nvvk::Buffer> m_clasLowDetailBlasBuffers;

  bool initClas();
  void deinitClas();

  // NV-DXVK P2.5: range-based bodies of init()/initClas(), mirroring
  // SceneStreaming::initGeometries' per-geometry structure. Both only touch
  // geometries [firstGeometry, firstGeometry + geometryCount) plus their
  // entries of the (capacity-sized) geometry table.
  bool initGeometries(size_t firstGeometry, size_t geometryCount);
  bool initClasGeometries(size_t firstGeometry, size_t geometryCount);
};
}  // namespace lodclusters
