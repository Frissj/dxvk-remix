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

// NV-DXVK: This file originates from nvpro-samples/vk_lod_clusters
// (src/renderer.hpp) and was trimmed for the RTX Remix integration:
// - The sample's viewer subsystems are removed: SceneTextures (Remix owns all
//   materials/textures), the raster renderer, the "basic" fullscreen/bbox
//   debug pipelines and their shaders, and DLSS.
// - The ray tracing renderer no longer owns a TLAS or any ray tracing
//   pipeline: it produces cluster BLASes and patches their addresses into an
//   externally provided TlasInstance array (Remix's AccelManager owns the
//   TLAS and the path tracer consumes it).
// - Render instances are provided by Remix per frame (RtInstances) instead of
//   being derived from Scene::m_instances once at init. The renderer is
//   initialized with a fixed instance capacity; the per-frame count may be
//   anything up to that capacity.
// Everything kept below is unchanged from the sample wherever possible.

#pragma once

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <memory>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "resources.hpp"
#include "scene.hpp"
#include "scene_preloaded.hpp"
#include "scene_streaming.hpp"

namespace lodclusters {

// There are two implementations for a renderable scene.
// Everything is preloaded or we stream in data dynamically.
class RenderScene
{
public:
  const Scene*   scene        = nullptr;
  bool           useStreaming = false;
  ScenePreloaded scenePreloaded;
  SceneStreaming sceneStreaming;

  // pointers must stay valid during lifetime
  // NV-DXVK P2.5/P3: preloadedGeometryCapacity sizes the geometry table (and,
  // for streaming, the reserved persistent prefix) for incremental appends
  // (0 = exact count, sample behavior)
  bool init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_, uint32_t preloadedGeometryCapacity = 0);
  void deinit();

  // NV-DXVK P2.5/P3: uploads scene geometries [firstGeometry, firstGeometry +
  // geometryCount) appended to the Scene after init (preloaded: full data,
  // streaming: persistent lowest-detail data). Returns false when the append
  // cannot be accommodated (the caller then performs a full generation
  // rebuild).
  bool appendGeometries(size_t firstGeometry, size_t geometryCount);

  void streamingReset();

  bool updateClasRequired(bool state);

  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const;
  size_t                                       getClasSize(bool reserved) const;
  size_t                                       getBlasSize(bool reserved) const;
  size_t                                       getOperationsSize() const;
  size_t                                       getGeometrySize(bool reserved) const;
};

struct RendererConfig
{
  bool flipWinding               = false;
  bool forceTwoSided             = false;
  bool useForcedInvisibleCulling = false;
  bool useSorting                = false;
  bool useRenderStats            = false;
  bool useCulling                = true;
  bool useBlasSharing            = true;
  bool useBlasMerging            = true;
  bool useBlasCaching            = false;
  bool usePersistentTraversal    = true;

  // the maximum number of renderable clusters per frame in bits i.e. (1 << number)
  uint32_t numRenderClusterBits = 20;
  // the maximum number of traversal intermediate tasks
  uint32_t numTraversalTaskBits = 20;

  // build flags for the cluster BLAS
  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags = 0;

  // NV-DXVK: maximum number of render instances the renderer's buffers are
  // sized for. The per-frame instance count may be anything up to this.
  uint32_t maxRenderInstances = 1;

  // NV-DXVK P2.5: maximum number of geometries the renderer's geometry-count
  // dependent buffers (BLAS-sharing build infos, geometry histograms) are
  // sized for. 0 = exactly the scene's geometry count at init (sample
  // behavior). A larger capacity lets the scene grow via appendGeometries
  // without re-initializing the renderer.
  uint32_t maxGeometries = 0;
};

// NV-DXVK: per-frame inputs that in the sample were derived from
// Renderer-internal state (TLAS instances buffer) or set once at init.
struct RendererFrameInput
{
  // number of render instances active this frame (<= config.maxRenderInstances)
  uint32_t numRenderInstances = 0;

  // NV-DXVK P2.5: number of geometries valid this frame (<= the capacity the
  // renderer was sized for). 0 keeps the count from init. Driven by the
  // caller so it only advances once an appended range is fully uploaded.
  uint32_t numGeometries = 0;

  // device address of the shaderio::TlasInstance array instance_assign_blas
  // patches (the renderer-owned staging buffer, see getTlasInstancesBuffer)
  uint64_t tlasInstancesAddress = 0;
};

class Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, const RendererFrameInput& frameInput, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};  // Defined only so that inherited classes also have virtual destructors. Use deinit().

  // NV-DXVK P4: called after Resources recreated the HiZ images (render
  // resolution change) so descriptors referencing them are rewritten. The
  // sample's version also refreshed its color/DLSS targets; only the HiZ
  // binding remains in Remix.
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) {}

  // NV-DXVK P2.5: true when the renderer's init-time sizing (geometry
  // capacity, per-BLAS cluster maxima) still accommodates the - possibly
  // appended-to - scene. When false the renderer must be re-initialized
  // (full generation rebuild).
  virtual bool canRenderScene(const RenderScene& /*rscene*/) const { return true; }

  struct ResourceUsageInfo
  {
    size_t rtTlasMemBytes{};
    size_t rtBlasMemBytes{};
    size_t rtClasMemBytes{};
    size_t operationsMemBytes{};
    size_t geometryMemBytes{};

    void add(const ResourceUsageInfo& other)
    {
      rtTlasMemBytes += other.rtTlasMemBytes;
      rtBlasMemBytes += other.rtBlasMemBytes;
      rtClasMemBytes += other.rtClasMemBytes;
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }
    size_t getTotalSum() const
    {
      return rtTlasMemBytes + rtBlasMemBytes + rtClasMemBytes + geometryMemBytes + operationsMemBytes;
    }
  };

  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

  uint32_t getMaxRenderClusters() const { return m_maxRenderClusters; }
  uint32_t getMaxTraversalTasks() const { return m_maxTraversalTasks; }
  uint32_t getMaxBlasBuilds() const { return m_maxBlasBuilds; }

  // NV-DXVK: Remix uploads the per-frame shaderio::RenderInstance array into
  // this buffer (sized for config.maxRenderInstances at init).
  const nvvk::Buffer& getRenderInstanceBuffer() const { return m_renderInstanceBuffer; }

  // NV-DXVK: staging array of shaderio::TlasInstance that instance_assign_blas
  // patches; Remix uploads CPU-known fields per frame and copies patched
  // entries out into AccelManager's instance buffer.
  virtual const nvvk::Buffer& getTlasInstancesBuffer() const = 0;

  // vk_lod_clusters c19a250: current (possibly adaptively raised) LoD pixel error
  inline float getLodError() const { return m_lodPixelError; }

protected:
  void initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config);
  void deinitBasics(Resources& res);

  // vk_lod_clusters c19a250: computes errorOverDistanceThreshold from frame.lodPixelError,
  // optionally scaled up under streaming memory pressure (frame.adaptiveError). Replaces the
  // old stateless clusterLodErrorOverDistance helper.
  float updateLodPixelError(Resources& res, RenderScene& rscene, const FrameConfig& frame);

  RendererConfig m_config;
  uint32_t       m_maxRenderClusters = 0;
  uint32_t       m_maxTraversalTasks = 0;
  uint32_t       m_maxBlasBuilds     = 0;
  uint32_t       m_frameIndex        = 0;
  // vk_lod_clusters c19a250: adaptive LoD error state
  float          m_lodPixelError{};
  float          m_smoothedLoadFactor{};

  nvvk::Buffer m_renderInstanceBuffer;

  // NV-DXVK: the compute kernels' shared binding block declares the render
  // material SSBO; Remix shades through its own materials, so a single zeroed
  // dummy entry keeps the descriptor valid.
  nvvk::Buffer m_renderMaterialBuffer;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  nvvk::Buffer m_sortingAuxBuffer;
};

//////////////////////////////////////////////////////////////////////////

// vk_lod_clusters c19a250: clusterLodErrorOverDistance removed, its math lives in
// Renderer::updateLodPixelError now (which also applies the adaptive scaling)

std::unique_ptr<Renderer> makeRendererRayTraceClustersLod();

}  // namespace lodclusters
