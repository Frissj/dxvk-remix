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
// (src/renderer.cpp) and was trimmed for the RTX Remix integration (see
// renderer.hpp for the rationale):
// - RenderScene:: methods are kept verbatim except for the removal of the
//   SceneTextures member (Remix owns materials/textures).
// - Renderer::initBasics no longer derives render instances/materials from
//   Scene::m_instances (Remix provides shaderio::RenderInstance data per
//   frame from its RtInstances); it creates the instance-capacity buffer, the
//   dummy material entry and the optional sorting scratch.
// - All "basic" viewer pipelines (fullscreen background/depth, bbox mesh
//   shaders, atomic raster) are removed.

#include <vector>

#include <volk.h>
#include <fmt/format.h>

#include "renderer.hpp"
#include "shaderio.h"


namespace lodclusters {

//////////////////////////////////////////////////////////////////////////

bool RenderScene::init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_, uint32_t preloadedGeometryCapacity)
{
  scene        = scene_;
  useStreaming = useStreaming_;

  if(useStreaming)
  {
    // NV-DXVK P3: the geometry-slot capacity feeds the streaming system's
    // reserved persistent prefix so appendGeometries works while live
    StreamingConfig streamingConfig            = streamingConfig_;
    streamingConfig.persistentGeometryCapacity = std::max(streamingConfig.persistentGeometryCapacity, preloadedGeometryCapacity);
    return sceneStreaming.init(res, scene_, streamingConfig);
  }
  else
  {
    ScenePreloaded::Config preloadConfig;
    preloadConfig.clasBuildFlags           = streamingConfig_.clasBuildFlags;
    preloadConfig.clasPositionTruncateBits = streamingConfig_.clasPositionTruncateBits;
    preloadConfig.geometryCapacity         = preloadedGeometryCapacity;
    return scenePreloaded.init(res, scene_, preloadConfig);
  }
}

// NV-DXVK P2.5/P3: incremental geometry addition for both paths. The
// preloaded path uploads everything; the streaming path adds persistent
// lowest-detail data into its reserved capacity (higher detail streams in
// on demand as usual).
bool RenderScene::appendGeometries(size_t firstGeometry, size_t geometryCount)
{
  if(useStreaming)
  {
    return sceneStreaming.appendGeometries(firstGeometry, geometryCount);
  }
  return scenePreloaded.appendGeometries(firstGeometry, geometryCount);
}

void RenderScene::deinit()
{
  scenePreloaded.deinit();
  sceneStreaming.deinit();
}

void RenderScene::streamingReset()
{
  if(useStreaming)
  {
    sceneStreaming.reset();
  }
}

bool RenderScene::updateClasRequired(bool state)
{
  if(useStreaming)
  {
    return sceneStreaming.updateClasRequired(state);
  }
  else
  {
    return scenePreloaded.updateClasRequired(state);
  }
}

const nvvk::BufferTyped<shaderio::Geometry>& RenderScene::getShaderGeometriesBuffer() const
{

  if(useStreaming)
    return sceneStreaming.getShaderGeometriesBuffer();
  else
    return scenePreloaded.getShaderGeometriesBuffer();
}

size_t RenderScene::getClasSize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getClasSize(reserved);
  else
    return scenePreloaded.getClasSize();
}

size_t RenderScene::getBlasSize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getBlasSize(reserved);
  else
    return scenePreloaded.getBlasSize();
}

size_t RenderScene::getGeometrySize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getGeometrySize(reserved);
  else
    return scenePreloaded.getGeometrySize();
}

size_t RenderScene::getOperationsSize() const
{
  if(useStreaming)
    return sceneStreaming.getOperationsSize();
  else
    return scenePreloaded.getOperationsSize();
}

//////////////////////////////////////////////////////////////////////////

void Renderer::initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  // NV-DXVK: render instances come from Remix per frame; allocate for the
  // configured capacity. Contents are uploaded by the caller each frame.
  res.createBuffer(m_renderInstanceBuffer, sizeof(shaderio::RenderInstance) * config.maxRenderInstances,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  NVVK_DBG_NAME(m_renderInstanceBuffer.buffer);
  m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_renderInstanceBuffer.bufferSize, "operations", "render instances");

  // NV-DXVK: single zeroed dummy material entry (see renderer.hpp)
  shaderio::RenderMaterial dummyMaterial = {};
  res.createBuffer(m_renderMaterialBuffer, sizeof(shaderio::RenderMaterial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NVVK_DBG_NAME(m_renderMaterialBuffer.buffer);
  res.simpleUploadBuffer(m_renderMaterialBuffer, &dummyMaterial);

  if(config.useSorting)
  {
    VrdxSorterStorageRequirements sorterRequirements = {};
    vrdxGetSorterKeyValueStorageRequirements(res.m_vrdxSorter, config.maxRenderInstances, &sorterRequirements);

    res.createBuffer(m_sortingAuxBuffer, sorterRequirements.size, sorterRequirements.usage);
    NVVK_DBG_NAME(m_sortingAuxBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sortingAuxBuffer.bufferSize, "operations", "traversal sorting");
  }
}

void Renderer::deinitBasics(Resources& res)
{
  res.m_allocator.destroyBuffer(m_renderMaterialBuffer);
  res.m_allocator.destroyBuffer(m_renderInstanceBuffer);
  res.m_allocator.destroyBuffer(m_sortingAuxBuffer);
}

// vk_lod_clusters c19a250
float Renderer::updateLodPixelError(Resources& res, RenderScene& rscene, const FrameConfig& frame)
{
  float lodPixelError = frame.lodPixelError;
  if(rscene.useStreaming && frame.adaptiveError)
  {
    m_lodPixelError  = std::max(frame.lodPixelError, m_lodPixelError);
    float loadFactor = rscene.sceneStreaming.getLoadFactor();
    // Smooth load factor to avoid reacting to single-frame spikes.
    m_smoothedLoadFactor = glm::mix(m_smoothedLoadFactor, loadFactor, 0.05f);
    // Deadband [0.70, 0.85]: no adjustment to prevent oscillation near the threshold.
    // Outside the band: increase error quickly when overloaded, recover slowly.
    if(m_smoothedLoadFactor > 0.85f)
      m_lodPixelError *= 1.02f;
    else if(m_smoothedLoadFactor < 0.70f)
      m_lodPixelError *= 0.995f;

    m_lodPixelError = std::max(frame.lodPixelError, m_lodPixelError);
    lodPixelError   = m_lodPixelError;
  }
  else
  {
    // keep synchronized so enabling adaptive mode starts from the
    // current error rather than a stale adaptive value
    m_lodPixelError = frame.lodPixelError;
  }

  glm::vec2 renderScale       = res.getFramebufferWindow2RenderScale();
  float     pixelScale        = std::min(renderScale.x, renderScale.y);
  float     errorSizeInPixels = lodPixelError * pixelScale;

  // note we use half-pixel sizes: error taken as radius, not as diameter.
  // otherwise there was more LoD popping.
  return (tanf(frame.traversalFov * 0.5f) * errorSizeInPixels / frame.traversalViewHeight);
}

}  // namespace lodclusters
