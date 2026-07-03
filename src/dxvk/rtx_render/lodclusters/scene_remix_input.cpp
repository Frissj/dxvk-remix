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

// NV-DXVK: Remix mesh input path for lodclusters::Scene.
//
// This file is the Remix counterpart of scene_gltf.cpp: it feeds Remix-provided
// mesh data (game draw-call CPU snapshots, replacement assets, Remix-API meshes)
// into the identical Scene processing pipeline (processGeometry / buildGeometryLod
// / cache machinery), replacing only the cgltf ingestion. The orchestration in
// loadFromMeshInputs and the per-geometry fill in loadGeometryRemix deliberately
// mirror loadGLTF / loadGeometryGLTF line-for-line wherever applicable.

#include <cinttypes>
#include <cstring>
#include <cfloat>
#include <algorithm>

#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_operations.hpp>

#include "scene.hpp"

namespace {

// based on meshopt_quantizeFloat
// https://github.com/zeux/meshoptimizer/blob/master/src/quantization.cpp
// (verbatim copy of the helpers in scene_gltf.cpp so compression behaves identically)
inline float quantizeFloat(float value, uint32_t dropBits)
{
  union
  {
    uint32_t u32;
    float    f32;
  } un;

  un.f32      = value;
  uint32_t ui = un.u32;

  const int32_t mask  = (1 << (dropBits)) - 1;
  const int32_t round = (1 << (dropBits)) >> 1;

  int32_t  e   = ui & 0x7f800000;
  uint32_t rui = (ui + round) & ~mask;

  // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
  ui = e == 0x7f800000 ? ui : rui;

  // flush denormals to zero
  ui = e == 0 ? 0 : ui;

  un.u32 = ui;
  return un.f32;
}

inline glm::vec2 quantizeFloat(const glm::vec2& vec, uint32_t dropBits)
{
  glm::vec2 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  return res;
}

inline glm::vec3 quantizeFloat(const glm::vec3& vec, uint32_t dropBits)
{
  glm::vec3 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  return res;
}

inline glm::vec4 quantizeFloat(const glm::vec4& vec, uint32_t dropBits)
{
  glm::vec4 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  res.w = quantizeFloat(vec.w, dropBits);
  return res;
}

}  // namespace

namespace lodclusters {

// glTF loading is intentionally not ported to Remix (scene_gltf.cpp stays in the
// sample; Remix feeds geometry through loadFromMeshInputs instead). This keeps the
// untouched Scene::init linkable should anything ever call the glTF entry point.
Scene::Result Scene::loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath)
{
  LOGE("Scene::loadGLTF is not available in Remix - geometry is fed via Scene::initFromMeshInputs\n");
  return SCENE_RESULT_ERROR;
}

Scene::Result Scene::initFromMeshInputs(const std::filesystem::path&    filePath,
                                        std::span<const RemixMeshInput> inputs,
                                        const SceneConfig&              config,
                                        const SceneLoaderConfig&        loaderConfig,
                                        const std::string&              cacheSuffix,
                                        bool                            skipCache)
{
  // structural clone of Scene::init with loadGLTF swapped for loadFromMeshInputs;
  // filePath is only ever used to derive cache file names.
  *this = {};

  m_filePath             = filePath;
  m_config               = config;
  m_loaderConfig         = loaderConfig;
  m_loadedFromCache      = false;
  m_cacheFilePath        = filePath;
  m_cachePartialFilePath = filePath;
  m_cacheFileSize        = 0;
  m_cacheSuffix          = cacheSuffix;

  std::string oldExtension = filePath.extension().string();
  m_cacheFilePath.replace_extension(oldExtension + cacheSuffix);
  m_cachePartialFilePath.replace_extension(oldExtension + cacheSuffix + "_partial");

  if(!skipCache && !m_loaderConfig.processingOnly && m_loaderConfig.autoLoadCache)
  {
    openCache();
  }

  ProcessingInfo processingInfo;
  processingInfo.init(m_loaderConfig.processingThreadsPct);

  Result loadResult = loadFromMeshInputs(processingInfo, inputs);
  if(loadResult == SCENE_RESULT_NEEDS_PREPROCESS || loadResult == SCENE_RESULT_CACHE_INVALID)
  {
    LOGI("Scene::initFromMeshInputs large scene or invalid cache detected\n  using dedicated preprocess pass\n");
    closeCache();

    m_loaderConfig.processingOnly = true;
    loadResult                    = loadFromMeshInputs(processingInfo, inputs);
    m_loaderConfig.processingOnly = false;
    if(loadResult == SCENE_RESULT_PREPROCESS_COMPLETED)
    {
      openCache();
      loadResult = loadFromMeshInputs(processingInfo, inputs);
    }
  }

  processingInfo.deinit();

  if(loadResult != SCENE_RESULT_SUCCESS)
  {
    closeCache();

    return loadResult;
  }

  if(m_loadedFromCache)
  {
    m_cacheFileView.getHistograms(m_histograms);
  }

  m_originalInstanceCount = m_instances.size();
  m_originalGeometryCount = m_geometryViews.size();
  m_activeGeometryCount   = m_originalGeometryCount;

  computeInstanceBBoxes();
  m_gridBbox = m_bbox;

  glm::vec3 modelExtent = m_bbox.hi - m_bbox.lo;
  m_isBig = modelExtent.y < 0.15f * std::max(modelExtent.x, modelExtent.z) && m_originalInstanceCount > 1024;

  for(auto& geometry : m_geometryViews)
  {
    m_hiPerGeometryTriangles = std::max(m_hiPerGeometryTriangles, geometry.hiTriangleCount);
    m_hiPerGeometryVertices  = std::max(m_hiPerGeometryVertices, geometry.hiVerticesCount);
    m_hiPerGeometryClusters  = std::max(m_hiPerGeometryClusters, geometry.hiClustersCount);
    m_hiPerGeometryGroups = std::max(m_hiPerGeometryGroups, geometry.lodLevels.empty() ? 0 : geometry.lodLevels[0].groupCount);

    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geometry.totalTriangleCount);
    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geometry.totalVerticesCount);
    m_maxPerGeometryClusters  = std::max(m_maxPerGeometryClusters, geometry.totalClustersCount);
    m_maxClusterVertices      = std::max(m_maxClusterVertices, geometry.clusterMaxVerticesCount);
    m_maxClusterTriangles     = std::max(m_maxClusterTriangles, geometry.clusterMaxTrianglesCount);
    m_maxLodLevelsCount       = std::max(m_maxLodLevelsCount, geometry.lodLevelsCount);

    m_hiTrianglesCount += geometry.hiTriangleCount;
    m_hiClustersCount += geometry.hiClustersCount;
    m_totalClustersCount += geometry.totalClustersCount;
    m_totalTrianglesCount += geometry.totalTriangleCount;
    m_totalVerticesCount += geometry.totalVerticesCount;

    if(geometry.localMaterialIDs.size() > 1)
    {
      m_geometryMultiMaterialCount += uint32_t(geometry.localMaterialIDs.size());
    }
  }
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    const GeometryView& geometry = m_geometryViews[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geometry.hiTriangleCount;
    m_hiClustersCountInstanced += geometry.hiClustersCount;
  }

  {
    // estimate depth of lod tree based on highest detail group count
    uint32_t hiGroups   = m_hiPerGeometryGroups;
    uint32_t hiNodes    = (hiGroups + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
    uint32_t rootPasses = 0;

    m_maxNodeTreeDepth = 1;
    while(hiNodes)
    {
      hiNodes = (hiNodes + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
      m_maxNodeTreeDepth++;
      if(hiNodes == 1)
        break;
    }
    // lod tree to root
    m_maxNodeTreeDepth++;
  }

  LOGI("cluster triangles: %d\n", m_config.clusterTriangles);
  LOGI("cluster vertices: %d\n", m_config.clusterTriangles);
  LOGI("cluster group: %d\n", m_config.clusterGroupSize);
  LOGI("clusters:  %" PRIu64 "\n", m_totalClustersCount);
  LOGI("triangles: %" PRIu64 "\n", m_totalTrianglesCount);
  LOGI("triangles/cluster: %.2f\n", double(m_totalTrianglesCount) / double(m_totalClustersCount));
  LOGI("vertices: %" PRIu64 "\n", m_totalVerticesCount);
  LOGI("vertices/cluster: %.2f\n", double(m_totalVerticesCount) / double(m_totalClustersCount));
  LOGI("hi clusters:  %" PRIu64 "\n", m_hiClustersCount);
  LOGI("hi triangles: %" PRIu64 "\n", m_hiTrianglesCount);
  LOGI("hi triangles/cluster: %.2f\n", double(m_hiTrianglesCount) / double(m_hiClustersCount));

  if(!m_loadedFromCache && m_loaderConfig.autoSaveCache)
  {
    saveCache();
  }

  if(m_loadedFromCache && !m_loaderConfig.memoryMappedCache)
  {
    // everything was loaded into system memory,
    // close file mappings
    closeCache();
  }

  return loadResult;
}

Scene::Result Scene::loadFromMeshInputs(ProcessingInfo& processingInfo, std::span<const RemixMeshInput> inputs)
{
  // One Scene material per input geometry. Only the two-sided / alpha-mask state is
  // consumed here (baked into cluster state bits during storeGroup); actual shading
  // always comes from Remix's own material system at hit resolution.
  m_materials.resize(inputs.size());
  m_materialNames.resize(inputs.size());
  for(size_t i = 0; i < inputs.size(); i++)
  {
    Material& material   = m_materials[i];
    material.twoSided    = inputs[i].twoSided;
    material.alphaMasked = inputs[i].alphaMasked;
    material.alphaCutOff = inputs[i].alphaCutOff;
    m_materialNames[i]   = inputs[i].name;

    m_hasTwoSided |= material.twoSided;
    m_hasAlphaMask |= material.alphaMasked;
  }

  // unlike glTF there is no mesh-dedup pass needed: every input is already a unique
  // geometry (keyed by Remix geometry hash before it gets here)
  std::vector<size_t> taskToGeometry(inputs.size());
  std::vector<size_t> geometryTriangleCount(inputs.size());

  uint64_t totalTriangleCount = 0;

  {
    size_t geometryMemoryEstimate = 0;

    for(size_t i = 0; i < inputs.size(); i++)
    {
      taskToGeometry[i] = i;

      size_t meshTriangleCount = inputs[i].indices.size() / 3;
      geometryTriangleCount[i] = meshTriangleCount;
      totalTriangleCount += meshTriangleCount;

      geometryMemoryEstimate += sizeof(glm::vec4) * inputs[i].vertexCount;
      geometryMemoryEstimate += sizeof(uint32_t) * inputs[i].indices.size();
    }

    // if there is too much geometry memory in the scene and we are not in processing only mode, early out
    if(!m_cacheFileView.isValid() && !m_loaderConfig.processingOnly
       && (geometryMemoryEstimate > size_t(m_loaderConfig.forcePreprocessMiB) * 1024 * 1024))
    {
      return SCENE_RESULT_NEEDS_PREPROCESS;
    }
  }

  m_geometryStorages.resize(inputs.size());
  m_geometryViews.resize(inputs.size());
  m_geometryNames.resize(inputs.size());

  beginProcessingOnly(inputs.size());

  // when we are resuming in processingOnly mode, we might have completed several geometries already,
  // which is passed to influence the decision about the parallelism mode.
  processingInfo.setupParallelism(inputs.size(), m_processingOnlyPartialCompleted, m_loaderConfig.processingMode);

  if(processingInfo.numOuterThreads > processingInfo.numInnerThreads)
  {
    // let's do the actual processing in a slightly different order (large meshes first).
    // This gives better work distribution across threads, avoids few long running threads
    // at the end. Thanks Arseny Kapoulkine for this suggestion.
    std::sort(taskToGeometry.begin(), taskToGeometry.end(),
              [&](size_t l, size_t r) { return geometryTriangleCount[l] > geometryTriangleCount[r]; });
  }

  auto fnLoadAndProcessGeometry = [&](uint64_t taskIndex, uint32_t threadOuterIdx) {
    uint64_t geometryIndex = taskToGeometry[taskIndex];

    loadGeometryRemix(processingInfo, geometryIndex, inputs[geometryIndex]);
  };

  // for partial files we don't have the completed triangle information
  processingInfo.logBegin(m_processingOnlyPartialFile ? 0 : totalTriangleCount);
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(0);
  }

  nvutils::parallel_batches_pooled<1>(inputs.size(), fnLoadAndProcessGeometry, processingInfo.numOuterThreads);

  processingInfo.logEnd();
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(100);
  }

  bool notCompleted = processingInfo.progressGeometriesCompleted != inputs.size();
  if(notCompleted)
  {
    LOGW("Error in processing geometries, completed / required mismatch\n");
  }
  else
  {
    computeHistogramMaxs();
  }

  if(endProcessingOnly(notCompleted))
  {
    return notCompleted ? SCENE_RESULT_ERROR : SCENE_RESULT_PREPROCESS_COMPLETED;
  }

  if(notCompleted)
  {
    return m_cacheFileView.isValid() ? SCENE_RESULT_CACHE_INVALID : SCENE_RESULT_ERROR;
  }

  // one identity-transform instance per geometry: keeps NVIDIA's bbox / statistics /
  // grid-stress machinery working unchanged. At render time the actual TLAS instances
  // come from Remix's live RtInstances, not from these.
  m_instances.resize(inputs.size());
  for(size_t i = 0; i < inputs.size(); i++)
  {
    Instance& instance  = m_instances[i];
    instance            = {};
    instance.matrix     = glm::mat4(1);
    instance.geometryID = uint32_t(i);
    instance.materialID = uint32_t(i);
    instance.twoSided   = m_materials[i].twoSided;
  }

  return SCENE_RESULT_SUCCESS;
}

void Scene::loadGeometryRemix(ProcessingInfo& processingInfo, uint64_t geometryIndex, const RemixMeshInput& input)
{
  // when resuming a partial processing, early out if it was already processed
  // second entry is dataSize
  if(m_processingOnlyPartialFile && m_processingOnlyGeometryOffsets[geometryIndex * 2 + 1])
  {
    uint32_t percentage = processingInfo.logCompletedGeometry();
    if(m_loaderConfig.progressPct)
    {
      m_loaderConfig.progressPct->store(percentage);
    }

    return;
  }

  GeometryStorage& geometry = m_geometryStorages[geometryIndex];
  geometry.bbox             = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  m_geometryNames[geometryIndex] = input.name;

  // single material per Remix geometry (index established by loadFromMeshInputs)
  geometry.localMaterialIDs.push_back(uint32_t(geometryIndex));

  const uint32_t triangleCount = uint32_t(input.indices.size() / 3);
  const uint32_t verticesCount = input.vertexCount;

  geometry.attributeBits = 0;
  if(input.normals && (m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
  {
    m_hasVertexNormals = true;
    geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;
  }
  if(input.tangents && (m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
  {
    m_hasVertexTangents = true;
    geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
  }
  if(input.texcoords0 && (m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
  {
    m_hasVertexTexCoord0 = true;
    geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0;
  }
  if(input.texcoords1 && (m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
  {
    m_hasVertexTexCoord1 = true;
    geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
  }

  // use memset 0 to avoid issues with padding within struct
  memset(&geometry.lodInfo, 0, sizeof(geometry.lodInfo));
  geometry.lodInfo.inputTriangleCount = triangleCount;
  geometry.lodInfo.inputVertexCount   = verticesCount;
  // unlike glTF we do have stable content hashes from Remix, store them for cache validation
  geometry.lodInfo.inputTriangleIndicesHash = input.indicesHash;
  geometry.lodInfo.inputVerticesHash        = input.verticesHash;

  // test if this mesh exists in the cache
  bool isCached = checkCache(geometry.lodInfo, geometryIndex);

  // invalid cache
  if(m_cacheFileView.isValid() && !isCached)
  {
    LOGW("geometry mismatches scene cache file\n");
    return;
  }

  // load vertices & index data
  if(!isCached)
  {
    // disable tangents if no TEXCOORDS or NORMALS are provided
    // might as well use automatic tangent space then
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }

    // disable TEX_1 if no TEX_0
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
    }

    bool hasMultiMaterial = geometry.localMaterialIDs.size() > 1;

    size_t attributeStride = (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL ? 3 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT ? 4 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 ? 2 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1 ? 2 : 0)
                             + (hasMultiMaterial ? 1 : 0);
    uint32_t attributeStart = 0;
    uint32_t attributeEnd   = uint32_t(attributeStride);

    // all attributes with simplification weights must come first due to how
    // meshoptimizer works
    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      if(m_config.simplifyNormalWeight > 0)
      {
        geometry.attributeNormalOffset = attributeStart;
        attributeStart += 3;
      }
      else
      {
        geometry.attributeNormalOffset = attributeEnd - 3;
        attributeEnd -= 3;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex0offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex0offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex1offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex1offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
    {
      if(m_config.simplifyTangentWeight > 0 && m_config.simplifyTangentSignWeight > 0)
      {
        geometry.attributeTangentOffset = attributeStart;
        attributeStart += 4;
      }
      else
      {
        geometry.attributeTangentOffset = attributeEnd - 4;
        attributeEnd -= 4;
      }
    }

    if(hasMultiMaterial)
    {
      if(m_config.simplifyMaterialWeight > 0)
      {
        geometry.attributeMaterialOffset = attributeStart;
        attributeStart += 1;
      }
      else
      {
        geometry.attributeMaterialOffset = attributeEnd - 1;
        attributeEnd -= 1;
      }
    }

    assert(attributeStart == attributeEnd);

    geometry.attributesWithWeights = attributeStart;
    geometry.vertexPositions.resize(verticesCount);
    geometry.vertexAttributes.resize(verticesCount * attributeStride, 0);
    geometry.triangles.resize(triangleCount);

    // fill pass

    // positions (+ bbox, + optional mantissa drop matching readAttributesGLTF)
    {
      const uint32_t dropBits = m_config.useCompressedData ? m_config.compressionPosDropBits : 0;

      for(uint32_t v = 0; v < verticesCount; v++)
      {
        const float* src = input.positions + size_t(v) * input.positionStride;

        glm::vec3 pos = {src[0], src[1], src[2]};
        if(dropBits)
        {
          pos = quantizeFloat(pos, dropBits);
        }

        geometry.vertexPositions[v] = pos;

        geometry.bbox.lo = glm::min(geometry.bbox.lo, pos);
        geometry.bbox.hi = glm::max(geometry.bbox.hi, pos);
      }
    }

    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      float* writeAttributes = geometry.vertexAttributes.data() + geometry.attributeNormalOffset;
      for(uint32_t v = 0; v < verticesCount; v++)
      {
        const float* src = input.normals + size_t(v) * input.normalStride;

        writeAttributes[size_t(v) * attributeStride + 0] = src[0];
        writeAttributes[size_t(v) * attributeStride + 1] = src[1];
        writeAttributes[size_t(v) * attributeStride + 2] = src[2];
      }
    }

    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
    {
      float* writeAttributes = geometry.vertexAttributes.data() + geometry.attributeTangentOffset;
      for(uint32_t v = 0; v < verticesCount; v++)
      {
        const float* src = input.tangents + size_t(v) * input.tangentStride;

        writeAttributes[size_t(v) * attributeStride + 0] = src[0];
        writeAttributes[size_t(v) * attributeStride + 1] = src[1];
        writeAttributes[size_t(v) * attributeStride + 2] = src[2];
        writeAttributes[size_t(v) * attributeStride + 3] = src[3];
      }
    }

    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
    {
      const uint32_t dropBits = m_config.useCompressedData ? m_config.compressionTexDropBits : 0;

      float* writeAttributes = geometry.vertexAttributes.data() + geometry.attributeTex0offset;
      for(uint32_t v = 0; v < verticesCount; v++)
      {
        const float* src = input.texcoords0 + size_t(v) * input.texcoord0Stride;

        glm::vec2 tex = {src[0], src[1]};
        if(dropBits)
        {
          tex = quantizeFloat(tex, dropBits);
        }

        writeAttributes[size_t(v) * attributeStride + 0] = tex.x;
        writeAttributes[size_t(v) * attributeStride + 1] = tex.y;
      }
    }

    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
    {
      const uint32_t dropBits = m_config.useCompressedData ? m_config.compressionTexDropBits : 0;

      float* writeAttributes = geometry.vertexAttributes.data() + geometry.attributeTex1offset;
      for(uint32_t v = 0; v < verticesCount; v++)
      {
        const float* src = input.texcoords1 + size_t(v) * input.texcoord1Stride;

        glm::vec2 tex = {src[0], src[1]};
        if(dropBits)
        {
          tex = quantizeFloat(tex, dropBits);
        }

        writeAttributes[size_t(v) * attributeStride + 0] = tex.x;
        writeAttributes[size_t(v) * attributeStride + 1] = tex.y;
      }
    }

    // indices (already a 0-based uint32 triangle list)
    memcpy(geometry.triangles.data(), input.indices.data(), sizeof(uint32_t) * input.indices.size());
  }

  processGeometry(processingInfo, geometryIndex, isCached);

  uint32_t percentage = processingInfo.logCompletedGeometry(triangleCount);
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(percentage);
  }
}

Scene::Result Scene::initFromCachedGeometries(std::span<const std::filesystem::path> cacheFilePaths,
                                              const SceneConfig&                     config,
                                              const SceneLoaderConfig&               loaderConfig)
{
  *this = {};

  m_config          = config;
  m_loaderConfig    = loaderConfig;
  m_loadedFromCache = true;
  m_cacheSuffix     = ".nvsngeo";

  // all geometry data stays memory-mapped in the per-geometry cache files
  m_loaderConfig.memoryMappedCache = true;

  // P2.5: the assembly itself is the shared append path (geometry range
  // [0, count) of an empty scene)
  Result result = appendCachedGeometries(cacheFilePaths);
  if(result != SCENE_RESULT_SUCCESS)
  {
    *this = {};
    return result;
  }

  LOGI("Scene::initFromCachedGeometries assembled %zu geometries\n", cacheFilePaths.size());
  LOGI("clusters:  %" PRIu64 "\n", m_totalClustersCount);
  LOGI("triangles: %" PRIu64 "\n", m_totalTrianglesCount);
  LOGI("hi clusters:  %" PRIu64 "\n", m_hiClustersCount);
  LOGI("hi triangles: %" PRIu64 "\n", m_hiTrianglesCount);

  return SCENE_RESULT_SUCCESS;
}

Scene::Result Scene::appendCachedGeometries(std::span<const std::filesystem::path> cacheFilePaths)
{
  if(cacheFilePaths.empty())
  {
    return SCENE_RESULT_ERROR;
  }

  const size_t firstGeometry = m_geometryViews.size();
  const size_t totalCount    = firstGeometry + cacheFilePaths.size();

  // Fallible phase: map and validate every new file into locals; the Scene's
  // members are untouched until all of them succeeded, so a bad file leaves
  // the (possibly rendering) Scene exactly as it was.
  std::vector<nvutils::FileReadMapping> newMappings(cacheFilePaths.size());
  std::vector<GeometryView>             newViews(cacheFilePaths.size());
  std::vector<Histograms>               newHistograms(cacheFilePaths.size());

  for(size_t i = 0; i < cacheFilePaths.size(); i++)
  {
    nvutils::FileReadMapping& mapping = newMappings[i];
    if(!mapping.open(cacheFilePaths[i]))
    {
      LOGE("Scene::appendCachedGeometries failed to map cache file:\n  %s\n",
           nvutils::utf8FromPath(cacheFilePaths[i]).c_str());
      return SCENE_RESULT_ERROR;
    }

    CacheFileView fileView;
    if(!fileView.init(mapping.size(), mapping.data()) || fileView.getGeometryCount() != 1)
    {
      LOGE("Scene::appendCachedGeometries invalid cache file:\n  %s\n",
           nvutils::utf8FromPath(cacheFilePaths[i]).c_str());
      return SCENE_RESULT_CACHE_INVALID;
    }

    if(firstGeometry == 0 && i == 0)
    {
      // adopt the cached SceneConfig, mirroring openCache(). The provider's
      // per-config-digest cache directory guarantees all files agree.
      fileView.getSceneConfig(m_config);
    }

    if(!fileView.getGeometryView(newViews[i], 0))
    {
      LOGE("Scene::appendCachedGeometries failed to load geometry view:\n  %s\n",
           nvutils::utf8FromPath(cacheFilePaths[i]).c_str());
      return SCENE_RESULT_CACHE_INVALID;
    }

    fileView.getHistograms(newHistograms[i]);
  }

  // Commit phase: nothing below fails. Mapped data pointers are stable across
  // the vector growth (FileReadMapping moves keep the OS mapping), so existing
  // GeometryViews and everything uploaded from them remain valid.
  for(size_t i = 0; i < cacheFilePaths.size(); i++)
  {
    m_cacheFileSize += newMappings[i].size();
    m_remixCacheMappings.push_back(std::move(newMappings[i]));
    m_geometryViews.push_back(newViews[i]);

    // merge cached histograms (statistics/GUI only)
    const Histograms& fileHistograms = newHistograms[i];

    auto mergeHistogram = [](auto& dst, const auto& src) {
      for(size_t e = 0; e < dst.size(); e++)
      {
        dst[e] += src[e];
      }
    };
    mergeHistogram(m_histograms.clusterTriangles, fileHistograms.clusterTriangles);
    mergeHistogram(m_histograms.clusterVertices, fileHistograms.clusterVertices);
    mergeHistogram(m_histograms.groupClusters, fileHistograms.groupClusters);
    mergeHistogram(m_histograms.nodeChildren, fileHistograms.nodeChildren);
    mergeHistogram(m_histograms.lodLevels, fileHistograms.lodLevels);
    m_histograms.clusterTrianglesMax = std::max(m_histograms.clusterTrianglesMax, fileHistograms.clusterTrianglesMax);
    m_histograms.clusterVerticesMax  = std::max(m_histograms.clusterVerticesMax, fileHistograms.clusterVerticesMax);
    m_histograms.groupClustersMax    = std::max(m_histograms.groupClustersMax, fileHistograms.groupClustersMax);
    m_histograms.nodeChildrenMax     = std::max(m_histograms.nodeChildrenMax, fileHistograms.nodeChildrenMax);
    m_histograms.lodLevelsMax        = std::max(m_histograms.lodLevelsMax, fileHistograms.lodLevelsMax);
  }

  // per-cluster state bits carry the actual two-sided / alpha flags; the ported
  // kernels were compiled with the conservative superset anyway
  m_hasTwoSided  = true;
  m_hasAlphaMask = true;

  // one Scene material and one identity-transform instance per geometry, exactly
  // like loadFromMeshInputs: keeps NVIDIA's bbox / statistics machinery working.
  // At render time the actual TLAS instances come from Remix's live RtInstances.
  m_geometryNames.resize(totalCount);
  m_materials.resize(totalCount);
  m_materialNames.resize(totalCount);
  m_instances.resize(totalCount);
  for(size_t i = firstGeometry; i < totalCount; i++)
  {
    Instance& instance  = m_instances[i];
    instance            = {};
    instance.matrix     = glm::mat4(1);
    instance.geometryID = uint32_t(i);
    instance.materialID = uint32_t(i);
  }

  // scene-level statistics/maxima: same tail as initFromMeshInputs, restricted
  // to the appended range (maxima only grow, sums accumulate)
  m_originalInstanceCount = m_instances.size();
  m_originalGeometryCount = m_geometryViews.size();
  m_activeGeometryCount   = m_originalGeometryCount;

  computeInstanceBBoxes();
  m_gridBbox = m_bbox;

  glm::vec3 modelExtent = m_bbox.hi - m_bbox.lo;
  m_isBig = modelExtent.y < 0.15f * std::max(modelExtent.x, modelExtent.z) && m_originalInstanceCount > 1024;

  for(size_t i = firstGeometry; i < totalCount; i++)
  {
    const GeometryView& geometry = m_geometryViews[i];

    m_hiPerGeometryTriangles = std::max(m_hiPerGeometryTriangles, geometry.hiTriangleCount);
    m_hiPerGeometryVertices  = std::max(m_hiPerGeometryVertices, geometry.hiVerticesCount);
    m_hiPerGeometryClusters  = std::max(m_hiPerGeometryClusters, geometry.hiClustersCount);
    m_hiPerGeometryGroups = std::max(m_hiPerGeometryGroups, geometry.lodLevels.empty() ? 0 : geometry.lodLevels[0].groupCount);

    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geometry.totalTriangleCount);
    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geometry.totalVerticesCount);
    m_maxPerGeometryClusters  = std::max(m_maxPerGeometryClusters, geometry.totalClustersCount);
    m_maxClusterVertices      = std::max(m_maxClusterVertices, geometry.clusterMaxVerticesCount);
    m_maxClusterTriangles     = std::max(m_maxClusterTriangles, geometry.clusterMaxTrianglesCount);
    m_maxLodLevelsCount       = std::max(m_maxLodLevelsCount, geometry.lodLevelsCount);

    m_hiTrianglesCount += geometry.hiTriangleCount;
    m_hiClustersCount += geometry.hiClustersCount;
    m_totalClustersCount += geometry.totalClustersCount;
    m_totalTrianglesCount += geometry.totalTriangleCount;
    m_totalVerticesCount += geometry.totalVerticesCount;

    if(geometry.localMaterialIDs.size() > 1)
    {
      m_geometryMultiMaterialCount += uint32_t(geometry.localMaterialIDs.size());
    }

    m_hasVertexNormals |= (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL) != 0;
    m_hasVertexTangents |= (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0;
    m_hasVertexTexCoord0 |= (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0;
    m_hasVertexTexCoord1 |= (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1) != 0;
  }
  for(size_t i = firstGeometry; i < m_instances.size(); i++)
  {
    const GeometryView& geometry = m_geometryViews[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geometry.hiTriangleCount;
    m_hiClustersCountInstanced += geometry.hiClustersCount;
  }

  {
    // estimate depth of lod tree based on highest detail group count
    uint32_t hiGroups   = m_hiPerGeometryGroups;
    uint32_t hiNodes    = (hiGroups + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
    uint32_t rootPasses = 0;

    m_maxNodeTreeDepth = 1;
    while(hiNodes)
    {
      hiNodes = (hiNodes + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
      m_maxNodeTreeDepth++;
      if(hiNodes == 1)
        break;
    }
    // lod tree to root
    m_maxNodeTreeDepth++;
  }

  if(firstGeometry != 0)
  {
    LOGI("Scene::appendCachedGeometries appended %zu geometries (%zu total)\n", cacheFilePaths.size(), totalCount);
  }

  return SCENE_RESULT_SUCCESS;
}

}  // namespace lodclusters
