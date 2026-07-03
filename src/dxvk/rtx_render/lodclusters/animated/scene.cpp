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

// NV-DXVK: This file originates from nvpro-samples/vk_animated_clusters
// (src/scene.cpp), trimmed for the RTX Remix Path B integration: the glTF
// loader entry (Scene::init/loadGLTF), viewer instance bbox computation and
// the per-geometry GPU buffer upload (initGpuBuffers) are removed - Remix
// feeds single-geometry CPU snapshots and the ClusterTemplateRenderer owns
// all GPU data. ProcessingInfo and the clusterization functions
// (buildGeometryClusters .. rebuildGeometryTriangles) are byte-identical to
// the sample. processSingleGeometry is the Remix-authored single-geometry
// driver (config + histogram setup mirroring Scene::init's tail).

#include <cassert>

#include <atomic>
#include <cfloat>
#include <cmath>

#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>

#include "scene.hpp"

namespace animatedclusters {

void Scene::ProcessingInfo::init(float processingThreadsPct)
{
  numPoolThreadsOriginal = nvutils::get_thread_pool().get_thread_count();

  numPoolThreads = numPoolThreadsOriginal;
  if(processingThreadsPct > 0.0f && processingThreadsPct < 1.0f)
  {
    numPoolThreads = std::min(numPoolThreadsOriginal,
                              std::max(1u, uint32_t(ceilf(float(numPoolThreadsOriginal) * processingThreadsPct))));

    if(numPoolThreads != numPoolThreadsOriginal)
      nvutils::get_thread_pool().reset(numPoolThreads);
  }
}

void Scene::ProcessingInfo::setupParallelism(size_t geometryCount_)
{
  geometryCount = geometryCount_;

  bool preferInnerParallelism = geometryCount < numPoolThreads;

  numOuterThreads = preferInnerParallelism ? 1 : numPoolThreads;
  numInnerThreads = preferInnerParallelism ? numPoolThreads : 1;
}

void Scene::ProcessingInfo::logBegin()
{
  LOGI("... geometry load & processing: geometries %llu, threads outer %d inner %d\n", geometryCount, numOuterThreads, numInnerThreads);

  startTime = clock.getMicroseconds();
}

void Scene::ProcessingInfo::logCompletedGeometry()
{
  std::lock_guard lock(progressMutex);

  progressGeometriesCompleted++;

  // statistics
  const uint32_t precentageGranularity = 5;
  uint32_t       percentage            = uint32_t(size_t(progressGeometriesCompleted * 100) / geometryCount);
  percentage                           = (percentage / precentageGranularity) * precentageGranularity;

  if(percentage > progressLastPercentage)
  {
    progressLastPercentage = percentage;
    LOGI("... geometry load & processing: %3d%%\n", percentage);
  }
}

void Scene::ProcessingInfo::logEnd()
{
  double endTime = clock.getMicroseconds();

  LOGI("... geometry load & processing: %f milliseconds\n", (endTime - startTime) / 1000.0f);
}

void Scene::ProcessingInfo::deinit()
{
  if(numPoolThreads != numPoolThreadsOriginal)
    nvutils::get_thread_pool().reset(numPoolThreadsOriginal);
}

// NV-DXVK: Remix-authored single-geometry driver replacing Scene::init's
// glTF orchestration - config/histogram setup and the maxima/histogram tail
// mirror the sample's init; processing itself is the sample's processGeometry
// with inner parallelism (geometryCount == 1).
bool Scene::processSingleGeometry(Geometry& geometry, const SceneConfig& config)
{
  m_config = config;

  m_clusterTriangleHistogram.assign(m_config.clusterTriangles + 1, 0);
  m_clusterVertexHistogram.assign(m_config.clusterVertices + 1, 0);
  m_maxClusterTriangles = 0;
  m_maxClusterVertices  = 0;

  ProcessingInfo processingInfo;
  processingInfo.init(config.processingThreadsPct);
  processingInfo.setupParallelism(1);

  // compute the reference-pose bbox (used for the template bbox bloat)
  geometry.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
  for(const glm::vec3& position : geometry.positions)
  {
    geometry.bbox.lo = glm::min(geometry.bbox.lo, position);
    geometry.bbox.hi = glm::max(geometry.bbox.hi, position);
  }

  processGeometry(processingInfo, geometry);

  processingInfo.deinit();

  if(!geometry.numClusters)
  {
    return false;
  }

  m_clusterTriangleHistogramMax = 0u;
  m_clusterVertexHistogramMax   = 0u;
  for(size_t i = 0; i < m_clusterTriangleHistogram.size(); i++)
  {
    m_clusterTriangleHistogramMax = std::max(m_clusterTriangleHistogramMax, m_clusterTriangleHistogram[i]);
    if(m_clusterTriangleHistogram[i])
      m_maxClusterTriangles = uint32_t(i);
  }
  for(size_t i = 0; i < m_clusterVertexHistogram.size(); i++)
  {
    m_clusterVertexHistogramMax = std::max(m_clusterVertexHistogramMax, m_clusterVertexHistogram[i]);
    if(m_clusterVertexHistogram[i])
      m_maxClusterVertices = uint32_t(i);
  }

  m_maxPerGeometryTriangles       = std::max(m_maxPerGeometryTriangles, geometry.numTriangles);
  m_maxPerGeometryVertices        = std::max(m_maxPerGeometryVertices, geometry.numVertices);
  m_maxPerGeometryClusters        = std::max(m_maxPerGeometryClusters, geometry.numClusters);
  m_maxPerGeometryClusterVertices = std::max(m_maxPerGeometryClusterVertices, geometry.numClusterVertices);
  m_numTriangles += geometry.numTriangles;
  m_numClusters += geometry.numClusters;

  return true;
}

void Scene::processGeometry(ProcessingInfo& processingInfo, Geometry& geometry)
{
  if(!geometry.numTriangles)
    return;

  buildGeometryClusters(processingInfo, geometry);

  if(!geometry.numClusters)
    return;

  optimizeGeometryClusters(processingInfo, geometry);

  buildGeometryClusterBboxes(processingInfo, geometry);

  if(m_config.clusterDedicatedVertices)
  {
    // give each cluster its own set of vertices, so require only
    // the local cluster 8-bit triangle indices
    buildGeometryClusterVertices(processingInfo, geometry);

    // no longer need vertex indirection
    geometry.clusterLocalVertices = std::vector<uint32_t>();
  }

  rebuildGeometryTriangles(processingInfo, geometry);
}

void Scene::buildGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry)
{
  uint32_t numInnerThreads = processingInfo.numInnerThreads;


  // we allow smaller clusters to be generated when that significantly improves their bounds
  size_t minTriangles = (m_config.clusterTriangles / 4) & ~3;

  std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(geometry.numTriangles * 3, m_config.clusterVertices, minTriangles));
  geometry.clusterLocalTriangles.resize(meshlets.size() * m_config.clusterTriangles * 3);
  geometry.clusterLocalVertices.resize(meshlets.size() * m_config.clusterVertices);

  size_t numClusters;

  if(m_config.clusterSpatial)
  {
    numClusters = meshopt_buildMeshletsSpatial(meshlets.data(), geometry.clusterLocalVertices.data(),
                                               geometry.clusterLocalTriangles.data(), (uint32_t*)geometry.triangles.data(),
                                               geometry.triangles.size() * 3, (float*)geometry.positions.data(),
                                               geometry.numVertices, sizeof(glm::vec3), m_config.clusterVertices,
                                               minTriangles, m_config.clusterTriangles, m_config.clusterMeshoptSpatialFill);
  }
  else
  {
    numClusters = meshopt_buildMeshletsFlex(meshlets.data(), geometry.clusterLocalVertices.data(),
                                            geometry.clusterLocalTriangles.data(), (uint32_t*)geometry.triangles.data(),
                                            geometry.triangles.size() * 3, (float*)geometry.positions.data(), geometry.numVertices,
                                            sizeof(glm::vec3), m_config.clusterVertices, minTriangles, m_config.clusterTriangles,
                                            m_config.clusterMeshoptFlexCone, m_config.clusterMeshoptFlexSplit);
  }

  geometry.numClusters = uint32_t(numClusters);

  if(geometry.numClusters)
  {
    geometry.clusters.resize(geometry.numClusters);
    geometry.clusters.shrink_to_fit();

    for(size_t c = 0; c < numClusters; c++)
    {
      meshopt_Meshlet&   meshlet = meshlets[c];
      shaderio::Cluster& cluster = geometry.clusters[c];

      cluster.numTriangles       = meshlet.triangle_count;
      cluster.numVertices        = meshlet.vertex_count;
      cluster.firstLocalTriangle = meshlet.triangle_offset;
      cluster.firstLocalVertex   = meshlet.vertex_offset;

      // update stats
      reinterpret_cast<std::atomic_uint32_t*>(m_clusterTriangleHistogram.data())[cluster.numTriangles]++;
      reinterpret_cast<std::atomic_uint32_t*>(m_clusterVertexHistogram.data())[cluster.numVertices]++;
    }
  }

  if(geometry.numClusters)
  {
    shaderio::Cluster& cluster = geometry.clusters[geometry.numClusters - 1];
    geometry.clusterLocalTriangles.resize(cluster.firstLocalTriangle + cluster.numTriangles * 3);
    geometry.clusterLocalVertices.resize(cluster.firstLocalVertex + cluster.numVertices);
    geometry.clusterLocalTriangles.shrink_to_fit();
    geometry.clusterLocalVertices.shrink_to_fit();

    geometry.numClusterVertices = uint32_t(geometry.clusterLocalVertices.size());
  }
}


void Scene::optimizeGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry)
{
  uint32_t numInnerThreads = processingInfo.numInnerThreads;

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          meshopt_optimizeMeshlet(&geometry.clusterLocalVertices[cluster.firstLocalVertex],
                                  &geometry.clusterLocalTriangles[cluster.firstLocalTriangle], cluster.numTriangles,
                                  cluster.numVertices);
        }
      },
      numInnerThreads);
}

void Scene::buildGeometryClusterBboxes(ProcessingInfo& processingInfo, Geometry& geometry)
{
  geometry.clusterBboxes.resize(geometry.numClusters);

  const glm::vec3* positions             = geometry.positions.data();
  const uint32_t*  clusterLocalVertices  = geometry.clusterLocalVertices.data();
  const uint8_t*   clusterLocalTriangles = geometry.clusterLocalTriangles.data();

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
          for(uint32_t v = 0; v < cluster.numVertices; v++)
          {
            uint32_t  vertexIndex = clusterLocalVertices[cluster.firstLocalVertex + v];
            glm::vec3 pos         = positions[vertexIndex];

            bbox.lo = glm::min(bbox.lo, pos);
            bbox.hi = glm::max(bbox.hi, pos);
          }

          geometry.clusterBboxes[idx] = bbox;
        }
      },
      processingInfo.numInnerThreads);
}

void Scene::buildGeometryClusterVertices(ProcessingInfo& processingInfo, Geometry& geometry)
{
  // build per-cluster vertices

  std::vector<glm::vec3> oldPositionsData = std::move(geometry.positions);

  geometry.positions.resize(geometry.numClusterVertices);
  geometry.numVertices = uint32_t(geometry.positions.size());

  const glm::vec3* oldPositions         = oldPositionsData.data();
  glm::vec3*       newPositions         = geometry.positions.data();
  uint32_t*        clusterLocalVertices = geometry.clusterLocalVertices.data();

  for(uint32_t c = 0; c < geometry.numClusters; c++)
  {
    shaderio::Cluster& cluster = geometry.clusters[c];

    for(uint32_t v = 0; v < cluster.numVertices; v++)
    {
      uint32_t oldIdx                                    = clusterLocalVertices[v + cluster.firstLocalVertex];
      clusterLocalVertices[v + cluster.firstLocalVertex] = v + cluster.firstLocalVertex;
      newPositions[v + cluster.firstLocalVertex]         = oldPositions[oldIdx];
    }
  }
}

void Scene::rebuildGeometryTriangles(ProcessingInfo& processingInfo, Geometry& geometry)
{
  // rebuild triangle buffer accounting for cluster order
  // in the rare event that cluster building filtered out original triangles

  uint32_t triOffset = 0;
  for(size_t c = 0; c < geometry.numClusters; c++)
  {
    shaderio::Cluster& cluster = geometry.clusters[c];

    cluster.firstTriangle = triOffset;
    triOffset += cluster.numTriangles;
  }

  geometry.triangles.resize(triOffset);
  geometry.numTriangles = triOffset;

  glm::uvec3*     triangles             = geometry.triangles.data();
  const uint32_t* clusterLocalVertices  = geometry.clusterLocalVertices.data();
  const uint8_t*  clusterLocalTriangles = geometry.clusterLocalTriangles.data();

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          for(uint32_t t = 0; t < cluster.numTriangles; t++)
          {
            glm::uvec3 localVertices = {clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 0],
                                        clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 1],
                                        clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 2]};

            assert(localVertices.x < cluster.numVertices);
            assert(localVertices.y < cluster.numVertices);
            assert(localVertices.z < cluster.numVertices);

            glm::uvec3 globalVertices = {localVertices.x + cluster.firstLocalVertex, localVertices.y + cluster.firstLocalVertex,
                                         localVertices.z + cluster.firstLocalVertex};

            if(!m_config.clusterDedicatedVertices)
            {
              // need one more indirection
              globalVertices = {clusterLocalVertices[globalVertices.x], clusterLocalVertices[globalVertices.y],
                                clusterLocalVertices[globalVertices.z]};
            }

            triangles[cluster.firstTriangle + t] = globalVertices;
          }
        }
      },
      processingInfo.numInnerThreads);
}

}  // namespace animatedclusters
