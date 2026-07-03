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

// NV-DXVK: This file originates from nvpro-samples/vk_animated_clusters
// (src/scene.hpp) and was trimmed for the RTX Remix Path B integration:
// only the clusterization structs and processing path are kept, exactly as
// planned (INTEGRATION_PLAN section 4): glTF loading, viewer scene management
// (instances/cameras/scene bbox) and the per-geometry GPU buffer creation are
// dropped - Remix feeds single geometries from CPU snapshots and the
// ClusterTemplateRenderer owns all GPU data. The processing functions
// themselves (buildGeometryClusters .. rebuildGeometryTriangles) are
// unchanged from the sample; ProcessingInfo and processGeometry moved from
// private to public so the Remix boundary can drive single-geometry
// processing directly.

#include <mutex>
#include <vector>

#include <glm/glm.hpp>
#include <nvutils/timers.hpp>

#include "shaderio_animated_host.h"

namespace animatedclusters {
struct SceneConfig
{
  uint32_t clusterVertices           = 64;
  uint32_t clusterTriangles          = 64;
  float    clusterMeshoptFlexSplit   = 2.0f;
  float    clusterMeshoptFlexCone    = 0.0f;
  float    clusterMeshoptSpatialFill = 0.5f;

  bool clusterDedicatedVertices = false;
  bool clusterSpatial           = true;

  // Influence the number of geometries that can be processed in parallel.
  // Percentage of threads of maximum hardware concurrency
  float processingThreadsPct = 0.5;
};

class Scene
{
public:
  struct Geometry
  {
    uint32_t numTriangles{};
    uint32_t numVertices{};
    uint32_t numClusters{};
    uint32_t numClusterVertices{};

    shaderio::BBox bbox;

    std::vector<glm::vec3>  positions;
    std::vector<glm::uvec3> triangles;
    std::vector<uint8_t>    clusterLocalTriangles;
    std::vector<uint32_t>   clusterLocalVertices;

    std::vector<shaderio::Cluster> clusters;
    std::vector<shaderio::BBox>    clusterBboxes;
  };

  SceneConfig m_config;

  std::vector<Geometry> m_geometries;

  size_t   m_sceneClusterMemBytes          = 0;
  size_t   m_sceneTriangleMemBytes         = 0;
  uint32_t m_maxClusterTriangles           = 0;
  uint32_t m_maxClusterVertices            = 0;
  uint32_t m_maxPerGeometryClusters        = 0;
  uint32_t m_maxPerGeometryTriangles       = 0;
  uint32_t m_maxPerGeometryVertices        = 0;
  uint32_t m_maxPerGeometryClusterVertices = 0;
  uint32_t m_numClusters                   = 0;
  uint32_t m_numTriangles                  = 0;

  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  uint32_t              m_clusterTriangleHistogramMax;
  uint32_t              m_clusterVertexHistogramMax;

  struct ProcessingInfo
  {
    // how we perform multi-threading:
    // - either over geometries (outer loop)
    // - or within a geometry (inner loops)

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;

    size_t geometryCount = 0;

    std::mutex processOnlySaveMutex;

    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    void setupParallelism(size_t geometryCount_);
    void deinit();

    void logBegin();
    void logCompletedGeometry();
    void logEnd();
  };

  // NV-DXVK: single-geometry entry for the Remix provider worker. Sets up the
  // config/histograms and runs the sample's processGeometry with inner
  // parallelism (one geometry at a time by design).
  bool processSingleGeometry(Geometry& geometry, const SceneConfig& config);

  void processGeometry(ProcessingInfo& processingInfo, Geometry& geometry);

private:
  void buildGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusterBboxes(ProcessingInfo& processingInfo, Geometry& geometry);
  void optimizeGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusterVertices(ProcessingInfo& processingInfo, Geometry& geometry);
  void rebuildGeometryTriangles(ProcessingInfo& processingInfo, Geometry& geometry);
};
}  // namespace animatedclusters
