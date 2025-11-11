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

#include "rtx_mega_geometry_autotune.h"
#include "rtx_mg_cluster.h"
#include "dxvk_device.h"
#include "../util/log/log.h"
#include "../util/util_math.h"

namespace dxvk {

  RtxMegaGeometryAutoTune::RtxMegaGeometryAutoTune(DxvkDevice* device)
    : m_device(device) {

    // SDK MATCH: Start small, grow dynamically, cap at 2M like NVIDIA sample
    // NVIDIA users configure this via GUI/args based on scene needs, not pre-allocated
    m_recommendedMaxClusters = 64 * 1024;  // Start with 64K clusters (conservative)
    m_recommendedTessellationDensity = 1.0f;
    m_recommendedClusterDataBufferSize = m_recommendedMaxClusters * kMaxClusterVertices * sizeof(ClusterVertex);
    m_recommendedClusterInfoBufferSize = m_recommendedMaxClusters * sizeof(ClusterShadingData);
    m_recommendedBLASBudget = 256 * 1024 * 1024;  // 256MB default

    Logger::info("[RTX Mega Geometry AutoTune] Initialized with 64K cluster budget, will grow dynamically up to 2M max (SDK MATCH)");
  }

  void RtxMegaGeometryAutoTune::updatePerFrame(
    uint32_t submittedVertices,
    uint32_t submittedTriangles,
    uint32_t submittedMeshes,
    uint32_t availableGpuMemoryMB) {

    // SDK MATCH: Check EVERY frame like NVIDIA sample (line 1265 calls UpdateMemoryAllocations every frame)
    // The check is cheap, reallocation only happens when needed
    computeRecommendations(submittedVertices, submittedTriangles, submittedMeshes, availableGpuMemoryMB);
  }

  void RtxMegaGeometryAutoTune::computeRecommendations(
    uint32_t submittedVertices,
    uint32_t submittedTriangles,
    uint32_t submittedMeshes,
    uint32_t availableGpuMemoryMB) {

    // SDK MATCH: Grow dynamically but cap at 2M like NVIDIA sample (they configure via GUI/args)
    // Estimate clusters needed based on triangle count (assume ~100 triangles per cluster)
    const uint32_t estimatedClustersNeeded = std::max(1u, submittedTriangles / 100) + submittedMeshes;

    // Add 50% headroom for growth
    const uint32_t clustersWithHeadroom = static_cast<uint32_t>(estimatedClustersNeeded * 1.5f);

    // Clamp to 2M maximum (SDK MATCH: kMaxApiClusterCount cap)
    const uint32_t kMaxClusters = 2 * 1024 * 1024;  // 2M max
    const uint32_t kMinClusters = 64 * 1024;        // 64K min
    uint32_t newMaxClusters = std::clamp(clustersWithHeadroom, kMinClusters, kMaxClusters);

    // Check for growth or shrink with hysteresis (50% threshold)
    // SDK MATCH: NVIDIA checks every frame, reallocates when needed
    bool needsGrowth = newMaxClusters > m_recommendedMaxClusters;
    bool needsShrink = newMaxClusters < m_recommendedMaxClusters / 2;  // Only shrink if <50% usage

    if (!needsGrowth && !needsShrink) {
      return;  // No change needed
    }

    if (needsGrowth) {
      Logger::info(str::format(
        "[RTX Mega Geometry AutoTune] Growing cluster budget: ",
        m_recommendedMaxClusters / 1024, "K -> ", newMaxClusters / 1024, "K clusters"));
    } else if (needsShrink) {
      Logger::info(str::format(
        "[RTX Mega Geometry AutoTune] Shrinking cluster budget: ",
        m_recommendedMaxClusters / 1024, "K -> ", newMaxClusters / 1024, "K clusters (fence-tracked release)"));
    }

    // Update recommendations
    m_recommendedMaxClusters = newMaxClusters;
    m_recommendedClusterDataBufferSize =
      static_cast<uint64_t>(newMaxClusters) * kMaxClusterVertices * sizeof(ClusterVertex);
    m_recommendedClusterInfoBufferSize =
      static_cast<uint64_t>(newMaxClusters) * sizeof(ClusterShadingData);

    m_needsBufferResize = true;
  }

  RtxMegaGeometryAutoTune::BLASEntry* RtxMegaGeometryAutoTune::tryReuseBLAS(
    uint64_t geometryHash,
    uint32_t primitiveCount,
    uint32_t currentFrame) {

    // Search for matching BLAS in pool
    for (auto& entry : m_blasPool) {
      if (!entry.inUse &&
          entry.geometryHash == geometryHash &&
          entry.primitiveCount == primitiveCount) {

        // Found reusable BLAS
        entry.inUse = true;
        entry.lastUsedFrame = currentFrame;
        m_memoryStats.blasPoolUsed++;
        return &entry;
      }
    }

    // Try to find an oversized BLAS that can fit this geometry
    for (auto& entry : m_blasPool) {
      if (!entry.inUse &&
          entry.primitiveCount >= primitiveCount &&
          entry.primitiveCount < primitiveCount * 1.5f) {  // Within 50% overhead

        // Reuse oversized BLAS
        entry.inUse = true;
        entry.lastUsedFrame = currentFrame;
        entry.geometryHash = geometryHash;
        entry.primitiveCount = primitiveCount;
        m_memoryStats.blasPoolUsed++;
        return &entry;
      }
    }

    return nullptr;  // No suitable BLAS found
  }

  void RtxMegaGeometryAutoTune::addBLASToPool(
    VkAccelerationStructureKHR blas,
    Rc<DxvkBuffer> buffer,
    uint64_t sizeBytes,
    uint32_t primitiveCount,
    uint64_t geometryHash,
    uint32_t currentFrame) {

    // Check if pool is full
    if (m_blasPool.size() >= m_maxBLASPoolSize) {
      // Find oldest unused entry to replace
      uint32_t oldestFrame = currentFrame;
      size_t oldestIndex = 0;
      bool foundUnused = false;

      for (size_t i = 0; i < m_blasPool.size(); ++i) {
        if (!m_blasPool[i].inUse && m_blasPool[i].lastUsedFrame < oldestFrame) {
          oldestFrame = m_blasPool[i].lastUsedFrame;
          oldestIndex = i;
          foundUnused = true;
        }
      }

      if (foundUnused) {
        // Destroy old BLAS
        if (m_blasPool[oldestIndex].blas != VK_NULL_HANDLE) {
          m_device->vkd()->vkDestroyAccelerationStructureKHR(
            m_device->vkd()->device(),
            m_blasPool[oldestIndex].blas,
            nullptr);
        }

        // Replace entry
        m_blasPool[oldestIndex] = BLASEntry{
          blas, buffer, sizeBytes, primitiveCount, geometryHash, currentFrame, false
        };
        return;
      }

      // Pool full with all entries in use - skip caching
      ONCE(Logger::warn("[RTX Mega Geometry AutoTune] BLAS pool full, skipping cache"));
      return;
    }

    // Add new entry to pool
    m_blasPool.push_back(BLASEntry{
      blas, buffer, sizeBytes, primitiveCount, geometryHash, currentFrame, false
    });
  }

  void RtxMegaGeometryAutoTune::cleanupOldBLAS(uint32_t currentFrame, uint32_t maxAge) {
    size_t cleanedCount = 0;
    uint64_t freedBytes = 0;

    for (auto it = m_blasPool.begin(); it != m_blasPool.end(); ) {
      const uint32_t age = currentFrame - it->lastUsedFrame;

      if (!it->inUse && age > maxAge) {
        // Destroy old BLAS
        if (it->blas != VK_NULL_HANDLE) {
          m_device->vkd()->vkDestroyAccelerationStructureKHR(
            m_device->vkd()->device(),
            it->blas,
            nullptr);
        }

        freedBytes += it->sizeBytes;
        it = m_blasPool.erase(it);
        cleanedCount++;
      } else {
        ++it;
      }
    }

    if (cleanedCount > 0) {
      Logger::debug(str::format(
        "[RTX Mega Geometry AutoTune] Cleaned up ", cleanedCount,
        " old BLAS entries, freed ", freedBytes / (1024 * 1024), "MB"));
    }

    // Reset usage counters for next frame
    m_memoryStats.blasPoolUsed = 0;
    for (auto& entry : m_blasPool) {
      entry.inUse = false;
    }
  }

} // namespace dxvk
