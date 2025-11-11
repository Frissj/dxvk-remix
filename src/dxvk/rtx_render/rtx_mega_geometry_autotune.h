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
#pragma once

#include "rtx_resources.h"
#include "rtx_option.h"
#include "dxvk_buffer.h"
#include "../vulkan/vulkan_loader.h"

namespace dxvk {

  class DxvkDevice;

  /**
   * \brief RTX Mega Geometry Auto-Tuning System
   *
   * Automatically manages:
   * - Dynamic memory allocation based on scene complexity
   * - BLAS pooling and reuse to avoid re-building
   * - Statistics-driven parameter adaptation
   * - GPU memory budgeting
   *
   * All parameters are automatically tuned - no user configuration needed.
   */
  class RtxMegaGeometryAutoTune {

  public:
    RtxMegaGeometryAutoTune(DxvkDevice* device);
    ~RtxMegaGeometryAutoTune() = default;

    /**
     * \brief Update auto-tuning based on frame statistics
     *
     * Call once per frame after geometry submission.
     *
     * \param [in] submittedVertices Number of vertices submitted this frame
     * \param [in] submittedTriangles Number of triangles submitted this frame
     * \param [in] submittedMeshes Number of meshes submitted this frame
     * \param [in] availableGpuMemoryMB Available GPU VRAM in megabytes
     */
    void updatePerFrame(
      uint32_t submittedVertices,
      uint32_t submittedTriangles,
      uint32_t submittedMeshes,
      uint32_t availableGpuMemoryMB);

    /**
     * \brief Get recommended max cluster count
     *
     * Automatically scales based on scene complexity and available memory.
     */
    uint32_t getRecommendedMaxClusters() const {
      return m_recommendedMaxClusters;
    }

    /**
     * \brief Get recommended tessellation density multiplier
     *
     * Automatically adjusted based on performance metrics.
     */
    float getRecommendedTessellationDensity() const {
      return m_recommendedTessellationDensity;
    }

    /**
     * \brief Get recommended buffer size for cluster data
     *
     * Automatically sized based on actual usage patterns.
     */
    uint64_t getRecommendedClusterDataBufferSize() const {
      return m_recommendedClusterDataBufferSize;
    }

    /**
     * \brief Get recommended buffer size for cluster info
     */
    uint64_t getRecommendedClusterInfoBufferSize() const {
      return m_recommendedClusterInfoBufferSize;
    }

    /**
     * \brief Get recommended BLAS budget in bytes
     *
     * Dynamically allocated from available GPU memory.
     */
    uint64_t getRecommendedBLASBudget() const {
      return m_recommendedBLASBudget;
    }

    /**
     * \brief Check if buffers need to be resized
     *
     * Returns true if current recommendations exceed last frame's allocations.
     */
    bool needsBufferResize() const {
      return m_needsBufferResize;
    }

    /**
     * \brief Notify that buffers have been resized
     *
     * Call after resizing buffers to reset the flag.
     */
    void acknowledgeBufferResize() {
      m_needsBufferResize = false;
      m_lastAllocatedMaxClusters = m_recommendedMaxClusters;
    }

    /**
     * \brief Get memory usage statistics
     */
    struct MemoryStats {
      uint64_t clusterDataMB = 0;
      uint64_t clusterInfoMB = 0;
      uint64_t tilingBufferMB = 0;
      uint64_t blasPoolMB = 0;
      uint64_t totalMB = 0;
      uint32_t blasPoolSize = 0;
      uint32_t blasPoolUsed = 0;
      float utilizationPercent = 0.0f;
    };

    void getMemoryStats(MemoryStats& outStats) const {
      outStats = m_memoryStats;
    }

    /**
     * \brief BLAS Pool Management
     */
    struct BLASEntry {
      VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
      Rc<DxvkBuffer> buffer;
      uint64_t sizeBytes = 0;
      uint32_t primitiveCount = 0;
      uint64_t geometryHash = 0;
      uint32_t lastUsedFrame = 0;
      bool inUse = false;
    };

    /**
     * \brief Try to reuse existing BLAS from pool
     *
     * \param [in] geometryHash Hash of geometry data
     * \param [in] primitiveCount Number of primitives
     * \param [in] currentFrame Current frame index
     * \return Pointer to reusable BLAS entry, or nullptr if none available
     */
    BLASEntry* tryReuseBLAS(uint64_t geometryHash, uint32_t primitiveCount, uint32_t currentFrame);

    /**
     * \brief Add BLAS to pool for reuse
     *
     * \param [in] blas Vulkan acceleration structure
     * \param [in] buffer Buffer backing the BLAS
     * \param [in] sizeBytes Size in bytes
     * \param [in] primitiveCount Number of primitives
     * \param [in] geometryHash Hash of geometry
     * \param [in] currentFrame Current frame index
     */
    void addBLASToPool(
      VkAccelerationStructureKHR blas,
      Rc<DxvkBuffer> buffer,
      uint64_t sizeBytes,
      uint32_t primitiveCount,
      uint64_t geometryHash,
      uint32_t currentFrame);

    /**
     * \brief Release unused BLAS entries older than N frames
     *
     * Automatically frees memory from stale entries.
     *
     * \param [in] currentFrame Current frame index
     * \param [in] maxAge Maximum age in frames before cleanup
     */
    void cleanupOldBLAS(uint32_t currentFrame, uint32_t maxAge = 120);

  private:
    DxvkDevice* m_device = nullptr;

    // Adaptive parameters (automatically tuned)
    uint32_t m_recommendedMaxClusters = 64 * 1024;  // Start 64K, grow to 2M max (SDK MATCH)
    float m_recommendedTessellationDensity = 1.0f;
    uint64_t m_recommendedClusterDataBufferSize = 0;
    uint64_t m_recommendedClusterInfoBufferSize = 0;
    uint64_t m_recommendedBLASBudget = 0;

    // Tracking for auto-scaling
    uint32_t m_lastAllocatedMaxClusters = 0;
    uint32_t m_peakVerticesPerFrame = 0;
    uint32_t m_peakTrianglesPerFrame = 0;
    uint32_t m_peakMeshesPerFrame = 0;
    uint32_t m_averageVerticesPerFrame = 0;
    uint32_t m_averageTrianglesPerFrame = 0;
    uint32_t m_framesSinceLastResize = 0;
    bool m_needsBufferResize = false;

    // BLAS pool for reuse
    std::vector<BLASEntry> m_blasPool;
    uint32_t m_maxBLASPoolSize = 256;  // Auto-adjust based on memory

    // Memory statistics
    MemoryStats m_memoryStats;

    // Compute optimal parameters based on observed usage
    void computeRecommendations(
      uint32_t submittedVertices,
      uint32_t submittedTriangles,
      uint32_t submittedMeshes,
      uint32_t availableGpuMemoryMB);
  };

} // namespace dxvk
