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

#include "../dxvk_context.h"
#include "rtx_types.h"

namespace dxvk {

  class RtxContext;
  class RtxMegaGeometry;
  class SceneManager;
  class DxvkDevice;
  struct DrawParameters;
  struct DrawCallState;

  /**
   * \brief Mega geometry statistics for UI
   */
  struct MegaGeometryStatistics {
    uint32_t totalClusters;
    uint32_t visibleClusters;
    uint32_t culledClusters;
    uint32_t totalVertices;
    uint32_t totalTriangles;
    float memoryUsedMB;
  };

  /**
   * \brief Initialize RTX Mega Geometry system
   *
   * Called once during device initialization.
   */
  void initializeMegaGeometry(DxvkDevice* device);

  /**
   * \brief Shutdown RTX Mega Geometry system
   *
   * Called during device cleanup.
   */
  void shutdownMegaGeometry();

  /**
   * \brief Get the global mega geometry instance
   */
  RtxMegaGeometry* getMegaGeometry();

  /**
   * \brief Process geometry through RTXMG cluster tessellation
   *
   * Intercepts geometry submission and converts to cluster-based tessellation.
   * Always-on by design - no fallback.
   *
   * \param [in] ctx The RTX context
   * \param [in] params Draw parameters
   * \param [in/out] drawCallState Draw call state
   * \param [in] sceneManager Scene manager for instance submission
   * \returns true if geometry was processed by RTXMG, false to use normal path
   */
  bool processGeometryWithMegaGeometry(
    RtxContext* ctx,
    const DrawParameters& params,
    DrawCallState& drawCallState,
    RasterGeometry& geometryData,
    SceneManager& sceneManager);

  /**
   * \brief Update RTXMG per-frame
   *
   * Called during injectRTX to update HiZ and build cluster acceleration structures.
   *
   * \param [in] ctx The RTX context
   * \param [in] sceneManager Scene manager
   * \param [in] depthBuffer Current depth buffer for HiZ
   */
  void updateMegaGeometryPerFrame(
    RtxContext* ctx,
    SceneManager& sceneManager,
    const Rc<DxvkImageView>& depthBuffer);

  /**
   * \brief Inject cluster BLASes into scene's instance system
   *
   * Called after cluster BLASes are built to set dynamicBlas on all instances that have cluster geometry.
   * This enables TLAS GPU patching to find and use the cluster BLASes.
   *
   * \param [in] ctx The RTX context
   * \param [in] sceneManager Scene manager with RtInstances
   * \param [in] megaGeometry The RTX Mega Geometry system with cluster builder
   */
  void injectClusterBlasesIntoScene(
    RtxContext* ctx,
    SceneManager& sceneManager,
    RtxMegaGeometry* megaGeometry);

  /**
   * \brief Render RTXMG debug visualization
   *
   * \param [in] ctx The RTX context
   * \param [in] outputImage Target image for debug output
   * \param [in] debugViewIndex The RTX debug view index (900-907 for mega geometry views)
   */
  void renderMegaGeometryDebugView(
    RtxContext* ctx,
    const Rc<DxvkImageView>& outputImage,
    uint32_t debugViewIndex);

  /**
   * \brief Check if geometry should use RTXMG
   *
   * In always-on mode, this always returns true.
   *
   * \param [in] drawCallState The draw call state
   * \returns true if RTXMG should be used (always in always-on mode)
   */
  bool shouldUseMegaGeometry(const DrawCallState& drawCallState);

  /**
   * \brief Get RTXMG statistics for UI display
   *
   * \param [out] outStats Statistics structure to fill
   */
  void getMegaGeometryStatistics(MegaGeometryStatistics& outStats);

} // namespace dxvk
