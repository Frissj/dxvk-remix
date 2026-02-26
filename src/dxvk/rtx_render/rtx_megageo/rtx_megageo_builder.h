/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "../rtx_resources.h"
#include "nvrhi_adapter/nvrhi_dxvk_device.h"
#include "cluster_builder/cluster_accel_builder.h"
#include "hiz/hiz_buffer.h"
#include "hiz/zbuffer.h"
#include "scene/camera.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>

// Forward declarations for RTX MG classes
struct ClusterLODData;
struct MeshletTemplateSet;

namespace dxvk {

  // Forward declarations
  class RtxContext;
  struct RaytraceGeometry;
  struct BlasEntry;

  /**
   * \brief Triangle mesh descriptor for meshoptimizer cluster path
   *
   * Describes a triangle mesh that will be clusterized via meshoptimizer
   * and rendered using per-meshlet CLAS templates. No subdivision involved.
   */
  struct TriangleMeshDesc {
    const uint32_t* indices = nullptr;
    size_t indexCount = 0;
    const float* vertexPositions = nullptr;
    size_t vertexCount = 0;
    size_t vertexPositionsStride = 12; // bytes between consecutive positions (default: packed float3)
    const float* texcoords = nullptr;       // optional, 2 floats per vertex
    const float* normals = nullptr;         // optional, 3 floats per vertex
    const char* debugName = nullptr;
  };

  /**
   * \brief RTX Mega Geometry Builder
   *
   * High-level wrapper around ClusterAccelBuilder that integrates RTX MG
   * into RTX Remix's scene graph and acceleration structure management.
   *
   * This class:
   * - Manages cluster meshes (meshoptimizer LOD DAG + CLAS templates)
   * - Builds cluster-based BLAS from triangle meshes
   * - Integrates with RtxAccelManager for unified AS management
   * - Provides hierarchical Z-buffer (HIZ) for visibility culling
   * - Handles lifecycle of RTX MG resources
   */
  class RtxMegaGeoBuilder : public RcObject {
  public:
    RtxMegaGeoBuilder(
      const Rc<DxvkDevice>& device,
      const Rc<RtxContext>& rtxContext);

    ~RtxMegaGeoBuilder();

    /**
     * \brief Initialize RTX MG systems
     *
     * Creates NVRHI adapter, ClusterAccelBuilder, HIZ buffer, etc.
     * Must be called once before using any other methods.
     */
    bool initialize();

    /**
     * \brief Create cluster mesh from triangle data (meshoptimizer path)
     *
     * Clusterizes a triangle mesh using meshoptimizer and builds per-meshlet
     * CLAS templates. The LOD DAG is built asynchronously on a worker thread.
     * Templates are built on the first frame after the LOD DAG is ready.
     *
     * \param [in] desc Triangle mesh descriptor
     * \param [out] surfaceId Unique ID for this surface
     * \return true on success
     */
    bool createClusterMesh(
      const TriangleMeshDesc& desc,
      uint32_t& surfaceId);

    /**
     * \brief Build cluster BLAS for all surfaces
     *
     * Executes the complete RTX MG pipeline:
     * 1. LOD selection per instance (CPU)
     * 2. Fill instantiation args + vertex positions (CPU)
     * 3. Instantiate CLAS templates (GPU)
     * 4. Build BLAS from clusters (GPU)
     *
     * \param [in] context Rendering context for command recording
     * \param [in] depthBuffer Optional depth buffer for HIZ culling
     * \param [in] camera Camera for LOD selection
     * \param [in] instanceTransforms Per-surface object-to-world transforms
     * \return true on success
     */
    bool buildClusterBlas(
      const Rc<RtxContext>& context,
      const Rc<DxvkImageView>& depthBuffer,
      const class RtCamera& camera,
      const std::unordered_map<uint32_t, Matrix4>& instanceTransforms = {});

    /**
     * \brief Check if surface is ready for ray tracing
     *
     * Returns true if the cluster mesh has been built and its
     * CLAS templates are ready for instantiation.
     *
     * \param [in] surfaceId Surface ID
     * \return true if ready
     */
    bool isSurfaceReady(uint32_t surfaceId) const;

    /**
     * \brief Get BLAS device address for a surface
     *
     * Returns the GPU virtual address of a surface's BLAS, looked up
     * from the downloaded BLAS addresses buffer.
     *
     * \param [in] surfaceId Surface ID
     * \return Device address or 0 if not built
     */
    VkDeviceAddress getSurfaceBlasAddress(uint32_t surfaceId) const;

    /**
     * \brief Get BLAS pointers buffer for GPU-side TLAS patching
     *
     * Returns the GPU buffer containing BLAS addresses for all instances.
     * This buffer is populated by the cluster BLAS build and can be used
     * with a compute shader to patch instance descriptors without GPU->CPU readback.
     *
     * \return NVRHI buffer handle or nullptr if not available
     */
    nvrhi::IBuffer* getBlasPointersBuffer() const;

    /**
     * \brief Get instance index for a surface ID
     *
     * Returns the RTXMG instance index corresponding to a surface ID.
     * Used for mapping between RTX Remix surfaces and RTXMG instances.
     *
     * \param [in] surfaceId Surface ID
     * \return Instance index or UINT32_MAX if not found
     */
    uint32_t getInstanceIndexForSurface(uint32_t surfaceId) const;

    /** Check if a surface ID still exists in the builder (not pruned). */
    bool hasSurface(uint32_t surfaceId) const { return m_clusterMeshes.count(surfaceId) > 0; }

    /**
     * \brief Patch cluster BLAS addresses in instance buffer (GPU-side)
     *
     * Runs a compute shader to copy BLAS addresses from blasPtrsBuffer to
     * the Vulkan instance buffer at the specified instance indices.
     * This matches the sample's approach of patching addresses on GPU.
     *
     * \param [in] instanceBuffer Raw pointer to Vulkan instance buffer
     * \param [in] mappings Vector of (remixInstanceIndex, rtxmgInstanceIndex) pairs
     * \param [in] instanceStride sizeof(VkAccelerationStructureInstanceKHR)
     */
    struct InstancePatchMapping {
      uint32_t remixInstanceIndex;
      uint32_t rtxmgInstanceIndex;
    };
    void patchClusterBlasAddresses(
      VkBuffer instanceBuffer,
      VkDeviceSize instanceBufferOffset,
      const std::vector<InstancePatchMapping>& mappings);

    /**
     * \\brief Patch cluster BLAS addresses using GPU compute shader
     *
     * More efficient than patchClusterBlasAddresses - patches directly on GPU
     * without CPU readback. Requires the instance buffer as an NVRHI buffer.
     *
     * \\param [in] instanceBuffer NVRHI buffer containing instance descriptors
     * \\param [in] instanceBufferOffset Offset in instances (not bytes) to start of relevant data
     * \\param [in] mappings Vector of (remixInstanceIndex, rtxmgInstanceIndex) pairs
     */
    void patchClusterBlasAddressesGPU(
      nvrhi::IBuffer* instanceBuffer,
      uint32_t instanceBufferOffset,
      const std::vector<InstancePatchMapping>& mappings);

    /**
     * \\brief Get downloaded BLAS address for an RTXMG instance
     *
     * Returns the BLAS address that was downloaded from GPU during
     * patchClusterBlasAddresses. Used to patch instance descriptors.
     *
     * \\param [in] rtxmgInstanceIndex Index in the RTXMG blasPtrsBuffer
     * \\return BLAS device address or 0 if not available
     */
    VkDeviceAddress getDownloadedBlasAddress(uint32_t rtxmgInstanceIndex) const;

    /**
     * \brief Download BLAS addresses from GPU after BuildAccel
     *
     * Downloads blasPtrsBuffer to CPU for direct use in addBlas().
     * \return true if at least one non-zero BLAS address was downloaded
     */
    bool downloadBlasAddresses();

    /**
     * \brief Get tessellation statistics
     */
    struct TessellationStats {
      uint32_t numClusters = 0;
      uint32_t numDesiredClusters = 0;
      uint32_t numTriangles = 0;
      uint32_t numVertices = 0;
      uint64_t clasMemoryBytes = 0;
      float cullRatio = 0.0f;
    };

    const TessellationStats& getStats() const { return m_stats; }

    /**
     * \brief Show ImGui debug UI
     */
    void showImguiSettings();

    /**
     * \brief Get NVRHI device adapter
     */
    NvrhiDxvkDevice* getNvrhiDevice() const { return m_nvrhiDevice; }

    /**
     * \brief Get cluster acceleration builder
     */
    ClusterAccelBuilder* getClusterAccelBuilder() const { return m_clusterBuilder.get(); }

    // Debug view settings
    enum class ColorMode {
      BASE_COLOR = 0,
      COLOR_BY_NORMAL,
      COLOR_BY_TEXCOORD,
      COLOR_BY_MATERIAL,
      COLOR_BY_GEOMETRY_INDEX,
      COLOR_BY_SURFACE_INDEX,
      COLOR_BY_CLUSTER_ID,
      COLOR_BY_MICROTRI_ID,
      COLOR_BY_CLUSTER_UV,
      COLOR_BY_MICROTRI_AREA,
      COLOR_BY_TOPOLOGY,
      COLOR_MODE_COUNT
    };

    // Debug settings accessors
    bool getWireframeMode() const { return m_wireframeMode; }
    void setWireframeMode(bool enabled) { m_wireframeMode = enabled; }

    float getWireframeThickness() const { return m_wireframeThickness; }
    void setWireframeThickness(float thickness) { m_wireframeThickness = thickness; }

    bool getShowMicroTriangles() const { return m_showMicroTriangles; }
    void setShowMicroTriangles(bool show) { m_showMicroTriangles = show; }

    ColorMode getColorMode() const { return m_colorMode; }
    void setColorMode(ColorMode mode) { m_colorMode = mode; }

    bool getEnableTessellation() const { return m_enableTessellation; }
    void setEnableTessellation(bool enabled) { m_enableTessellation = enabled; }

    /**
     * \brief Get ClusterShadingData buffer for shader binding
     */
    nvrhi::BufferHandle getClusterShadingDataBuffer() const;

    /**
     * \brief Get number of clusters in the ClusterShadingData buffer
     */
    uint32_t getClusterCount() const;

    /**
     * \brief Get the cluster vertex positions buffer
     */
    nvrhi::BufferHandle getClusterVertexPositionsBuffer() const;

    /**
     * \brief Get the cluster vertex normals buffer
     */
    nvrhi::BufferHandle getClusterVertexNormalsBuffer() const;

    /**
     * \brief Check if cluster buffers are ready for rendering
     */
    bool hasValidBuffers() const;

  private:
    // Worker thread infrastructure
    std::mutex m_pendingMutex; // Guards the CV wait
    std::condition_variable m_workerCV;
    std::vector<std::thread> m_workerThreads;
    std::atomic<bool> m_workerShouldExit{false};
    uint32_t m_numWorkerThreads = 4;

    void workerThreadFunc(uint32_t threadIndex);

    // RTX Remix integration
    Rc<DxvkDevice> m_device;
    Rc<RtxContext> m_rtxContext;

    // NVRHI adapter layer
    NvrhiDxvkDevice* m_nvrhiDevice = nullptr;
    nvrhi::CommandListHandle m_commandList;
    uint32_t m_frameIndex = 0;

    // RTX MG core systems
    std::unique_ptr<ClusterAccelBuilder> m_clusterBuilder;
    std::unique_ptr<HiZBuffer> m_hizBuffer;
    std::unique_ptr<ZBuffer> m_zBuffer;

    // Cluster acceleration structures
    std::unique_ptr<ClusterAccels> m_clusterAccels;
    ClusterStatistics m_clusterStats;
    nvrhi::BufferHandle m_scratchBuffer;

    // Cluster mesh entries (meshoptimizer path)
    struct RTXMGClusterMeshEntry {
      std::unique_ptr<ClusterLODData> lodData;       // LOD DAG from meshoptimizer (built async)
      std::unique_ptr<MeshletTemplateSet> templates;  // CLAS templates (built once, needs GPU)
      std::string debugName;
      bool isReady = false;         // LOD DAG is built
      bool templatesBuilt = false;  // CLAS templates are built
    };
    std::unordered_map<uint32_t, RTXMGClusterMeshEntry> m_clusterMeshes;

    // Async cluster mesh creation
    struct PendingClusterMesh {
      uint32_t surfaceId;
      std::vector<uint32_t> indices;
      std::vector<float> vertexPositions;
      size_t vertexCount;
      size_t vertexPositionsStride;
      std::string debugName;
    };
    struct CompletedClusterMesh {
      uint32_t surfaceId;
      std::unique_ptr<ClusterLODData> lodData;
      std::string debugName;
    };
    std::queue<PendingClusterMesh> m_pendingClusterMeshes;
    std::queue<CompletedClusterMesh> m_completedClusterMeshes;
    std::mutex m_pendingClusterMeshMutex;
    std::mutex m_completedClusterMeshMutex;

    uint32_t m_nextSurfaceId = 1;

    // Mapping from surfaceId to instance index in the scene (rebuilt each frame)
    std::unordered_map<uint32_t, uint32_t> m_surfaceToInstanceIndex;

    // Instance transforms from RTX Remix (surfaceId -> objectToWorld)
    std::unordered_map<uint32_t, Matrix4> m_instanceTransforms;

    // Track last frame each surface had a transform (for stale surface cleanup)
    std::unordered_map<uint32_t, uint32_t> m_surfaceLastSeenFrame;

    // Downloaded BLAS addresses from GPU (populated after BuildAccel)
    std::vector<VkDeviceAddress> m_downloadedBlasAddresses;

    // Tessellation statistics
    TessellationStats m_stats;

    // Camera for tessellation (updated each frame from RtCamera)
    Camera m_tessellationCamera;

    // Dirty tracking - skip redundant BuildAccel when nothing changed
    Vector3 m_prevCameraPosition = Vector3(0.0f, 0.0f, 0.0f);
    Vector3 m_prevCameraForward = Vector3(0.0f, 0.0f, -1.0f);
    float m_prevFovY = 0.0f;
    std::unordered_map<uint32_t, Matrix4> m_prevBuildTransforms;
    bool m_forceRebuild = true;

    // Initialization state
    bool m_initialized = false;

    void updateHiZBuffer(
      const Rc<DxvkImageView>& depthBuffer);

    void collectStatistics();

    // Debug view settings
    bool m_wireframeMode = false;
    float m_wireframeThickness = 1.0f;
    bool m_showMicroTriangles = false;
    ColorMode m_colorMode = ColorMode::BASE_COLOR;
    bool m_enableTessellation = true;

    // GPU-side BLAS address patching infrastructure
    nvrhi::BufferHandle m_patchParamsBuffer;
    nvrhi::BufferHandle m_patchMappingsBuffer;
    nvrhi::BindingLayoutHandle m_patchBlasAddressesBL;
    nvrhi::ComputePipelineHandle m_patchBlasAddressesPSO;
  };

} // namespace dxvk
