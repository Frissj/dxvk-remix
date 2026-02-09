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

// Include OpenSubdiv version to ensure OPENSUBDIV_VERSION macro is defined
#include "osd_lite/opensubdiv/version.h"

// Forward declarations for RTX MG classes
class SubdivisionSurface;
class TopologyCache;
struct Shape;

// Forward declarations for OpenSubdiv
namespace OpenSubdiv {
  namespace OPENSUBDIV_VERSION {
    namespace Tmr {
      class TopologyMap;
      struct SurfaceTable;
    }
  }
}

namespace dxvk {

  // Forward declarations
  class RtxContext;
  struct RaytraceGeometry;
  struct BlasEntry;

  /**
   * \brief Subdivision surface descriptor
   *
   * Describes a Catmull-Clark subdivision surface mesh that will be
   * tessellated via RTX Mega Geometry cluster-based acceleration structures.
   */
  struct SubdivisionSurfaceDesc {
    // Topology
    uint32_t numFaces = 0;                    // Number of faces (quads and/or triangles)
    uint32_t numVertices = 0;                 // Number of control vertices
    const uint32_t* faceVertexCounts = nullptr; // Vertices per face (4 for quads, 3 for triangles)
    const uint32_t* faceVertexIndices = nullptr; // Flattened vertex indices

    // Control point data
    const Vector3* controlPoints = nullptr;   // Vertex positions (object space)
    const Vector2* texcoords = nullptr;       // Optional texture coordinates
    const Vector3* normals = nullptr;         // Optional vertex normals

    // Material
    uint32_t materialIndex = 0;               // Index into material system

    // Tessellation parameters
    uint32_t isolationLevel = 2;              // Subdivision isolation level (0-4)
    float tessellationScale = 1.0f;           // Global tessellation density multiplier
    bool enableDisplacement = false;          // Enable displacement mapping
    int displacementTextureIndex = -1;        // Bindless texture index for displacement
    float displacementScale = 1.0f;           // Displacement magnitude

    // Debug
    const char* debugName = nullptr;          // Optional debug name
  };

  /**
   * \brief RTX Mega Geometry Builder
   *
   * High-level wrapper around ClusterAccelBuilder that integrates RTX MG
   * into RTX Remix's scene graph and acceleration structure management.
   *
   * This class:
   * - Manages subdivision surface data (topology, control points)
   * - Builds cluster-based BLAS from subdivision surfaces
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
     * \brief Create subdivision surface from mesh data
     *
     * Converts a mesh descriptor into RTX MG internal representation
     * and prepares it for cluster-based tessellation.
     *
     * \param [in] desc Subdivision surface descriptor
     * \param [out] surfaceId Unique ID for this surface (used for updates)
     * \return true on success
     */
    bool createSubdivisionSurface(
      const SubdivisionSurfaceDesc& desc,
      uint32_t& surfaceId);

    /**
     * \brief Update subdivision surface data
     *
     * Updates control points, materials, or tessellation parameters
     * for an existing surface. Does NOT rebuild topology.
     *
     * \param [in] surfaceId Surface ID from createSubdivisionSurface
     * \param [in] desc Updated surface descriptor
     * \return true on success
     */
    bool updateSubdivisionSurface(
      uint32_t surfaceId,
      const SubdivisionSurfaceDesc& desc);

    /**
     * \brief Remove subdivision surface
     *
     * Removes a surface and frees associated resources.
     *
     * \param [in] surfaceId Surface ID to remove
     */
    void removeSubdivisionSurface(uint32_t surfaceId);

    /**
     * \brief Build cluster BLAS for all surfaces
     *
     * Executes the complete RTX MG pipeline:
     * 1. Compute cluster tiling (tessellation + culling)
     * 2. Build cluster templates (CLAS)
     * 3. Instantiate clusters
     * 4. Build BLAS from clusters
     *
     * \param [in] context Rendering context for command recording
     * \param [in] depthBuffer Optional depth buffer for HIZ culling
     * \param [in] viewProj View-projection matrix for frustum culling
     * \return true on success
     */
    bool buildClusterBlas(
      const Rc<RtxContext>& context,
      const Rc<DxvkImageView>& depthBuffer,
      const class RtCamera& camera,
      const std::unordered_map<uint32_t, Matrix4>& instanceTransforms = {});

    /**
     * \brief Get built BLAS for a surface
     *
     * Retrieves the VkAccelerationStructureKHR handle for a surface's
     * cluster-based BLAS. Valid after buildClusterBlas() succeeds.
     *
     * \param [in] surfaceId Surface ID
     * \return BLAS handle or VK_NULL_HANDLE if not built
     */
    VkAccelerationStructureKHR getSurfaceBlas(uint32_t surfaceId) const;

    /**
     * \brief Get BLAS device address
     *
     * Returns the GPU virtual address of a surface's BLAS for use
     * in ray tracing shader binding tables.
     *
     * \param [in] surfaceId Surface ID
     * \return Device address or 0 if not built
     */
    VkDeviceAddress getSurfaceBlasAddress(uint32_t surfaceId) const;

    /**
     * \brief Check if surface is ready for ray tracing
     *
     * Returns true if the surface has been tessellated and its
     * cluster BLAS is ready to be added to a TLAS.
     *
     * \param [in] surfaceId Surface ID
     * \return true if BLAS is ready
     */
    bool isSurfaceReady(uint32_t surfaceId) const;

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
    bool hasSurface(uint32_t surfaceId) const { return m_surfaces.count(surfaceId) > 0; }

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
     *
     * Returns information about the most recent tessellation pass:
     * - Number of clusters generated
     * - Number of triangles generated
     * - Number of vertices generated
     * - Culling statistics (frustum, backface, HIZ)
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
     *
     * Displays RTX Mega Geometry debug controls and statistics in ImGui.
     */
    void showImguiSettings();

    /**
     * \brief Get NVRHI device adapter
     *
     * Provides access to the NVRHI→DXVK adapter for advanced use cases.
     */
    NvrhiDxvkDevice* getNvrhiDevice() const { return m_nvrhiDevice; }

    /**
     * \brief Get cluster acceleration builder
     *
     * Provides direct access to ClusterAccelBuilder for advanced scenarios.
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
     *
     * Returns the GPU buffer containing per-cluster shading information
     * (surface ID, cluster offset, edge segments, etc.) needed for
     * MegaGeo debug views and material evaluation.
     *
     * \return NVRHI buffer handle or nullptr if not initialized
     */
    nvrhi::BufferHandle getClusterShadingDataBuffer() const;

    /**
     * \brief Get number of clusters in the ClusterShadingData buffer
     *
     * Returns the current number of clusters for bounds checking
     * when accessing the cluster shading data buffer.
     *
     * \return Number of clusters
     */
    uint32_t getClusterCount() const;

    /**
     * \brief Get the cluster vertex positions buffer
     *
     * Returns the buffer containing tessellated vertex positions for all clusters.
     * Used for surface interaction vertex lookup.
     *
     * \return NVRHI buffer handle or nullptr if not initialized
     */
    nvrhi::BufferHandle getClusterVertexPositionsBuffer() const;

    /**
     * \brief Get the cluster vertex normals buffer
     *
     * Returns the buffer containing tessellated vertex normals for all clusters.
     * Used for surface interaction normal lookup.
     *
     * \return NVRHI buffer handle or nullptr if not initialized
     */
    nvrhi::BufferHandle getClusterVertexNormalsBuffer() const;

    /**
     * \brief Check if cluster buffers are ready for rendering
     *
     * Returns true if all cluster buffers (shading data, vertex positions,
     * vertex normals) are valid and ready for shader access. When this
     * returns false, cluster surfaces should not be added to the TLAS.
     *
     * \return true if all buffers are valid, false otherwise
     */
    bool hasValidBuffers() const;

    /**
     * \brief Process completed async subdivision surface creations
     *
     * Called once per frame to integrate subdivision surfaces that were
     * created asynchronously on the worker thread.
     */
    void processCompletedSurfaces();

  private:
    // Async subdivision surface creation
    struct PendingSurface {
      uint32_t surfaceId;
      std::unique_ptr<Shape> shape;
      XXH64_hash_t topologyHash;
    };

    struct CompletedSurface {
      uint32_t surfaceId;
      std::unique_ptr<Shape> shape;
      XXH64_hash_t topologyHash;
    };

    std::queue<PendingSurface> m_pendingSurfaces;
    std::queue<CompletedSurface> m_completedSurfaces;
    std::mutex m_pendingMutex;
    std::mutex m_completedMutex;
    std::condition_variable m_workerCV;
    std::vector<std::thread> m_workerThreads;
    std::atomic<bool> m_workerShouldExit{false};
    uint32_t m_numWorkerThreads = 4; // Use 4 worker threads for parallel processing

    void workerThreadFunc(uint32_t threadIndex);

    // RTX Remix integration
    Rc<DxvkDevice> m_device;
    Rc<RtxContext> m_rtxContext;

    // NVRHI adapter layer
    NvrhiDxvkDevice* m_nvrhiDevice = nullptr;
    nvrhi::CommandListHandle m_commandList;
    uint32_t m_frameIndex = 0;  // Frame counter for transient resource management

    // RTX MG core systems
    std::unique_ptr<ClusterAccelBuilder> m_clusterBuilder;
    std::unique_ptr<HiZBuffer> m_hizBuffer;
    std::unique_ptr<ZBuffer> m_zBuffer;
    std::unique_ptr<class RTXMGScene> m_scene;

    // Cluster acceleration structures
    std::unique_ptr<ClusterAccels> m_clusterAccels;
    ClusterStatistics m_clusterStats;
    nvrhi::BufferHandle m_scratchBuffer;  // Scratch memory for cluster operations

    // Subdivision surface management - uses actual SubdivisionSurface class
    struct RTXMGSubdivisionSurfaceEntry {
      std::unique_ptr<SubdivisionSurface> subdivSurface;
      // Store only value-type data from descriptor (pointers become dangling after creation)
      std::string debugName;
      uint32_t numVertices = 0;
      uint32_t numFaces = 0;
      uint32_t isolationLevel = 2;
      float tessellationScale = 1.0f;
      VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
      VkDeviceAddress blasAddress = 0;
      bool isDirty = true;
      bool isReady = false;
      bool m_hasDisplacementMaterial = false;
      float displacementScale = 1.0f;
      size_t gpuMemoryBytes = 0;
    };

    std::unordered_map<uint32_t, RTXMGSubdivisionSurfaceEntry> m_surfaces;
    uint32_t m_nextSurfaceId = 1;

    // Deferred destruction for SubdivisionSurface objects.
    // When a surface is pruned, its SubdivisionSurface is moved here temporarily.
    // In buildClusterBlas, these are transferred to the DXVK command list's lifetime
    // tracker (via DxvkResource wrapping), which holds them until the GPU fence signals
    // that all in-flight commands have completed. This is the same pattern used for
    // binding sets (DxvkBindingSetHolder) and cluster operation buffers.
    // Frame counting doesn't work because GPU can fall behind CPU with heavy cluster loads.
    std::vector<std::unique_ptr<SubdivisionSurface>> m_pendingGpuDestructions;

    // Mapping from surfaceId to instance index in the scene (rebuilt each frame)
    // Used to look up BLAS addresses after cluster build
    std::unordered_map<uint32_t, uint32_t> m_surfaceToInstanceIndex;

    // Mapping from surfaceId to mesh index in the scene (persistent)
    // Used to rebuild instances each frame
    std::unordered_map<uint32_t, uint32_t> m_surfaceToMeshIndex;

    // Instance transforms from RTX Remix (surfaceId -> objectToWorld)
    std::unordered_map<uint32_t, Matrix4> m_instanceTransforms;

    // Track last frame each surface had a transform (for stale surface cleanup)
    std::unordered_map<uint32_t, uint32_t> m_surfaceLastSeenFrame;

    // Total GPU memory used by all surfaces in the cache
    size_t m_surfaceCacheVramBytes = 0;

    // Dynamic memory sizing - high-water mark (only grows, never shrinks to avoid reallocation churn)
    uint32_t m_hwmClusters = 0;
    size_t m_hwmClasBytes = 0;
    size_t m_hwmVertexBytes = 0;

    // Downloaded BLAS addresses from GPU (populated after BuildAccel)
    std::vector<VkDeviceAddress> m_downloadedBlasAddresses;

    // Topology caches - one per worker thread for thread-safe parallel processing
    std::vector<std::unique_ptr<TopologyCache>> m_topologyCaches;

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

    // Topology management (OpenSubdiv)
    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap> m_topologyMap;
    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::SurfaceTable> m_surfaceTable;

    // Internal methods
    std::unique_ptr<Shape> convertDescToShape(const SubdivisionSurfaceDesc& desc);

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
