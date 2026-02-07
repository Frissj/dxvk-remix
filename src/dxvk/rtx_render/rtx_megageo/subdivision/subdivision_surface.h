/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// RTX Remix replacements for Donut includes
#include "../../../util/util_math.h"
#include "../nvrhi_adapter/nvrhi_types.h"
#include "box3.h"
#include <opensubdiv/tmr/surfaceTable.h>

// Forward declarations and types to replace donut dependencies
namespace donut {
  namespace engine {
    class DescriptorTableManager;
    using DescriptorHandle = uint32_t;  // Placeholder type

    // ShaderMacro for shader compilation
    struct ShaderMacro {
      std::string name;
      std::string definition;

      ShaderMacro(const std::string& _name, const std::string& _definition)
        : name(_name), definition(_definition) {}
    };
  }
}

struct Shape;
class TopologyCache;
struct TopologyMap;

class SubdivisionSurface
{
public:
    // hashes the mesh topology into a TopologyCache and
    // initializes the device-side data structures corresponding to the
    // Tmr::SurfaceTables for 'vertex' and 'face-vayring' data (position &
    // texcoords).
    SubdivisionSurface(TopologyCache& topologyCache, std::unique_ptr<Shape> shape,
        const std::vector<std::unique_ptr<Shape>>& keyFrames,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
        nvrhi::ICommandList* commandList);

    bool HasAnimation() const;
    uint32_t NumKeyframes() const;

    size_t getGpuMemoryBytes() const;

    void Animate(float animTime, float frameRate);

    uint32_t NumVertices() const;
    uint32_t SurfaceCount() const;

    // AABBs are in object-space !
    std::vector<box3> m_aabbKeyframes;
    box3 m_aabb;

    // Animation space
    int m_f0 = 0;
    int m_f1 = 0;
    float m_dt = 0.f;
    float m_frameOffset = 0.0f;

public:
    struct SurfaceTableDeviceData
    {
        nvrhi::BufferHandle surfaceDescriptors;
        nvrhi::BufferHandle controlPointIndices;

        nvrhi::BufferHandle patchPoints;
        nvrhi::BufferHandle patchPointsOffsets;
    };

    //
    // 'vertex' limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#vertex-and-varying-data
    //
    SurfaceTableDeviceData m_vertexDeviceData;

    std::vector<nvrhi::BufferHandle> m_positionKeyframeBuffers;
    nvrhi::BufferHandle m_positionsBuffer;
    nvrhi::BufferHandle m_positionsPrevBuffer;

    //
    // 'face-varying' (texcoords) limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#face-varying-data-and-topology
    //
    SurfaceTableDeviceData m_texcoordDeviceData;
    nvrhi::BufferHandle m_texcoordsBuffer;
    nvrhi::BufferHandle m_surfaceToGeometryIndexBuffer;
    nvrhi::BufferHandle m_topologyQualityBuffer;

    donut::engine::DescriptorHandle m_vertexSurfaceDescriptorDescriptor;
    donut::engine::DescriptorHandle m_vertexControlPointIndicesDescriptor;
    donut::engine::DescriptorHandle m_positionsDescriptor;
    donut::engine::DescriptorHandle m_positionsPrevDescriptor;
    donut::engine::DescriptorHandle m_surfaceToGeometryIndexDescriptor;   

    // Surface Types
    enum class SurfaceType : uint32_t
    {
        PureBSpline,
        RegularBSpline,
        Limit,
        NoLimit,
        Count
    };
    std::array<uint32_t, size_t(SurfaceType::Count)> m_surfaceOffsets;
    uint32_t m_surfaceCount = 0;
    
    bool m_hasDisplacementMaterial = false;
    donut::engine::DescriptorHandle m_topologyQualityDescriptor;

public:
    Shape const* GetShape() const { return m_shape.get(); }
    TopologyMap const* GetTopologyMap() const { return m_topology_map; }

protected:
    TopologyMap const* m_topology_map = nullptr;

    void InitDeviceData(nvrhi::ICommandList* commandList);

    std::unique_ptr<Shape> m_shape;

    std::unique_ptr<const OpenSubdiv::Tmr::SurfaceTable> m_surface_table;
    std::unique_ptr<const OpenSubdiv::Tmr::LinearSurfaceTable>
        m_texcoord_surface_table;
};