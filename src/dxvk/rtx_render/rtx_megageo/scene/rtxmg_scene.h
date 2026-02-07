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

#include <vector>
#include <memory>
#include "../nvrhi_adapter/nvrhi_types.h"
#include "instance.h"

// Include subdivision surface from C++ directory
#include "../subdivision/subdivision_surface.h"

// SceneGraph for scene hierarchy management
class SceneGraph {
public:
    SceneGraph() = default;

    uint32_t GetMaxGeometryCountPerMesh() const { return m_maxGeometryCountPerMesh; }
    void SetMaxGeometryCountPerMesh(uint32_t count) { m_maxGeometryCountPerMesh = count; }

private:
    uint32_t m_maxGeometryCountPerMesh = 1;
};

// RTXMGScene - Main scene class for RTX Mega Geometry integration
class RTXMGScene {
public:
    RTXMGScene(nvrhi::DeviceHandle device);
    ~RTXMGScene() = default;

    // Subdivision mesh management
    const std::vector<SubdivisionSurface*>& GetSubdMeshes() const { return m_subdMeshes; }
    void AddSubdMesh(SubdivisionSurface* mesh) { m_subdMeshes.push_back(mesh); }
    void NullifySubdMesh(uint32_t index) { if (index < m_subdMeshes.size()) m_subdMeshes[index] = nullptr; }

    // Instance management
    const std::vector<Instance>& GetSubdMeshInstances() const { return m_instances; }
    std::vector<Instance>& GetSubdMeshInstances() { return m_instances; }
    void AddInstance(const Instance& instance) { m_instances.push_back(instance); }
    void ClearInstances() { m_instances.clear(); }

    // Buffer accessors for shader bindings
    nvrhi::BufferHandle GetGeometryBuffer() const { return m_geometryBuffer; }
    nvrhi::BufferHandle GetMaterialBuffer() const { return m_materialBuffer; }
    nvrhi::SamplerHandle GetDisplacementSampler() const { return m_displacementSampler; }

    // Statistics
    uint32_t TotalSubdPatchCount() const;

    // Scene graph accessor
    const SceneGraph* GetSceneGraph() const { return &m_sceneGraph; }
    SceneGraph* GetSceneGraph() { return &m_sceneGraph; }

    // Initialization
    void Initialize();
    void CreateBuffers();

private:
    nvrhi::DeviceHandle m_device;
    SceneGraph m_sceneGraph;

    std::vector<SubdivisionSurface*> m_subdMeshes;
    std::vector<Instance> m_instances;

    // GPU buffers
    nvrhi::BufferHandle m_geometryBuffer;
    nvrhi::BufferHandle m_materialBuffer;
    nvrhi::SamplerHandle m_displacementSampler;
};
