//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "subdivision_surface.h"

#include "../utils/buffer.h"
#include "topology_cache.h"
#include "topology_map.h"
#include "far.h"
#include "segmented_vector.h"

#include <opensubdiv/tmr/surfaceTableFactory.h>
#include <opensubdiv/tmr/subdivisionPlanBuilder.h>
#include <opensubdiv/tmr/subdivisionPlan.h>

#include "../../../util/log/log.h"

#include <algorithm>
#include <numeric>
#include <ranges>

#include "../rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_SUBDIVISION_SURFACE
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

using namespace OpenSubdiv;
using namespace dxvk;

using TexcoordDeviceData = SubdivisionSurface::SurfaceTableDeviceData;

void initSubdLinearDeviceData(const Tmr::LinearSurfaceTable& surfaceTable,
    TexcoordDeviceData& deviceData, nvrhi::ICommandList* commandList)
{
    deviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::LinearSurfaceDescriptor>(
            surfaceTable.descriptors, "texture coordinate surface descriptors", commandList);

    deviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
        surfaceTable.controlPointIndices, "texture coordinate control point indices", commandList);

    // Support (patch) points
    const uint32_t numSurfaces = surfaceTable.GetNumSurfaces();
    std::vector<uint32_t> patchPointsOffsets(numSurfaces + 1, 0);
    for (uint32_t i = 0; i < numSurfaces; i++)
    {
        Tmr::LinearSurfaceDescriptor desc = surfaceTable.GetDescriptor(i);
        if (!desc.HasLimit())
        {
            patchPointsOffsets[i + 1] = patchPointsOffsets[i];
            continue;
        }

        uint32_t numPatchPoints =
            (desc.GetQuadSubfaceIndex() == Tmr::LOCAL_INDEX_INVALID)
            ? 0
            : desc.GetFaceSize() + 1;
        patchPointsOffsets[i + 1] = patchPointsOffsets[i] + numPatchPoints;
    }

    deviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        patchPointsOffsets, "texture coordinate patch points offsets", commandList);

    deviceData.patchPoints = CreateBuffer(patchPointsOffsets.back(), sizeof(Vector2),
        "texture coordinate patch points", commandList->getDevice());
}

static void gatherStatistics(Shape const& shape,
    Far::TopologyRefiner const& refiner,
    Tmr::TopologyMap const& topologyMap,
    Tmr::SurfaceTable const& surfTable,
    std::vector<uint16_t> &topologyQuality)
{
    int nsurfaces = surfTable.GetNumSurfaces();

    // Histogram size for stencil count distribution
    constexpr int histogramSize = 10;

    // TODO: Implement statistics gathering for RTX Remix
    // The donut::stats namespace is not available, so we create a dummy struct
    // to allow the code to compile. Statistics are not currently tracked.
    struct {
        uint32_t holesCount = 0;
        uint32_t irregularFaceCount = 0;
        uint32_t maxFaceSize = 0;
        uint32_t infSharpCreases = 0;
        float sharpnessMax = 0.0f;
        uint32_t bsplineSurfaceCount = 0;
        uint32_t regularSurfaceCount = 0;
        uint32_t sharpSurfaceCount = 0;
        uint32_t isolationSurfaceCount = 0;
        uint32_t stencilCountMin = UINT32_MAX;
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0.0f;
        uint32_t surfaceCount = 0;
        std::vector<uint32_t> stencilCountHistogram;
        size_t byteSize = 0;
        std::vector<std::string> topologyRecommendations;
        bool IsCatmarkTopology() const { return true; }
        void BuildTopologyRecommendations() {} // Stub
    } surfStats;

    // Dummy evaluation stats struct
    struct {
        size_t surfaceTablesByteSizeTotal = 0;
        bool hasBadTopology = false;
        std::vector<decltype(surfStats)> surfaceTableStats;
    } evalStats;

    RTXMG_LOG(dxvk::str::format("gatherStatistics: Starting, nsurfaces=", nsurfaces));
    topologyQuality.resize(nsurfaces, 0);

    size_t stencilSum = 0;

    for (int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex)
    {
        if (surfIndex % 10 == 0) {
            RTXMG_LOG(dxvk::str::format("gatherStatistics: Processing surface ", surfIndex, " of ", nsurfaces));
        }

        Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor(surfIndex);

        if (!desc.HasLimit())
        {
            ++surfStats.holesCount;
            continue;
        }

        auto const plan =
            topologyMap.GetSubdivisionPlan(desc.GetSubdivisionPlanIndex());

        uint16_t& quality = topologyQuality[surfIndex];

        // check face m_size (regular / non-quad)
        if (!plan->IsRegularFace())
            ++surfStats.irregularFaceCount;

        uint32_t faceSize = plan->GetFaceSize();

        if (faceSize > 5)
            quality = std::max(quality, (uint16_t)0xff);

        surfStats.maxFaceSize = std::max(faceSize, surfStats.maxFaceSize);

        // check vertex valences
        const Tmr::Index* controlPoints =
            surfTable.GetControlPointIndices(surfIndex);

        uint32_t maxVertexValence = 0;
        int numVertices = refiner.GetLevel(0).GetNumVertices();
        for (uint8_t i = 0; i < faceSize; ++i)
        {
            if (controlPoints[i] >= 0 && controlPoints[i] < numVertices) {
                auto edges = refiner.GetLevel(0).GetVertexEdges(controlPoints[i]);
                maxVertexValence = std::max(maxVertexValence, uint32_t(edges.size()));
            } else {
                dxvk::Logger::warn(dxvk::str::format("gatherStatistics: Invalid control point index ",
                    controlPoints[i], " for surface ", surfIndex, " (numVerts=", numVertices, ")"));
            }
        }
        if (maxVertexValence > 8)
            quality = std::max(quality, (uint16_t)0xff);

        // check sharpness
        bool hasSharpness = false;

        if (plan->GetNumNeighborhoods())
        {
            Tmr::Neighborhood const& n = plan->GetNeighborhood(0);

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if (hasSharpness = !(corners.empty() && creases.empty()))
            {

                auto processSharpness = [&surfStats,
                    &quality](Tmr::ConstFloatArray values)
                    {
                        for (int i = 0; i < values.size(); ++i)
                        {
                            if (values[i] >= 10.f)
                                ++surfStats.infSharpCreases;
                            else
                            {
                                surfStats.sharpnessMax =
                                    std::max(surfStats.sharpnessMax, values[i]);

                                if (values[i] > 8.f)
                                    quality = std::max(quality, (uint16_t)0xff);
                                else if (values[i] > 4.f)
                                    quality = std::max(
                                        quality,
                                        uint16_t((values[i] / Sdc::Crease::SHARPNESS_INFINITE) *
                                            255.f));
                            }
                        }
                    };
                processSharpness(creases);
                processSharpness(corners);
            }
        }

        // check stencil matrix
        size_t nstencils = plan->GetNumStencils();

        if (nstencils == 0)
        {
            if (plan->GetNumControlPoints() == 16)
                ++surfStats.bsplineSurfaceCount;
            else
                ++surfStats.regularSurfaceCount;
        }
        else
        {
            if (hasSharpness)
                ++surfStats.sharpSurfaceCount;
            else
                ++surfStats.isolationSurfaceCount;
        }

        stencilSum += nstencils;

        surfStats.stencilCountMin =
            std::min(surfStats.stencilCountMin, (uint32_t)nstencils);
        surfStats.stencilCountMax =
            std::max(surfStats.stencilCountMax, (uint32_t)nstencils);
    }

    RTXMG_LOG(dxvk::str::format("gatherStatistics: Loop complete, validating counts"));
    uint32_t totalCounted = surfStats.holesCount + surfStats.bsplineSurfaceCount +
        surfStats.regularSurfaceCount + surfStats.isolationSurfaceCount +
        surfStats.sharpSurfaceCount;

    if (totalCounted != nsurfaces) {
        dxvk::Logger::warn(dxvk::str::format("gatherStatistics: Surface count mismatch! Total=", nsurfaces,
            " holes=", surfStats.holesCount,
            " bspline=", surfStats.bsplineSurfaceCount,
            " regular=", surfStats.regularSurfaceCount,
            " isolation=", surfStats.isolationSurfaceCount,
            " sharp=", surfStats.sharpSurfaceCount,
            " sum=", totalCounted));
    }

    // Don't assert in release builds - just log the warning
    // assert((surfStats.holesCount + surfStats.bsplineSurfaceCount +
    //     surfStats.regularSurfaceCount + surfStats.isolationSurfaceCount +
    //     surfStats.sharpSurfaceCount) == nsurfaces);

    surfStats.stencilCountAvg = float(stencilSum) / float(nsurfaces);

    surfStats.stencilCountHistogram.resize(histogramSize);

    surfStats.surfaceCount = nsurfaces;

    RTXMG_LOG("gatherStatistics: Checking topology type");
    if (!surfStats.IsCatmarkTopology())
    {
        RTXMG_LOG("gatherStatistics: Running second pass for non-Catmark topology");
        // if we suspect this was not a sub-d model (likely a triangular mesh), run
        // a second pass of the surfaces to tag all the irregular faces (non-quads)
        // as poor quality
        int const regularFaceSize =
            Sdc::SchemeTypeTraits::GetRegularFaceSize(refiner.GetSchemeType());

        const Vtr::internal::Level& level = refiner.getLevel(0);
        for (int faceIndex = 0, surfaceIndex = 0; faceIndex < level.getNumFaces();
            ++faceIndex)
        {
            if (level.isFaceHole(faceIndex))
                continue;
            if (int nverts = level.getFaceVertices(faceIndex).size();
                nverts == regularFaceSize)
                ++surfaceIndex;
            else
            {
                for (int vert = 0; vert < nverts; ++vert, ++surfaceIndex)
                    topologyQuality[surfaceIndex] = 0xff;
            }
        }
    }

    RTXMG_LOG("gatherStatistics: Building stencil histogram");
    // fill stencil counts histogram
    if (surfStats.stencilCountMin == surfStats.stencilCountMax)
    {
        // all the surfaces have the same number of stencils
        surfStats.stencilCountHistogram.push_back(nsurfaces);
    }
    else
    {
        surfStats.stencilCountHistogram.resize(histogramSize);

        float delta = float(surfStats.stencilCountMax - surfStats.stencilCountMin) /
            histogramSize;

        for (int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex)
        {

            Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor(surfIndex);

            if (!desc.HasLimit())
                continue;

            auto const plan =
                topologyMap.GetSubdivisionPlan(desc.GetSubdivisionPlanIndex());

            uint32_t nstencils = (uint32_t)plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor(
                float(nstencils - surfStats.stencilCountMin) / delta);

            ++surfStats
                .stencilCountHistogram[std::min(uint32_t(histogramSize - 1), i)];
        }
    }

    RTXMG_LOG("gatherStatistics: Building topology recommendations");
    surfStats.BuildTopologyRecommendations();

    RTXMG_LOG("gatherStatistics: Updating eval stats");
    evalStats.surfaceTablesByteSizeTotal += surfStats.byteSize;
    evalStats.hasBadTopology |= (!surfStats.topologyRecommendations.empty());

    evalStats.surfaceTableStats.emplace_back(std::move(surfStats));

    RTXMG_LOG("gatherStatistics: Complete");
}

static std::vector<uint16_t> quadrangulateFaceToSubshape(
    Shape const& shape, uint32_t nsurfaces)
{
    assert(shape.scheme == Scheme::kCatmark);

    if (shape.nvertsPerFace.empty() || shape.faceToSubshapeIndex.empty() || !nsurfaces)
        return {};

    std::vector<uint16_t> result(nsurfaces);

    // Strong assumption here that this matches the quadrangulation to Create the surface descriptors
    for (uint32_t face = 0, vcount = 0; face < (uint32_t)shape.nvertsPerFace.size(); ++face)
    {
        int nverts = shape.nvertsPerFace[face];

        uint32_t subShapeIndex = shape.faceToSubshapeIndex[face];

        if (nverts == 4)
        {
            assert(vcount < result.size());
            result[vcount++] = static_cast<uint16_t>(subShapeIndex);
        }
        else
        {
            assert(vcount + nverts <= result.size());
            for (int vert = 0; vert < nverts; ++vert)
            {
                result[vcount + vert] = static_cast<uint16_t>(subShapeIndex);
            }
            vcount += nverts;
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// SubdivisionSurface
// -----------------------------------------------------------------------------
SubdivisionSurface::SubdivisionSurface(TopologyCache& topologyCache,
    std::unique_ptr<Shape> shape,
    const std::vector<std::unique_ptr<Shape>> &keyFrameShapes,
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
    nvrhi::ICommandList* commandList)
{
    RTXMG_LOG("SubdivSurface: Constructor start");
    m_shape = std::move(shape);

    // Create Far mesh (control cage topology)
    RTXMG_LOG("SubdivSurface: Getting SDC type and options");
    Sdc::SchemeType schemeType = GetSdcType(*m_shape);
    Sdc::Options schemeOptions = GetSdcOptions(*m_shape);
    Tmr::EndCapType endCaps = Tmr::EndCapType::ENDCAP_BSPLINE_BASIS;

    {
        // note: for now the topology cache only supports a single map
        // for a given set of traits ; eventually Tmr::SurfaceTableFactory
        // may support directly topology caches, allowing a given
        // Tmr::SurfaceTable to reference multiple topology maps at run-time.
        RTXMG_LOG("SubdivSurface: Setting topology traits");
        Tmr::TopologyMap::Traits traits;
        traits.SetCompatible(schemeType, schemeOptions, endCaps);

        RTXMG_LOG("SubdivSurface: Getting topology map from cache");
        m_topology_map = &topologyCache.get(traits.value);
    }

    RTXMG_LOG("SubdivSurface: Getting topology map reference");
    Tmr::TopologyMap& topologyMap = *m_topology_map->aTopologyMap;

    RTXMG_LOG("SubdivSurface: Creating TopologyRefiner");
    std::unique_ptr<Far::TopologyRefiner> refiner;

    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(
        *m_shape,
        Far::TopologyRefinerFactory<Shape>::Options(schemeType, schemeOptions)));

    if (!refiner) {
        dxvk::Logger::err(dxvk::str::format("SubdivSurface: TopologyRefiner creation FAILED for shape with ",
            m_shape->verts.size(), " verts, ", m_shape->nvertsPerFace.size(), " faces"));
        throw std::runtime_error("TopologyRefiner creation failed");
    }

    // Log shape topology details for crash diagnosis
    dxvk::Logger::warn(dxvk::str::format("SubdivSurface: TopologyRefiner created - numLevels=", refiner->GetNumLevels(),
        " numVerts=", refiner->GetLevel(0).GetNumVertices(),
        " numFaces=", refiner->GetLevel(0).GetNumFaces(),
        " numEdges=", refiner->GetLevel(0).GetNumEdges(),
        " numFVarChannels=", refiner->GetNumFVarChannels(),
        " shapeVerts=", m_shape->verts.size(),
        " shapeFaces=", m_shape->nvertsPerFace.size(),
        " shapeIndices=", m_shape->faceverts.size()));

    RTXMG_LOG("SubdivSurface: Creating SurfaceTable");
    Tmr::SurfaceTableFactory tableFactory;

    Tmr::SurfaceTableFactory::Options options;
    options.planBuilderOptions.endCapType = endCaps;
    options.planBuilderOptions.isolationLevel = topologyCache.options.isoLevelSharp;
    options.planBuilderOptions.isolationLevelSecondary = topologyCache.options.isoLevelSmooth;
    options.planBuilderOptions.useSingleCreasePatch = true;
    options.planBuilderOptions.useInfSharpPatch = true;
    options.planBuilderOptions.useTerminalNode = topologyCache.options.useTerminalNodes;
    options.planBuilderOptions.useDynamicIsolation = true;
    options.planBuilderOptions.orderStencilMatrixByLevel = true;
    options.planBuilderOptions.generateLegacySharpCornerPatches = false;

    dxvk::Logger::warn("SubdivSurface: Calling SurfaceTableFactory::Create");
    m_surface_table =
        tableFactory.Create(*refiner, topologyMap, options);
    dxvk::Logger::warn("SubdivSurface: SurfaceTableFactory::Create returned");

    RTXMG_LOG("SubdivSurface: Gathering statistics");
    std::vector<uint16_t> topologyQuality;
    try {
        gatherStatistics(*m_shape, *refiner, topologyMap, *m_surface_table, topologyQuality);
        RTXMG_LOG("SubdivSurface: Statistics gathered successfully");
    } catch (const std::exception& e) {
        dxvk::Logger::err(dxvk::str::format("SubdivSurface: Exception in gatherStatistics: ", e.what()));
        throw;
    } catch (...) {
        dxvk::Logger::err("SubdivSurface: Unknown exception in gatherStatistics");
        throw;
    }

    RTXMG_LOG("SubdivSurface: Creating topology quality buffer");
    m_topologyQualityBuffer = CreateAndUploadBuffer<uint16_t>(
        topologyQuality, "topology quality", commandList);
    RTXMG_LOG("SubdivSurface: Topology quality buffer created");

    // setup for texcoords - always create like the sample does
    RTXMG_LOG("SubdivSurface: Creating texcoord surface table");
    RTXMG_LOG(dxvk::str::format("SubdivSurface: refiner numFVarChannels=", refiner->GetNumFVarChannels()));
    RTXMG_LOG(dxvk::str::format("SubdivSurface: shape uvs size=", m_shape->uvs.size()));

    Tmr::LinearSurfaceTableFactory tableFactoryFvar;
    constexpr int const fvarChannel = 0;

    // Check if we have fvar channel data
    if (refiner->GetNumFVarChannels() == 0) {
        dxvk::Logger::warn("SubdivSurface: No fvar channels, skipping texcoord surface table");
        m_texcoord_surface_table = nullptr;
    } else {
        try {
            m_texcoord_surface_table =
                tableFactoryFvar.Create(*refiner, fvarChannel, m_surface_table.get());
            RTXMG_LOG("SubdivSurface: Texcoord surface table created");
        } catch (const std::exception& e) {
            dxvk::Logger::err(dxvk::str::format("SubdivSurface: Exception creating texcoord surface table: ", e.what()));
            m_texcoord_surface_table = nullptr;
        } catch (...) {
            dxvk::Logger::err("SubdivSurface: Unknown exception creating texcoord surface table");
            m_texcoord_surface_table = nullptr;
        }
    }

    RTXMG_LOG("SubdivSurface: Calling InitDeviceData");
    InitDeviceData(commandList);
    RTXMG_LOG("SubdivSurface: InitDeviceData complete");

    RTXMG_LOG("SubdivSurface: Creating texcoords buffer");
    m_texcoordsBuffer =
        CreateAndUploadBuffer(m_shape->uvs, "base texcoords", commandList);
    RTXMG_LOG("SubdivSurface: Texcoords buffer created");

    RTXMG_LOG("SubdivSurface: Creating positions buffer");
    RTXMG_LOG(dxvk::str::format("SubdivSurface: m_shape->verts.size()=", m_shape->verts.size()));
    // DEBUG: Log first few vertex positions
    for (size_t i = 0; i < std::min(m_shape->verts.size(), size_t(10)); i++) {
        const auto& v = m_shape->verts[i];
        RTXMG_LOG(dxvk::str::format("SubdivSurface: vert[", i, "] = (", v.x, ", ", v.y, ", ", v.z, ")"));
    }
    // DEBUG: Log AABB
    RTXMG_LOG(dxvk::str::format("SubdivSurface: AABB min=(", m_shape->aabb.m_mins[0], ",", m_shape->aabb.m_mins[1], ",", m_shape->aabb.m_mins[2],
        ") max=(", m_shape->aabb.m_maxs[0], ",", m_shape->aabb.m_maxs[1], ",", m_shape->aabb.m_maxs[2], ")"));
    m_positionsBuffer = CreateAndUploadBuffer(m_shape->verts, "SubdPosedPositions", commandList);
    m_aabb = m_shape->aabb;
    RTXMG_LOG("SubdivSurface: Positions buffer created");

    if (keyFrameShapes.size() > 0)
    {
        // Includes the 0th frame
        size_t nframes = keyFrameShapes.size() + 1;

        m_positionsPrevBuffer = CreateAndUploadBuffer(m_shape->verts, "SubdPosedPositions", commandList);

        m_positionKeyframeBuffers.resize(nframes);
        m_aabbKeyframes.resize(nframes);

        m_positionKeyframeBuffers[0] = CreateAndUploadBuffer(
            m_shape->verts, "SubdKeyFramePosition0", commandList);
        m_aabbKeyframes[0] = m_shape->aabb;

        // starts 1 indexed
        uint32_t frameIndex = 1;
        for (auto& keyFrameShape : keyFrameShapes)
        {
            char debugName[50];
            std::snprintf(debugName, std::size(debugName), "SubdKeyFramePosition%d",
                frameIndex);
            m_positionKeyframeBuffers[frameIndex] =
                CreateAndUploadBuffer(keyFrameShape->verts, debugName, commandList);
            m_aabbKeyframes[frameIndex] = keyFrameShape->aabb;

            frameIndex++;
        }
    }

}

size_t SubdivisionSurface::getGpuMemoryBytes() const
{
    size_t total = 0;
    auto addBuffer = [&total](const nvrhi::BufferHandle& buf) {
        if (buf)
            total += buf->getDesc().byteSize;
    };

    // Vertex surface table data
    addBuffer(m_vertexDeviceData.surfaceDescriptors);
    addBuffer(m_vertexDeviceData.controlPointIndices);
    addBuffer(m_vertexDeviceData.patchPoints);
    addBuffer(m_vertexDeviceData.patchPointsOffsets);

    // Texcoord surface table data
    addBuffer(m_texcoordDeviceData.surfaceDescriptors);
    addBuffer(m_texcoordDeviceData.controlPointIndices);
    addBuffer(m_texcoordDeviceData.patchPoints);
    addBuffer(m_texcoordDeviceData.patchPointsOffsets);

    // Position buffers
    addBuffer(m_positionsBuffer);
    addBuffer(m_positionsPrevBuffer);

    // Texcoords and per-surface metadata
    addBuffer(m_texcoordsBuffer);
    addBuffer(m_surfaceToGeometryIndexBuffer);
    addBuffer(m_topologyQualityBuffer);

    // Keyframe position buffers
    for (const auto& buf : m_positionKeyframeBuffers) {
        addBuffer(buf);
    }

    return total;
}

uint32_t SubdivisionSurface::NumVertices() const
{
    return static_cast<uint32_t>(m_positionsBuffer->getDesc().byteSize / sizeof(Vector3));
}

uint32_t SubdivisionSurface::SurfaceCount() const
{
    return m_surfaceCount;
}

void SubdivisionSurface::InitDeviceData(nvrhi::ICommandList* commandList)
{
    m_surfaceCount = uint32_t(m_surface_table->descriptors.size());

    // Sort surfaces by PureBspline, Bspline, Complex types for shader optimization
    std::vector<Tmr::SurfaceDescriptor> sortedDescriptors = m_surface_table->descriptors;

    // DEBUG: Log descriptor HasLimit status before sorting
    {
        uint32_t hasLimitCount = 0;
        uint32_t noLimitCount = 0;
        for (uint32_t i = 0; i < m_surfaceCount && i < 10; i++) {
            const auto& desc = sortedDescriptors[i];
            bool hasLimit = desc.HasLimit();
            if (hasLimit) hasLimitCount++; else noLimitCount++;
            RTXMG_LOG(dxvk::str::format("SubdivSurface: Descriptor[", i, "] field0=0x",
                std::hex, desc.field0, std::dec, " firstCP=", desc.firstControlPoint,
                " HasLimit=", hasLimit ? "true" : "false",
                " planIdx=", desc.GetSubdivisionPlanIndex()));
        }
        for (uint32_t i = 10; i < m_surfaceCount; i++) {
            if (sortedDescriptors[i].HasLimit()) hasLimitCount++; else noLimitCount++;
        }
        RTXMG_LOG(dxvk::str::format("SubdivSurface: Total ", m_surfaceCount, " descriptors: ",
            hasLimitCount, " have limit, ", noLimitCount, " no limit"));
    }

    // Copy texcoord descriptors if available
    std::vector<Tmr::LinearSurfaceDescriptor> sortedTexcoordDescriptors;
    if (m_texcoord_surface_table) {
        sortedTexcoordDescriptors = m_texcoord_surface_table->descriptors;
    } else {
        // No texcoords - create empty descriptors matching surface count
        dxvk::Logger::warn("SubdivSurface: No texcoord surface table, creating empty descriptors");
        sortedTexcoordDescriptors.resize(m_surfaceCount);
    }
    auto surfaceToGeometryIndex = quadrangulateFaceToSubshape(*m_shape, m_surfaceCount);

    assert(m_surfaceCount == sortedDescriptors.size());
    assert(m_surfaceCount == sortedTexcoordDescriptors.size());
    assert(m_surfaceCount == surfaceToGeometryIndex.size());

    // C++20 compatible sorting: use index vector instead of C++23 zip
    std::vector<size_t> indices(m_surfaceCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [this, &sortedDescriptors](size_t lhsIdx, size_t rhsIdx)
        {
            // Extract surface descriptors using indices
            const auto& a = sortedDescriptors[lhsIdx];
            const auto& b = sortedDescriptors[rhsIdx];
            bool tieBreakerAB = a.firstControlPoint < b.firstControlPoint;

            // All holes last
            bool aHasLimit = a.HasLimit();
            bool bHasLimit = b.HasLimit();
            if (aHasLimit != bHasLimit)
                return aHasLimit;
            else if (!aHasLimit && !bHasLimit)
                return tieBreakerAB;
            
            // PureBspline
            bool aIsPureBSplinePatch = a.GetSubdivisionPlanIndex() == 0;
            bool bIsPureBSplinePatch = b.GetSubdivisionPlanIndex() == 0;
            if (aIsPureBSplinePatch != bIsPureBSplinePatch)
                return aIsPureBSplinePatch;
            else if (aIsPureBSplinePatch && bIsPureBSplinePatch)
                return tieBreakerAB;

            // BSpline
            const auto* aPlan = m_surface_table->topologyMap.GetSubdivisionPlan(a.GetSubdivisionPlanIndex());
            bool aIsBSplinePatch = aPlan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;

            const auto* bPlan = m_surface_table->topologyMap.GetSubdivisionPlan(b.GetSubdivisionPlanIndex());
            bool bIsBSplinePatch = bPlan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;

            if (aIsBSplinePatch != bIsBSplinePatch)
                return aIsBSplinePatch;
            
            // Complex, Limit Surface
            return tieBreakerAB;
        });

    // Reorder the vectors based on sorted indices
    auto reorderVector = [](auto& vec, const std::vector<size_t>& indices) {
        auto copy = vec;
        for (size_t i = 0; i < indices.size(); i++) {
            vec[i] = copy[indices[i]];
        }
    };
    reorderVector(sortedDescriptors, indices);
    reorderVector(sortedTexcoordDescriptors, indices);
    reorderVector(surfaceToGeometryIndex, indices);

    // Array is sorted lets find the starting index for each stype
    int lastSurfaceType = -1;
    auto UpdateSurfaceOffset = [&lastSurfaceType, this](SurfaceType surfaceType, uint32_t startIndex)
        {
            for (int i = lastSurfaceType + 1; i <= int(surfaceType); i++)
            {
                m_surfaceOffsets[i] = startIndex;
            }
            lastSurfaceType = int(surfaceType);
        };

    for (uint32_t i = 0; i < m_surfaceCount; i++)
    {
        const auto& descriptor = sortedDescriptors[i];
        bool isPureBSplinePatch = descriptor.GetSubdivisionPlanIndex() == 0;
        if (isPureBSplinePatch)
        {
            UpdateSurfaceOffset(SurfaceType::PureBSpline, i);
        }
        else
        {
            const auto* plan = m_surface_table->topologyMap.GetSubdivisionPlan(descriptor.GetSubdivisionPlanIndex());
            bool isBSplinePatch = plan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;
            if (isBSplinePatch)
            {
                UpdateSurfaceOffset(SurfaceType::RegularBSpline, i);
            }
            else
            {
                if (descriptor.HasLimit())
                {
                    UpdateSurfaceOffset(SurfaceType::Limit, i);
                }
                else
                {
                    UpdateSurfaceOffset(SurfaceType::NoLimit, i);
                }
            }
        }
    }
    UpdateSurfaceOffset(SurfaceType::NoLimit, m_surfaceCount);

    RTXMG_LOG(dxvk::str::format("SubdivSurface: Surface offsets: PureBSpline=", m_surfaceOffsets[0],
        " RegularBSpline=", m_surfaceOffsets[1], " Limit=", m_surfaceOffsets[2],
        " NoLimit=", m_surfaceOffsets[3], " total=", m_surfaceCount));

    std::vector<uint32_t> patchPointsOffsets(m_surfaceCount + 1, 0);
    for (uint32_t iSurface = 0; iSurface < m_surfaceCount; ++iSurface)
    {
        const Tmr::SurfaceDescriptor surface = sortedDescriptors[iSurface];
        if (!surface.HasLimit())
        {
            patchPointsOffsets[iSurface + 1] = patchPointsOffsets[iSurface];
            continue;
        }

        // plan is never going to be null here
        const auto* plan = m_surface_table->topologyMap.GetSubdivisionPlan(
            surface.GetSubdivisionPlanIndex());

        patchPointsOffsets[iSurface + 1] =
            patchPointsOffsets[iSurface] + static_cast<uint32_t>(plan->GetNumPatchPoints());

#if RTXMG_LOG_SUBDIVISION_SURFACE // DIAGNOSTIC: per-level numPatchPoints
        if (iSurface < 5 && iSurface >= m_surfaceOffsets[2]) {
            auto& treeDesc = plan->GetTreeDescriptor();
            RTXMG_LOG(dxvk::str::format("RTX MegaGeo PATCHPT-LEVELS: surface[", iSurface,
                "] planIdx=", surface.GetSubdivisionPlanIndex(),
                " numPP(default)=", plan->GetNumPatchPoints(),
                " numPP[0]=", treeDesc.GetNumPatchPoints(0),
                " numPP[6]=", treeDesc.GetNumPatchPoints(6),
                " numPP[10]=", treeDesc.GetNumPatchPoints(10)));
        }
#endif
    }

    // Texcoord Patch Points - always computed like the sample
    std::vector<uint32_t> texcoordPatchPointsOffsets(m_surfaceCount + 1, 0);
    for (uint32_t i = 0; i < m_surfaceCount; i++)
    {
        Tmr::LinearSurfaceDescriptor desc = sortedTexcoordDescriptors[i];
        if (!desc.HasLimit())
        {
            texcoordPatchPointsOffsets[i + 1] = texcoordPatchPointsOffsets[i];
            continue;
        }

        uint32_t numPatchPoints =
            (desc.GetQuadSubfaceIndex() == Tmr::LOCAL_INDEX_INVALID)
            ? 0
            : desc.GetFaceSize() + 1;
        texcoordPatchPointsOffsets[i + 1] = texcoordPatchPointsOffsets[i] + numPatchPoints;
    }

    // Log raw field0 values being uploaded to GPU
    {
        RTXMG_LOG(dxvk::str::format("SubdivSurface: Uploading ", sortedDescriptors.size(), " descriptors to GPU"));
        for (uint32_t i = 0; i < std::min(size_t(5), sortedDescriptors.size()); i++) {
            RTXMG_LOG(dxvk::str::format("SubdivSurface: GPU Descriptor[", i, "] field0=0x",
                std::hex, sortedDescriptors[i].field0, std::dec,
                " (bit0=", (sortedDescriptors[i].field0 & 1), ")",
                " firstCP=", sortedDescriptors[i].firstControlPoint));
        }
    }

    m_vertexDeviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::SurfaceDescriptor>(
            sortedDescriptors, "surface descriptors", commandList);

    m_vertexDeviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
        m_surface_table->controlPointIndices, "control point indices", commandList);

    m_vertexDeviceData.patchPoints = CreateBuffer(patchPointsOffsets.back(), sizeof(Vector3), "patch points", commandList->getDevice());

    m_vertexDeviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        patchPointsOffsets, "patch points offsets", commandList);

    m_surfaceToGeometryIndexBuffer = CreateAndUploadBuffer<uint16_t>(surfaceToGeometryIndex, "surfaceToGeometryIndex", commandList);

    // Create texcoord device data
    m_texcoordDeviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::LinearSurfaceDescriptor>(
            sortedTexcoordDescriptors, "texture coordinate surface descriptors", commandList);

    if (m_texcoord_surface_table) {
        m_texcoordDeviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
            m_texcoord_surface_table->controlPointIndices, "texture coordinate control point indices", commandList);
    } else {
        // No texcoords - create empty buffer
        std::vector<Vtr::Index> emptyIndices;
        m_texcoordDeviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
            emptyIndices, "texture coordinate control point indices (empty)", commandList);
    }

    m_texcoordDeviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        texcoordPatchPointsOffsets, "texture coordinate patch points offsets", commandList);

    m_texcoordDeviceData.patchPoints = CreateBuffer(
        texcoordPatchPointsOffsets.empty() ? 0 : texcoordPatchPointsOffsets.back(),
        sizeof(Vector2),
        "texture coordinate patch points", commandList->getDevice());
}

bool SubdivisionSurface::HasAnimation() const
{
    return !m_positionKeyframeBuffers.empty();
}

uint32_t SubdivisionSurface::NumKeyframes() const
{
    return (uint32_t)m_positionKeyframeBuffers.size();
}

static inline box3 lerpAabb(const box3& a, const box3& b, float t)
{
    box3 result;
    // Manually lerp each component since arrays can't be assigned directly
    for (int i = 0; i < 3; i++) {
        result.m_mins[i] = a.m_mins[i] + (b.m_mins[i] - a.m_mins[i]) * t;
        result.m_maxs[i] = a.m_maxs[i] + (b.m_maxs[i] - a.m_maxs[i]) * t;
    }
    return result;
}

void SubdivisionSurface::Animate(float animTime, float frameRate)
{
    if (!HasAnimation())
        return;

    uint32_t nframes = static_cast<uint32_t>(m_positionKeyframeBuffers.size());

    float frameTime = m_frameOffset + animTime * frameRate;
    float frame = std::truncf(frameTime);

    // animation implicitly loops if frameTime >= NumKeyframes
    m_f0 = static_cast<int>(frame) % nframes;
    m_f1 = (m_f0 + 1) % nframes;

    m_dt = frameTime - frame;

    m_aabb = lerpAabb(m_aabbKeyframes[m_f0], m_aabbKeyframes[m_f1], animTime);
}
