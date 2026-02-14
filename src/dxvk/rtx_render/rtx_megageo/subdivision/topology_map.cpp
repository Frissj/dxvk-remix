//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "topology_map.h"

#include "segmented_vector.h"
#include "subdivision_plan_hlsl.h"
#include "../utils/buffer.h"

#include <opensubdiv/tmr/neighborhood.h>
#include <opensubdiv/tmr/topologyMap.h>

#include <algorithm>
#include <fstream>
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"

// Undefine min/max macros if they exist (Windows headers)
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

using namespace OpenSubdiv;

// Define stats namespace and TopologyMapStats structure
namespace stats {
    struct TopologyMapStats {
        uint32_t plansCount = 0;
        size_t plansByteSize = 0;
        uint32_t regularFacePlansCount = 0;
        uint32_t maxFaceSize = 0;
        uint32_t sharpnessCount = 0;
        float sharpnessMax = 0.f;
        uint32_t stencilCountMin = UINT32_MAX;
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0.f;
        std::vector<uint32_t> stencilCountHistogram;

        // Hash table statistics
        double pslMean = 0.0;
        uint32_t hashCount = 0;
        uint32_t addressCount = 0;
        double loadFactor = 0.0;
    };
}

TopologyMap::TopologyMap(
    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
    atopologyMap)
    : aTopologyMap(std::move(atopologyMap))
{
}

void TopologyMap::InitDeviceData(nvrhi::ICommandList* commandList, bool keepHostData)
{
    // Skip if already initialized (avoid recreating buffers every frame)
    if (plansBuffer) {
        return;
    }

    assert(aTopologyMap);

    Sdc::SchemeType schemeType = aTopologyMap->GetTraits().getSchemeType();
    Tmr::EndCapType endcapType = aTopologyMap->GetTraits().getEndCapType();

    const int numPlans = aTopologyMap->GetNumSubdivisionPlans();

    std::vector<SubdivisionPlanHLSL> gpuPlans(numPlans);

    segmented_vector<uint32_t> gpuTrees;
    gpuTrees.Reserve(numPlans);

    segmented_vector<int> gpuPatchPointIndexArrays;
    gpuPatchPointIndexArrays.Reserve(numPlans);

    segmented_vector<float> gpuStencilMatrixArrays;
    gpuStencilMatrixArrays.Reserve(numPlans);

    for (auto planIndex = 0; planIndex < numPlans; ++planIndex)
    {
        SubdivisionPlanHLSL& gpuPlan = gpuPlans[planIndex];
        // Data is set in the plan, but shader code references hardcodes these
        gpuPlan.scheme = (SchemeType)schemeType;
        gpuPlan.endCap = (EndCapType)endcapType;

        if (const auto* plan = aTopologyMap->GetSubdivisionPlan(planIndex))
        {
            const auto& tree_desc = plan->GetTreeDescriptor();
            gpuPlan.numControlPoints = tree_desc.GetNumControlPoints();
            gpuPlan.coarseFaceQuadrant = tree_desc.GetSubfaceIndex();
            gpuPlan.coarseFaceSize = tree_desc.GetFaceSize();

            gpuPatchPointIndexArrays.Append(plan->GetPatchPoints());
            gpuTrees.Append(plan->GetPatchTreeData());
            gpuStencilMatrixArrays.Append(plan->GetStencilMatrix());
        }
        else
        {
            gpuPlan.numControlPoints = 0;
            gpuPlan.coarseFaceSize = 4;
            gpuPlan.coarseFaceQuadrant = -1;
            static std::vector<int> empty{};
            gpuPatchPointIndexArrays.Append(empty);
            gpuTrees.Append(empty);
            gpuStencilMatrixArrays.Append(
                reinterpret_cast<std::vector<float> &>(empty));
        }
    }

    subpatchTreesArraysBuffer = CreateAndUploadBuffer<uint32_t>(gpuTrees.elements, "subpatch trees arrays", commandList);
    patchPointIndicesArraysBuffer = CreateAndUploadBuffer<int>(gpuPatchPointIndexArrays.elements, "patch point indices arrays", commandList);
    stencilMatrixArraysBuffer = CreateAndUploadBuffer<float>(gpuStencilMatrixArrays.elements, "stencil matrix arrays", commandList);

    for (auto i_plan = 0; i_plan < numPlans; ++i_plan)
    {
        gpuPlans[i_plan].treeOffset = gpuTrees.offsets[i_plan];
        gpuPlans[i_plan].treeSize = gpuTrees.sizes[i_plan];

        gpuPlans[i_plan].patchPointsOffset = gpuPatchPointIndexArrays.offsets[i_plan];
        gpuPlans[i_plan].patchPointsSize = gpuPatchPointIndexArrays.sizes[i_plan];

        gpuPlans[i_plan].stencilMatrixOffset = gpuStencilMatrixArrays.offsets[i_plan];
        gpuPlans[i_plan].stencilMatrixSize = gpuStencilMatrixArrays.sizes[i_plan];
    }

    plansBuffer = CreateAndUploadBuffer<SubdivisionPlanHLSL>(gpuPlans, "plans", commandList);

    // CPU-side plan data dump for debugging
    {
        static bool s_dumpedOnce = false;
        if (!s_dumpedOnce) {
            s_dumpedOnce = true;
            dxvk::Logger::warn(dxvk::str::format("TopologyMap: numPlans=", numPlans,
                " totalPatchPoints=", gpuPatchPointIndexArrays.elements.size(),
                " totalTrees=", gpuTrees.elements.size(),
                " totalStencils=", gpuStencilMatrixArrays.elements.size()));
            // Dump a few plans including plan 33
            for (int pi : {0, 1, 2, 33, 34}) {
                if (pi >= numPlans) continue;
                auto& p = gpuPlans[pi];
                dxvk::Logger::warn(dxvk::str::format("  Plan[", pi, "] numCP=", p.numControlPoints,
                    " faceSize=", p.coarseFaceSize, " quadrant=", p.coarseFaceQuadrant,
                    " treeOff=", p.treeOffset, " treeSz=", p.treeSize,
                    " ppOff=", p.patchPointsOffset, " ppSz=", p.patchPointsSize,
                    " stOff=", p.stencilMatrixOffset, " stSz=", p.stencilMatrixSize,
                    " scheme=", (int)p.scheme, " endCap=", (int)p.endCap));
                // Dump patchPointIndices for this plan
                if (p.patchPointsSize > 0) {
                    std::string ppStr = "    patchPoints[" + std::to_string(pi) + "]: ";
                    int count = std::min((int)p.patchPointsSize, 32);
                    for (int j = 0; j < count; j++) {
                        if (j > 0) ppStr += ",";
                        ppStr += std::to_string(gpuPatchPointIndexArrays.elements[p.patchPointsOffset + j]);
                    }
                    dxvk::Logger::warn(ppStr);
                }
                // Dump first 16 tree words for this plan
                if (p.treeSize > 0) {
                    std::string treeStr = "    tree[" + std::to_string(pi) + "]: ";
                    int count = std::min((int)p.treeSize, 20);
                    for (int j = 0; j < count; j++) {
                        if (j > 0) treeStr += ",";
                        char buf[16];
                        snprintf(buf, sizeof(buf), "0x%x", gpuTrees.elements[p.treeOffset + j]);
                        treeStr += buf;
                    }
                    dxvk::Logger::warn(treeStr);
                }
            }
        }
    }

    if (!keepHostData)
        aTopologyMap.reset();
}

stats::TopologyMapStats
TopologyMap::ComputeStatistics(const Tmr::TopologyMap& topologyMap,
    int histogramSize)
{
    using namespace OpenSubdiv;

    stats::TopologyMapStats stats;

    {
        auto hashStats = topologyMap.ComputeHashTableStatistics();

        stats.pslMean = hashStats.pslMean;
        stats.hashCount = hashStats.hashCount;
        stats.addressCount = hashStats.addressCount;
        stats.loadFactor = hashStats.loadFactor;
    }

    uint32_t nplans = static_cast<uint32_t>(topologyMap.GetNumSubdivisionPlans());

    if (nplans == 0)
        return stats;

    size_t stencilSum = 0;

    for (uint32_t planIndex = 0; planIndex < nplans; ++planIndex)
    {
        Tmr::SubdivisionPlan const* plan =
            topologyMap.GetSubdivisionPlan(planIndex);

        if (planIndex == 0 && !plan)
        {
            assert(Tmr::TopologyMap::kRegularPlanAtIndexZero);
            continue;
        }

        if (plan->IsRegularFace())
            ++stats.regularFacePlansCount;

        stats.maxFaceSize =
            std::max((uint32_t)plan->GetFaceSize(), stats.maxFaceSize);

        if (plan->GetNumNeighborhoods())
        {

            Tmr::Neighborhood const& n = plan->GetNeighborhood(0);

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if (bool hasSharpness = !(corners.empty() && creases.empty()))
            {
                ++stats.sharpnessCount;
                for (int i = 0; i < corners.size(); ++i)
                    stats.sharpnessMax = std::max(stats.sharpnessMax, (float)corners[i]);
                for (int i = 0; i < creases.size(); ++i)
                    stats.sharpnessMax = std::max(stats.sharpnessMax, (float)creases[i]);
            }
        }

        size_t nstencils = plan->GetNumStencils();

        stencilSum += nstencils;

        stats.stencilCountMin =
            std::min(stats.stencilCountMin, (uint32_t)nstencils);
        stats.stencilCountMax =
            std::max(stats.stencilCountMax, (uint32_t)nstencils);

        stats.plansByteSize += plan->GetByteSize(true);
    }

    stats.plansCount = nplans - (topologyMap.GetSubdivisionPlan(0) == nullptr);

    stats.stencilCountAvg = float(stencilSum) / float(stats.plansCount);

    // fill stencil counts histogram
    if (stats.stencilCountMin == stats.stencilCountMax)
    {
        // all the plans have the same number of stencils (ex. single cube)
        stats.stencilCountHistogram.push_back(stats.plansCount);
    }
    else
    {
        stats.stencilCountHistogram.resize(histogramSize);

        float delta =
            float(stats.stencilCountMax - stats.stencilCountMin) / histogramSize;

        for (uint32_t planIndex = 0; planIndex < nplans; ++planIndex)
        {
            Tmr::SubdivisionPlan const* plan =
                topologyMap.GetSubdivisionPlan(planIndex);

            if (planIndex == 0 && !plan)
                continue;

            size_t nstencils = plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor(
                float(nstencils - stats.stencilCountMin) / delta);

            ++stats.stencilCountHistogram[std::min(uint32_t(histogramSize - 1), i)];
        }
    }
    return stats;
}
