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

// Lightweight profiler implementation for RTX Remix integration
// Real profiling without donut/ImGui dependencies

#pragma once

#include <cstdint>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

// Forward declare ICommandList outside of stats namespace to avoid nesting
namespace nvrhi {
    class ICommandList;
}

namespace stats {

// GPU Timer - tracks elapsed time using CPU timestamps
// For true GPU timing, would integrate with DXVK's query pools
class GPUTimer {
public:
    void Start(nvrhi::ICommandList* commandList = nullptr) {
        // commandList parameter for future GPU query integration
        // For now, just use CPU timing
        (void)commandList;  // Suppress unused parameter warning
        m_startTime = std::chrono::high_resolution_clock::now();
        m_running = true;
    }

    void Stop() {
        if (m_running) {
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> elapsed = endTime - m_startTime;
            m_lastElapsed = elapsed.count();
            m_running = false;

            // Keep rolling history of last 60 samples
            m_history.push_back(m_lastElapsed);
            if (m_history.size() > 60) {
                m_history.erase(m_history.begin());
            }
        }
    }

    float GetElapsedTime() const {
        return m_lastElapsed;
    }

    float GetAverageElapsedTime() const {
        if (m_history.empty()) return 0.0f;
        return std::accumulate(m_history.begin(), m_history.end(), 0.0f) / m_history.size();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
    float m_lastElapsed = 0.0f;
    bool m_running = false;
    std::vector<float> m_history;
};

// Sampler - tracks statistical samples
template<typename T>
class Sampler {
public:
    std::string name;

    void Add(T value) {
        m_samples.push_back(value);
        if (m_samples.size() > 60) {
            m_samples.erase(m_samples.begin());
        }
    }

    T GetLatest() const {
        return m_samples.empty() ? T{} : m_samples.back();
    }

    T GetAverage() const {
        if (m_samples.empty()) return T{};
        return std::accumulate(m_samples.begin(), m_samples.end(), T{}) / static_cast<T>(m_samples.size());
    }

    T GetMax() const {
        if (m_samples.empty()) return T{};
        return *std::max_element(m_samples.begin(), m_samples.end());
    }

private:
    std::vector<T> m_samples;
};

// Cluster acceleration samplers
struct ClusterAccelSamplers {
    std::string name = "AccelBuilder";

    GPUTimer fillClustersTime;
    GPUTimer clusterTilingTime;
    GPUTimer buildClasTime;
    GPUTimer buildBlasTime;

    Sampler<uint32_t> numClusters;
    Sampler<uint32_t> numTriangles;

    struct {
        int x = 0;
        int y = 0;
    } renderSize;
};

// Global instance
extern ClusterAccelSamplers clusterAccelSamplers;

} // namespace stats
