/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_draw_call_cache.h"
#include "../d3d9/d3d9_state.h"

namespace dxvk 
{

namespace {
  bool isSky(CameraType::Enum t) {
    return t == CameraType::Sky;
  }

  bool exactMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    if (isSky(drawCall.cameraType) != isSky(blas.input.cameraType)) {
      return false;
    }

    return drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash()
        && drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>() == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
        && drawCall.getSkinningState().boneHash == blas.input.getSkinningState().boneHash;
  }

  // Relaxed match for ClusterBlas entries: material + camera type only.
  // FullGeometryHash changes every frame for D3D9 dynamic vertex buffers,
  // but MegaGeo handles vertex data separately via subdivision surfaces.
  bool clusterBlasMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    if (isSky(drawCall.cameraType) != isSky(blas.input.cameraType)) {
      return false;
    }
    return drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash();
  }
}

DrawCallCache::DrawCallCache(DxvkDevice* device) : CommonDeviceObject(device) {
  m_entries.reserve(1024);
}
DrawCallCache::~DrawCallCache() {}

DrawCallCache::CacheState DrawCallCache::get(const DrawCallState& drawCall, BlasEntry** out) {
  // First, find the right bucket:
  const XXH64_hash_t hash = drawCall.getGeometryData().getHashForRule<rules::TopologicalHash>();
  auto range = m_entries.equal_range(hash);
  if (range.first == m_entries.end()) {
    // New bucket
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  // Handle buckets with 1 entry:
  auto iter = range.first;
  iter++;
  if (iter == range.second) {
    // Only 1 element
    BlasEntry& entry = range.first->second;
    const bool updatedThisFrame = entry.frameLastTouched == m_device->getCurrentFrameId();

    // ClusterBlas: relaxed matching by material + camera type only.
    // FullGeometryHash changes every frame for D3D9 dynamic VBs, so exactMatch
    // would always fail and recreate BlasEntries every frame.
    // Allow matching even when updatedThisFrame: D3D9 games submit the same mesh
    // in multiple render passes (depth, color, shadow). Without this, each pass
    // creates a duplicate BlasEntry → overlapping CLAS → z-fighting/flickering.
    // The InstanceManager handles per-position dedup via findSimilarInstance.
    if (entry.isClusterBlas()) {
      if (clusterBlasMatch(drawCall, entry)) {
        m_clusterHits++;
        *out = &entry;
        return CacheState::kExisted;
      } else {
        m_clusterMisses++;
        *out = allocateEntry(hash, drawCall);
        return CacheState::kNew;
      }
    }
    m_nonClusterCalls++;
    const bool vertexDataMatches = entry.input.getGeometryData().getHashForRule<rules::VertexDataHash>() == drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
    const bool boneHashesMatch = entry.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    const bool materialHashesMatch = entry.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();

    if (exactMatch(drawCall, entry) || !updatedThisFrame && (vertexDataMatches && boneHashesMatch || materialHashesMatch)) {
      // Exact vertex match that is reusable for the current draw call,
      // or something that hasn't been updated this frame and is similar enough.
      // Matching the logic in the multi-element loop below.
      *out = &entry;
      return CacheState::kExisted;
    } else {
      // First frame of having two mismatching instances, and the first instance has already 
      // been paired with the existing BlasEntry.
      *out = allocateEntry(hash, drawCall);
      return CacheState::kNew;
    }
  }

  // Bucket has multiple BlasEntries

  float bestScore = std::numeric_limits<float>::min();
  Matrix4 newTransform = drawCall.getTransformData().objectToWorld;
  const Vector3 newWorldPosition = drawCall.getGeometryData().boundingBox.getTransformedCentroid(newTransform);

  for (auto bucketIter = range.first; bucketIter != range.second; bucketIter++) {
    BlasEntry& blas  = bucketIter->second;
    if (exactMatch(drawCall, blas)) {
      *out = &blas;
      return CacheState::kExisted;
    }
    // ClusterBlas: relaxed match by material + proximity (no vertex hash).
    // Allow matching even when updatedThisFrame — D3D9 multi-pass rendering
    // submits the same mesh multiple times per frame.
    if (blas.isClusterBlas()) {
      if (!clusterBlasMatch(drawCall, blas)) {
        continue;
      }
      float score = 1000.f; // material matched
      Matrix4 oldTransform = blas.input.getTransformData().objectToWorld;
      const Vector3 worldPosition = blas.input.getGeometryData().boundingBox.getTransformedCentroid(oldTransform);
      score -= lengthSqr(newWorldPosition - worldPosition);
      if (score > bestScore) {
        bestScore = score;
        *out = &blas;
      }
      continue;
    }
    if (blas.frameLastTouched == m_device->getCurrentFrameId()) {
      continue;
    }
    // TODO these heuristics could use more refinement.
    float score = 0;
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexPosition] == drawCall.getGeometryData().hashes[HashComponents::VertexPosition] &&
        blas.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash) {
      score += 1000.f;
    }
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexTexcoord] == drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord]) {
      score += 1000.f;
    }
    if (blas.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash()) {
      score += 1000.f;
    }
    // TODO this is only checking the distance to the first instance that created the BlasEntry, not to
    // each instance.  It also doesn't include the portal logic from InstanceManager.
    Matrix4 oldTransform = blas.input.getTransformData().objectToWorld;
    const Vector3 worldPosition = blas.input.getGeometryData().boundingBox.getTransformedCentroid(oldTransform);
    score -= lengthSqr(newWorldPosition - worldPosition);
    if (score > bestScore) {
      bestScore = score;
      *out = &blas;
    }
  }
  if (*out == nullptr) {
    // Failed to find similar blas, so allocate a new one
    m_clusterMisses++;
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  m_clusterMultiHits++;
  return CacheState::kExisted;

}

void DrawCallCache::logFrameStats() {
  // Per-frame cache hit/miss stats for ClusterBlas entries
  static uint32_t s_frameCount = 0;
  s_frameCount++;
  if (m_clusterHits + m_clusterMisses + m_clusterMultiHits > 0) {
    Logger::info(str::format("RTX MegaGeo CACHE[", s_frameCount, "]: ",
        "clusterHits=", m_clusterHits,
        " clusterMultiHits=", m_clusterMultiHits,
        " clusterMisses=", m_clusterMisses,
        " nonCluster=", m_nonClusterCalls,
        " totalEntries=", m_entries.size()));
  }
  m_clusterHits = 0;
  m_clusterMisses = 0;
  m_clusterMultiHits = 0;
  m_nonClusterCalls = 0;
}

BlasEntry* DrawCallCache::allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall) {
  auto iter = m_entries.emplace(hash, drawCall);
  BlasEntry* result = &iter->second;
  result->frameCreated = m_device->getCurrentFrameId();
  return result;
}

}  // namespace nvvk
