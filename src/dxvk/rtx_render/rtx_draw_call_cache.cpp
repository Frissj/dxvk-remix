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

#include <mutex>

namespace dxvk 
{

namespace {
  bool exactMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    auto isSky = [](CameraType::Enum t) {
      return t == CameraType::Sky;
    };

    if (isSky(drawCall.cameraType) != isSky(blas.input.cameraType)) {
      return false;
    }

    return drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash()
        && drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>() == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
        && drawCall.getSkinningState().boneHash == blas.input.getSkinningState().boneHash;
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
    const bool vertexDataMatches = entry.input.getGeometryData().getHashForRule<rules::VertexDataHash>() == drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
    const bool boneHashesMatch = entry.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    const bool materialHashesMatch = entry.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();

    // World-position guard on the loose material-ONLY reuse (see below): a material
    // match with MISMATCHED vertex data rebinds this one entry to any same-material
    // draw in the topological bucket, ignoring where it sits in the world. For
    // promotion-CAPTURED geometry that is the LEGO class-churn root: many placements
    // of one mesh at different baked (anisotropic) scales share a topological hash, so
    // across frames the single entry ping-pongs between placements. Its per-BlasEntry
    // capture (the promotion solve's positionBuffer, refreshed only by the first draw
    // each frame) then flips content, capSig jumps (e.g. 124 -> 0.25), the shape class
    // churns, and the building pops Path A<->B. Require the incoming draw to sit at
    // (nearly) the cached entry's world position before allowing a material-only
    // rebind, so each placement keeps its OWN stable entry/capture. Gated to captured
    // draws so all other geometry keeps its exact prior behavior; the vertexData-match
    // branch (a rigid instance that merely MOVED, verts unchanged) is untouched.
    bool materialMatchAllowed = materialHashesMatch;
    if (materialHashesMatch && !vertexDataMatches && drawCall.preCaptureVertexData != nullptr
        && drawCall.getGeometryData().boundingBox.isValid()) {
      const Matrix4& newXf = drawCall.getTransformData().objectToWorld;
      const AxisAlignedBoundingBox& newBox = drawCall.getGeometryData().boundingBox;
      const Matrix4& oldXf = entry.input.getTransformData().objectToWorld;
      const AxisAlignedBoundingBox& oldBox = entry.input.getGeometryData().boundingBox;
      const Vector3 newCentroid = newBox.getTransformedCentroid(newXf);
      const Vector3 oldCentroid = oldBox.getTransformedCentroid(oldXf);
      const Vector3 newWorldMin = (newXf * Vector4(newBox.minPos, 1.0f)).xyz();
      const Vector3 newWorldMax = (newXf * Vector4(newBox.maxPos, 1.0f)).xyz();
      const float newDiagSq = lengthSqr(newWorldMax - newWorldMin);
      const float deltaSq = lengthSqr(newCentroid - oldCentroid);
      // POSITION gate: a rigid instance moves << its own size per frame (stays within one
      // world-space bbox diagonal); a placement elsewhere is many diagonals away.
      const bool positionOk = deltaSq <= newDiagSq;
      // SCALE gate: LEGO places one mesh at different BAKED (anisotropic) scales at the
      // (near) same spot - same topo hash + material, different content. The position gate
      // is blind to these (identical centroid), yet the shared entry's promotion capture
      // still flips scale between them -> the solve fits the OLD reference to new-scale
      // content -> residual spikes -> a promoted building DEMOTES and, solved forever
      // against a mismatched class reference, never recovers (the promote->demote->stuck
      // symptom). Split on a world-space SIZE mismatch too. diagSq ~ size^2 ~ capSig, so a
      // >1.5x world-size step is a different placement; <=1.5x absorbs the per-frame
      // pooled-capture jitter of ONE placement (observed fine capSig steps were ~1.1-1.15x,
      // the placement flip ~1.8x).
      bool scaleOk = true;
      if (oldBox.isValid()) {
        const Vector3 oldWorldMin = (oldXf * Vector4(oldBox.minPos, 1.0f)).xyz();
        const Vector3 oldWorldMax = (oldXf * Vector4(oldBox.maxPos, 1.0f)).xyz();
        const float oldDiagSq = lengthSqr(oldWorldMax - oldWorldMin);
        const float lo = newDiagSq < oldDiagSq ? newDiagSq : oldDiagSq;
        const float hi = newDiagSq < oldDiagSq ? oldDiagSq : newDiagSq;
        scaleOk = hi <= 1.5f * lo;
      }
      materialMatchAllowed = positionOk && scaleOk;
      if (!materialMatchAllowed) {
        // [CacheSplit] the guard rejected a material-only rebind of a captured entry to a
        // different placement (position far and/or scale mismatch) -> allocateEntry gives
        // it a fresh, content-stable BlasEntry, so one entry's promotion capture no longer
        // flips between placements. Throttled per topo-bucket per 10 frames.
        static std::mutex s_csMutex;
        static std::unordered_map<XXH64_hash_t, uint32_t> s_csLast;
        const uint32_t frame = m_device->getCurrentFrameId();
        std::lock_guard<std::mutex> lk(s_csMutex);
        uint32_t& last = s_csLast[hash];
        if (last == 0u || frame - last > 10u) {
          last = frame;
          Logger::info(str::format("[CacheSplit] captured topo 0x", std::hex, hash, std::dec,
                                   " reject ", (positionOk ? "" : "POS "), (scaleOk ? "" : "SCALE "),
                                   "(deltaSq ", deltaSq, " newDiagSq ", newDiagSq,
                                   ") - split to a new stable BlasEntry (frame ", frame, ")"));
        }
      }
    }

    if (exactMatch(drawCall, entry) || !updatedThisFrame && (vertexDataMatches && boneHashesMatch || materialMatchAllowed)) {
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
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  return CacheState::kExisted;

}

BlasEntry* DrawCallCache::allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall) {
  auto iter = m_entries.emplace(hash, drawCall);
  BlasEntry* result = &iter->second;
  result->frameCreated = m_device->getCurrentFrameId();
  return result;
}

}  // namespace nvvk
