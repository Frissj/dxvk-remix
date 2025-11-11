/*
* GPU-side geometry hashing for mega geometry cache
* Hashes vertex position data directly on GPU to avoid CPU readback
*/

#include "rtx/utility/common.slangh"

layout(binding = 0)
StructuredBuffer<uint> VertexPositions;

layout(binding = 1)
StructuredBuffer<uint> Indices;

layout(binding = 2)
RWStructuredBuffer<uint> HashOutput;

struct PushConstants {
  uint vertexCount;
  uint indexCount;
  uint positionStride;  // In floats (e.g., 3 for vec3, 4 for padded vec4)
  uint hasIndices;
  uint positionBufferOffset;  // Byte offset into position buffer
  uint indexBufferOffset;     // Byte offset into index buffer
  uint indexIs16Bit;          // 1 if indices are uint16, 0 if uint32
};

layout(push_constant)
ConstantBuffer<PushConstants> cb;

// FNV-1a 64-bit constants (constructed from two 32-bit values)
static const uint64_t FNV_OFFSET_BASIS = (uint64_t(0xcbf29ce4) << 32) | uint64_t(0x84222325);
static const uint64_t FNV_PRIME = (uint64_t(0x00000100) << 32) | uint64_t(0x000001b3);

// Simple 64-bit hash function (FNV-1a variant)
uint64_t hash64(uint64_t hash, uint value) {
  hash ^= uint64_t(value);
  hash *= FNV_PRIME;
  return hash;
}

groupshared uint64_t localHashes[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_GroupThreadID, uint3 globalId : SV_DispatchThreadID) {
  uint tid = threadId.x;
  uint gid = globalId.x;

  uint64_t myHash = FNV_OFFSET_BASIS;

  // Hash vertex positions (only x,y,z components, skip padding)
  // StructuredBuffer<uint> uses uint indices, so divide byte offsets by 4
  uint verticesPerThread = (cb.vertexCount + 63) / 64;
  for (uint i = 0; i < verticesPerThread; i++) {
    uint vertexIndex = gid + i * 64;
    if (vertexIndex < cb.vertexCount) {
      // Convert byte offset to uint index (divide by 4)
      uint uintIndex = cb.positionBufferOffset / 4 + vertexIndex * cb.positionStride;

      // Hash x, y, z as raw bits
      uint x = VertexPositions[uintIndex + 0];
      uint y = VertexPositions[uintIndex + 1];
      uint z = VertexPositions[uintIndex + 2];

      myHash = hash64(myHash, x);
      myHash = hash64(myHash, y);
      myHash = hash64(myHash, z);
    }
  }

  // Hash indices if present
  if (cb.hasIndices != 0) {
    uint indicesPerThread = (cb.indexCount + 63) / 64;
    for (uint i = 0; i < indicesPerThread; i++) {
      uint indexIdx = gid + i * 64;
      if (indexIdx < cb.indexCount) {
        uint indexValue;
        if (cb.indexIs16Bit != 0) {
          // 16-bit indices: packed two per uint
          uint uintIndex = cb.indexBufferOffset / 4 + indexIdx / 2;
          uint dword = Indices[uintIndex];
          // Extract correct 16-bit value
          indexValue = (indexIdx & 1) ? (dword >> 16) : (dword & 0xFFFF);
        } else {
          // 32-bit indices
          uint uintIndex = cb.indexBufferOffset / 4 + indexIdx;
          indexValue = Indices[uintIndex];
        }
        myHash = hash64(myHash, indexValue);
      }
    }
  }

  // Store local hash
  localHashes[tid] = myHash;
  GroupMemoryBarrierWithGroupSync();

  // Parallel reduction to combine all hashes
  for (uint stride = 32; stride > 0; stride >>= 1) {
    if (tid < stride) {
      localHashes[tid] = hash64(localHashes[tid], uint(localHashes[tid + stride]));
    }
    GroupMemoryBarrierWithGroupSync();
  }

  // Thread 0 writes final result
  if (tid == 0) {
    // Store 64-bit hash as two 32-bit values
    uint64_t finalHash = localHashes[0];
    HashOutput[0] = uint(finalHash & 0xFFFFFFFF);
    HashOutput[1] = uint(finalHash >> 32);
  }
}
