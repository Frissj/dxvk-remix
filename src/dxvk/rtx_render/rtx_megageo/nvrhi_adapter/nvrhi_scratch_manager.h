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

#include <list>
#include <memory>
#include "../../dxvk_buffer.h"

namespace dxvk {

  class DxvkDevice;

  // Buffer chunk for scratch allocations (matches NVRHI's BufferChunk)
  struct ScratchBufferChunk {
    static constexpr uint64_t c_sizeAlignment = 65536;  // 64KB alignment for chunks

    Rc<DxvkBuffer> buffer;
    uint64_t bufferSize = 0;
    uint64_t writePointer = 0;
    uint64_t version = 0;  // Tracks when chunk was used (for GPU completion checking)
  };

  // Scratch buffer manager (matches NVRHI's UploadManager in scratch mode)
  // Manages a pool of buffer chunks that can be suballocated for scratch memory
  class ScratchManager {
  public:
    ScratchManager(DxvkDevice* device, uint64_t defaultChunkSize, uint64_t memoryLimit = 0)
      : m_device(device)
      , m_defaultChunkSize(defaultChunkSize)
      , m_memoryLimit(memoryLimit)
    {}

    // Suballocate scratch memory from the pool
    // Returns true if allocation succeeded
    // pBuffer receives the buffer containing the allocation
    // pOffset receives the offset within the buffer
    bool suballocateBuffer(
      uint64_t size,
      Rc<DxvkBuffer>* pBuffer,
      uint64_t* pOffset,
      uint32_t alignment = 256);

    // Called when command list is submitted - moves current chunk to pool
    void submitChunks(uint64_t currentVersion, uint64_t submittedVersion);

    // Called at start of frame - reset write pointers so chunks can be reused
    // DXVK handles GPU sync at frame boundaries, so previous frame's chunks are safe
    void resetWritePointers();

  private:
    std::shared_ptr<ScratchBufferChunk> createChunk(uint64_t size);

    static uint64_t align(uint64_t value, uint64_t alignment) {
      return (value + alignment - 1) & ~(alignment - 1);
    }

    DxvkDevice* m_device;
    uint64_t m_defaultChunkSize;
    uint64_t m_memoryLimit;
    uint64_t m_allocatedMemory = 0;

    std::shared_ptr<ScratchBufferChunk> m_currentChunk;
    std::list<std::shared_ptr<ScratchBufferChunk>> m_chunkPool;
  };

} // namespace dxvk
