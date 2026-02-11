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

#include "nvrhi_scratch_manager.h"
#include "../../dxvk_device.h"
#include "../../../util/log/log.h"

#include "../rtxmg_log.h"
#if RTXMG_LOG_NVRHI_SCRATCH_MANAGER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

namespace dxvk {

  std::shared_ptr<ScratchBufferChunk> ScratchManager::createChunk(uint64_t size) {
    auto chunk = std::make_shared<ScratchBufferChunk>();

    DxvkBufferCreateInfo info;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                  VK_ACCESS_TRANSFER_WRITE_BIT;

    chunk->buffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXAccelerationStructure, "ScratchBufferChunk");
    chunk->bufferSize = size;
    chunk->writePointer = 0;
    chunk->version = 0;

    m_allocatedMemory += size;
    RTXMG_LOG(str::format("RTX MegaGeo: ScratchManager created chunk, size=", size,
                             " totalAllocated=", m_allocatedMemory));

    return chunk;
  }

  bool ScratchManager::suballocateBuffer(
    uint64_t size,
    Rc<DxvkBuffer>* pBuffer,
    uint64_t* pOffset,
    uint32_t alignment)
  {
    std::shared_ptr<ScratchBufferChunk> chunkToRetire;

    // Try to fit in current chunk
    if (m_currentChunk) {
      uint64_t alignedOffset = align(m_currentChunk->writePointer, (uint64_t)alignment);
      uint64_t endOfDataInChunk = alignedOffset + size;

      if (endOfDataInChunk <= m_currentChunk->bufferSize) {
        m_currentChunk->writePointer = endOfDataInChunk;
        *pBuffer = m_currentChunk->buffer;
        *pOffset = alignedOffset;
        return true;
      }

      // Current chunk is full, retire it
      chunkToRetire = m_currentChunk;
      m_currentChunk.reset();
    }

    // Look for a free chunk in the pool that's big enough
    // Simplified: chunks become free when resetWritePointers() is called (at start of frame)
    for (auto it = m_chunkPool.begin(); it != m_chunkPool.end(); ++it) {
      std::shared_ptr<ScratchBufferChunk> chunk = *it;

      // Chunk is free (writePointer reset) and big enough
      if (chunk->writePointer == 0 && chunk->bufferSize >= size) {
        m_chunkPool.erase(it);
        m_currentChunk = chunk;
        break;
      }
    }

    // Retire the old chunk to the pool
    if (chunkToRetire) {
      m_chunkPool.push_back(chunkToRetire);
    }

    // Need to create a new chunk
    if (!m_currentChunk) {
      uint64_t sizeToAllocate = align(std::max(size, m_defaultChunkSize), ScratchBufferChunk::c_sizeAlignment);

      if (m_memoryLimit > 0 && m_allocatedMemory + sizeToAllocate > m_memoryLimit) {
        Logger::err(str::format("RTX MegaGeo: ScratchManager memory limit exceeded, need=", sizeToAllocate,
                                " allocated=", m_allocatedMemory, " limit=", m_memoryLimit));
        return false;
      }

      m_currentChunk = createChunk(sizeToAllocate);
      if (!m_currentChunk) {
        Logger::err("RTX MegaGeo: ScratchManager failed to create chunk");
        return false;
      }
    }

    m_currentChunk->writePointer = size;

    *pBuffer = m_currentChunk->buffer;
    *pOffset = 0;
    return true;
  }

  void ScratchManager::submitChunks(uint64_t currentVersion, uint64_t submittedVersion) {
    // Move current chunk to pool
    if (m_currentChunk) {
      m_chunkPool.push_back(m_currentChunk);
      m_currentChunk.reset();
    }
  }

  void ScratchManager::resetWritePointers() {
    // Called at start of frame - reset all chunk write pointers so they can be reused
    // DXVK handles GPU synchronization at frame boundaries, so chunks from previous frame are safe to reuse
    for (auto& chunk : m_chunkPool) {
      chunk->writePointer = 0;
    }
    if (m_currentChunk) {
      m_currentChunk->writePointer = 0;
    }
  }

} // namespace dxvk
