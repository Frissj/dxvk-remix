/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Adapted from NVIDIA RTX Mega Geometry SDK for RTX Remix integration
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*/

#pragma once

#include "../../dxvk_buffer.h"
#include "../../../util/util_error.h"
#include "../../../util/rc/util_rc_ptr.h"
#include "../../../util/util_string.h"
#include "../../../util/log/log.h"
#include <vector>
#include <cstring>

namespace dxvk {

class DxvkDevice;

// DXVK buffer wrapper for RTXMG
// Simplified version of NVRHI RTXMGBuffer<T>
template<typename T>
class RtxmgBuffer {
private:
  Rc<DxvkBuffer> m_buffer;

public:
  using ElementType = T;

  RtxmgBuffer() = default;

  // Get underlying buffer
  Rc<DxvkBuffer> getBuffer() const { return m_buffer; }
  DxvkBuffer* ptr() const { return m_buffer.ptr(); }

  // Size queries
  size_t elementBytes() const { return sizeof(T); }
  uint32_t numElements() const {
    return m_buffer != nullptr ? uint32_t(m_buffer->info().size / sizeof(T)) : 0;
  }
  size_t bytes() const {
    return m_buffer != nullptr ? m_buffer->info().size : 0;
  }

  // GPU address
  VkDeviceAddress getDeviceAddress() const {
    return m_buffer != nullptr ? m_buffer->getDeviceAddress() : 0;
  }

  VkDeviceAddress deviceAddress() const {
    return getDeviceAddress();
  }

  // Check if buffer exists
  bool isValid() const { return m_buffer != nullptr; }

  // Release buffer
  void release() { m_buffer = nullptr; }

  // Create or resize buffer
  void create(DxvkDevice* device, size_t nElements, const char* name,
              VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    size_t requiredSize = nElements * sizeof(T);

    // Reuse existing buffer if large enough AND usage flags match
    if (m_buffer != nullptr && m_buffer->info().size >= requiredSize && m_buffer->info().usage == usage) {
      return;
    }

    // Create new buffer
    DxvkBufferCreateInfo info;
    info.size = requiredSize;
    info.usage = usage;
    info.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    info.access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    try {
      m_buffer = device->createBuffer(info, memFlags,
        DxvkMemoryStats::Category::RTXBuffer, name);

      if (m_buffer == nullptr) {
        throw DxvkError(str::format("Failed to create buffer '", name,
                                    "' (", requiredSize / (1024 * 1024), " MB)"));
      }
    } catch (const DxvkError& e) {
      Logger::err(str::format("[RTXMG] Buffer allocation failed: ", name,
                              " (", requiredSize / (1024 * 1024), " MB)"));
      throw;
    }
  }

  // Upload data to buffer
  void upload(const std::vector<T>& data) {
    if (m_buffer == nullptr || data.empty()) {
      return;
    }

    size_t uploadSize = std::min(data.size() * sizeof(T), m_buffer->info().size);
    void* mapped = m_buffer->mapPtr(0);
    if (mapped) {
      std::memcpy(mapped, data.data(), uploadSize);
    }
  }

  // Wrap existing DxvkBuffer
  void wrapExisting(const Rc<DxvkBuffer>& buffer) {
    m_buffer = buffer;
  }

  // Upload single element
  void uploadElement(const T& data, uint32_t index) {
    if (!m_buffer) {
      return;
    }

    void* mapped = m_buffer->mapPtr(index * sizeof(T));
    if (mapped) {
      std::memcpy(mapped, &data, sizeof(T));
    }
  }

  // Download data from buffer (requires host-visible memory)
  std::vector<T> download() const {
    std::vector<T> result;

    if (m_buffer == nullptr) {
      return result;
    }

    size_t numElems = m_buffer->info().size / sizeof(T);
    result.resize(numElems);

    const void* mapped = m_buffer->mapPtr(0);
    if (mapped) {
      std::memcpy(result.data(), mapped, numElems * sizeof(T));
    }

    return result;
  }

  // Clear buffer memory
  void clear() {
    if (!m_buffer) {
      return;
    }

    void* mapped = m_buffer->mapPtr(0);
    if (mapped) {
      std::memset(mapped, 0, m_buffer->info().size);
    }
  }
};

} // namespace dxvk
