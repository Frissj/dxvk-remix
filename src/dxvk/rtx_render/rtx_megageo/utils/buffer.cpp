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

#include "buffer.h"
#include "../../../dxvk_include.h"

#include "../rtxmg_log.h"
#undef RTXMG_LOG
#if RTXMG_LOG_BUFFER
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

nvrhi::BufferDesc GetGenericDesc(size_t nElements, uint32_t elementSize, const char* name, nvrhi::Format format)
{
    nElements = std::max(1ull, nElements);
    // Vulkan's vkCmdUpdateBuffer requires size to be a multiple of 4
    // Round up buffer size to avoid alignment issues during writes
    size_t byteSize = nElements * elementSize;
    size_t alignedByteSize = (byteSize + 3) & ~3;  // Round up to multiple of 4
    return nvrhi::BufferDesc{
        .byteSize = alignedByteSize,
        .debugName = name,
        .structStride = elementSize,
        .format = format,
        .canHaveUAVs = true,
        .canHaveTypedViews = true,
        .canHaveRawViews = true,
        .initialState = nvrhi::ResourceStates::UnorderedAccess,
        .keepInitialState = true
    };
}

nvrhi::BufferDesc GetReadbackDesc(const nvrhi::BufferDesc& desc)
{
    // CRITICAL: Must set cpuAccess to Read for the buffer to be mappable
    return nvrhi::BufferDesc{
        .byteSize = desc.byteSize,
        .debugName = "Readback Buffer",
        .format = desc.format,
        .cpuAccess = nvrhi::CpuAccessMode::Read,
        .initialState = nvrhi::ResourceStates::CopyDest,
        .keepInitialState = true
    };
}

void DownloadBuffer(nvrhi::IBuffer* src, void* dest, nvrhi::IBuffer* staging, bool async, nvrhi::ICommandList* commandList)
{
    size_t numBytes = src->getDesc().byteSize;
    size_t stagingBytes = staging->getDesc().byteSize;

    // Validate buffer sizes to prevent heap corruption
    if (stagingBytes < numBytes) {
        dxvk::Logger::err(dxvk::str::format("RTX MegaGeo: DownloadBuffer ERROR - staging buffer too small! staging=", stagingBytes, " src=", numBytes));
        return;
    }

    RTXMG_LOG(dxvk::str::format("RTX MegaGeo: DownloadBuffer entry, numBytes=", numBytes, " stagingBytes=", stagingBytes));

    RTXMG_LOG("RTX MegaGeo: DownloadBuffer - copyBuffer");
    commandList->copyBuffer(staging, 0, src, 0, numBytes);

    if (!async)
    {
        // For synchronous downloads, we must wait for the copy to complete
        // DXVK's mapPtr() can only synchronize if commands have been submitted,
        // and flushCommandList() alone doesn't submit to GPU - it just prepares them.
        // So we need waitForIdle() here to ensure the copy completes before mapping.
        RTXMG_LOG("RTX MegaGeo: DownloadBuffer - close");
        commandList->close();
        RTXMG_LOG("RTX MegaGeo: DownloadBuffer - executeCommandList");
        commandList->getDevice()->executeCommandList(commandList);
        RTXMG_LOG("RTX MegaGeo: DownloadBuffer - waitForIdle (required for sync download)");
        commandList->getDevice()->waitForIdle();
    }

    RTXMG_LOG("RTX MegaGeo: DownloadBuffer - mapBuffer");
    void* mappedBuffer = commandList->getDevice()->mapBuffer(staging, nvrhi::CpuAccessMode::Read);
    RTXMG_LOG(dxvk::str::format("RTX MegaGeo: DownloadBuffer - mappedBuffer=", mappedBuffer));
    if (mappedBuffer)
        memcpy(dest, mappedBuffer, numBytes);
    else
        memset(dest, 0, numBytes);
    RTXMG_LOG("RTX MegaGeo: DownloadBuffer - unmapBuffer");
    commandList->getDevice()->unmapBuffer(staging);

    if (!async)
    {
        // Reopen command list for subsequent commands
        RTXMG_LOG("RTX MegaGeo: DownloadBuffer - open");
        commandList->open();
    }
    RTXMG_LOG("RTX MegaGeo: DownloadBuffer - complete");
}