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

// Slang-compatible version of shader_debug.h
// Uses a different approach - stores only predicate IDs in globals, passes buffer to each call
// This avoids the SPIR-V "pointer in global variable" limitation

#ifndef SHADER_DEBUG_SLANG_H
#define SHADER_DEBUG_SLANG_H

#define ENABLE_SHADER_DEBUG 0

// Slots 0..SHADER_DEBUG_RESERVED_SLOTS-1 are reserved for custom debug writes.
// ShaderDebug_AllocateSlot() starts allocating from this offset.
#define SHADER_DEBUG_RESERVED_SLOTS 16

// PayloadType constants (standalone for Slang compatibility)
static const uint PayloadType_None = 0;
static const uint PayloadType_Uint = 1;
static const uint PayloadType_Uint2 = 2;
static const uint PayloadType_Uint3 = 3;
static const uint PayloadType_Uint4 = 4;
static const uint PayloadType_Int = 5;
static const uint PayloadType_Int2 = 6;
static const uint PayloadType_Int3 = 7;
static const uint PayloadType_Int4 = 8;
static const uint PayloadType_Float = 9;
static const uint PayloadType_Float2 = 10;
static const uint PayloadType_Float3 = 11;
static const uint PayloadType_Float4 = 12;

struct ShaderDebugElement
{
    uint4 uintData;
    float4 floatData;
    uint payloadType;
    uint lineNumber;
    uint2 pad0;
};

#if ENABLE_SHADER_DEBUG

// Store only primitive types in globals (not buffer references)
static uint3 g_ShaderDebug_predicateID;
static uint3 g_ShaderDebug_currentID;

uint ShaderDebug_AllocateSlot(RWStructuredBuffer<ShaderDebugElement> output)
{
    uint bufferSize, bufferStride;
    output.GetDimensions(bufferSize, bufferStride);
    // Reserve first SHADER_DEBUG_RESERVED_SLOTS slots for custom debug writes.
    // ShaderDebug allocates in the remaining range [SHADER_DEBUG_RESERVED_SLOTS, bufferSize-1].
    uint availableSlots = bufferSize - SHADER_DEBUG_RESERVED_SLOTS;

    uint result;
    InterlockedAdd(output[0].payloadType, 1, result);
    return (result % availableSlots) + SHADER_DEBUG_RESERVED_SLOTS;
}

void ShaderDebug_WriteFloat4(RWStructuredBuffer<ShaderDebugElement> output, float4 value, uint lineNumber, uint payloadType, bool checkPredicate)
{
    if (!checkPredicate || all(g_ShaderDebug_predicateID == g_ShaderDebug_currentID))
    {
        ShaderDebugElement element = (ShaderDebugElement)0;
        element.payloadType = payloadType;
        element.lineNumber = lineNumber;
        element.floatData = value;
        element.uintData = uint4(0,0,0,0);
        output[ShaderDebug_AllocateSlot(output)] = element;
    }
}

void ShaderDebug_WriteUint4(RWStructuredBuffer<ShaderDebugElement> output, uint4 value, uint lineNumber, uint payloadType, bool checkPredicate)
{
    if (!checkPredicate || all(g_ShaderDebug_predicateID == g_ShaderDebug_currentID))
    {
        ShaderDebugElement element = (ShaderDebugElement)0;
        element.payloadType = payloadType;
        element.lineNumber = lineNumber;
        element.floatData = float4(0,0,0,0);
        element.uintData = value;
        output[ShaderDebug_AllocateSlot(output)] = element;
    }
}

// Overloaded functions for different types
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, uint4 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, value, lineNumber, PayloadType_Uint4, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, uint3 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0), lineNumber, PayloadType_Uint3, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, uint2 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0, 0), lineNumber, PayloadType_Uint2, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, uint value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0, 0, 0), lineNumber, PayloadType_Uint, checkPredicate);
}

void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, int4 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value), lineNumber, PayloadType_Int4, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, int3 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0), lineNumber, PayloadType_Int3, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, int2 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0, 0), lineNumber, PayloadType_Int2, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, int value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteUint4(output, uint4(value, 0, 0, 0), lineNumber, PayloadType_Int, checkPredicate);
}

void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, float4 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteFloat4(output, value, lineNumber, PayloadType_Float4, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, float3 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteFloat4(output, float4(value, 0), lineNumber, PayloadType_Float3, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, float2 value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteFloat4(output, float4(value, 0, 0), lineNumber, PayloadType_Float2, checkPredicate);
}
void ShaderDebug_Write(RWStructuredBuffer<ShaderDebugElement> output, float value, uint lineNumber, bool checkPredicate)
{
    ShaderDebug_WriteFloat4(output, float4(value, 0, 0, 0), lineNumber, PayloadType_Float, checkPredicate);
}

void InitShaderDebugger_Impl3(uint3 predicateID, uint3 currentID)
{
    g_ShaderDebug_predicateID = predicateID;
    g_ShaderDebug_currentID = currentID;
}

void InitShaderDebugger_Impl2(uint2 predicateID, uint2 currentID)
{
    g_ShaderDebug_predicateID = uint3(predicateID.x, predicateID.y, 0);
    g_ShaderDebug_currentID = uint3(currentID.x, currentID.y, 0);
}

void InitShaderDebugger_Impl1(uint predicateID, uint currentID)
{
    g_ShaderDebug_predicateID = uint3(predicateID, 0, 0);
    g_ShaderDebug_currentID = uint3(currentID, 0, 0);
}

// Macros that pass u_Debug buffer directly
// Note: These macros assume u_Debug is declared in the shader
#define SHADER_DEBUG(value) ShaderDebug_Write(u_Debug, value, __LINE__, true)
#define SHADER_DEBUG_FORCE(value) ShaderDebug_Write(u_Debug, value, __LINE__, false)
// Use _Generic-style overloading - caller must use correct suffix or we detect by argument count
// For Slang: just call the right version directly based on type
#define SHADER_DEBUG_INIT(outputBuffer, predicateID, currentID) _ShaderDebugInit(predicateID, currentID)

// Slang function overloading
void _ShaderDebugInit(uint3 predicateID, uint3 currentID) { InitShaderDebugger_Impl3(predicateID, currentID); }
void _ShaderDebugInit(uint2 predicateID, uint2 currentID) { InitShaderDebugger_Impl2(predicateID, currentID); }
void _ShaderDebugInit(uint predicateID, uint currentID) { InitShaderDebugger_Impl1(predicateID, currentID); }

#else
#define SHADER_DEBUG(value)
#define SHADER_DEBUG_FORCE(value)
#define SHADER_DEBUG_INIT(outputBuffer, predicateID, currentID)
#endif

#endif /* SHADER_DEBUG_SLANG_H */
