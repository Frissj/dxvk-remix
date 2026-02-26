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

#include "nvrhi_types.h"
#include "../../dxvk_shader.h"

namespace dxvk {

  // NVRHI IShader implementation wrapping DxvkShader
  class NvrhiDxvkShader : public nvrhi::IShader {
  public:
    NvrhiDxvkShader(const Rc<DxvkShader>& dxvkShader)
      : m_dxvkShader(dxvkShader)
    {
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      if (type == nvrhi::ObjectType::VK_ShaderModule) {
        nvrhi::Object obj;
        obj.pointer = m_dxvkShader.ptr();
        obj.type = nvrhi::ObjectType::VK_ShaderModule;
        return obj;
      }
      return nvrhi::Object();
    }

    // Adapter-specific methods
    const Rc<DxvkShader>& getDxvkShader() const { return m_dxvkShader; }

  private:
    Rc<DxvkShader> m_dxvkShader;
  };

  // Helper function to wrap DxvkShader in nvrhi::ShaderHandle
  inline nvrhi::ShaderHandle wrapShader(const Rc<DxvkShader>& dxvkShader) {
    if (dxvkShader.ptr() == nullptr)
      return nullptr;
    return new NvrhiDxvkShader(dxvkShader);
  }

} // namespace dxvk
