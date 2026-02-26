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
#include "../../dxvk_sampler.h"

namespace dxvk {

  // NVRHI ISampler implementation wrapping DxvkSampler
  class NvrhiDxvkSampler : public nvrhi::ISampler {
  public:
    NvrhiDxvkSampler(const nvrhi::SamplerDesc& desc, const Rc<DxvkSampler>& dxvkSampler)
      : m_desc(desc)
      , m_dxvkSampler(dxvkSampler)
    {
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      // Note: ObjectType doesn't have VK_Sampler - return raw pointer
      nvrhi::Object obj;
      obj.pointer = (void*)(uintptr_t)m_dxvkSampler->handle();
      return obj;
    }

    // Adapter-specific accessors
    const Rc<DxvkSampler>& getDxvkSampler() const { return m_dxvkSampler; }
    const nvrhi::SamplerDesc& getDesc() const { return m_desc; }

  private:
    nvrhi::SamplerDesc m_desc;
    Rc<DxvkSampler> m_dxvkSampler;
  };

} // namespace dxvk
