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

// Guard to prevent type redefinition with nvrhiHLSL.h
#define NVRHI_TYPES_ALREADY_DEFINED

#include "../../dxvk_buffer.h"
#include "../../dxvk_device.h"
#include "../../dxvk_context.h"

// Note: Shader math types (float2/3/4, uint2/3/4, float4x4) are provided by RTX Remix's MathLib
// (include/MathLib/MathLib_f.h) which is included via dxvk headers.
// IMPORTANT: MathLib float3 is 16 bytes (contains __m128 xmm for SSE), NOT 12 bytes!
// The sample uses donut::math::vector<float,3> which is 12 bytes (just {float x,y,z}).
// Slang GPU float3 with scalar layout is also 12 bytes.
// Any buffer indexed as float3 on the GPU must use 12-byte stride, not sizeof(float3)=16.
// See cluster_accels.h and cluster_accel_builder.cpp for the kGpuFloat3Stride pattern.

typedef uint32_t uint;

// Minimal NVRHI type definitions for RTX MG compatibility
// This provides just enough NVRHI API surface to run ClusterAccelBuilder

namespace nvrhi {

  // Buffer range specifier (offset + size)
  struct BufferRange {
    uint64_t byteOffset = 0;
    uint64_t byteSize = 0;

    BufferRange() = default;
    BufferRange(uint64_t offset, uint64_t size)
      : byteOffset(offset), byteSize(size) {}
  };

  // EntireBuffer constant (used as default parameter)
  static const BufferRange EntireBuffer = BufferRange(0, ~0ull);

  // Texture subresource types
  typedef uint32_t MipLevel;
  typedef uint32_t ArraySlice;

  // Texture subresource set
  struct TextureSubresourceSet {
    static constexpr MipLevel AllMipLevels = MipLevel(-1);
    static constexpr ArraySlice AllArraySlices = ArraySlice(-1);

    MipLevel baseMipLevel = 0;
    MipLevel numMipLevels = 1;
    ArraySlice baseArraySlice = 0;
    ArraySlice numArraySlices = 1;

    TextureSubresourceSet() = default;

    TextureSubresourceSet(MipLevel _baseMipLevel, MipLevel _numMipLevels, ArraySlice _baseArraySlice, ArraySlice _numArraySlices)
        : baseMipLevel(_baseMipLevel)
        , numMipLevels(_numMipLevels)
        , baseArraySlice(_baseArraySlice)
        , numArraySlices(_numArraySlices)
    {
    }

    bool operator ==(const TextureSubresourceSet& other) const {
      return baseMipLevel == other.baseMipLevel &&
             numMipLevels == other.numMipLevels &&
             baseArraySlice == other.baseArraySlice &&
             numArraySlices == other.numArraySlices;
    }
    bool operator !=(const TextureSubresourceSet& other) const { return !(*this == other); }
  };

  // AllSubresources constant
  static const TextureSubresourceSet AllSubresources = TextureSubresourceSet(0, TextureSubresourceSet::AllMipLevels, 0, TextureSubresourceSet::AllArraySlices);

  // Forward declarations
  class IResource;
  class IBuffer;
  class ITexture;
  class IShader;
  class ISampler;
  class IComputePipeline;
  class IBindingLayout;
  class IBindingSet;
  class ICommandList;
  class IDevice;

  // Smart pointers (ref-counted) with null-checking operator
  template<typename T>
  class RefCountPtr : public dxvk::Rc<T> {
  public:
    using dxvk::Rc<T>::Rc;
    RefCountPtr() : dxvk::Rc<T>() {}
    RefCountPtr(T* ptr) : dxvk::Rc<T>(ptr) {}
    RefCountPtr(const dxvk::Rc<T>& other) : dxvk::Rc<T>(other) {}

    // Null-check operator
    explicit operator bool() const { return this->ptr() != nullptr; }

    // Get raw pointer
    T* Get() const { return this->ptr(); }
  };

  using BufferHandle = RefCountPtr<IBuffer>;
  using TextureHandle = RefCountPtr<ITexture>;
  using ShaderHandle = RefCountPtr<IShader>;
  using SamplerHandle = RefCountPtr<ISampler>;
  using ComputePipelineHandle = RefCountPtr<IComputePipeline>;
  using BindingLayoutHandle = RefCountPtr<IBindingLayout>;
  using BindingSetHandle = RefCountPtr<IBindingSet>;
  using BindlessLayoutHandle = RefCountPtr<IBindingLayout>;
  using CommandListHandle = RefCountPtr<ICommandList>;
  using DeviceHandle = RefCountPtr<IDevice>;

  // GPU virtual address (Vulkan device address)
  using GpuVirtualAddress = VkDeviceAddress;

  // Forward declare nested namespaces before use
  namespace rt {
    namespace cluster {
      struct OperationParams;
      struct OperationSizeInfo;
      struct OperationDesc;
    }
  }

  // Format enum (subset of commonly used formats in RTX MG)
  enum class Format : uint32_t {
    UNKNOWN = 0,
    R32_FLOAT = VK_FORMAT_R32_SFLOAT,
    RGB32_FLOAT = VK_FORMAT_R32G32B32_SFLOAT,
    RGBA32_FLOAT = VK_FORMAT_R32G32B32A32_SFLOAT,
  };

  // Object types for getNativeObject()
  enum class ObjectType {
    VK_Buffer,
    VK_Image,
    VK_Device,
    VK_PhysicalDevice,
    VK_CommandBuffer,
    VK_ShaderModule,
  };

  // Object wrapper for native handles
  struct Object {
    void* pointer = nullptr;
    ObjectType type;
  };

  // CPU access modes for buffer mapping
  enum class CpuAccessMode {
    None,
    Read,
    Write,
    ReadWrite,
  };

  // Resource states (simplified - we let DXVK handle barriers automatically)
  enum class ResourceStates : uint32_t {
    Unknown = 0,
    Common = 1,
    ConstantBuffer = 2,
    VertexBuffer = 4,
    IndexBuffer = 8,
    IndirectArgument = 16,
    ShaderResource = 32,
    UnorderedAccess = 64,
    RenderTarget = 128,
    DepthWrite = 256,
    DepthRead = 512,
    CopyDest = 1024,
    CopySource = 2048,
    AccelStructRead = 4096,
    AccelStructWrite = 8192,
    AccelStructBuildInput = 16384,
  };

  inline ResourceStates operator|(ResourceStates a, ResourceStates b) {
    return static_cast<ResourceStates>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
  }

  inline ResourceStates operator&(ResourceStates a, ResourceStates b) {
    return static_cast<ResourceStates>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
  }

  // Buffer descriptor
  struct BufferDesc {
    uint64_t byteSize = 0;
    const char* debugName = nullptr;
    uint32_t structStride = 0;
    Format format = Format::UNKNOWN;
    CpuAccessMode cpuAccess = CpuAccessMode::None;  // CPU access mode for mapping
    bool isConstantBuffer = false;
    bool canHaveUAVs = false;
    bool canHaveTypedViews = false;
    bool canHaveRawViews = false;
    bool isDrawIndirectArgs = false;
    bool isVertexBuffer = false;
    bool isIndexBuffer = false;
    bool isAccelStructBuildInput = false;
    bool isAccelStructStorage = false;
    bool isVolatile = false;
    uint32_t maxVersions = 0;
    ResourceStates initialState = ResourceStates::Common;
    bool keepInitialState = false;
  };

  // Texture descriptor
  struct TextureDesc {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    uint32_t arraySize = 1;
    uint32_t mipLevels = 1;
    uint32_t sampleCount = 1;
    Format format = Format::UNKNOWN;
    bool isRenderTarget = false;
    bool isUAV = false;
    bool isTypeless = false;
    ResourceStates initialState = ResourceStates::Common;
    bool keepInitialState = false;
    const char* debugName = nullptr;
  };

  // Sampler address mode
  enum class SamplerAddressMode {
    Clamp,
    Wrap,
    Border,
    Mirror,
    ClampToEdge,
    MirrorOnce
  };

  // Sampler reduction type
  enum class SamplerReductionType {
    Standard,
    Comparison,
    Minimum,
    Maximum
  };

  // Sampler descriptor
  struct SamplerDesc {
    bool minFilter = true;
    bool magFilter = true;
    bool mipFilter = true;
    SamplerAddressMode addressU = SamplerAddressMode::Clamp;
    SamplerAddressMode addressV = SamplerAddressMode::Clamp;
    SamplerAddressMode addressW = SamplerAddressMode::Clamp;
    float mipBias = 0.0f;
    float maxAnisotropy = 1.0f;
    SamplerReductionType reductionType = SamplerReductionType::Standard;
    bool borderColor = false;
    const char* debugName = nullptr;

    void setAllFilters(bool filter) { minFilter = magFilter = mipFilter = filter; }
    void setAllAddressModes(SamplerAddressMode mode) { addressU = addressV = addressW = mode; }
  };

  // Shader type
  enum class ShaderType {
    Compute,
    Vertex,
    Hull,
    Domain,
    Geometry,
    Pixel,
    Amplification,
    Mesh,
    AllGraphics,
    AllRayTracing,
    All,
  };

  // Shader descriptor
  struct ShaderDesc {
    ShaderType shaderType = ShaderType::Compute;
    const char* debugName = nullptr;

    ShaderDesc() = default;
    ShaderDesc(ShaderType type) : shaderType(type) {}
  };

  // Compute pipeline descriptor
  struct ComputePipelineDesc {
    ShaderHandle computeShader;
    std::vector<BindingLayoutHandle> bindingLayouts;
    const char* debugName = nullptr;

    // Builder methods
    ComputePipelineDesc& setComputeShader(const ShaderHandle& shader) {
      computeShader = shader;
      return *this;
    }
    ComputePipelineDesc& addBindingLayout(const BindingLayoutHandle& layout) {
      bindingLayouts.push_back(layout);
      return *this;
    }
  };

  // Compute state
  struct ComputeState {
    ComputePipelineHandle pipeline;
    std::vector<BindingSetHandle> bindingSets;
    BufferHandle indirectParamsBuffer;

    // Builder methods
    ComputeState& setPipeline(const ComputePipelineHandle& p) {
      pipeline = p;
      return *this;
    }
    ComputeState& addBindingSet(const BindingSetHandle& set) {
      bindingSets.push_back(set);
      return *this;
    }
    ComputeState& setIndirectParams(const BufferHandle& buffer) {
      indirectParamsBuffer = buffer;
      return *this;
    }
  };

  // Binding layout descriptor
  struct BindingLayoutItem {
    uint32_t slot = 0;
    ShaderType type = ShaderType::All;
    uint32_t size = 1; // Array size
    enum class ResourceType {
      None,
      Texture_SRV,
      Texture_UAV,
      TypedBuffer_SRV,
      TypedBuffer_UAV,
      StructuredBuffer_SRV,
      StructuredBuffer_UAV,
      RawBuffer_SRV,
      RawBuffer_UAV,
      ConstantBuffer,
      Sampler,
      PushConstants,
    } resourceType = ResourceType::None;

    // Builder methods
    BindingLayoutItem& setSize(uint32_t s) { size = s; return *this; }
    BindingLayoutItem& setShaderType(ShaderType t) { type = t; return *this; }

    // Static factory methods
    static BindingLayoutItem Texture_SRV(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::Texture_SRV;
      return item;
    }
    static BindingLayoutItem Texture_UAV(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::Texture_UAV;
      return item;
    }
    static BindingLayoutItem StructuredBuffer_SRV(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::StructuredBuffer_SRV;
      return item;
    }
    static BindingLayoutItem StructuredBuffer_UAV(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::StructuredBuffer_UAV;
      return item;
    }
    static BindingLayoutItem ConstantBuffer(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::ConstantBuffer;
      return item;
    }
    static BindingLayoutItem Sampler(uint32_t slot) {
      BindingLayoutItem item;
      item.slot = slot;
      item.resourceType = ResourceType::Sampler;
      return item;
    }
  };

  struct BindingLayoutDesc {
    std::vector<BindingLayoutItem> bindings;
    ShaderType visibility = ShaderType::All;
    const char* debugName = nullptr;
    uint32_t registerSpace = 0;
    bool registerSpaceIsDescriptorSet = false;

    // Builder methods
    BindingLayoutDesc& addItem(const BindingLayoutItem& item) {
      bindings.push_back(item);
      return *this;
    }

    BindingLayoutDesc& setVisibility(ShaderType vis) {
      visibility = vis;
      return *this;
    }

    BindingLayoutDesc& setRegisterSpace(uint32_t value) {
      registerSpace = value;
      return *this;
    }

    BindingLayoutDesc& setRegisterSpaceIsDescriptorSet(bool value) {
      registerSpaceIsDescriptorSet = value;
      return *this;
    }
  };

  // Bindless layout descriptor
  struct BindlessLayoutDesc {
    enum class LayoutType {
      Immutable = 0,
      MutableSrvUavCbv,
      MutableCounters,
      MutableSampler
    };

    uint32_t firstSlot = 0;
    uint32_t maxCapacity = 0;
    ShaderType visibility = ShaderType::All;
    LayoutType layoutType = LayoutType::Immutable;
    bool registerSpaces = false;
    const char* debugName = nullptr;
  };

  // Binding set descriptor
  struct BindingSetItem {
    uint32_t slot = 0;
    uint32_t arrayElement = 0; // For array bindings
    enum class Type {
      None,
      ConstantBuffer,
      Texture_SRV,
      Texture_UAV,
      TypedBuffer_SRV,
      TypedBuffer_UAV,
      StructuredBuffer_SRV,
      StructuredBuffer_UAV,
      RawBuffer_SRV,
      RawBuffer_UAV,
      Sampler,
      PushConstants,
    } type = Type::None;

    IResource* resourceHandle = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0;
    Format format = Format::UNKNOWN;
    BufferRange range = EntireBuffer;

    // Builder methods
    BindingSetItem& setArrayElement(uint32_t element) { arrayElement = element; return *this; }
    BindingSetItem& setOffset(uint64_t off) { offset = off; return *this; }
    BindingSetItem& setSize(uint64_t sz) { size = sz; return *this; }

    // Static factory methods (accept both raw pointers and Handles)
    // Implementations are after class definitions so inheritance is known
    static BindingSetItem Texture_SRV(uint32_t slot, const TextureHandle& texture);
    static BindingSetItem Texture_UAV(uint32_t slot, const TextureHandle& texture);
    // Overloads for 2-parameter calls (matches sample code)
    static BindingSetItem StructuredBuffer_SRV(uint32_t slot, const BufferHandle& buffer);
    static BindingSetItem StructuredBuffer_UAV(uint32_t slot, const BufferHandle& buffer);
    static BindingSetItem ConstantBuffer(uint32_t slot, const BufferHandle& buffer);
    // Full parameter versions
    static BindingSetItem StructuredBuffer_SRV(uint32_t slot, const BufferHandle& buffer, Format format, BufferRange range);
    static BindingSetItem StructuredBuffer_UAV(uint32_t slot, const BufferHandle& buffer, Format format, BufferRange range);
    static BindingSetItem ConstantBuffer(uint32_t slot, const BufferHandle& buffer, BufferRange range);
    static BindingSetItem Sampler(uint32_t slot, const SamplerHandle& sampler);
  };

  struct BindingSetDesc {
    std::vector<BindingSetItem> bindings;

    // Builder method
    BindingSetDesc& addItem(const BindingSetItem& item) {
      bindings.push_back(item);
      return *this;
    }
  };

  // Base resource interface
  class IResource : public dxvk::RcObject {
  public:
    virtual ~IResource() = default;
    virtual Object getNativeObject(ObjectType type) = 0;
  };

  // IBuffer interface
  class IBuffer : public IResource {
  public:
    virtual ~IBuffer() = default;
    virtual const BufferDesc& getDesc() const = 0;
    virtual GpuVirtualAddress getGpuVirtualAddress() const = 0;
  };

  // ITexture interface
  class ITexture : public IResource {
  public:
    virtual ~ITexture() = default;
    virtual const TextureDesc& getDesc() const = 0;
  };

  // IShader interface
  class IShader : public IResource {
  public:
    virtual ~IShader() = default;
  };

  // ISampler interface
  class ISampler : public IResource {
  public:
    virtual ~ISampler() = default;
  };

  // IBindingLayout interface
  class IBindingLayout : public IResource {
  public:
    virtual ~IBindingLayout() = default;
  };

  // IBindingSet interface
  class IBindingSet : public IResource {
  public:
    virtual ~IBindingSet() = default;
  };

  // IComputePipeline interface
  class IComputePipeline : public IResource {
  public:
    virtual ~IComputePipeline() = default;
  };

  // Forward declare IDevice for ICommandList
  class IDevice;

  // Helper types for texture clearing (must be before ICommandList)
  struct Color {
    float r, g, b, a;
    Color(float r_ = 0.0f, float g_ = 0.0f, float b_ = 0.0f, float a_ = 0.0f)
      : r(r_), g(g_), b(b_), a(a_) {}
  };

  // ICommandList interface
  class ICommandList : public IResource {
  public:
    virtual ~ICommandList() = default;

    // Lifecycle
    virtual void open() = 0;
    virtual void close() = 0;
    virtual void clearState() = 0;

    // Buffer operations
    virtual void writeBuffer(IBuffer* buffer, const void* data, size_t size, uint64_t offset = 0) = 0;
    virtual void clearBufferUInt(IBuffer* buffer, uint32_t value) = 0;
    virtual void copyBuffer(IBuffer* dst, uint64_t dstOffset, IBuffer* src, uint64_t srcOffset, uint64_t size) = 0;

    // Helper overloads that accept Handle types
    void writeBuffer(const BufferHandle& buffer, const void* data, size_t size, uint64_t offset = 0) {
      writeBuffer(buffer.Get(), data, size, offset);
    }
    void clearBufferUInt(const BufferHandle& buffer, uint32_t value) {
      clearBufferUInt(buffer.Get(), value);
    }

    // Texture operations
    virtual void copyTexture(ITexture* dst, ITexture* src) = 0;
    virtual void clearTextureFloat(ITexture* texture, const TextureSubresourceSet& subresources, const Color& clearColor) = 0;

    // Compute
    virtual void setComputeState(const ComputeState& state) = 0;
    virtual void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;
    virtual void dispatchIndirect(IBuffer* buffer, uint64_t offset = 0) = 0;
    // Overload that uses the indirect params buffer from the current compute state
    virtual void dispatchIndirect(uint64_t offset) = 0;

    // Cluster operations
    virtual void executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) = 0;

    // Barriers
    virtual void bufferBarrier(IBuffer* buffer, ResourceStates stateBefore, ResourceStates stateAfter) = 0;
    virtual void textureBarrier(ITexture* texture, ResourceStates stateBefore, ResourceStates stateAfter) = 0;
    virtual void globalBarrier(ResourceStates stateBefore, ResourceStates stateAfter) = 0;

    // Device access
    virtual IDevice* getDevice() = 0;
  };

  // Command list parameters
  struct CommandListParameters {
    bool enableImmediateExecution = true;
  };

  // IDevice interface
  class IDevice : public IResource {
  public:
    virtual ~IDevice() = default;

    // Buffer management
    virtual BufferHandle createBuffer(const BufferDesc& desc) = 0;
    virtual void* mapBuffer(IBuffer* buffer, CpuAccessMode access) = 0;
    virtual void unmapBuffer(IBuffer* buffer) = 0;

    // Texture management
    virtual TextureHandle createTexture(const TextureDesc& desc) = 0;

    // Sampler management
    virtual SamplerHandle createSampler(const SamplerDesc& desc) = 0;

    // Shader/Pipeline
    virtual ShaderHandle createShader(const ShaderDesc& desc, const void* binary, size_t size) = 0;
    virtual ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) = 0;

    // Binding layouts
    virtual BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) = 0;
    virtual BindlessLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) = 0;
    virtual BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) = 0;

    // Helper overload that accepts Handle type
    BindingSetHandle createBindingSet(const BindingSetDesc& desc, const BindingLayoutHandle& layout) {
      return createBindingSet(desc, layout.Get());
    }

    // Create a descriptor table (binding set) for bindless resources.
    // This creates an empty binding set that satisfies pipeline binding requirements
    // when a bindless layout is used. The returned binding set can be passed to
    // addBindingSet() in compute/graphics state.
    virtual BindingSetHandle createDescriptorTable(IBindingLayout* layout) = 0;

    // Helper overload that accepts Handle type
    BindingSetHandle createDescriptorTable(const BindlessLayoutHandle& layout) {
      return createDescriptorTable(layout.Get());
    }

    // Cluster operations
    virtual rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) = 0;

    // Command list creation
    virtual CommandListHandle createCommandList(const CommandListParameters& params = {}) = 0;

    // Command list execution and synchronization
    virtual uint64_t executeCommandList(ICommandList* commandList) = 0;
    virtual void waitForIdle() = 0;
  };

  // NVRHI utility namespace
  namespace utils {
    // Scoped debug marker for command lists
    class ScopedMarker {
    public:
      ScopedMarker(ICommandList* commandList, const char* name)
        : m_commandList(commandList), m_name(name) {
        // In DXVK/Vulkan, debug markers are typically handled by the backend
        // For now, this is a no-op placeholder
      }

      ~ScopedMarker() {
        // End marker
      }

    private:
      ICommandList* m_commandList;
      const char* m_name;
    };

    // Helper for texture UAV barriers
    inline void TextureUavBarrier(ICommandList* commandList, ITexture* texture) {
      // In Vulkan, UAV barriers are handled automatically by DXVK
      // This is a no-op for now as DXVK tracks resource states
    }

    // Helper to create both binding layout and binding set from a binding set descriptor
    inline bool CreateBindingSetAndLayout(
        IDevice* device,
        ShaderType shaderType,
        uint32_t registerSpace,
        const BindingSetDesc& bindingSetDesc,
        BindingLayoutHandle& outBindingLayout,
        BindingSetHandle& outBindingSet) {

      // Create binding layout if it doesn't exist yet
      if (!outBindingLayout) {
        BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = shaderType;

        // Convert binding set items to binding layout items
        for (const auto& item : bindingSetDesc.bindings) {
          BindingLayoutItem layoutItem;
          layoutItem.slot = item.slot;

          // Map BindingSetItem::Type to BindingLayoutItem::ResourceType
          switch (item.type) {
            case BindingSetItem::Type::ConstantBuffer:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::ConstantBuffer;
              break;
            case BindingSetItem::Type::Texture_SRV:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::Texture_SRV;
              break;
            case BindingSetItem::Type::Texture_UAV:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::Texture_UAV;
              break;
            case BindingSetItem::Type::StructuredBuffer_SRV:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::StructuredBuffer_SRV;
              break;
            case BindingSetItem::Type::StructuredBuffer_UAV:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::StructuredBuffer_UAV;
              break;
            case BindingSetItem::Type::Sampler:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::Sampler;
              break;
            default:
              layoutItem.resourceType = BindingLayoutItem::ResourceType::None;
              break;
          }

          layoutDesc.bindings.push_back(layoutItem);
        }

        outBindingLayout = device->createBindingLayout(layoutDesc);
        if (!outBindingLayout) {
          return false;
        }
      }

      // Create the binding set
      outBindingSet = device->createBindingSet(bindingSetDesc, outBindingLayout);
      return outBindingSet != nullptr;
    }

    // Overload that accepts DeviceHandle
    inline bool CreateBindingSetAndLayout(
        const DeviceHandle& device,
        ShaderType shaderType,
        uint32_t registerSpace,
        const BindingSetDesc& bindingSetDesc,
        BindingLayoutHandle& outBindingLayout,
        BindingSetHandle& outBindingSet) {
      return CreateBindingSetAndLayout(device.Get(), shaderType, registerSpace,
                                        bindingSetDesc, outBindingLayout, outBindingSet);
    }
  }

  // BindingSetItem factory method implementations (after class definitions so inheritance is known)
  inline BindingSetItem BindingSetItem::Texture_SRV(uint32_t slot, const TextureHandle& texture) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::Texture_SRV;
    item.resourceHandle = static_cast<IResource*>(texture.Get());
    return item;
  }

  inline BindingSetItem BindingSetItem::Texture_UAV(uint32_t slot, const TextureHandle& texture) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::Texture_UAV;
    item.resourceHandle = static_cast<IResource*>(texture.Get());
    return item;
  }

  // Overload with 2 parameters (slot, buffer) - matches sample code
  inline BindingSetItem BindingSetItem::StructuredBuffer_SRV(uint32_t slot, const BufferHandle& buffer) {
    return StructuredBuffer_SRV(slot, buffer, Format::UNKNOWN, BufferRange());
  }

  inline BindingSetItem BindingSetItem::StructuredBuffer_SRV(uint32_t slot, const BufferHandle& buffer, Format format, BufferRange range) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::StructuredBuffer_SRV;
    item.resourceHandle = static_cast<IResource*>(buffer.Get());
    item.format = format;
    item.range = range;
    return item;
  }

  // Overload with 2 parameters (slot, buffer) - matches sample code
  inline BindingSetItem BindingSetItem::StructuredBuffer_UAV(uint32_t slot, const BufferHandle& buffer) {
    return StructuredBuffer_UAV(slot, buffer, Format::UNKNOWN, BufferRange());
  }

  inline BindingSetItem BindingSetItem::StructuredBuffer_UAV(uint32_t slot, const BufferHandle& buffer, Format format, BufferRange range) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::StructuredBuffer_UAV;
    item.resourceHandle = static_cast<IResource*>(buffer.Get());
    item.format = format;
    item.range = range;
    return item;
  }

  // Overload with 2 parameters (slot, buffer) - matches sample code
  inline BindingSetItem BindingSetItem::ConstantBuffer(uint32_t slot, const BufferHandle& buffer) {
    return ConstantBuffer(slot, buffer, BufferRange());
  }

  inline BindingSetItem BindingSetItem::ConstantBuffer(uint32_t slot, const BufferHandle& buffer, BufferRange range) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::ConstantBuffer;
    item.resourceHandle = static_cast<IResource*>(buffer.Get());
    item.range = range;
    return item;
  }

  inline BindingSetItem BindingSetItem::Sampler(uint32_t slot, const SamplerHandle& sampler) {
    BindingSetItem item;
    item.slot = slot;
    item.type = Type::Sampler;
    item.resourceHandle = static_cast<IResource*>(sampler.Get());
    return item;
  }

  // Cluster acceleration structure types (VK_NV_cluster_acceleration_structure)
  namespace rt::cluster {

    // CLAS byte alignment requirement
    static const uint32_t kClasByteAlignment = 128;

    enum class OperationType {
      ClasBuildTemplates,        // Build reusable topology templates
      ClasInstantiateTemplates,  // Instantiate templates with vertex data
      BlasBuild,                 // Build BLAS from CLASes
    };

    enum class OperationMode {
      GetSizes,                  // Query memory requirements
      ExplicitDestinations,      // App provides output buffers
      ImplicitDestinations,      // Driver allocates
    };

    // CRITICAL: These are bitmask values, NOT sequential enum values!
    // Must match VkClusterAccelerationStructureIndexFormatFlagBitsNV
    enum class OperationIndexFormat : uint32_t {
      IndexFormat8bit = 0x1,   // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
      IndexFormat16bit = 0x2,  // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV
      IndexFormat32bit = 0x4   // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV
    };

    enum class OperationFlags : uint8_t {
      None = 0x0,
      FastTrace = 0x1,
      FastBuild = 0x2,
      NoOverlap = 0x4,
      AllowOMM = 0x8
    };

    // GPU virtual address with stride - matches VkStridedDeviceAddressNV
    // NOTE: Vulkan uses VkDeviceSize (8 bytes) for strideInBytes, not uint32_t!
    struct GpuVirtualAddressAndStride {
      GpuVirtualAddress startAddress = 0;   // 8 bytes
      uint64_t strideInBytes = 0;           // 8 bytes (VkDeviceSize)
    };

    // Indirect instantiate template arguments - matches VkClusterAccelerationStructureInstantiateClusterInfoNV
    // CRITICAL: Layout must exactly match the Vulkan structure (32 bytes total)
    struct IndirectInstantiateTemplateArgs {
      uint32_t clusterIdOffset = 0;           // 4 bytes - Offset added to clusterId in template
      uint32_t geometryIndexOffsetPacked = 0; // 4 bytes - Lower 24 bits: geometryIndexOffset, Upper 8 bits: reserved (must be 0)
      GpuVirtualAddress clusterTemplate = 0;  // 8 bytes - Address of cluster template
      GpuVirtualAddressAndStride vertexBuffer; // 16 bytes - Vertex buffer for instantiation
    };
    static_assert(sizeof(GpuVirtualAddressAndStride) == 16, "GpuVirtualAddressAndStride must be 16 bytes to match VkStridedDeviceAddressNV");
    static_assert(sizeof(IndirectInstantiateTemplateArgs) == 32, "IndirectInstantiateTemplateArgs must be 32 bytes to match VkClusterAccelerationStructureInstantiateClusterInfoNV");

    // Indirect BLAS from CLAS arguments
    // Maps to VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV in Vulkan
    struct IndirectArgs {
      uint32_t clusterCount = 0;              // Number of CLASes (clusterReferencesCount in Vulkan)
      uint32_t clusterReferencesStride = 0;   // Stride between VkDeviceAddress elements in clusterAddresses array (must be 8)
      GpuVirtualAddress clusterAddresses = 0; // Address of CLAS address array (clusterReferences in Vulkan)
    };
    static_assert(sizeof(IndirectArgs) == 16, "IndirectArgs must be 16 bytes to match VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV");

    // Clone of NVAPI_D3D12_RAYTRACING_ACCELERATION_STRUCTURE_MULTI_INDIRECT_TRIANGLE_TEMPLATE_ARGS
    // Must match exactly for Vulkan cluster operations
    struct IndirectTriangleTemplateArgs
    {
      uint32_t          clusterId;                         // The user specified cluster Id to encode in the cluster template
      uint32_t          clusterFlags;                      // Values of cluster flags
      uint32_t          triangleCount : 9;                 // The number of triangles used by the cluster template (max 256)
      uint32_t          vertexCount : 9;                   // The number of vertices used by the cluster template (max 256)
      uint32_t          positionTruncateBitCount : 6;      // The number of bits to truncate from the position values
      uint32_t          indexFormat : 4;                   // The index format to use for the indexBuffer
      uint32_t          opacityMicromapIndexFormat : 4;    // The index format to use for the opacityMicromapIndexBuffer
      uint32_t          baseGeometryIndexAndFlags;         // The base geometry index (lower 24 bit) and base geometry flags
      uint16_t          indexBufferStride;                 // The stride of the elements of indexBuffer, in bytes
      uint16_t          vertexBufferStride;                // The stride of the elements of vertexBuffer, in bytes
      uint16_t          geometryIndexAndFlagsBufferStride; // The stride of the elements of geometryIndexBuffer, in bytes
      uint16_t          opacityMicromapIndexBufferStride;  // The stride of the elements of opacityMicromapIndexBuffer, in bytes
      GpuVirtualAddress indexBuffer;                       // The index buffer to construct the cluster template
      GpuVirtualAddress vertexBuffer;                      // (optional) The vertex buffer to optimize the cluster template
      GpuVirtualAddress geometryIndexAndFlagsBuffer;       // (optional) Address of an array of 32bit geometry indices
      GpuVirtualAddress opacityMicromapArray;              // (optional) Address of a valid OMM array
      GpuVirtualAddress opacityMicromapIndexBuffer;        // (optional) Address of an array of indices into the OMM array
      GpuVirtualAddress instantiationBoundingBoxLimit;     // (optional) Pointer to 6 floats for bounding box limits
    };
    static_assert(sizeof(IndirectTriangleTemplateArgs) == 72, "IndirectTriangleTemplateArgs must be 72 bytes to match VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV");

    // Cluster geometry parameters
    struct ClasGeometryParams {
      VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
      uint32_t vertexStride = 12;
      uint32_t maxGeometryIndex = 0;
      uint32_t maxUniqueGeometryCount = 0;
      uint32_t maxTriangleCount = 0;
      uint32_t maxVertexCount = 0;
      uint32_t maxTotalTriangleCount = 0;
      uint32_t maxTotalVertexCount = 0;
      uint32_t minPositionTruncateBitCount = 0;
    };

    // BLAS parameters
    struct BlasParams {
      uint32_t maxClasPerBlasCount = 0;
      uint32_t maxTotalClasCount = 0;
    };

    // Operation parameters
    struct OperationParams {
      uint32_t maxArgCount = 0;
      OperationType type = OperationType::ClasBuildTemplates;
      OperationMode mode = OperationMode::GetSizes;
      OperationFlags flags = OperationFlags::None;
      ClasGeometryParams clas;
      BlasParams blas;
    };

    // Size query result
    struct OperationSizeInfo {
      uint64_t resultMaxSizeInBytes = 0;
      uint64_t scratchSizeInBytes = 0;
    };

    // Operation descriptor
    struct OperationDesc {
      OperationParams params;

      uint64_t scratchSizeInBytes = 0;                        // Size of scratch resource

      // Input Resources
      IBuffer* inIndirectArgCountBuffer = nullptr;            // Buffer containing the number of AS to build
      uint64_t inIndirectArgCountOffsetInBytes = 0;           // Offset to where the count is
      IBuffer* inIndirectArgsBuffer = nullptr;                // Buffer of descriptor array
      uint64_t inIndirectArgsOffsetInBytes = 0;               // Offset to where the descriptor array starts

      // In/Out Resources
      IBuffer* inOutAddressesBuffer = nullptr;                // Array of addresses of CLAS, CLAS Templates, or BLAS
      uint64_t inOutAddressesOffsetInBytes = 0;               // Offset to where the addresses array starts

      // Output Resources
      IBuffer* outSizesBuffer = nullptr;                      // Sizes (in bytes) of CLAS, CLAS Templates, or BLAS
      uint64_t outSizesOffsetInBytes = 0;                     // Offset to where the output sizes array starts
      IBuffer* outAccelerationStructuresBuffer = nullptr;     // Destination buffer for CLAS, CLAS Template, or BLAS data
      uint64_t outAccelerationStructuresOffsetInBytes = 0;    // Offset to where the output acceleration structures start

      // Backward compatibility aliases for old nvrhi adapter code
      uint32_t maxArgCount = 0;                               // Alias for params.maxArgCount
      IBuffer* outResultBuffer = nullptr;                     // Alias for outAccelerationStructuresBuffer
      IBuffer* outScratchBuffer = nullptr;                    // Scratch buffer
    };

  } // namespace rt::cluster

  // Ray tracing instance descriptor (shader-friendly)
  namespace rt {

    struct IndirectInstanceDesc {
#ifdef __cplusplus
      float transform[12];
#else
      float4 transform[3];
#endif
      uint32_t instanceID : 24;
      uint32_t instanceMask : 8;
      uint32_t instanceContributionToHitGroupIndex : 24;
      uint32_t flags : 8;
      GpuVirtualAddress blasDeviceAddress;
    };

  } // namespace rt

  // Utils namespace for helper functions
  namespace utils {
    // Helper to create volatile constant buffer descriptors
    inline BufferDesc CreateVolatileConstantBufferDesc(size_t byteSize, const char* debugName, uint32_t maxVersions) {
      BufferDesc desc;
      desc.byteSize = byteSize;
      desc.debugName = debugName;
      desc.isConstantBuffer = true;
      desc.isVolatile = true;
      desc.maxVersions = maxVersions;
      desc.initialState = ResourceStates::ConstantBuffer;
      desc.keepInitialState = true;
      return desc;
    }

    // Helper to convert dxvk::Matrix4 to column-major float array (for shaders)
    inline void affineToColumnMajor(const dxvk::Matrix4& matrix, float* outArray) {
      // dxvk::Matrix4 is row-major in memory: data[row][col]
      // We need column-major output for shaders
      for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
          outArray[col * 4 + row] = matrix.data[row][col];
        }
      }
    }

    // Helper to convert dxvk::Matrix4 to affine transform (3x4 column-major)
    inline void matrixToAffineColumnMajor(const dxvk::Matrix4& matrix, float* outArray12) {
      // Extract 3x4 affine part (ignore last row which is typically [0,0,0,1])
      // Output as column-major for shaders
      for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 3; ++row) {
          outArray12[col * 3 + row] = matrix.data[row][col];
        }
      }
    }

    // Helper to convert Matrix4 to float3x4 (used by cluster tiling params)
    // float3x4 is 3 rows of float4, row-major layout
    struct float3x4_helper {
      float data[3][4];
    };

    inline void matrixToFloat3x4(const dxvk::Matrix4& matrix, float3x4_helper& out) {
      // Extract upper 3x4 part of the matrix (affine transform)
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
          out.data[row][col] = matrix.data[row][col];
        }
      }
    }

  } // namespace utils

} // namespace nvrhi

// Global helper function for backward compatibility with sample code
inline void affineToColumnMajor(const dxvk::Matrix4& matrix, float* outData) {
  // This converts to float3x4 row-major layout (3 rows x 4 columns)
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      outData[row * 4 + col] = matrix.data[row][col];
    }
  }
}

// Utility function: divide and round up (ceiling division)
inline uint32_t div_ceil(uint32_t numerator, uint32_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// Donut adapter namespace - provides compatibility layer for donut-based code
namespace donut {
  namespace engine {
    // Constants
    static constexpr uint32_t c_MaxRenderPassConstantBufferVersions = 16;

    // Forward declarations from RTX Remix
    class RtxShaderManager;
  }
}

// Forward declaration for RTX Remix context
namespace dxvk {
  class RtxContext;
}

namespace donut {
  namespace engine {

    // Forward declaration of ShaderMacro
    struct ShaderMacro;

    // Adapter for shader compilation - wraps RTX Remix's RtxShaderManager
    class ShaderFactory {
    public:
      ShaderFactory(dxvk::RtxContext* ctx);
      ~ShaderFactory();

      // Creates a shader using RTX Remix's shader system
      nvrhi::ShaderHandle CreateShader(const char* path, const char* entryPoint,
                                       const std::vector<ShaderMacro>* macros,
                                       nvrhi::ShaderType shaderType);

      // Overload that accepts ShaderDesc
      nvrhi::ShaderHandle CreateShader(const char* path, const char* entryPoint,
                                       const std::vector<ShaderMacro>* macros,
                                       const nvrhi::ShaderDesc& desc) {
        return CreateShader(path, entryPoint, macros, desc.shaderType);
      }

      // Get the HiZ descriptor set layout (for binding set 1)
      // Returns VK_NULL_HANDLE if not yet created
      VkDescriptorSetLayout getHiZDescriptorSetLayout() const { return m_hiZDescriptorSetLayout; }

    private:
      dxvk::RtxContext* m_rtxContext = nullptr;
      VkDevice m_vkDevice = VK_NULL_HANDLE;
      VkDescriptorSetLayout m_hiZDescriptorSetLayout = VK_NULL_HANDLE;  // HiZ textures for descriptor set 1

      // Creates the HiZ descriptor set layout (called once on first use)
      void ensureHiZDescriptorSetLayout();
    };

    // Adapter for common render passes - minimal implementation for RTX Remix
    class CommonRenderPasses {
    public:
      CommonRenderPasses(nvrhi::DeviceHandle device) : m_device(device) {
        // Create default samplers
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.minFilter = true;
        samplerDesc.magFilter = true;
        samplerDesc.mipFilter = true;
        samplerDesc.addressU = nvrhi::SamplerAddressMode::Clamp;
        samplerDesc.addressV = nvrhi::SamplerAddressMode::Clamp;
        samplerDesc.addressW = nvrhi::SamplerAddressMode::Clamp;
        m_LinearClampSampler = m_device->createSampler(samplerDesc);
      }

      nvrhi::DeviceHandle GetDevice() const { return m_device; }

      nvrhi::SamplerHandle m_LinearClampSampler;

    private:
      nvrhi::DeviceHandle m_device;
    };

  } // namespace engine
} // namespace donut

// Descriptor table adapter types
namespace nvrhi {
  // DescriptorTableHandle - for bindless descriptor management
  using DescriptorTableHandle = uint32_t;

  static constexpr DescriptorTableHandle c_InvalidDescriptorHandle = ~0u;
}
