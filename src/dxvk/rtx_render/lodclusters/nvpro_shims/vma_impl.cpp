/*
* NV-DXVK: VMA implementation translation unit for the vendored nvvk
* allocator (nvpro_core2 defines VMA_IMPLEMENTATION in its application TU;
* here the lodclusters static library is the "application").
*
* VMA_STATIC_VULKAN_FUNCTIONS=0 / VMA_DYNAMIC_VULKAN_FUNCTIONS=1 are defined
* library-wide by the build so declarations match: nvvk's ResourceAllocator
* hands VMA volk's vkGetInstanceProcAddr (see resource_allocator.cpp), which
* resolves through dxvk's loaded vulkan-1.dll.
*/
#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
