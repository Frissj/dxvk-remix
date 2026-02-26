# RTX Mega Geometry (RTX MG) Integration
# Based on NVIDIA's RTX MG sample, adapted for RTX Remix

# =============================================================================
# meshoptimizer - Vendored Library (mesh clusterization + LOD DAG)
# =============================================================================

meshopt_sources = files([
  'meshoptimizer/allocator.cpp',
  'meshoptimizer/clusterizer.cpp',
  'meshoptimizer/clusterlod_impl.cpp',
  'meshoptimizer/indexanalyzer.cpp',
  'meshoptimizer/indexcodec.cpp',
  'meshoptimizer/indexgenerator.cpp',
  'meshoptimizer/overdrawoptimizer.cpp',
  'meshoptimizer/partition.cpp',
  'meshoptimizer/quantization.cpp',
  'meshoptimizer/rasterizer.cpp',
  'meshoptimizer/simplifier.cpp',
  'meshoptimizer/spatialorder.cpp',
  'meshoptimizer/stripifier.cpp',
  'meshoptimizer/vcacheoptimizer.cpp',
  'meshoptimizer/vertexcodec.cpp',
  'meshoptimizer/vertexfilter.cpp',
  'meshoptimizer/vfetchoptimizer.cpp',
])

meshopt_include = include_directories('meshoptimizer')

meshopt_lib = static_library('meshoptimizer', meshopt_sources,
  include_directories : meshopt_include,
  override_options : ['cpp_std=c++20']
)

meshopt_dep = declare_dependency(
  link_with : meshopt_lib,
  include_directories : meshopt_include
)

# =============================================================================
# RTX MG Core Modules
# =============================================================================

rtx_mg_include = include_directories('.')

rtx_mg_sources = files([
  # NVRHI Adapter (4 files - critical for DXVK integration)
  'nvrhi_adapter/nvrhi_dxvk_device.cpp',
  'nvrhi_adapter/nvrhi_dxvk_command_list.cpp',
  'nvrhi_adapter/nvrhi_scratch_manager.cpp',
  'nvrhi_adapter/donut_adapter.cpp',

  # High-Level Wrapper (1 file - RTX Remix integration point)
  'rtx_megageo_builder.cpp',

  # Cluster Builder (3 files)
  'cluster_builder/cluster_accel_builder.cpp',
  'cluster_builder/cluster_lod_builder.cpp',
  'cluster_builder/meshlet_template_builder.cpp',

  # HIZ (Hierarchical Z-Buffer) (2 files)
  'hiz/hiz_buffer.cpp',
  'hiz/zbuffer.cpp',

  # Scene (1 file - camera for tessellation)
  'scene/camera.cpp',

  # Utils (1 file - buffer helpers)
  'utils/buffer.cpp',

  # Profiler (1 file - stub implementation)
  'profiler/profiler_stub.cpp',
])

# Build RTX MG as a static library
# Note: Needs all DXVK dependencies including USD for RTX Remix headers
# Note: include_directories('../..') adds src/dxvk/ so dxvk_include.h can be found
rtx_mg_lib = static_library('rtx_mg', rtx_mg_sources,
  include_directories : [rtx_mg_include, include_directories('../..'), dxvk_include_path, remix_api_include_path, dxvk_shader_include_path],
  dependencies : [meshopt_dep, usd_dep],
  override_options : ['cpp_std=c++20']
)

# RTX MG Shaders - Cluster Builder
# Note: Actual shaders come from pre-compiled Slang versions via donut_adapter.cpp
rtx_mg_cluster_shaders = files([])

# RTX MG Shaders - HIZ
# Note: Actual shaders come from pre-compiled Slang versions in src/dxvk/shaders/rtx/pass/rtx_megageo/hiz/
rtx_mg_hiz_shaders = files([])

# RTX MG Dependency (to be used by main dxvk build)
rtx_mg_dep = declare_dependency(
  include_directories : rtx_mg_include,
  link_with : [rtx_mg_lib, meshopt_lib]
)

# Export sources and shaders to parent build
# These will be added to dxvk_src and shader lists in parent meson.build
