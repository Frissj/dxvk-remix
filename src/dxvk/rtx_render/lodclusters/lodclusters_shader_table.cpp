/*
* NV-DXVK: prebuilt shader variant lookup for the cluster LOD system.
*
* The vk_lod_clusters / vk_animated_clusters samples compile their GLSL
* kernels at runtime with shaderc, passing config macros via
* CompileOptions::AddMacroDefinition. In Remix the same kernels are compiled
* at BUILD time by compile_shaders.py into one SPIR-V blob per //!variant
* combination (see the //!variant annotations in
* shaders/rtx/pass/lodclusters/*.comp).
*
* This file resolves the sample's original (file name, macro set) request to
* the matching prebuilt blob, so every NVIDIA shader-init call site stays
* byte-identical. Matching policy:
*   - Each file has SELECTOR macros (exactly the axes of its //!variant
*     matrix, in declaration order). The requested value of each selector
*     picks the variant; a selector missing from the request uses its first
*     (default) value, mirroring the axis "-" defaults.
*   - Macros in the request that are NOT selectors were compiled as fixed
*     values chosen conservatively (e.g. HAS_ALPHA_TEST=1, USE_RENDER_STATS=1,
*     SUBGROUP_SIZE=32). These are intentionally ignored here; the fixed
*     values are correct supersets of the sample's optimized specializations,
*     and hard device constraints (subgroup size 32) are enforced by the
*     ClusterLodManager support gate.
*
* The variant enumeration for matrix files comes from the X-macro headers
* compile_shaders.py generates (<base>_variants.h), so shader-side matrix
* changes flow into this table mechanically.
*/

#include <cstring>
#include <iterator>
#include <string>

#include <nvutils/logger.hpp>
#include <nvvkglsl/glsl.hpp>

// Matrix-variant headers (generated at build time; each also includes all of
// its variants' SPIR-V array headers).
#include <rtx_shaders/blas_clusters_insert_variants.h>
#include <rtx_shaders/build_setup_variants.h>
#include <rtx_shaders/instance_classify_lod_variants.h>
#include <rtx_shaders/stream_update_clas_geometry_indices_variants.h>
#include <rtx_shaders/stream_update_scene_variants.h>
#include <rtx_shaders/traversal_blas_merging_variants.h>
#include <rtx_shaders/traversal_init_variants.h>
#include <rtx_shaders/traversal_init_blas_sharing_variants.h>
#include <rtx_shaders/traversal_run_variants.h>
#include <rtx_shaders/traversal_run_groups_variants.h>
#include <rtx_shaders/cluster_blas_instances_variants.h>
#include <rtx_shaders/nvhiz_update_variants.h>

// Explicit-variant headers.
#include <rtx_shaders/geometry_blas_sharing.h>
#include <rtx_shaders/geometry_blas_sharing_strm.h>
#include <rtx_shaders/geometry_blas_sharing_strm_merge.h>
#include <rtx_shaders/geometry_blas_sharing_strm_cache.h>
#include <rtx_shaders/geometry_blas_sharing_strm_merge_cache.h>
#include <rtx_shaders/instance_assign_blas.h>
#include <rtx_shaders/instance_assign_blas_strm.h>
#include <rtx_shaders/instance_assign_blas_share.h>
#include <rtx_shaders/instance_assign_blas_share_strm.h>
#include <rtx_shaders/instance_assign_blas_share_strm_cache.h>

// Single-variant headers.
#include <rtx_shaders/blas_setup_insertion.h>
#include <rtx_shaders/blas_caching_setup_build.h>
#include <rtx_shaders/blas_caching_setup_copy.h>
#include <rtx_shaders/traversal_presort.h>
#include <rtx_shaders/stream_setup.h>
#include <rtx_shaders/stream_agefilter_groups.h>
#include <rtx_shaders/stream_compaction_new_clas.h>
#include <rtx_shaders/stream_compaction_old_clas.h>
#include <rtx_shaders/stream_allocator_build_freegaps.h>
#include <rtx_shaders/stream_allocator_freegaps_insert.h>
#include <rtx_shaders/stream_allocator_setup_insertion.h>
#include <rtx_shaders/stream_allocator_unload_groups.h>
#include <rtx_shaders/stream_allocator_load_groups.h>
// P4: Remix-authored HiZ source conversion (game depth -> reversed-Z)
#include <rtx_shaders/remix_depth_flip.h>
// P4c: Remix-authored rigid-capture promotion solve/gate/patch (plan 7.7)
#include <rtx_shaders/promotion_solve.h>
// P4b: Remix-authored per-pose vertex gather (global -> cluster-local order;
// the animated CLAS build's razor-triangle fix)
#include <rtx_shaders/anim_gather_positions.h>

namespace lodclusters {

namespace {

constexpr uint32_t kMaxSelectors = 6;

struct ShaderSelector {
  // Macro name in the sample's AddMacroDefinition call.
  const char* macro;
  // Accepted value strings; position = value index. First entry is the
  // default used when the request does not define the macro.
  const char* values[3];
};

struct ShaderVariantEntry {
  uint8_t values[kMaxSelectors];
  const uint32_t* data;
  size_t size;
};

struct ShaderFileTable {
  const char* fileName;  // the sample's original file name incl. ".glsl"
  const ShaderSelector* selectors;
  uint32_t selectorCount;
  const ShaderVariantEntry* entries;
  uint32_t entryCount;
};

#define SPV(sym) sym, sizeof(sym)

// -- Common selector definitions ---------------------------------------------

constexpr ShaderSelector kSelStreaming   = { "USE_STREAMING", { "0", "1" } };
constexpr ShaderSelector kSelMerging     = { "USE_BLAS_MERGING", { "0", "1" } };
constexpr ShaderSelector kSelCaching     = { "USE_BLAS_CACHING", { "0", "1" } };
constexpr ShaderSelector kSelSharing     = { "USE_BLAS_SHARING", { "0", "1" } };
constexpr ShaderSelector kSelCulling     = { "USE_CULLING", { "0", "1" } };
constexpr ShaderSelector kSelSorting     = { "USE_SORTING", { "0", "1" } };
constexpr ShaderSelector kSelPersistent  = { "USE_PERSISTENT_TRAVERSAL_KERNEL", { "0", "1" } };
constexpr ShaderSelector kSelFiCulling   = { "USE_FORCED_INVISIBLE_CULLING", { "0", "1" } };
constexpr ShaderSelector kSelRaster      = { "TARGETS_RASTERIZATION", { "0", "1" } };
constexpr ShaderSelector kSelClusterVtx  = { "CLUSTER_VERTEX_COUNT", { "128", "64" } };
constexpr ShaderSelector kSelClusterTri  = { "CLUSTER_TRIANGLE_COUNT", { "128", "64" } };
constexpr ShaderSelector kSelDedicated   = { "CLUSTER_DEDICATED_VERTICES", { "0", "1" } };
constexpr ShaderSelector kSelHizIsFirst  = { "NV_HIZ_IS_FIRST", { "1", "0" } };
constexpr ShaderSelector kSelHizOutNear  = { "NV_HIZ_OUTPUT_NEAR", { "0", "1" } };
constexpr ShaderSelector kSelHizRevZ     = { "NV_HIZ_REVERSED_Z", { "0", "1" } };

// -- Matrix files (entries enumerated via generated X-macros) ----------------

// blas_clusters_insert: axes (streaming)
#define X1(a, name) { { uint8_t(a) }, SPV(name) },
constexpr ShaderSelector kSelectorsBlasClustersInsert[] = { kSelStreaming };
constexpr ShaderVariantEntry kEntriesBlasClustersInsert[] = {
  RTX_SHADER_VARIANT_MATRIX_BLAS_CLUSTERS_INSERT_STREAMING_NONE_COMP(X1)
  RTX_SHADER_VARIANT_MATRIX_BLAS_CLUSTERS_INSERT_STREAMING_STRM_COMP(X1)
};

// build_setup: axes (traversal)
constexpr ShaderSelector kSelectorsBuildSetup[] = { kSelPersistent };
constexpr ShaderVariantEntry kEntriesBuildSetup[] = {
  RTX_SHADER_VARIANT_MATRIX_BUILD_SETUP_TRAVERSAL_NONE_COMP(X1)
  RTX_SHADER_VARIANT_MATRIX_BUILD_SETUP_TRAVERSAL_PERSIST_COMP(X1)
};

// traversal_blas_merging: axes (caching)
constexpr ShaderSelector kSelectorsTraversalBlasMerging[] = { kSelCaching };
constexpr ShaderVariantEntry kEntriesTraversalBlasMerging[] = {
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_BLAS_MERGING_CACHING_NONE_COMP(X1)
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_BLAS_MERGING_CACHING_CACHE_COMP(X1)
};

// cluster_blas_instances: axes (dedicated)
constexpr ShaderSelector kSelectorsClusterBlasInstances[] = { kSelDedicated };
constexpr ShaderVariantEntry kEntriesClusterBlasInstances[] = {
  RTX_SHADER_VARIANT_MATRIX_CLUSTER_BLAS_INSTANCES_DEDICATED_NONE_COMP(X1)
  RTX_SHADER_VARIANT_MATRIX_CLUSTER_BLAS_INSTANCES_DEDICATED_DEDIC_COMP(X1)
};

// stream_update_clas_geometry_indices: axes (clustersize) -> selectors
// (CLUSTER_VERTEX_COUNT, CLUSTER_TRIANGLE_COUNT), both driven by one axis.
#define XCS(s, name) { { uint8_t(s), uint8_t(s) }, SPV(name) },
constexpr ShaderSelector kSelectorsStreamUpdateClasGeometryIndices[] = { kSelClusterVtx, kSelClusterTri };
constexpr ShaderVariantEntry kEntriesStreamUpdateClasGeometryIndices[] = {
  RTX_SHADER_VARIANT_MATRIX_STREAM_UPDATE_CLAS_GEOMETRY_INDICES_CLUSTERSIZE_NONE_COMP(XCS)
  RTX_SHADER_VARIANT_MATRIX_STREAM_UPDATE_CLAS_GEOMETRY_INDICES_CLUSTERSIZE_C64_COMP(XCS)
};

// instance_classify_lod: axes (culling, ficulling)
#define X2(a, b, name) { { uint8_t(a), uint8_t(b) }, SPV(name) },
constexpr ShaderSelector kSelectorsInstanceClassifyLod[] = { kSelCulling, kSelFiCulling };
constexpr ShaderVariantEntry kEntriesInstanceClassifyLod[] = {
  RTX_SHADER_VARIANT_MATRIX_INSTANCE_CLASSIFY_LOD_CULLING_NONE_COMP(X2)
  RTX_SHADER_VARIANT_MATRIX_INSTANCE_CLASSIFY_LOD_CULLING_CULL_COMP(X2)
};

// stream_update_scene: axes (raster, caching, clustersize) -> selectors
// (TARGETS_RASTERIZATION, USE_BLAS_CACHING, CLUSTER_VERTEX, CLUSTER_TRIANGLE)
#define XUS(r, c, s, name) { { uint8_t(r), uint8_t(c), uint8_t(s), uint8_t(s) }, SPV(name) },
constexpr ShaderSelector kSelectorsStreamUpdateScene[] = { kSelRaster, kSelCaching, kSelClusterVtx, kSelClusterTri };
constexpr ShaderVariantEntry kEntriesStreamUpdateScene[] = {
  RTX_SHADER_VARIANT_MATRIX_STREAM_UPDATE_SCENE_RASTER_NONE_COMP(XUS)
  RTX_SHADER_VARIANT_MATRIX_STREAM_UPDATE_SCENE_RASTER_RASTER_COMP(XUS)
};

// traversal_init: axes (culling, sorting, ficulling)
#define X3(a, b, c, name) { { uint8_t(a), uint8_t(b), uint8_t(c) }, SPV(name) },
constexpr ShaderSelector kSelectorsTraversalInit[] = { kSelCulling, kSelSorting, kSelFiCulling };
constexpr ShaderVariantEntry kEntriesTraversalInit[] = {
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_INIT_CULLING_NONE_COMP(X3)
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_INIT_CULLING_CULL_COMP(X3)
};

// traversal_init_blas_sharing: axes (merging, caching, culling, sorting, ficulling)
#define X5(a, b, c, d, e, name) { { uint8_t(a), uint8_t(b), uint8_t(c), uint8_t(d), uint8_t(e) }, SPV(name) },
constexpr ShaderSelector kSelectorsTraversalInitBlasSharing[] = { kSelMerging, kSelCaching, kSelCulling, kSelSorting, kSelFiCulling };
constexpr ShaderVariantEntry kEntriesTraversalInitBlasSharing[] = {
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_INIT_BLAS_SHARING_MERGING_NONE_COMP(X5)
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_INIT_BLAS_SHARING_MERGING_MERGE_COMP(X5)
};

// traversal_run / traversal_run_groups: axes (streaming, merging, culling, traversal, ficulling)
constexpr ShaderSelector kSelectorsTraversalRun[] = { kSelStreaming, kSelMerging, kSelCulling, kSelPersistent, kSelFiCulling };
constexpr ShaderVariantEntry kEntriesTraversalRun[] = {
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_RUN_STREAMING_NONE_COMP(X5)
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_RUN_STREAMING_STRM_COMP(X5)
};
constexpr ShaderVariantEntry kEntriesTraversalRunGroups[] = {
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_RUN_GROUPS_STREAMING_NONE_COMP(X5)
  RTX_SHADER_VARIANT_MATRIX_TRAVERSAL_RUN_GROUPS_STREAMING_STRM_COMP(X5)
};

// nvhiz_update: axes (pass, revz) -> selectors (IS_FIRST, OUTPUT_NEAR, REVERSED_Z).
// Axis "pass" drives two macros; only the sample's valid combinations exist.
#define XHZ(p, r, name) /* expanded manually below */
constexpr ShaderSelector kSelectorsNvhizUpdate[] = { kSelHizIsFirst, kSelHizOutNear, kSelHizRevZ };
#define XHZ_FAR(p, r, name)     { { 0 /*IS_FIRST=1*/, 0 /*OUT_NEAR=0*/, uint8_t(r) }, SPV(name) },
#define XHZ_FARNEAR(p, r, name) { { 0, 1, uint8_t(r) }, SPV(name) },
#define XHZ_FARREST(p, r, name) { { 1 /*IS_FIRST=0*/, 0, uint8_t(r) }, SPV(name) },
constexpr ShaderVariantEntry kEntriesNvhizUpdate[] = {
  RTX_SHADER_VARIANT_MATRIX_NVHIZ_UPDATE_PASS_FAR_COMP(XHZ_FAR)
  RTX_SHADER_VARIANT_MATRIX_NVHIZ_UPDATE_PASS_FARNEAR_COMP(XHZ_FARNEAR)
  RTX_SHADER_VARIANT_MATRIX_NVHIZ_UPDATE_PASS_FARREST_COMP(XHZ_FARREST)
};

// -- Explicit-variant files (valid combinations only) -------------------------

constexpr ShaderSelector kSelectorsGeometryBlasSharing[] = { kSelStreaming, kSelMerging, kSelCaching };
constexpr ShaderVariantEntry kEntriesGeometryBlasSharing[] = {
  { { 0, 0, 0 }, SPV(geometry_blas_sharing) },
  { { 1, 0, 0 }, SPV(geometry_blas_sharing_strm) },
  { { 1, 1, 0 }, SPV(geometry_blas_sharing_strm_merge) },
  { { 1, 0, 1 }, SPV(geometry_blas_sharing_strm_cache) },
  { { 1, 1, 1 }, SPV(geometry_blas_sharing_strm_merge_cache) },
};

constexpr ShaderSelector kSelectorsInstanceAssignBlas[] = { kSelSharing, kSelStreaming, kSelCaching };
constexpr ShaderVariantEntry kEntriesInstanceAssignBlas[] = {
  { { 0, 0, 0 }, SPV(instance_assign_blas) },
  { { 0, 1, 0 }, SPV(instance_assign_blas_strm) },
  { { 1, 0, 0 }, SPV(instance_assign_blas_share) },
  { { 1, 1, 0 }, SPV(instance_assign_blas_share_strm) },
  { { 1, 1, 1 }, SPV(instance_assign_blas_share_strm_cache) },
};

// -- Single-variant files ------------------------------------------------------

#define SINGLE_ENTRY(sym) \
  constexpr ShaderVariantEntry kEntriesSingle_##sym[] = { { { 0 }, SPV(sym) } };
SINGLE_ENTRY(blas_setup_insertion)
SINGLE_ENTRY(blas_caching_setup_build)
SINGLE_ENTRY(blas_caching_setup_copy)
SINGLE_ENTRY(traversal_presort)
SINGLE_ENTRY(stream_setup)
SINGLE_ENTRY(stream_agefilter_groups)
SINGLE_ENTRY(stream_compaction_new_clas)
SINGLE_ENTRY(stream_compaction_old_clas)
SINGLE_ENTRY(stream_allocator_build_freegaps)
SINGLE_ENTRY(stream_allocator_freegaps_insert)
SINGLE_ENTRY(stream_allocator_setup_insertion)
SINGLE_ENTRY(stream_allocator_unload_groups)
SINGLE_ENTRY(stream_allocator_load_groups)
SINGLE_ENTRY(remix_depth_flip)
SINGLE_ENTRY(promotion_solve)
SINGLE_ENTRY(anim_gather_positions)
#undef SINGLE_ENTRY

// -- File registry -------------------------------------------------------------

#define FILE_MATRIX(name, glslName, selectors, entries) \
  { glslName, selectors, uint32_t(std::size(selectors)), entries, uint32_t(std::size(entries)) }
#define FILE_SINGLE(sym, glslName) \
  { glslName, nullptr, 0, kEntriesSingle_##sym, 1 }

constexpr ShaderFileTable kShaderFiles[] = {
  FILE_MATRIX(blas_clusters_insert, "blas_clusters_insert.comp.glsl", kSelectorsBlasClustersInsert, kEntriesBlasClustersInsert),
  FILE_MATRIX(build_setup, "build_setup.comp.glsl", kSelectorsBuildSetup, kEntriesBuildSetup),
  FILE_MATRIX(instance_classify_lod, "instance_classify_lod.comp.glsl", kSelectorsInstanceClassifyLod, kEntriesInstanceClassifyLod),
  FILE_MATRIX(stream_update_clas_geometry_indices, "stream_update_clas_geometry_indices.comp.glsl",
              kSelectorsStreamUpdateClasGeometryIndices, kEntriesStreamUpdateClasGeometryIndices),
  FILE_MATRIX(stream_update_scene, "stream_update_scene.comp.glsl", kSelectorsStreamUpdateScene, kEntriesStreamUpdateScene),
  FILE_MATRIX(traversal_blas_merging, "traversal_blas_merging.comp.glsl", kSelectorsTraversalBlasMerging, kEntriesTraversalBlasMerging),
  FILE_MATRIX(traversal_init, "traversal_init.comp.glsl", kSelectorsTraversalInit, kEntriesTraversalInit),
  FILE_MATRIX(traversal_init_blas_sharing, "traversal_init_blas_sharing.comp.glsl",
              kSelectorsTraversalInitBlasSharing, kEntriesTraversalInitBlasSharing),
  FILE_MATRIX(traversal_run, "traversal_run.comp.glsl", kSelectorsTraversalRun, kEntriesTraversalRun),
  FILE_MATRIX(traversal_run_groups, "traversal_run_groups.comp.glsl", kSelectorsTraversalRun, kEntriesTraversalRunGroups),
  FILE_MATRIX(cluster_blas_instances, "cluster_blas_instances.comp.glsl", kSelectorsClusterBlasInstances, kEntriesClusterBlasInstances),
  FILE_MATRIX(nvhiz_update, "nvhiz-update.comp.glsl", kSelectorsNvhizUpdate, kEntriesNvhizUpdate),

  FILE_MATRIX(geometry_blas_sharing, "geometry_blas_sharing.comp.glsl", kSelectorsGeometryBlasSharing, kEntriesGeometryBlasSharing),
  FILE_MATRIX(instance_assign_blas, "instance_assign_blas.comp.glsl", kSelectorsInstanceAssignBlas, kEntriesInstanceAssignBlas),

  FILE_SINGLE(blas_setup_insertion, "blas_setup_insertion.comp.glsl"),
  FILE_SINGLE(blas_caching_setup_build, "blas_caching_setup_build.comp.glsl"),
  FILE_SINGLE(blas_caching_setup_copy, "blas_caching_setup_copy.comp.glsl"),
  FILE_SINGLE(traversal_presort, "traversal_presort.comp.glsl"),
  FILE_SINGLE(stream_setup, "stream_setup.comp.glsl"),
  FILE_SINGLE(stream_agefilter_groups, "stream_agefilter_groups.comp.glsl"),
  FILE_SINGLE(stream_compaction_new_clas, "stream_compaction_new_clas.comp.glsl"),
  FILE_SINGLE(stream_compaction_old_clas, "stream_compaction_old_clas.comp.glsl"),
  FILE_SINGLE(stream_allocator_build_freegaps, "stream_allocator_build_freegaps.comp.glsl"),
  FILE_SINGLE(stream_allocator_freegaps_insert, "stream_allocator_freegaps_insert.comp.glsl"),
  FILE_SINGLE(stream_allocator_setup_insertion, "stream_allocator_setup_insertion.comp.glsl"),
  FILE_SINGLE(stream_allocator_unload_groups, "stream_allocator_unload_groups.comp.glsl"),
  FILE_SINGLE(stream_allocator_load_groups, "stream_allocator_load_groups.comp.glsl"),
  FILE_SINGLE(remix_depth_flip, "remix_depth_flip.comp.glsl"),
  FILE_SINGLE(promotion_solve, "promotion_solve.comp.glsl"),
  FILE_SINGLE(anim_gather_positions, "anim_gather_positions.comp.glsl"),
};

#undef FILE_MATRIX
#undef FILE_SINGLE
#undef SPV
#undef X1
#undef X2
#undef X3
#undef X5
#undef XCS
#undef XUS
#undef XHZ
#undef XHZ_FAR
#undef XHZ_FARNEAR
#undef XHZ_FARREST

}  // namespace

shaderc::SpvCompilationResult lookupPrebuiltShader(const char* fileName, const shaderc::CompileOptions& options)
{
  for(const ShaderFileTable& file : kShaderFiles)
  {
    if(std::strcmp(file.fileName, fileName) != 0)
    {
      continue;
    }

    // Resolve each selector macro to its value index.
    uint8_t requested[kMaxSelectors] = {};
    for(uint32_t s = 0; s < file.selectorCount; s++)
    {
      const ShaderSelector& selector = file.selectors[s];

      std::string value;
      if(!options.getMacro(selector.macro, value))
      {
        // Missing selector -> first (default) value, matching the axis "-" default.
        requested[s] = 0;
        continue;
      }

      bool found = false;
      for(uint8_t v = 0; v < 3 && selector.values[v]; v++)
      {
        if(value == selector.values[v])
        {
          requested[s] = v;
          found        = true;
          break;
        }
      }
      if(!found)
      {
        LOGE("lodclusters: shader '%s' selector '%s' has unsupported value '%s'\n", fileName, selector.macro, value.c_str());
        return {};
      }
    }

    for(uint32_t e = 0; e < file.entryCount; e++)
    {
      const ShaderVariantEntry& entry = file.entries[e];
      if(std::memcmp(entry.values, requested, file.selectorCount) == 0)
      {
        return shaderc::SpvCompilationResult(entry.data, entry.size);
      }
    }

    LOGE("lodclusters: shader '%s' has no variant for the requested selector combination\n", fileName);
    return {};
  }

  LOGE("lodclusters: unknown shader file '%s'\n", fileName);
  return {};
}

}  // namespace lodclusters
