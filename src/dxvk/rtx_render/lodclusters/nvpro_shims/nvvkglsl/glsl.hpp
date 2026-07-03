/*
* NV-DXVK: stand-in for nvpro_core2's <nvvkglsl/glsl.hpp> (shaderc-based
* runtime GLSL compiler).
*
* dxvk-remix compiles the cluster kernels at BUILD time through
* compile_shaders.py (glslang), producing one SPIR-V blob per //!variant
* combination. This header keeps the shaderc/nvvkglsl API surface that the
* ported NVIDIA code uses, but "compilation" becomes a lookup into the
* prebuilt variant table:
*
*   - shaderc::CompileOptions::AddMacroDefinition records the macro values
*     the sample would have compiled with.
*   - Resources::compileShader resolves (source file name, macro values) to
*     the matching prebuilt variant and returns it as a
*     shaderc::SpvCompilationResult.
*
* This keeps every NVIDIA shader-init call site byte-identical while the
* actual SPIR-V comes from the Remix build. The lookup itself is implemented
* in lodclusters_shader_table.cpp.
*/
#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>

#include <volk.h>

// Matches the C enum from shaderc/shaderc.h (only the values the ported code
// references, plus common error states for logging).
typedef enum shaderc_compilation_status {
  shaderc_compilation_status_success = 0,
  shaderc_compilation_status_invalid_stage = 1,
  shaderc_compilation_status_compilation_error = 2,
  shaderc_compilation_status_internal_error = 3,
  shaderc_compilation_status_null_result_object = 4,
  shaderc_compilation_status_invalid_assembly = 5,
  shaderc_compilation_status_validation_error = 6,
  shaderc_compilation_status_transformation_error = 7,
  shaderc_compilation_status_configuration_error = 8,
} shaderc_compilation_status;

namespace shaderc {

// Records macro definitions; consumed by the prebuilt-variant lookup.
class CompileOptions {
public:
  CompileOptions() = default;

  void AddMacroDefinition(const std::string& name, const std::string& value) {
    m_macros[name] = value;
  }
  void AddMacroDefinition(const std::string& name) {
    m_macros[name] = "1";
  }

  // Lookup helpers (shim-side additions, not part of real shaderc; NVIDIA
  // code never calls these).
  const std::map<std::string, std::string>& getMacros() const { return m_macros; }
  bool getMacro(const std::string& name, std::string& value) const {
    auto it = m_macros.find(name);
    if (it == m_macros.end()) {
      return false;
    }
    value = it->second;
    return true;
  }

private:
  std::map<std::string, std::string> m_macros;
};

// A non-owning view of a prebuilt SPIR-V blob (the blobs are compiled into
// the binary as C arrays, so they live for the program's lifetime).
class SpvCompilationResult {
public:
  SpvCompilationResult() = default;
  SpvCompilationResult(const uint32_t* data, size_t sizeInBytes)
      : m_data(data)
      , m_sizeInBytes(sizeInBytes)
      , m_status(shaderc_compilation_status_success) {
  }

  shaderc_compilation_status GetCompilationStatus() const { return m_status; }

  const uint32_t* data() const { return m_data; }
  size_t sizeInBytes() const { return m_sizeInBytes; }

private:
  const uint32_t* m_data = nullptr;
  size_t m_sizeInBytes = 0;
  shaderc_compilation_status m_status = shaderc_compilation_status_null_result_object;
};

}  // namespace shaderc

namespace nvvkglsl {

// API-compatible subset of nvpro_core2's GlslCompiler. Compilation is
// replaced by prebuilt lookup, so this only carries the base CompileOptions
// and the static SPIR-V accessors used at pipeline-creation time.
class GlslCompiler {
public:
  shaderc::CompileOptions& options() { return m_options; }

  static const uint32_t* getSpirv(const shaderc::SpvCompilationResult& compiled) {
    return compiled.data();
  }
  static size_t getSpirvSize(const shaderc::SpvCompilationResult& compiled) {
    return compiled.sizeInBytes();
  }
  static VkShaderModuleCreateInfo makeShaderModuleCreateInfo(const shaderc::SpvCompilationResult& compiled) {
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = compiled.sizeInBytes();
    info.pCode = compiled.data();
    return info;
  }

private:
  shaderc::CompileOptions m_options;
};

}  // namespace nvvkglsl

namespace lodclusters {

// Implemented in lodclusters_shader_table.cpp: resolves an original sample
// shader file name (e.g. "traversal_run.comp.glsl") plus the macro values in
// `options` to the matching build-time compiled variant. Returns a
// null_result_object result (and logs) if no variant matches - which means
// the //!variant matrix in the shader and the lookup table are out of sync.
shaderc::SpvCompilationResult lookupPrebuiltShader(const char* fileName, const shaderc::CompileOptions& options);

}  // namespace lodclusters
