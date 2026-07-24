find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

option(
  GGML_HRX_ENABLE_HSA_DESCRIPTOR_TESTS
  "Register descriptor execution tests that require HSA/GPU runtime resources"
  ON
)

if(NOT DEFINED GGML_HRX_TOOL_DIR)
  set(_ggml_hrx_tool_dir_default "")
  if(DEFINED ENV{GGML_HRX_TOOL_DIR} AND NOT "$ENV{GGML_HRX_TOOL_DIR}" STREQUAL "")
    set(_ggml_hrx_tool_dir_default "$ENV{GGML_HRX_TOOL_DIR}")
  endif()
  set(
    GGML_HRX_TOOL_DIR
    "${_ggml_hrx_tool_dir_default}"
    CACHE STRING
    "Optional PATH-style search list containing loom-link, loom-compile, ggml-hrx-run-loom, and iree-test-loom"
  )
endif()

if(NOT DEFINED GGML_HRX_ROCM_PATH)
  set(_ggml_hrx_rocm_path_default "")
  if(DEFINED ENV{GGML_HRX_ROCM_PATH} AND NOT "$ENV{GGML_HRX_ROCM_PATH}" STREQUAL "")
    set(_ggml_hrx_rocm_path_default "$ENV{GGML_HRX_ROCM_PATH}")
  endif()
  set(GGML_HRX_ROCM_PATH "${_ggml_hrx_rocm_path_default}" CACHE PATH "Optional ROCm root for runtime tests")
endif()

include(GgmlHrxRequiredTools)
include(GgmlHrxDescriptorSuites)

add_subdirectory(infra)
add_subdirectory(kernels)
add_subdirectory(models)

add_custom_target(
  kernel-route-import-v2
  DEPENDS
    kernel-llama-cpp-yaml-route-import-v2
    kernel-llama-cpp-mul-mat-shards-yaml-route-import-v2
    kernel-model-llama-3-3-8b-q8-0-yaml-route-import-v2
)

add_custom_target(
  kernel-descriptor-prepare-v2
  DEPENDS
    kernel-prepare-llama-cpp-v2-generated
    kernel-prepare-llama-cpp-mul-mat-shards-v2-generated
    kernel-prepare-model-llama-3-3-8b-q8-0-v2-generated
)

add_custom_target(
  kernel-yaml-route-import-v2
  DEPENDS
    kernel-route-import-v2
    kernel-descriptor-prepare-v2
)
