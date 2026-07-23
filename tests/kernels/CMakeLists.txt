set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_core.v2.yaml)
set(GGML_HRX_EXPECTED_YAML_ROUTE_IMPORT_COVERAGE_V2 ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp.import-coverage.json)
set(GGML_HRX_EXPECTED_NATIVE_ROUTE_COUNT_V2 ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp.native-route-count.json)
set(GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2 ${CMAKE_CURRENT_BINARY_DIR}/artifacts/llama-cpp-yaml-route-import-v2)

add_yaml_route_import_descriptor_suite(
  NAME llama-cpp-v2
  ROUTE_IMPORT_TARGET kernel-llama-cpp-yaml-route-import-v2
  GROUPED_YAML ${GGML_HRX_GROUPED_YAML}
  GENERATED_IMPORT_DIR ${GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2}
  EXPECTED_COVERAGE ${GGML_HRX_EXPECTED_YAML_ROUTE_IMPORT_COVERAGE_V2}
  EXPECTED_NATIVE_ROUTE_COUNT ${GGML_HRX_EXPECTED_NATIVE_ROUTE_COUNT_V2}
  ENABLE_CONFIGURED_HSA_EXECUTION
)

set(GGML_HRX_LLAMA_CPP_MUL_MAT_SHARDS
  src0-bf16-src1-f32-dst-f32
  src0-f16-src1-f16-dst-f32
  src0-f16-src1-f32-dst-f32
  src0-f32-src1-f16-dst-f32
  src0-f32-src1-f32-dst-f32
  src0-iq1-m-src1-f32-dst-f32
  src0-iq1-s-src1-f32-dst-f32
  src0-iq2-s-src1-f32-dst-f32
  src0-iq2-xs-src1-f32-dst-f32
  src0-iq2-xxs-src1-f16-dst-f32
  src0-iq2-xxs-src1-f32-dst-f32
  src0-iq3-s-src1-f32-dst-f32
  src0-iq3-xxs-src1-f32-dst-f32
  src0-iq4-nl-src1-f32-dst-f32
  src0-iq4-xs-src1-f32-dst-f32
  src0-mxfp4-src1-f16-dst-f32
  src0-mxfp4-src1-f32-dst-f32
  src0-nvfp4-src1-f16-dst-f32
  src0-nvfp4-src1-f32-dst-f32
  src0-q2-k-src1-f32-dst-f32
  src0-q3-k-src1-f32-dst-f32
  src0-q4-0-src1-f16-dst-f32
  src0-q4-0-src1-f32-dst-f32
  src0-q4-1-src1-f16-dst-f32
  src0-q4-1-src1-f32-dst-f32
  src0-q4-k-src1-f16-dst-f32
  src0-q4-k-src1-f32-dst-f32
  src0-q5-0-src1-f32-dst-f32
  src0-q5-1-src1-f32-dst-f32
  src0-q5-k-src1-f32-dst-f32
  src0-q6-k-src1-f32-dst-f32
  src0-q8-0-src1-f16-dst-f32
  src0-q8-0-src1-f32-dst-f32
)

set(GGML_HRX_LLAMA_CPP_MUL_MAT_ROUTE_IMPORT_TARGETS)
set(GGML_HRX_LLAMA_CPP_MUL_MAT_PREPARE_TARGETS)

foreach(shard IN LISTS GGML_HRX_LLAMA_CPP_MUL_MAT_SHARDS)
  set(shard_yaml ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp/mul_mat/llamacpp-mul-mat-${shard}.v2.yaml)
  set(shard_coverage ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp/mul_mat/llamacpp-mul-mat-${shard}.import-coverage.json)
  set(shard_native_route_count ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp/mul_mat/llamacpp-mul-mat-${shard}.native-route-count.json)
  set(shard_route_import_target kernel-llama-cpp-mul-mat-${shard}-yaml-route-import-v2)
  set(shard_prepare_target kernel-prepare-llama-cpp-mul-mat-${shard}-v2-generated)

  add_yaml_route_import_descriptor_suite(
    NAME llama-cpp-mul-mat-${shard}-v2
    ROUTE_IMPORT_TARGET ${shard_route_import_target}
    GROUPED_YAML ${shard_yaml}
    GENERATED_IMPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/llama-cpp-mul-mat-${shard}-route-import-v2
    EXPECTED_COVERAGE ${shard_coverage}
    EXPECTED_NATIVE_ROUTE_COUNT ${shard_native_route_count}
    PREPARE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-prepare-llama-cpp-mul-mat-${shard}-v2
    PREPARE_TARGET ${shard_prepare_target}
    ENABLE_CONFIGURED_HSA_EXECUTION
  )

  list(APPEND GGML_HRX_LLAMA_CPP_MUL_MAT_ROUTE_IMPORT_TARGETS ${shard_route_import_target})
  list(APPEND GGML_HRX_LLAMA_CPP_MUL_MAT_PREPARE_TARGETS ${shard_prepare_target})
endforeach()

add_custom_target(
  kernel-llama-cpp-mul-mat-shards-yaml-route-import-v2
  DEPENDS ${GGML_HRX_LLAMA_CPP_MUL_MAT_ROUTE_IMPORT_TARGETS}
)

add_custom_target(
  kernel-prepare-llama-cpp-mul-mat-shards-v2-generated
  DEPENDS ${GGML_HRX_LLAMA_CPP_MUL_MAT_PREPARE_TARGETS}
)
