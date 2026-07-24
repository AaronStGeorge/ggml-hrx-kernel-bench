include_guard(GLOBAL)

function(add_loom_benchmark_script_materialization_target)
  set(options)
  set(one_value_args
    NAME
    GENERATED_IMPORT_DIR
    PREPARE_OUTPUT_DIR
    PREPARE_TARGET
    OUTPUT_DIR
    ASSET_ROOT
    SUITE
    SUMMARY_OUTPUT_DIR
    OP
    COMMENT
  )
  set(multi_value_args DEPENDS)
  cmake_parse_arguments(GGML_HRX_LBST "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_LBST_NAME)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires NAME")
  endif()
  if(NOT GGML_HRX_LBST_GENERATED_IMPORT_DIR)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires GENERATED_IMPORT_DIR")
  endif()
  if(NOT GGML_HRX_LBST_PREPARE_OUTPUT_DIR)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires PREPARE_OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_LBST_PREPARE_TARGET)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires PREPARE_TARGET")
  endif()
  if(NOT GGML_HRX_LBST_OUTPUT_DIR)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_LBST_OP)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires OP")
  endif()

  set(asset_root ${GGML_HRX_LBST_ASSET_ROOT})
  if(NOT asset_root)
    set(asset_root ${GGML_HRX_ASSET_ROOT})
  endif()
  if(NOT asset_root)
    message(FATAL_ERROR "add_loom_benchmark_script_materialization_target requires ASSET_ROOT or GGML_HRX_ASSET_ROOT")
  endif()

  set(index_path ${GGML_HRX_LBST_OUTPUT_DIR}/catalog/v2/index.json)
  set(summary_output_dir ${GGML_HRX_LBST_SUMMARY_OUTPUT_DIR})
  if(NOT summary_output_dir)
    if(GGML_HRX_LBST_SUITE)
      set(summary_output_dir ${CMAKE_BINARY_DIR}/benchmarks/${GGML_HRX_LBST_SUITE})
    else()
      set(summary_output_dir ${GGML_HRX_LBST_OUTPUT_DIR}/summary)
    endif()
  endif()
  set(summary_json_path ${summary_output_dir}/benchmark-route-summary.json)
  set(summary_markdown_path ${summary_output_dir}/benchmark-route-summary.md)
  set(materialize_command
    ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/infra/materialize_loom_benchmarks.py
    --prepare-root ${GGML_HRX_LBST_PREPARE_OUTPUT_DIR}
    --repo-root ${CMAKE_SOURCE_DIR}
    --asset-root ${asset_root}
    --output-root ${GGML_HRX_LBST_OUTPUT_DIR}
    --summary-output-dir ${summary_output_dir}
    --op ${GGML_HRX_LBST_OP}
  )
  if(GGML_HRX_LBST_SUITE)
    list(APPEND materialize_command --suite ${GGML_HRX_LBST_SUITE})
  endif()
  if(GGML_HRX_TOOL_DIR)
    list(APPEND materialize_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()

  set(materialize_depends
    ${GGML_HRX_LBST_PREPARE_TARGET}
    ${GGML_HRX_LBST_GENERATED_IMPORT_DIR}/route-import.stamp
    ${GGML_HRX_LBST_DEPENDS}
    ${CMAKE_SOURCE_DIR}/tests/infra/materialize_loom_benchmarks.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/common.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/discovery.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/materialize.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/summary.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/workbench.py
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/loom_execution_descriptor.py
  )
  if(GGML_HRX_TOOL_BUILD_TARGET)
    list(APPEND materialize_depends ${GGML_HRX_TOOL_BUILD_TARGET})
  endif()

  set(command_comment ${GGML_HRX_LBST_COMMENT})
  if(NOT command_comment)
    set(command_comment "Generating Loom benchmark scripts for ${GGML_HRX_LBST_NAME}")
  endif()

  add_custom_command(
    OUTPUT ${index_path}
    BYPRODUCTS
      ${summary_json_path}
      ${summary_markdown_path}
    COMMAND ${materialize_command}
    DEPENDS
      ${materialize_depends}
    COMMENT ${command_comment}
    VERBATIM
  )

  add_custom_target(
    ${GGML_HRX_LBST_NAME}
    DEPENDS
      ${index_path}
      ${summary_json_path}
      ${summary_markdown_path}
  )
endfunction()
