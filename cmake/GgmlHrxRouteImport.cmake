include_guard(GLOBAL)

function(add_yaml_route_import_target)
  set(options NO_ALL)
  set(one_value_args
    NAME
    OUTPUT_DIR
    ROUTING_DIR
    EXPECTED_COVERAGE
    EXPECTED_NATIVE_ROUTE_COUNT
  )
  set(multi_value_args YAML_PATHS DEPENDS)
  cmake_parse_arguments(GGML_HRX_YRI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_YRI_NAME)
    message(FATAL_ERROR "add_yaml_route_import_target requires NAME")
  endif()
  if(NOT GGML_HRX_YRI_OUTPUT_DIR)
    message(FATAL_ERROR "add_yaml_route_import_target requires OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_YRI_ROUTING_DIR)
    message(FATAL_ERROR "add_yaml_route_import_target requires ROUTING_DIR")
  endif()
  if(NOT GGML_HRX_YRI_EXPECTED_COVERAGE)
    message(FATAL_ERROR "add_yaml_route_import_target requires EXPECTED_COVERAGE")
  endif()
  if(NOT GGML_HRX_YRI_YAML_PATHS)
    message(FATAL_ERROR "add_yaml_route_import_target requires YAML_PATHS")
  endif()
  if(GGML_HRX_TOOL_BUILD_TARGET)
    list(APPEND GGML_HRX_YRI_DEPENDS ${GGML_HRX_TOOL_BUILD_TARGET})
  endif()
  list(APPEND GGML_HRX_YRI_DEPENDS ggml-hrx-v2-route-selector)

  set(route_queries_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-queries.jsonl)
  set(route_query_import_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-query-import.json)
  set(query_import_stamp_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-query-import.stamp)
  set(route_import_stamp_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-import.stamp)
  set(query_import_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/import_yaml_route_queries.py
    ${GGML_HRX_YRI_OUTPUT_DIR}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
  )
  set(route_config_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/generate_route_configs.py
    ${GGML_HRX_YRI_OUTPUT_DIR}
    --route-queries ${route_queries_path}
    --import-metadata ${route_query_import_path}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
    --expected-coverage ${GGML_HRX_YRI_EXPECTED_COVERAGE}
  )
  foreach(yaml_path IN LISTS GGML_HRX_YRI_YAML_PATHS)
    list(APPEND query_import_command --yaml ${yaml_path})
  endforeach()
  set(yaml_route_import_environment
    "GGML_HRX_V2_ROUTE_SELECTOR=$<TARGET_FILE:ggml-hrx-v2-route-selector>"
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND query_import_command --tool-dir ${GGML_HRX_TOOL_DIR})
    list(APPEND route_config_command --tool-dir ${GGML_HRX_TOOL_DIR})
    list(APPEND yaml_route_import_environment
      "PATH=${GGML_HRX_TOOL_DIR}:$ENV{PATH}"
    )
  endif()
  set(query_import_env_command
    ${CMAKE_COMMAND} -E env
    ${yaml_route_import_environment}
    ${query_import_command}
  )
  set(route_config_env_command
    ${CMAKE_COMMAND} -E env
    ${yaml_route_import_environment}
    ${route_config_command}
  )
  add_custom_command(
    OUTPUT ${query_import_stamp_path}
    BYPRODUCTS
      ${route_queries_path}
      ${route_query_import_path}
    COMMAND ${query_import_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${query_import_stamp_path}
    DEPENDS
      ${GGML_HRX_YRI_YAML_PATHS}
      ${GGML_HRX_YRI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/import_yaml_route_queries.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/route_query_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/selection.py
    COMMENT "Importing descriptor YAML route queries for ${GGML_HRX_YRI_NAME}"
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${route_import_stamp_path}
    BYPRODUCTS
      ${GGML_HRX_YRI_OUTPUT_DIR}/generated-kernel-tests.json
      ${GGML_HRX_YRI_OUTPUT_DIR}/route-import-summary.json
    COMMAND ${route_config_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${route_import_stamp_path}
    DEPENDS
      ${query_import_stamp_path}
      ${route_queries_path}
      ${route_query_import_path}
      ${GGML_HRX_YRI_EXPECTED_COVERAGE}
      ${GGML_HRX_YRI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/check_yaml_route_import.py
      ${GGML_HRX_TESTS_ROOT}/infra/generate_route_configs.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/kernel_test_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/route_query_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/selection.py
    COMMENT "Generating descriptor route configs for ${GGML_HRX_YRI_NAME}"
    VERBATIM
  )
  if(GGML_HRX_YRI_NO_ALL)
    set(_ggml_hrx_yri_all_arg)
  else()
    set(_ggml_hrx_yri_all_arg ALL)
  endif()
  add_custom_target(${GGML_HRX_YRI_NAME} ${_ggml_hrx_yri_all_arg} DEPENDS ${route_import_stamp_path})

  set(route_selector_check_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_route_selector_parity.py
    --route-queries ${route_queries_path}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
    --python-selector ${CMAKE_SOURCE_DIR}/tools/v2-route-selector/python_v2_route_selector.py
    --native-selector $<TARGET_FILE:ggml-hrx-v2-route-selector>
  )
  add_test(
    NAME ${GGML_HRX_YRI_NAME}-route-selector-parity
    COMMAND ${route_selector_check_command} --mode parity
  )
  if(GGML_HRX_YRI_EXPECTED_NATIVE_ROUTE_COUNT)
    add_test(
      NAME ${GGML_HRX_YRI_NAME}-native-route-count
      COMMAND
        ${route_selector_check_command}
        --mode native-route-count
        --expected-native-route-count ${GGML_HRX_YRI_EXPECTED_NATIVE_ROUTE_COUNT}
    )
  endif()
  set_tests_properties(
    ${GGML_HRX_YRI_NAME}-route-selector-parity
    PROPERTIES
      LABELS "routing;parity"
      # Keep this in sync with PARITY_MISMATCH_SKIP_RETURN_CODE in the checker.
      SKIP_RETURN_CODE 77
      TIMEOUT 60
  )
  if(GGML_HRX_YRI_EXPECTED_NATIVE_ROUTE_COUNT)
    set_tests_properties(
      ${GGML_HRX_YRI_NAME}-native-route-count
      PROPERTIES
        LABELS "routing;native-route-count"
        TIMEOUT 60
    )
  endif()
endfunction()
