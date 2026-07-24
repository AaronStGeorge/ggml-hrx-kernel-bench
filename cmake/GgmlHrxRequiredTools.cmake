include_guard(GLOBAL)

function(add_required_tool_test tool_name)
  set(test_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_required_tool.py
    ${tool_name}
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND test_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  add_test(
    NAME required-tool-${tool_name}
    COMMAND ${test_command}
  )
endfunction()
