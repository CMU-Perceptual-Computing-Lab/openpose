
# Creates an export header similar to generate_export_header, but for templates.
# The main difference is that for MSVC, templates must not get exported.
# When the file ${export_file} is included in source code, the macro ${target_id}_TEMPLATE_API
# may get used to define public visibility for templates on GCC and Clang platforms.
function(generate_template_export_header target target_id export_file)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        configure_file(${PROJECT_SOURCE_DIR}/src/codegeneration/template_msvc_api.h.in ${CMAKE_CURRENT_BINARY_DIR}/${export_file})
    else()
        configure_file(${PROJECT_SOURCE_DIR}/src/codegeneration/template_api.h.in ${CMAKE_CURRENT_BINARY_DIR}/${export_file})
    endif()
endfunction()
