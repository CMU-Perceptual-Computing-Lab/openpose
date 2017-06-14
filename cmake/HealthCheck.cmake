
include(${CMAKE_CURRENT_LIST_DIR}/Cppcheck.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ClangTidy.cmake)

set(OPTION_CPPCHECK_ENABLED Off)
set(OPTION_CLANG_TIDY_ENABLED Off)

# Function to register a target for enabled health checks
function(perform_health_checks target)
    if(NOT TARGET check-all)
        add_custom_target(check-all)
    
        set_target_properties(check-all
            PROPERTIES
            FOLDER "Maintenance"
            EXCLUDE_FROM_DEFAULT_BUILD 1
        )
    endif()
    
    add_custom_target(check-${target})
    
    set_target_properties(check-${target}
        PROPERTIES
        FOLDER "Maintenance"
        EXCLUDE_FROM_DEFAULT_BUILD 1
    )
    
    if (OPTION_CPPCHECK_ENABLED)
        perform_cppcheck(cppcheck-${target} ${target} ${ARGN})
        add_dependencies(check-${target} cppcheck-${target})
    endif()
    
    if (OPTION_CLANG_TIDY_ENABLED)
        perform_clang_tidy(clang-tidy-${target} ${target} ${ARGN})
        add_dependencies(check-${target} clang-tidy-${target})
    endif()
    
    add_dependencies(check-all check-${target})
endfunction()

# Enable or disable cppcheck for health checks
function(enable_cppcheck status)
    if(NOT ${status})
        set(OPTION_CPPCHECK_ENABLED ${status} PARENT_SCOPE)
        message(STATUS "Check cppcheck skipped: Manually disabled")
        
        return()
    endif()
    
    find_package(cppcheck)
    
    if(NOT cppcheck_FOUND)
        set(OPTION_CPPCHECK_ENABLED Off PARENT_SCOPE)
        message(STATUS "Check cppcheck skipped: cppcheck not found")
        
        return()
    endif()
    
    set(OPTION_CPPCHECK_ENABLED ${status} PARENT_SCOPE)
    message(STATUS "Check cppcheck")
endfunction()

# Enable or disable clang-tidy for health checks
function(enable_clang_tidy status)
    if(NOT ${status})
        set(OPTION_CLANG_TIDY_ENABLED ${status} PARENT_SCOPE)
        message(STATUS "Check clang-tidy skipped: Manually disabled")
        
        return()
    endif()
    
    find_package(clang_tidy)
    
    if(NOT clang_tidy_FOUND)
        set(OPTION_CLANG_TIDY_ENABLED Off PARENT_SCOPE)
        message(STATUS "Check clang-tidy skipped: clang-tidy not found")
        
        return()
    endif()
    
    set(OPTION_CLANG_TIDY_ENABLED ${status} PARENT_SCOPE)
    message(STATUS "Check clang-tidy")
    
    set(CMAKE_EXPORT_COMPILE_COMMANDS On PARENT_SCOPE)
endfunction()

# Configure cmake target to check for cmake-init template
function(add_check_template_target current_template_sha)
    add_custom_target(
        check-template
        COMMAND ${CMAKE_COMMAND}
            -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DPROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}
            -DAPPLIED_CMAKE_INIT_SHA=${current_template_sha}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckTemplate.cmake
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    
    set_target_properties(check-template
        PROPERTIES
        FOLDER "Maintenance"
        EXCLUDE_FROM_DEFAULT_BUILD 1
    )
endfunction()
