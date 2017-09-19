
# Function to register a target for cppcheck
function(perform_cppcheck check_target target)
    set(includes "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>")
    
    add_custom_target(
        ${check_target}
        COMMAND
            ${cppcheck_EXECUTABLE}
                "$<$<BOOL:${includes}>:-I$<JOIN:${includes},\t-I>>"
                --check-config
                --enable=warning,performance,portability,information,missingInclude
                --quiet
                --std=c++11
                --verbose
                --suppress=missingIncludeSystem
                ${ARGN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_target_properties(${check_target}
        PROPERTIES
        FOLDER "Maintenance"
        EXCLUDE_FROM_DEFAULT_BUILD 1
    )
    
    add_dependencies(${check_target} ${target})
endfunction()
