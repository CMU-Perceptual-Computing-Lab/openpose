add_subdirectory(tracking)
add_custom_target(openpose_experimental)
add_dependencies(openpose_experimental openpose_tracking)

set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} PARENT_SCOPE)

