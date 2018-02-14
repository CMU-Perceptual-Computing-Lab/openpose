add_subdirectory(tracking)

if (UNIX AND NOT APPLE)
    add_custom_target(openpose_experimental)
    add_dependencies(openpose_experimental openpose_tracking)
endif (UNIX AND NOT APPLE)

set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} PARENT_SCOPE)
