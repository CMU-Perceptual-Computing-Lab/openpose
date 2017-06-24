
# Check if examples are enabled
if(NOT OPTION_BUILD_EXAMPLES)
    return()
endif()

if(OPTION_USE_CAFFE)
    ADD_DEFINITIONS( -DUSE_CAFFE )
endif()

# Example applications
add_subdirectory(openpose)
add_subdirectory(tutorial_pose)
add_subdirectory(tutorial_thread)
