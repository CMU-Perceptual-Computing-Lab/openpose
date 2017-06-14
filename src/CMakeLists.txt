if(OPTION_USE_CAFFE)
    ADD_DEFINITIONS( -DUSE_CAFFE )
endif()

add_subdirectory(openpose)
