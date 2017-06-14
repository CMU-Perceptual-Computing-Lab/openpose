if(OPTION_USE_CAFFE)
    ADD_DEFINITIONS( -DUSE_CAFFE=1 )
endif()

add_subdirectory(openpose)
