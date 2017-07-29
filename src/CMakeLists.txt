if(OPTION_USE_CAFFE)
    ADD_DEFINITIONS( -DUSE_CAFFE )
endif()
if(CPU_ONLY)
    ADD_DEFINITIONS( -DCPU_ONLY )
endif()

add_subdirectory(openpose)
