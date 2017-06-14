if(OPTION_USE_CAFFE AND NOT ${CAFFE_FOUND})
	add_subdirectory(caffe)
endif()

