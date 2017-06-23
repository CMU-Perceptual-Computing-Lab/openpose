if(OPTION_USE_CAFFE AND NOT ${CAFFE_FOUND})
	set(BUILD_docs OFF)
	add_subdirectory(caffe)
	add_custom_command(
		OUTPUT "${CMAKE_BINARY_DIR}/"
    	COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/3rdparty/caffe/include/caffe/proto" OUTPUT "${CMAKE_SOURCE_DIR}/include/caffe/"
		COMMENT "Copying protobuf files for caffe."
		DEPENDS caffe)
endif()

