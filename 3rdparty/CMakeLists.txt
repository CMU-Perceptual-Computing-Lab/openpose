if(OPTION_USE_CAFFE AND NOT Caffe_FOUND)
	set(BUILD_docs OFF)
	add_subdirectory(caffe)
	add_custom_target(caffe_proto ALL DEPENDS caffe)
	add_custom_command( TARGET caffe_proto POST_BUILD
    	COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/3rdparty/caffe/include/caffe/proto/" "${CMAKE_SOURCE_DIR}/3rdparty/caffe/include/caffe/proto/"
		COMMENT "Copying protobuf files for caffe. ${CMAKE_BINARY_DIR}/3rdparty/caffe/include/caffe/proto/ -> ${CMAKE_SOURCE_DIR}/include/caffe/proto"
		)
endif()

