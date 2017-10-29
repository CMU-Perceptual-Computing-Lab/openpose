set(EXAMPLE_FILES 
    1_custom_post_processing.cpp)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)
  message(STATUS "Adding Example ${SOURCE_NAME}")
  add_executable(${SOURCE_NAME}.bin ${EXAMPLE_FILE})
  target_link_libraries( ${SOURCE_NAME}.bin
      openpose ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS}
  )

endforeach()
