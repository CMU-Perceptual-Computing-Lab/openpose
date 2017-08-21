set(EXAMPLE_FILES 
  tutorial_thread/2_user_processing_function.cpp
  tutorial_thread/4_user_input_processing_output_and_datum.cpp
  tutorial_thread/1_openpose_read_and_display.cpp
  tutorial_thread/3_user_input_processing_and_output.cpp
  tutorial_pose/2_extract_pose_or_heatmat_from_image.cpp
  tutorial_pose/1_extract_from_image.cpp
  tutorial_wrapper/2_user_synchronous.cpp
  tutorial_wrapper/1_user_asynchronous.cpp
  openpose/openpose.cpp)

foreach(file ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS "Adding Example ${SOURCE_NAME}")
  
  add_executable(${SOURCE_NAME}.bin ${file})

  target_link_libraries( ${SOURCE_NAME}.bin
     wrapper utilities ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS}
  )

endforeach()
