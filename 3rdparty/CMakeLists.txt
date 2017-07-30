add_subdirectory(caffe)

# Reference --> [Here](https://github.com/BVLC/caffe/issues/1761)
add_custom_target(caffe_proto ALL DEPENDS caffe)
add_custom_command( 
  TARGET caffe_proto
  PRE_BUILD
  COMMAND protoc ${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/proto/caffe.proto 
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/proto 
    --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/proto
  COMMAND ${CMAKE_COMMAND} -E make_directory 
    ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/proto/caffe.pb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto
    COMMENT "Copying from here to there.")