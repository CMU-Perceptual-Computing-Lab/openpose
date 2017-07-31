add_subdirectory(caffe)

# Reference --> [Here](https://github.com/BVLC/caffe/issues/1761)
# Is there a better way ?
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

# add_custom_target(copy_caffe_includes ALL DEPENDS caffe)
# add_custom_command(
#   TARGET copy_caffe_includes
#   PRE_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include 
#     ${CMAKE_SOURCE_DIR}/build/3rdparty/caffe/include
#   COMMENT "Copying caffe includes to build directory.")

