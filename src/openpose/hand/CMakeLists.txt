set(SOURCES 
  defineTemplates.cpp
  handDetector.cpp
  handDetectorFromTxt.cpp
  handExtractor.cpp
  handRenderer.cpp
  renderHand.cpp
  renderHand.cu)

cuda_add_library(hand SHARED ${SOURCES})

# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()

# #  cuda_add_library(hand ${SOURCES} ${CMAKE_SOURCE_DIR}/include ${Caffe_INCLUDE_DIRS})

# if (${Caffe_LIBRARIES})
#   cuda_add_library(hand SHARED ${SOURCES} ${CMAKE_SOURCE_DIR}/include ${Caffe_INCLUDE_DIRS})
#   target_include_directories(hand ${Caffe_INCLUDE_DIRS})
#   target_link_libraries(hand ${Caffe_LIBRARIES}  utilities core)
# else(${Caffe_LIBRARIES})
#   cuda_add_library(hand  ${SOURCES} ${CMAKE_SOURCE_DIR}/include ${Caffe_INCLUDE_DIRS})
#   target_link_libraries(hand caffe utilities core)
#   # add_dependencies(hand caffe)
# endif (${Caffe_LIBRARIES})