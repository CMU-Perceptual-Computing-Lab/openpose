set(SOURCES_OP_HAND
    defineTemplates.cpp
    handDetector.cpp
    handDetectorFromTxt.cpp
    handExtractorCaffe.cpp
    handExtractorNet.cpp
    handCpuRenderer.cpp
    handGpuRenderer.cpp
    handRenderer.cpp
    renderHand.cpp
    renderHand.cu)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_HAND_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_HAND})
set(SOURCES_OP_HAND_WITH_CP ${SOURCES_OP_HAND_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_HAND_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_hand ${SOURCES_OP_HAND})
  else()
    add_library(openpose_hand ${SOURCES_OP_HAND})
  endif ()
  
  if (BUILD_CAFFE)
    add_dependencies(openpose_hand openpose_caffe)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_hand
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
