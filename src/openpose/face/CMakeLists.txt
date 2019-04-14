set(SOURCES_OP_FACE
    defineTemplates.cpp
    faceDetector.cpp
    faceDetectorOpenCV.cpp
    faceExtractorCaffe.cpp
    faceExtractorNet.cpp
    faceCpuRenderer.cpp
    faceGpuRenderer.cpp
    faceRenderer.cpp
    renderFace.cpp
    renderFace.cu)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_FACE_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_FACE})
set(SOURCES_OP_FACE_WITH_CP ${SOURCES_OP_FACE_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_FACE_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_face ${SOURCES_OP_FACE})
  else ()
    add_library(openpose_face ${SOURCES_OP_FACE})
  endif ()

  target_link_libraries(openpose_face openpose_core)
  
  if (BUILD_CAFFE)
    add_dependencies(openpose_face openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_face 
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
