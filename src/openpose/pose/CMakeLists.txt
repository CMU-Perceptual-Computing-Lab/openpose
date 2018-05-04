set(SOURCES_OP_POSE 
    bodyPartConnectorBase.cpp
    bodyPartConnectorBase.cu
    bodyPartConnectorCaffe.cpp
    defineTemplates.cpp
    poseCpuRenderer.cpp
    poseExtractor.cpp
    poseExtractorCaffe.cpp
    poseExtractorNet.cpp
    poseGpuRenderer.cpp
    poseParameters.cpp
    poseParametersRender.cpp
    poseRenderer.cpp
    renderPose.cpp
    renderPose.cu)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_POSE_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_POSE})
set(SOURCES_OP_POSE_WITH_CP ${SOURCES_OP_POSE_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_POSE_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_pose ${SOURCES_OP_POSE})
  else()
    add_library(openpose_pose ${SOURCES_OP_POSE})
  endif ()
  
  if (BUILD_CAFFE)
    add_dependencies(openpose_pose openpose_caffe)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_pose
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
