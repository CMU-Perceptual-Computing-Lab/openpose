set(SOURCES_OP_POSE
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

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_pose ${SOURCES_OP_POSE})
  else ()
    add_library(openpose_pose ${SOURCES_OP_POSE})
  endif ()

  target_link_libraries(openpose_pose openpose_core)

  if (BUILD_CAFFE)
    add_dependencies(openpose_pose openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_pose
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
