list(APPEND SOURCES
    bodyPartConnectorBase.cpp
    bodyPartConnectorBase.cu
    bodyPartConnectorCaffe.cpp
    defineTemplates.cpp
    poseCpuRenderer.cpp
    poseExtractor.cpp
    poseExtractorCaffe.cpp
    poseGpuRenderer.cpp
    poseParameters.cpp
    poseParametersRender.cpp
    poseRenderer.cpp
    renderPose.cpp
    renderPose.cu
)

if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_pose ${SOURCES})
else()
    add_library(openpose_pose ${SOURCES})
endif ()

if (BUILD_CAFFE)
  add_dependencies(openpose_pose openpose_caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_pose
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
