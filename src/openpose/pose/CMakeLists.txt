set(SOURCES 
    bodyPartConnectorBase.cpp
    bodyPartConnectorBase.cu
    bodyPartConnectorCaffe.cpp
    defineTemplates.cpp
    poseCpuRenderer.cpp
    poseExtractor.cpp
    poseExtractorCaffe.cpp
    poseGpuRenderer.cpp
    poseParameters.cpp
    poseRenderer.cpp
    renderPose.cpp
    renderPose.cu)

cuda_add_library(openpose_pose ${SOURCES})
if (BUILD_CAFFE)
  add_dependencies(openpose_pose openpose_caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_pose
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)

