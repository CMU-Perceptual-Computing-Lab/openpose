set(SOURCES 
    bodyPartConnectorBase.cpp
    bodyPartConnectorBase.cu
    bodyPartConnectorCaffe.cpp
    defineTemplates.cpp
    poseExtractor.cpp
    poseExtractorCaffe.cpp
    poseParameters.cpp
    poseRenderer.cpp
    renderPose.cpp
    renderPose.cu)

cuda_add_library(pose ${SOURCES})
if (BUILD_CAFFE)
  add_dependencies(pose caffe)
endif (BUILD_CAFFE)

install(TARGETS pose
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)

