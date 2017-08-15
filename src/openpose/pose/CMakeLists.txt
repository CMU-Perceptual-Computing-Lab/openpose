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

cuda_add_library(pose SHARED ${SOURCES})
target_link_libraries(pose ${Caffe_LIBS})
add_dependencies(pose caffe)