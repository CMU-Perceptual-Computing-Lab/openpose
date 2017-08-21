set(SOURCES
  defineTemplates.cpp
  faceDetector.cpp
  faceExtractor.cpp
  faceRenderer.cpp
  renderFace.cpp
  renderFace.cu
)

cuda_add_library(face SHARED ${SOURCES})
target_link_libraries(face ${Caffe_LIBS})
if (BUILD_CAFFE)
  add_dependencies(face caffe)
endif (BUILD_CAFFE)
