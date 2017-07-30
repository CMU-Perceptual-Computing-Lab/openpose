set(SOURCES SHARED
  defineTemplates.cpp
  faceDetector.cpp
  faceExtractor.cpp
  faceRenderer.cpp
  renderFace.cpp
  renderFace.cu
)

cuda_add_library(face ${SOURCES})
target_link_libraries(face caffe)
# target_include_directories()