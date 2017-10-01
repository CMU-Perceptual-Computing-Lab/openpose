set(SOURCES
    defineTemplates.cpp
    faceDetector.cpp
    faceExtractor.cpp
    faceRenderer.cpp
    renderFace.cpp
    renderFace.cu
)

cuda_add_library(openpose_face ${SOURCES})
if (BUILD_CAFFE)
  add_dependencies(openpose_face caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_face 
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
