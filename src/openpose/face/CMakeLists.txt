set(SOURCES
    defineTemplates.cpp
    faceDetector.cpp
    faceDetectorOpenCV.cpp
    faceExtractor.cpp
    faceExtractorCaffe.cpp
    faceCpuRenderer.cpp
    faceGpuRenderer.cpp
    renderFace.cpp
    renderFace.cu
)

cuda_add_library(openpose_face ${SOURCES})
if (UNIX AND NOT APPLE)
  if (BUILD_CAFFE)
    add_dependencies(openpose_face openpose_caffe)
  endif (BUILD_CAFFE)
endif (UNIX AND NOT APPLE)

install(TARGETS openpose_face 
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
