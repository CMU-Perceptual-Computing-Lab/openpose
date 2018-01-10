list(APPEND SOURCES
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

if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_face ${SOURCES})
else()
    add_library(openpose_face ${SOURCES})
endif ()

if (BUILD_CAFFE)
  add_dependencies(openpose_face openpose_caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_face
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
