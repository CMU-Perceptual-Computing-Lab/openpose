set(SOURCES 
    defineTemplates.cpp
    handDetector.cpp
    handDetectorFromTxt.cpp
    handExtractor.cpp
    handExtractorCaffe.cpp
    handCpuRenderer.cpp
    handGpuRenderer.cpp
    renderHand.cpp
    renderHand.cu)

cuda_add_library(openpose_hand ${SOURCES})
if (UNIX AND NOT APPLE)
  if (BUILD_CAFFE)
    add_dependencies(openpose_hand openpose_caffe)
  endif (BUILD_CAFFE)
endif (UNIX AND NOT APPLE)

install(TARGETS openpose_hand
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
