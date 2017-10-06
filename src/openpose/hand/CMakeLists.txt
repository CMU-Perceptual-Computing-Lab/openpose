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

cuda_add_library(hand ${SOURCES})
if (BUILD_CAFFE)
  add_dependencies(hand caffe)
endif (BUILD_CAFFE)
