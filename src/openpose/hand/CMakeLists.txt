set(SOURCES 
    defineTemplates.cpp
    handDetector.cpp
    handDetectorFromTxt.cpp
    handExtractor.cpp
    handRenderer.cpp
    renderHand.cpp
    renderHand.cu)

cuda_add_library(hand ${SOURCES})
if (BUILD_CAFFE)
  add_dependencies(hand caffe)
endif (BUILD_CAFFE)

install(TARGETS hand
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
