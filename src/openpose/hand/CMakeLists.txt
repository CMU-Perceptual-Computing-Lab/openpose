set(SOURCES 
  defineTemplates.cpp
  handDetector.cpp
  handDetectorFromTxt.cpp
  handExtractor.cpp
  handRenderer.cpp
  renderHand.cpp
  renderHand.cu)

cuda_add_library(hand SHARED ${SOURCES})
target_link_libraries(hand ${Caffe_LIBS})
add_dependencies(hand caffe)