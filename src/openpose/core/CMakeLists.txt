cuda_add_library(core
    array.cpp
    cvMatToOpInput.cpp
    cvMatToOpOutput.cpp
    datum.cpp
    defineTemplates.cpp
    gpuRenderer.cpp
    keypointScaler.cpp
    maximumBase.cpp
    maximumBase.cu
    maximumCaffe.cpp
    netCaffe.cpp
    nmsBase.cpp
    nmsBase.cu
    nmsCaffe.cpp
    opOutputToCvMat.cpp
    point.cpp
    rectangle.cpp
    renderer.cpp
    resizeAndMergeBase.cpp
    resizeAndMergeBase.cu
    resizeAndMergeCaffe.cpp)

target_link_libraries(core ${Caffe_LIBS})
if (BUILD_CAFFE)
  add_dependencies(core caffe)
endif (BUILD_CAFFE)
