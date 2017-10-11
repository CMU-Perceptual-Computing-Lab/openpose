cuda_add_library(openpose_core
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
    resizeAndMergeCaffe.cpp
    scaleAndSizeExtractor)

add_library(caffe SHARED IMPORTED)
set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 

target_link_libraries(openpose_core caffe)

if (BUILD_CAFFE)
  add_dependencies(openpose_core openpose_caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_core
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
