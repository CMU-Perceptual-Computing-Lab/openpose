cuda_add_library(openpose_core
    array.cpp
    cvMatToOpInput.cpp
    cvMatToOpOutput.cpp
    datum.cpp
    defineTemplates.cpp
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

target_link_libraries(openpose_core ${Caffe_LIBS})
if (BUILD_CAFFE)
  add_dependencies(openpose_core caffe)
endif (BUILD_CAFFE)

install(TARGETS openpose_core
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
