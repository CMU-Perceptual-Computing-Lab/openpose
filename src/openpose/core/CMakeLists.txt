set(SOURCES_OP_CORE
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
    scaleAndSizeExtractor.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_CORE_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_CORE})
set(SOURCES_OP_CORE_WITH_CP ${SOURCES_OP_CORE_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_CORE_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_core ${SOURCES_OP_CORE})
  else()
    add_library(openpose_core ${SOURCES_OP_CORE})
  endif ()

  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 
  target_link_libraries(openpose_core caffe ${MKL_LIBS})

  if (BUILD_CAFFE)
    add_dependencies(openpose_core openpose_caffe)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_core
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
