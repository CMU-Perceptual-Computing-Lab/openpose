set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;mm;CPP;cl)
set(SOURCES_OP_CORE
    array.cpp
    arrayCpuGpu.cpp
    cvMatToOpInput.cpp
    cvMatToOpOutput.cpp
    datum.cpp
    defineTemplates.cpp
    gpuRenderer.cpp
    keepTopNPeople.cpp
    keypointScaler.cpp
    opOutputToCvMat.cpp
    point.cpp
    rectangle.cpp
    renderer.cpp
    scaleAndSizeExtractor.cpp
    verbosePrinter.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_CORE_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_CORE})
set(SOURCES_OP_CORE_WITH_CP ${SOURCES_OP_CORE_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_CORE_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_core ${SOURCES_OP_CORE})
  else ()
    add_library(openpose_core ${SOURCES_OP_CORE})
  endif ()

  if (APPLE)
    target_link_libraries(openpose_core openpose caffe  ${OpenCV_LIBS} ${Caffe_LIBS} ${GLUT_LIBRARY} ${SPINNAKER_LIB} ${OpenCL_LIBRARIES}
          ${GLOG_LIBRARY} ${OpenCV_LIBS} ${Caffe_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${MKL_LIBS} ${GLUT_LIBRARY} ${SPINNAKER_LIB})
    add_library(caffe SHARED IMPORTED)
    set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS})
  endif (APPLE)

  if (BUILD_CAFFE)
    add_dependencies(openpose_core openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_core
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
