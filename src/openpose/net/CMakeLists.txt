set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;mm;CPP;cl)
set(SOURCES_OP_NET
    bodyPartConnectorBase.cpp
    bodyPartConnectorBase.cu
    bodyPartConnectorBaseCL.cpp
    bodyPartConnectorCaffe.cpp
    maximumBase.cpp
    maximumBase.cu
    maximumCaffe.cpp
    netCaffe.cpp
    netOpenCv.cpp
    nmsBase.cpp
    nmsBase.cu
    nmsBaseCL.cpp
    nmsCaffe.cpp
    resizeAndMergeBase.cpp
    resizeAndMergeBase.cu
    resizeAndMergeBaseCL.cpp
    resizeAndMergeCaffe.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_NET_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_NET})
set(SOURCES_OP_NET_WITH_CP ${SOURCES_OP_NET_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_NET_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_net ${SOURCES_OP_NET})
  else ()
    add_library(openpose_net ${SOURCES_OP_NET})
  endif ()

  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 
  target_link_libraries(openpose_net caffe ${MKL_LIBS} openpose_core)

  if (BUILD_CAFFE)
    add_dependencies(openpose_net openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_net
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
