set(SOURCES_OP_3D
    cameraParameterReader.cpp
    defineTemplates.cpp
    poseTriangulation.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_3D_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_3D})
set(SOURCES_OP_3D_WITH_CP ${SOURCES_OP_3D_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_3D_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_3d ${SOURCES_OP_3D})
  else()
    add_library(openpose_3d ${SOURCES_OP_3D})
  endif ()

  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 
  target_link_libraries(openpose_3d caffe ${MKL_LIBS})

  if (BUILD_CAFFE)
    add_dependencies(openpose_3d openpose_caffe)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_3d
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
