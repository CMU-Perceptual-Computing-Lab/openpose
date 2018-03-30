if (WITH_3D_ADAM_MODEL)
  file(GLOB JSON_SOURCES_OP_ADAM RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "lib_json/*.cpp")
  file(GLOB SOURCES_OP_ADAM RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
  # set(SOURCES_OP_ADAM
  #     CMeshModelinstance.cpp
  #     FitToBody.cpp
  #     poseTriangulation.cpp)

  include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
  prepend(SOURCES_OP_ADAM_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_ADAM} ${JSON_SOURCES_OP_ADAM})
  set(SOURCES_OP_ADAM_WITH_CP ${SOURCES_OP_ADAM_WITH_CP} PARENT_SCOPE)
  set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_ADAM_WITH_CP} PARENT_SCOPE)

  if (UNIX AND NOT APPLE)
    if (${GPU_MODE} MATCHES "CUDA")
      cuda_add_library(openpose_adam ${SOURCES_OP_ADAM} ${JSON_SOURCES_OP_ADAM})
    else()
      add_library(openpose_adam ${SOURCES_OP_ADAM} ${JSON_SOURCES_OP_ADAM})
    endif ()

    add_library(caffe SHARED IMPORTED)
    set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS}) 
    target_link_libraries(openpose_adam caffe ${MKL_LIBS})
    target_link_libraries(openpose_adam ${OpenCV_LIBS} ${CERES_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY} ${FREE_IMAGE_LIBRARY})

    if (BUILD_CAFFE)
      add_dependencies(openpose_adam openpose_caffe)
    endif (BUILD_CAFFE)

    install(TARGETS openpose_adam
        EXPORT OpenPose
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/openpose)
  endif (UNIX AND NOT APPLE)
endif (WITH_3D_ADAM_MODEL)
