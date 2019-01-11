set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;mm;CPP;cl)
set(SOURCES_OP_CALIBRATION
    cameraParameterEstimation.cpp
    gridPatternFunctions.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_CALIBRATION_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_CALIBRATION})
set(SOURCES_OP_CALIBRATION_WITH_CP ${SOURCES_OP_CALIBRATION_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_CALIBRATION_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_calibration ${SOURCES_OP_CALIBRATION})
  else ()
    add_library(openpose_calibration ${SOURCES_OP_CALIBRATION})
  endif ()

  target_link_libraries(openpose_calibration openpose_core)

  if (BUILD_CAFFE)
    add_dependencies(openpose_calibration openpose)
  endif (BUILD_CAFFE)

  install(TARGETS openpose_calibration
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
