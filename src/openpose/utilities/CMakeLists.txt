set(SOURCES_OP_UTILITIES
    errorAndLog.cpp
    fileSystem.cpp
    flagsToOpenPose.cpp
    keypoint.cpp
    openCv.cpp
    profiler.cpp
    string.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_UTILITIES_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_UTILITIES})
set(SOURCES_OP_UTILITIES_WITH_CP ${SOURCES_OP_UTILITIES_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_UTILITIES_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_utilities ${SOURCES_OP_UTILITIES})
  else ()
    add_library(openpose_utilities ${SOURCES_OP_UTILITIES})
  endif ()

  target_link_libraries(openpose_utilities openpose_producer openpose_filestream)

  install(TARGETS openpose_utilities
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
