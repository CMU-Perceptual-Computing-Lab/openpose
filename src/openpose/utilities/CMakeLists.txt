set(SOURCES_OP_UTILITIES 
    cuda.cpp
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

if (UNIX AND NOT APPLE)
<<<<<<< HEAD
  cuda_add_library(openpose_utilities ${SOURCES})
  target_link_libraries(openpose_utilities openpose_producer openpose_filestream
      debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} debug ${Boost_SYSTEM_LIBRARY_DEBUG}
      optimized ${Boost_FILESYSTEM_LIBRARY_OPTIMIZED} optimized ${Boost_SYSTEM_LIBRARY_OPTIMIZED})
=======
  cuda_add_library(openpose_utilities ${SOURCES_OP_UTILITIES})
  target_link_libraries(openpose_utilities openpose_producer openpose_filestream
      ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
>>>>>>> 5a640b34e58050acb0eff2dcb2024a8b17185db6

  install(TARGETS openpose_utilities
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
<<<<<<< HEAD
endif (UNIX AND NOT APPLE)
=======
endif (UNIX AND NOT APPLE)
>>>>>>> 5a640b34e58050acb0eff2dcb2024a8b17185db6
