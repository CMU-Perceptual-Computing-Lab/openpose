set(SOURCES  
    cuda.cpp
    errorAndLog.cpp
    fileSystem.cpp
    flagsToOpenPose.cpp
    keypoint.cpp
    openCv.cpp
    profiler.cpp
    string.cpp)

find_package(Boost COMPONENTS system filesystem REQUIRED)
  
cuda_add_library(openpose_utilities ${SOURCES})
target_link_libraries(openpose_utilities openpose_producer openpose_filestream
    ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

install(TARGETS openpose_utilities
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
