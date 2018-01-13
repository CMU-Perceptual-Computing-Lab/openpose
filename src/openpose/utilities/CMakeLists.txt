set(SOURCES
    cuda.cpp
    errorAndLog.cpp
    fileSystem.cpp
    flagsToOpenPose.cpp
    keypoint.cpp
    openCv.cpp
    profiler.cpp
    string.cpp)

cuda_add_library(openpose_utilities ${SOURCES})
target_link_libraries(openpose_utilities openpose_filestream openpose_producer)

install(TARGETS openpose_utilities
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
