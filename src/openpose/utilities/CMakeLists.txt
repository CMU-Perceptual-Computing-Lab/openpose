list(APPEND SOURCES
    cuda.cpp
    errorAndLog.cpp
    fileSystem.cpp
    flagsToOpenPose.cpp
    keypoint.cpp
    openCv.cpp
    profiler.cpp
    string.cpp
)

if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_utilities ${SOURCES})
else()
    add_library(openpose_utilities ${SOURCES})
endif ()

find_package(Boost COMPONENTS system filesystem REQUIRED)

target_link_libraries(openpose_utilities openpose_filestream openpose_producer
    ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${MKL_LIBS}
    )

install(TARGETS openpose_utilities
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
