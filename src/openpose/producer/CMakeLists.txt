set(SOURCES
    defineTemplates.cpp
    imageDirectoryReader.cpp
    ipCameraReader.cpp
    producer.cpp
    videoCaptureReader.cpp
    videoReader.cpp
    webcamReader.cpp)

add_library(openpose_producer ${SOURCES})
target_link_libraries(openpose_producer ${OpenCV_LIBS} openpose_core 
    openpose_thread openpose_filestream)

install(TARGETS openpose_producer
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
