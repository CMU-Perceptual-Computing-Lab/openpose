set(SOURCES
    defineTemplates.cpp
    imageDirectoryReader.cpp
    producer.cpp
    videoCaptureReader.cpp
    videoReader.cpp
    webcamReader.cpp)

add_library(producer ${SOURCES})
target_link_libraries(producer ${OpenCV_LIBS} core thread filestream)

install(TARGETS producer
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
