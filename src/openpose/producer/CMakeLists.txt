set(SOURCES
  defineTemplates.cpp
  imageDirectoryReader.cpp
  producer.cpp
  videoCaptureReader.cpp
  videoReader.cpp
  webcamReader.cpp)

add_library(producer SHARED ${SOURCES})
target_link_libraries(producer ${OpenCV_LIBS} core thread filestream)