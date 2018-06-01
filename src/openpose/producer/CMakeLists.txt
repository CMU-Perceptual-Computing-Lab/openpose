set(SOURCES_OP_PRODUCER
    defineTemplates.cpp
    flirReader.cpp
    imageDirectoryReader.cpp
    ipCameraReader.cpp
    producer.cpp
    spinnakerWrapper.cpp
    videoCaptureReader.cpp
    videoReader.cpp
    webcamReader.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_PRODUCER_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_PRODUCER})
set(SOURCES_OP_PRODUCER_WITH_CP ${SOURCES_OP_PRODUCER_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_PRODUCER_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  add_library(openpose_producer ${SOURCES_OP_PRODUCER})
  target_link_libraries(openpose_producer ${OpenCV_LIBS} openpose_core 
      openpose_thread openpose_filestream)

  install(TARGETS openpose_producer
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
