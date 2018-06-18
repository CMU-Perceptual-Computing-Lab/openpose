set(SOURCES_OP_FILESTREAM bvhSaver.cpp
    cocoJsonSaver.cpp
    defineTemplates.cpp
    fileSaver.cpp
    fileStream.cpp
    heatMapSaver.cpp
    imageSaver.cpp
    jsonOfstream.cpp
    keypointSaver.cpp
    peopleJsonSaver.cpp
    udpSender.cpp
    videoSaver.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_FILESTREAM_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_FILESTREAM})
set(SOURCES_OP_FILESTREAM_WITH_CP ${SOURCES_OP_FILESTREAM_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_FILESTREAM_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  add_library(openpose_filestream ${SOURCES_OP_FILESTREAM})

  target_link_libraries(openpose_filestream openpose_core)

  install(TARGETS openpose_filestream
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
