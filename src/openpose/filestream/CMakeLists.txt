set(SOURCES cocoJsonSaver.cpp
    defineTemplates.cpp
    fileSaver.cpp
    fileStream.cpp
    heatMapSaver.cpp
    imageSaver.cpp
    jsonOfstream.cpp
    keypointSaver.cpp
    peopleJsonSaver.cpp
    videoSaver.cpp)

add_library(openpose_filestream ${SOURCES})

install(TARGETS openpose_filestream
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
