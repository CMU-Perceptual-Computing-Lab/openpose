set(SOURCES cocoJsonSaver.cpp
    defineTemplates.cpp
    fileSaver.cpp
    fileStream.cpp
    heatMapSaver.cpp
    imageSaver.cpp
    jsonOfstream.cpp
    keypointJsonSaver.cpp
    keypointSaver.cpp
    videoSaver.cpp)

add_library(filestream ${SOURCES})

install(TARGETS filestream
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
