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

add_library(filestream SHARED ${SOURCES})