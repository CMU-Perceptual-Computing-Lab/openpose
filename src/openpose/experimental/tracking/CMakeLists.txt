set(SOURCES
    defineTemplates.cpp
    personIdExtractor.cpp)

add_library(openpose_tracking ${SOURCES})

install(TARGETS openpose_tracking
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
