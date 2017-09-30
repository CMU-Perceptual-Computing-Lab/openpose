set(SOURCES
    defineTemplates.cpp)

add_library(thread ${SOURCES})

install(TARGETS thread 
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
