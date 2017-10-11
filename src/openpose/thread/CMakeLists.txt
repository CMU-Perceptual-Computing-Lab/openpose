set(SOURCES
    defineTemplates.cpp)

add_library(openpose_thread ${SOURCES})

install(TARGETS openpose_thread 
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
