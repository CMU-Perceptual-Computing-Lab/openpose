set(SOURCES
    defineTemplates.cpp)

add_library(thread ${SOURCES})

install(TARGETS thread 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
