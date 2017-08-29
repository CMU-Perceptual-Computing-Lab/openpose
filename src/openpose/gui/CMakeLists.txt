set(SOURCES
    CMakeLists.txt
    defineTemplates.cpp
    frameDisplayer.cpp
    gui.cpp
    guiInfoAdder.cpp
)

add_library(gui SHARED ${SOURCES})
target_link_libraries(gui pose)