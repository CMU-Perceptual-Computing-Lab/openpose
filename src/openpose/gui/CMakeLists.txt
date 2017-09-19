set(SOURCES
    CMakeLists.txt
    defineTemplates.cpp
    frameDisplayer.cpp
    gui.cpp
    guiInfoAdder.cpp
)

add_library(gui ${SOURCES})
target_link_libraries(gui pose ${OpenCV_LIBS})
