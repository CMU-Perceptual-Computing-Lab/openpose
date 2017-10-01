set(SOURCES
    CMakeLists.txt
    defineTemplates.cpp
    frameDisplayer.cpp
    gui.cpp
    guiInfoAdder.cpp
)

add_library(openpose_gui ${SOURCES})
target_link_libraries(openpose_gui openpose_pose ${OpenCV_LIBS})

install(TARGETS openpose_gui
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)

