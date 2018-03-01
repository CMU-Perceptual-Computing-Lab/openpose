set(SOURCES_OP_GUI
    defineTemplates.cpp
    frameDisplayer.cpp
    gui.cpp
    gui3D.cpp
    guiInfoAdder.cpp
)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
set(SOURCES_OP_GUI_WITH_CP "" PARENT_SCOPE)
prepend(SOURCES_OP_GUI_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_GUI})
set(SOURCES_OP_GUI_WITH_CP ${SOURCES_OP_GUI_WITH_CP} PARENT_SCOPE)
set(SOURCES_OP_FILESTREAM_WITH_CP ${SOURCES_OP_FILESTREAM_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_GUI_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  add_library(openpose_gui ${SOURCES_OP_GUI})
  target_link_libraries(openpose_gui openpose_pose ${OpenCV_LIBS})

  install(TARGETS openpose_gui
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)

