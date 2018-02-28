set(SOURCES_OP_WRAPPER
    defineTemplates.cpp
    wrapperAuxiliary.cpp
    wrapperStructFace.cpp
    wrapperStructHand.cpp
    wrapperStructInput.cpp
    wrapperStructOutput.cpp
    wrapperStructPose.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_WRAPPER_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_WRAPPER})
set(SOURCES_OP_WRAPPER_WITH_CP ${SOURCES_OP_WRAPPER_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_WRAPPER_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
  add_library(openpose_wrapper ${SOURCES_OP_WRAPPER})
  target_link_libraries(openpose_wrapper openpose_thread openpose_pose openpose_hand 
      openpose_core openpose_face openpose_filestream openpose_gui openpose_producer 
      openpose_utilities)

  install(TARGETS openpose_wrapper
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
