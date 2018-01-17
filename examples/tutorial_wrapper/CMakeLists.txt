set(EXAMPLE_FILES 
    1_user_asynchronous_output.cpp
    2_user_synchronous.cpp
    3_user_asynchronous.cpp)
  
include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)
  
  if (UNIX AND NOT APPLE)
    set(EXE_NAME "${SOURCE_NAME}_bin")
  elseif (WIN32)
    set(EXE_NAME "${SOURCE_NAME}")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  add_executable(${EXE_NAME} ${EXAMPLE_FILE})
  target_link_libraries( ${EXE_NAME} openpose ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS})
  
  if (WIN32)
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples/Tutorial/Wrapper")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose.vcxproj.user 
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY) 
  endif (WIN32)

endforeach()