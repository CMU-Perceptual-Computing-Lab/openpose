set(EXAMPLE_FILES 
    calibration.cpp)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)

  if (UNIX AND NOT APPLE)
    set(EXE_NAME "${SOURCE_NAME}.bin")
  elseif (WIN32)
    set(EXE_NAME "Calibration")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  add_executable(${EXE_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${EXE_NAME} openpose ${examples_3rdparty_libraries})

  if (WIN32)
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${CL}.vcxproj.user
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
    # Properties->General->Output Directory
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  endif (WIN32)

endforeach()
