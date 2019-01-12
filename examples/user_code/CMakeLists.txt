# Uncomment these lines with your custom file names
# set(USER_CODE_FILES 
#     ADD_HERE_YOUR_FILE1.cpp
#     ADD_HERE_YOUR_FILE1.hpp
#     ADD_HERE_YOUR_FILE2.cpp
#     # ...
#     ADD_HERE_YOUR_FILEn.cpp)

foreach(USER_CODE_FILE ${USER_CODE_FILES})

  get_filename_component(SOURCE_NAME ${USER_CODE_FILE} NAME_WE)

  if (UNIX OR APPLE)
    set(EXE_NAME "${SOURCE_NAME}.bin")
  elseif (WIN32)
    set(EXE_NAME "UserCustomCode")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  add_executable(${EXE_NAME} ${USER_CODE_FILE})
  target_link_libraries(${EXE_NAME} openpose ${examples_3rdparty_libraries})

  if (WIN32)
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "User Code")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${VCXPROJ_FILE_GPU_MODE}.vcxproj.user
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
    # Properties->General->Output Directory
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  endif (WIN32)

endforeach()
