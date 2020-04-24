set(EXAMPLE_FILES
    01_body_from_image_default.cpp
    02_whole_body_from_image_default.cpp
    03_keypoints_from_image.cpp
    04_keypoints_from_images.cpp
    05_keypoints_from_images_multi_gpu.cpp
    06_face_from_image.cpp
    07_hand_from_image.cpp
    08_heatmaps_from_image.cpp
    09_keypoints_from_heatmaps.cpp
    10_asynchronous_custom_input.cpp
    11_asynchronous_custom_input_multi_camera.cpp
    12_asynchronous_custom_output.cpp
    13_asynchronous_custom_input_output_and_datum.cpp
    14_synchronous_custom_input.cpp
    15_synchronous_custom_preprocessing.cpp
    16_synchronous_custom_postprocessing.cpp
    17_synchronous_custom_output.cpp
    18_synchronous_custom_all_and_datum.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)

  if (UNIX OR APPLE)
    set(EXE_NAME "${SOURCE_NAME}.bin")
  elseif (WIN32)
    set(EXE_NAME "${SOURCE_NAME}")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  add_executable(${EXE_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${EXE_NAME} openpose ${examples_3rdparty_libraries})

  if (WIN32)
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples/Tutorial/C++ API")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${VCXPROJ_FILE_GPU_MODE}.vcxproj.user
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
    # Properties->General->Output Directory
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  endif (WIN32)

endforeach()
