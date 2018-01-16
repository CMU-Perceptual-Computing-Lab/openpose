if (WIN32)

  set(OPENPOSE_3D_INCLUDES include)
  file(GLOB_RECURSE OPENPOSE_3D_SOURCES "*.cpp")

  set(EXAMPLE_FILES openpose3d.cpp)

  include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)

  foreach (EXAMPLE_FILE ${EXAMPLE_FILES})

    get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)
    
    set(EXE_NAME "${SOURCE_NAME}")
    
    message(STATUS "Adding Example ${EXE_NAME}")
    add_executable(${EXE_NAME} ${EXAMPLE_FILE} ${OPENPOSE_3D_SOURCES})
    target_include_directories(${EXE_NAME} PUBLIC ${OPENPOSE_3D_INCLUDES})
    target_link_libraries(${EXE_NAME} openpose ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS} 
        ${Freeglut_LIB} ${Spinnaker_LIB})
    
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples/Tutorial/OpenPose3DReconstruction")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose3D.vcxproj.user 
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY) 
  
  endforeach ()

endif (WIN32)