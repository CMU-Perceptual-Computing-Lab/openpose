
file(GLOB_RECURSE EXAMPLE_FILES *.cpp)

foreach(file ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS " Adding Example ${SOURCE_NAME} ")
  
  add_executable(${SOURCE_NAME}.bin ${file} )

  # set_target_properties(${SOURCE_NAME}
  #   PROPERTIES
  #   ${DEFAULT_PROJECT_OPTIONS}
  #   FOLDER "${IDE_FOLDER}"
  # )

  target_link_libraries( ${SOURCE_NAME}.bin
     wrapper utilities ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBRARIES}
  )

  # install(TARGETS ${SOURCE_NAME}
  #   RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
  #   BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
  # )

endforeach()