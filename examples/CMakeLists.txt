
# Check if examples are enabled
if(NOT OPTION_BUILD_EXAMPLES)
    return()
endif()

if(OPTION_USE_CAFFE)
    ADD_DEFINITIONS( -DUSE_CAFFE )
endif()

file(GLOB_RECURSE EXAMPLE_FILES *.cpp)

foreach(file ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS " Adding Example ${SOURCE_NAME} ")
  
  add_executable(${SOURCE_NAME} ${file} )

  set_target_properties(${SOURCE_NAME}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
  )

  include_directories(${DEFAULT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/include)

  target_compile_options(${SOURCE_NAME} PUBLIC ${DEFAULT_COMPILE_OPTIONS})
  
  target_link_libraries(${SOURCE_NAME}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
  )

  target_link_libraries(${SOURCE_NAME}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
  )

  target_compile_definitions(${SOURCE_NAME}
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
  )

  install(TARGETS ${SOURCE_NAME}
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
  )

endforeach()