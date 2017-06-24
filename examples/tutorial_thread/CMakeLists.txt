
# 
# External dependencies
# 

# find_package(THIRDPARTY REQUIRED)
find_package(CUDA QUIET)

# 
# Executable name and options
# 

# Target name
set(target tutorial_thread)

# Exit here if required dependencies are not met
message(STATUS "Example ${target}")

# 
# Sources
# 

set(openPoseTutorialThread1sources
    1_openpose_read_and_display.cpp
)

set(openPoseTutorialThread2sources
    2_user_processing_function.cpp
)

set(openPoseTutorialThread3sources
    3_user_input_processing_and_output.cpp
)

set(openPoseTutorialThread4sources
    4_user_input_processing_output_and_datum.cpp
)

# 
# Create executable
# 

# Build executable
add_executable(openPoseTutorialThread1
    MACOSX_BUNDLE
    ${openPoseTutorialThread1sources}
)
add_executable(openPoseTutorialThread2
    MACOSX_BUNDLE
    ${openPoseTutorialThread2sources}
)
add_executable(openPoseTutorialThread3
    MACOSX_BUNDLE
    ${openPoseTutorialThread3sources}
)
add_executable(openPoseTutorialThread4
    MACOSX_BUNDLE
    ${openPoseTutorialThread4sources}
)

# Create namespaced alias
add_executable(${META_PROJECT_NAME}::openPoseTutorialThread1 ALIAS openPoseTutorialThread1)
add_executable(${META_PROJECT_NAME}::openPoseTutorialThread2 ALIAS openPoseTutorialThread2)
add_executable(${META_PROJECT_NAME}::openPoseTutorialThread3 ALIAS openPoseTutorialThread3)
add_executable(${META_PROJECT_NAME}::openPoseTutorialThread4 ALIAS openPoseTutorialThread4)

# 
# Project options
# 

set_target_properties(openPoseTutorialThread1
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)
set_target_properties(openPoseTutorialThread2
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)
set_target_properties(openPoseTutorialThread3
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)
set_target_properties(openPoseTutorialThread4
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)


# 
# Include directories
# 

target_include_directories(openPoseTutorialThread1
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(openPoseTutorialThread2
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(openPoseTutorialThread3
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(openPoseTutorialThread4
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)


# 
# Libraries
# 

target_link_libraries(openPoseTutorialThread1
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)
target_link_libraries(openPoseTutorialThread2
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)
target_link_libraries(openPoseTutorialThread3
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)
target_link_libraries(openPoseTutorialThread4
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)


# 
# Compile definitions
# 

target_compile_definitions(openPoseTutorialThread1
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)
target_compile_definitions(openPoseTutorialThread2
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)
target_compile_definitions(openPoseTutorialThread3
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)
target_compile_definitions(openPoseTutorialThread4
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)


# 
# Compile options
# 

target_compile_options(openPoseTutorialThread1
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_options(openPoseTutorialThread2
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_options(openPoseTutorialThread3
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_options(openPoseTutorialThread4
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)


# 
# Linker options
# 

target_link_libraries(openPoseTutorialThread1
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
target_link_libraries(openPoseTutorialThread2
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
target_link_libraries(openPoseTutorialThread3
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
target_link_libraries(openPoseTutorialThread4
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)

# 
# Deployment
# 

# Executable
install(TARGETS openPoseTutorialThread1
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
install(TARGETS openPoseTutorialThread2
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
install(TARGETS openPoseTutorialThread3
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
install(TARGETS openPoseTutorialThread4
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)