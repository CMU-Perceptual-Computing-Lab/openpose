
# 
# External dependencies
# 

# find_package(THIRDPARTY REQUIRED)
find_package(CUDA QUIET)

# 
# Executable name and options
# 

# Target name
set(target tutorial_wrapper)

# Exit here if required dependencies are not met
message(STATUS "Example ${target}")

# 
# Sources
# 

set(openPoseTutorialWrapper1sources
    1_user_asynchronous.cpp
)

set(openPoseTutorialWrapper2sources
    2_user_synchronous.cpp
)

# 
# Create executable
# 

# Build executable
add_executable(openPoseTutorialWrapper1
    MACOSX_BUNDLE
    ${openPoseTutorialWrapper1sources}
)
add_executable(openPoseTutorialWrapper2
    MACOSX_BUNDLE
    ${openPoseTutorialWrapper2sources}
)

# Create namespaced alias
add_executable(${META_PROJECT_NAME}::openPoseTutorialWrapper1 ALIAS openPoseTutorialWrapper1)
add_executable(${META_PROJECT_NAME}::openPoseTutorialWrapper2 ALIAS openPoseTutorialWrapper2)

# 
# Project options
# 

set_target_properties(openPoseTutorialWrapper1
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)
set_target_properties(openPoseTutorialWrapper2
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)


# 
# Include directories
# 

target_include_directories(openPoseTutorialWrapper1
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(openPoseTutorialWrapper2
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
)


# 
# Libraries
# 

target_link_libraries(openPoseTutorialWrapper1
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)
target_link_libraries(openPoseTutorialWrapper2
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::wrapper
)


# 
# Compile definitions
# 

target_compile_definitions(openPoseTutorialWrapper1
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)
target_compile_definitions(openPoseTutorialWrapper2
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)


# 
# Compile options
# 

target_compile_options(openPoseTutorialWrapper1
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_options(openPoseTutorialWrapper2
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)


# 
# Linker options
# 

target_link_libraries(openPoseTutorialWrapper1
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
target_link_libraries(openPoseTutorialWrapper2
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)

# 
# Deployment
# 

# Executable
install(TARGETS openPoseTutorialWrapper1
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
install(TARGETS openPoseTutorialWrapper2
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)