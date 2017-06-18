
# 
# External dependencies
# 

# find_package(THIRDPARTY REQUIRED)
find_package(CUDA QUIET)

# 
# Executable name and options
# 

# Target name
set(target tutorial_pose)

# Exit here if required dependencies are not met
message(STATUS "Example ${target}")

# 
# Sources
# 

set(openPoseTutorialPose1sources
    1_extract_from_image.cpp
)

set(openPoseTutorialPose2sources
    2_extract_pose_or_heatmat_from_image.cpp
)

# 
# Create executable
# 

# Build executable
add_executable(openPoseTutorialPose1
    MACOSX_BUNDLE
    ${openPoseTutorialPose1sources}
)
add_executable(openPoseTutorialPose2
    MACOSX_BUNDLE
    ${openPoseTutorialPose1sources}
)

# Create namespaced alias
add_executable(${META_PROJECT_NAME}::openPoseTutorialPose1 ALIAS openPoseTutorialPose1)
add_executable(${META_PROJECT_NAME}::openPoseTutorialPose2 ALIAS openPoseTutorialPose2)


# 
# Project options
# 

set_target_properties(openPoseTutorialPose1
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)
set_target_properties(openPoseTutorialPose2
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)


# 
# Include directories
# 

target_include_directories(openPoseTutorialPose1
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
    ${Caffe_INCLUDE_DIRS}
)
target_include_directories(openPoseTutorialPose2
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}/include
    ${Caffe_INCLUDE_DIRS}
)


# 
# Libraries
# 

target_link_libraries(openPoseTutorialPose1
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${Caffe_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${META_PROJECT_NAME}::core
    ${META_PROJECT_NAME}::face
    ${META_PROJECT_NAME}::producer
    ${META_PROJECT_NAME}::wrapper
)
target_link_libraries(openPoseTutorialPose2
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${Caffe_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${META_PROJECT_NAME}::core
    ${META_PROJECT_NAME}::face
    ${META_PROJECT_NAME}::producer
    ${META_PROJECT_NAME}::wrapper
)


# 
# Compile definitions
# 

target_compile_definitions(openPoseTutorialPose1
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)
target_compile_definitions(openPoseTutorialPose2
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)


# 
# Compile options
# 

target_compile_options(openPoseTutorialPose1
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_options(openPoseTutorialPose2
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)


# 
# Linker options
# 

target_link_libraries(openPoseTutorialPose1
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
target_link_libraries(openPoseTutorialPose2
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)


#
# Target Health
#

perform_health_checks(
    openPoseTutorialPose1
    ${openPoseTutorialPose1sources}
)
perform_health_checks(
    openPoseTutorialPose2
    ${openPoseTutorialPose2sources}
)

# 
# Deployment
# 

# Executable
install(TARGETS openPoseTutorialPose1
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
install(TARGETS openPoseTutorialPose2
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
