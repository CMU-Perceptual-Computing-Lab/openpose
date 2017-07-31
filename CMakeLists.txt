# CMake version
project(openpose)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Turn on C++11
add_definitions(-std=c++11)

# Project options
option(BUILD_DOCS  "Build documentation." OFF)
option(BUILD_CAFFE "Build using caffe" ON)
option(CPU_ONLY "Build only for cpu" OFF)

# Flag for OpenPose Macros
add_definitions(-DUSE_CAFFE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# include(FindCaffe)

# Find the packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
# find_package(Caffe) # Build if not found

# Set CUDA Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/3rdparty/caffe/include)

# message("\${Caffe_FOUND} = ${Caffe_FOUND}")
# if (Caffe_FOUND)
#   set(caffe ${Caffe_LIBS})
# else (Caffe_FOUND)
#   add_subdirectory(3rdparty)
#   set(Caffe_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/caffe/include)
#   message(\${Caffe_INCLUDE_DIRS} = ${Caffe_INCLUDE_DIRS})
# endif (Caffe_FOUND)
# add_definitions(${Caffe_INCLUDE_DIRS})

# Add the subdirectories
add_subdirectory(3rdparty)
add_subdirectory(src)
add_subdirectory(examples)

message(STATUS "\${OpenCV_INCLUDE_DIRS} = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "\${OpenCV_LIBS} = ${OpenCV_LIBS}")
message("\${CMAKE_SOURCE_DIR} = ${CMAKE_SOURCE_DIR}")
# message("\${Caffe_INCLUDE_DIRS} = ${Caffe_INCLUDE_DIRS}")
# message("\${Caffe_LIBRARIES} = ${Caffe_LIBRARIES}")
message(\${GLOG_INCLUDE_DIR} ${GLOG_INCLUDE_DIR})
message(\${GLOG_LIBRARY} ${GLOG_LIBRARY})
message(\${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS})

message(\${CMAKE_CURRENT_BINARY_DIR} = ${CMAKE_CURRENT_BINARY_DIR})

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_FILE ${CMAKE_SOURCE_DIR}/doc/doc_autogeneration.doxygen)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_FILE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)