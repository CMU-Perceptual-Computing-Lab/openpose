# CMake version
project(openpose)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_definitions(-std=c++11)

# Project options
option(OPTION_BUILD_DOCS     "Build documentation."                                   OFF)
option(OPTION_USE_CAFFE      "Build using caffe"                                      ON)
option(CPU_ONLY              "Build only for cpu"                                     OFF)

find_package(CUDA REQUIRED ${CUDA_INCLUDE_DIRS})

# message(${CUDA_INCLUDE_DIRS})

add_definitions(-DUSE_CAFFE)

if (CUDA_FOUND)
	message(STATUS "CUDA Found")
endif (CUDA_FOUND)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(FindCaffe)
find_package(Caffe)
find_package(Glog)
find_package(GFlags)

find_package( OpenCV REQUIRED )
message(STATUS "\${OpenCV_INCLUDE_DIRS} = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "\${OpenCV_LIBS} = ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include 
                      ${Caffe_INCLUDE_DIRS} 
                      ${CUDA_INCLUDE_DIRS}
                      /home/bikram/test/my_version/openpose/3rdparty/caffe/include/
                      ${GLOG_INCLUDE_DIR}
                      ${GFLAGS_INCLUDE_DIR})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

add_subdirectory(src)
add_subdirectory(examples)

message("\${CMAKE_SOURCE_DIR} = ${CMAKE_SOURCE_DIR}")
message("\${Caffe_INCLUDE_DIRS} = ${Caffe_INCLUDE_DIRS}")
message("\${Caffe_LIBRARIES} = ${Caffe_LIBRARIES}")
message(\${GLOG_INCLUDE_DIR} ${GLOG_INCLUDE_DIR})
message(\${GLOG_LIBRARY} ${GLOG_LIBRARY})
message(\${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS})