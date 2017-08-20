# CMake version
project(openpose)
# Check the min
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Turn on C++11
add_definitions(-std=c++11)
# Flag for OpenPose Macros
add_definitions(-DUSE_CAFFE)

# Testing code starts here

### Testing code ends here

# Project options
option(BUILD_DOCS  "Build documentation." OFF)
option(WITH_CAFFE "Build using caffe" ON)
option(BUILD_CAFFE "Build using caffe" ON)
option(CPU_ONLY "Build only for cpu" OFF) # This option is not used right now.
option(DOWNLOAD_MODELS "Download the models" OFF)
option(BUILD_EXAMPLES "Build OpenPose examples" ON)
option(USE_CUDNN "Build OpenPose with cuDNN library support" ON)

# find_package(Caffe)
if (NOT Caffe_FOUND)
  message(STATUS "Failed to find caffe, will build from source now.")
  # Build Caffe
  include(ExternalProject)

  set(CAFFE_PREFIX caffe)
  set(CAFFE_URL ${CMAKE_SOURCE_DIR}/3rdparty/caffe)

  ExternalProject_Add(caffe
    SOURCE_DIR ${CAFFE_URL} # Is the url needed to be specified? 
    PREFIX ${CAFFE_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DUSE_CUDNN=${USE_CUDNN})

  ExternalProject_Get_Property(caffe install_dir)

  # Include caffe directories
  ## include_directories(${install_dir}/include)

  set(Caffe_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/build/caffe/include)
  set(Caffe_LIBS ${CMAKE_SOURCE_DIR}/build/caffe/lib/libcaffe.so)  
endif (NOT Caffe_FOUND)

# if (WITH_CAFFE)
#   find_package(caffe)
# endif (WITH_CAFFE)

# Include sub options
include(CMakeDependentOption)
if (DOWNLOAD_MODELS)
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_COCO_MODEL "Download coco model" OFF
      "DOWNLOAD_MODELS" OFF)
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_MPI_MODEL "Download mpi model" OFF
      "DOWNLOAD_MODELS" OFF)
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_HAND_MODEL "Download hand model" OFF
      "DOWNLOAD_MODELS" OFF)  
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_FACE_MODEL "Download face model" OFF
      "DOWNLOAD_MODELS" OFF)
endif (DOWNLOAD_MODELS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Download the models

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
  ${Caffe_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

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
# add_subdirectory(3rdparty)
add_subdirectory(src)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)

message(STATUS "\${OpenCV_INCLUDE_DIRS} = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "\${OpenCV_LIBS} = ${OpenCV_LIBS}")
message("\${CMAKE_SOURCE_DIR} = ${CMAKE_SOURCE_DIR}")
message("\${Caffe_INCLUDE_DIRS} = ${Caffe_INCLUDE_DIRS}")
message("\${Caffe_LIBS} = ${Caffe_LIBS}")
message(\${GLOG_INCLUDE_DIR} ${GLOG_INCLUDE_DIR})
message(\${GLOG_LIBRARY} ${GLOG_LIBRARY})
message(\${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS})

message(\${CMAKE_CURRENT_BINARY_DIR} = ${CMAKE_CURRENT_BINARY_DIR})

### GENERATE DOCUMENTATION

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

### DOWNLOAD MODELS

include(cmake/Utils.cmake)
# Download the models if flag is set
# TODO -- remove hardcoded ubuntu paths
if (DOWNLOAD_MODELS)

  message(STATUS "Download the models.")
  set(OPENPOSE_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/models/")
  
  # Body (COCO)
  download_model("body (coco)" ${DOWNLOAD_COCO_MODEL} pose/coco/pose_iter_440000.caffemodel)
  # Body (MPI)
  download_model("body (mpi)" ${DOWNLOAD_MPI_MODEL} pose/mpi/pose_iter_160000.caffemodel)
  # Face
  download_model("face" ${DOWNLOAD_FACE_MODEL} face/pose_iter_116000.caffemodel)
  # Hand
  download_model("hand" ${DOWNLOAD_HAND_MODEL} hand/pose_iter_102000.caffemodel)

  message(STATUS "Models Downloaded.")

endif (DOWNLOAD_MODELS)

message(\${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR})
