project(openpose)
cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR) # min. cmake version recommmended by Caffe
### FLAGS

# Turn on C++11
add_definitions(-std=c++11)

# Openpose flags
add_definitions(-DUSE_CAFFE)

### PROJECT OPTIONS

# Select the DL Framework
SET(DL_FRAMEWORK CAFFE CACHE STRING "Select Deep Learning Framework")
# SET_PROPERTY(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE CAFFE2 TENSORFLOW) 

# Suboptions for Caffe DL Framework
include(CMakeDependentOption)
if (${DL_FRAMEWORK} MATCHES "CAFFE")
  CMAKE_DEPENDENT_OPTION(WITH_CAFFE "Build using Caffe" ON
      "DL_FRAMEWORK" ON)
  CMAKE_DEPENDENT_OPTION(BUILD_CAFFE "Build Caffe as part of OpenPose" OFF
      "DL_FRAMEWORK" ON)
endif (${DL_FRAMEWORK} MATCHES "CAFFE")

# Set the acceleration library
SET(ACCELERATION_LIB CUDA CACHE STRING "Select the acceleration library or CPU otherwise.")
# SET_PROPERTY(CACHE ACCELERATION_LIB PROPERTY STRINGS CUDA OPENCL CPU_ONLY) 

# Suboptions for acceleration library
include(CMakeDependentOption)
if (${ACCELERATION_LIB} MATCHES "CUDA")
  option(USE_CUDNN "Build OpenPose with cuDNN library support" ON)
endif (${ACCELERATION_LIB} MATCHES "CUDA")
 
# Download the models
option(DOWNLOAD_COCO_MODEL "Download coco model" ON)
option(DOWNLOAD_MPI_MODEL "Download mpi model" OFF)
option(DOWNLOAD_HAND_MODEL "Download hand model" ON)  
option(DOWNLOAD_FACE_MODEL "Download face model" ON)

# More options
option(BUILD_EXAMPLES "Build OpenPose examples" ON)
option(BUILD_DOCS "Build OpenPose documentation" OFF)

### FIND REQUIRED PACKAGES

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(Boost COMPONENTS system filesystem)
find_package(CUDA)
find_package(CuDNN)
find_package(GFlags)
find_package(Glog)
find_package(OpenCV)

if (NOT Boost_FOUND)
  message(FATAL_ERROR "Boost not found. Install Boost from the command line using the command(s) --\
      sudo apt-get install libboost-all-dev")
endif (NOT Boost_FOUND)

if (NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found. Install CUDA from the command line using the command(s)")
  execute_process(COMMAND cat install_cuda.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)
endif (NOT CUDA_FOUND)

if (USE_CUDNN AND NOT CUDNN_FOUND)
  message(FATAL_ERROR "cuDNN not found. Install cuDNN from the command line using the command(s)")
  execute_process(COMMAND cat install_cuda.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)
endif (USE_CUDNN AND NOT CUDNN_FOUND)

if (NOT GLOG_FOUND)
  message(FATAL_ERROR "Glog not found. Install Glog from the command line using the command(s) -\ 
      sudo apt-get install libgoogle-glog-dev")
endif (NOT GLOG_FOUND)

if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "GFlags not found. Install GFlags from the command line using the command(s) --\ 
    sudo apt-get install libgflags-dev")
endif (NOT GFLAGS_FOUND)

if (NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found. Install OpenCV from the command line using the command(s) --\
    sudo apt-get install libopencv-dev")
endif (NOT OpenCV_FOUND)

# Set CUDA Flags

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

### CAFFE


if (${DL_FRAMEWORK} MATCHES "CAFFE")

  # Check if the user specified caffe paths
  if (Caffe_INCLUDE_DIRS AND Caffe_LIBS AND WITH_CAFFE)
    message(STATUS "\${Caffe_INCLUDE_DIRS} set by the user to " ${Caffe_INCLUDE_DIRS})
    message(STATUS "\${Caffe_LIBS} set by the user to " ${Caffe_LIBS})
    set(Caffe_FOUND ON)
  endif (Caffe_INCLUDE_DIRS AND Caffe_LIBS AND WITH_CAFFE)

  # Check if caffe is installed in known paths
  if (NOT Caffe_FOUND AND WITH_CAFFE) 
    message(STATUS "Looking for caffe around in expected paths.")
    find_package(Caffe)
  endif (NOT Caffe_FOUND AND WITH_CAFFE)

  # Else build from scratch
  if (BUILD_CAFFE)
    message(STATUS "Failed to find caffe, will build from source now.")
    
    # Build Caffe
    include(ExternalProject)
    set(BUILD_CAFFE ON)
    set(CAFFE_PREFIX caffe)
    set(CAFFE_URL ${CMAKE_SOURCE_DIR}/3rdparty/caffe)

    ExternalProject_Add(${CAFFE_PREFIX}
        SOURCE_DIR ${CAFFE_URL}
        PREFIX ${CAFFE_PREFIX}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> 
            -DUSE_CUDNN=${USE_CUDNN}
            -DOpenCV_DIR=${OpenCV_DIR})

    ExternalProject_Get_Property(caffe install_dir)

    set(Caffe_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/build/caffe/include)
    set(Caffe_LIBS ${CMAKE_SOURCE_DIR}/build/caffe/lib/libcaffe.so)  
  endif (BUILD_CAFFE)

  if (NOT Caffe_FOUND AND NOT BUILD_CAFFE)
    message(FATAL_ERROR "Caffe not found. Either turn on the BUILD_CAFFE option or specify the path of Caffe includes and libs using -DCaffe_INCLUDE_DIRS and -DCaffe_LIBS")
  endif (NOT Caffe_FOUND AND NOT BUILD_CAFFE)

endif (${DL_FRAMEWORK} MATCHES "CAFFE")

### PROJECT INCLUDES

# Specify the include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${Caffe_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

### ADD SUBDIRECTORIES

add_subdirectory(src)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)

### GENERATE DOCUMENTATION

if (BUILD_DOCS)
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    # Set input and output files
    set(DOXYGEN_FILE ${CMAKE_SOURCE_DIR}/doc/doc_autogeneration.doxygen)

    # Custom target to build the documentation
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_FILE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif (BUILD_DOCS)

### DOWNLOAD MODELS

# Download the models if flag is set
# TODO -- remove hardcoded ubuntu paths
include(cmake/Utils.cmake)
message(STATUS "Download the models.")

# URL to the models
set(OPENPOSE_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/models/")
  
# Body (COCO)
if (DOWNLOAD_COCO_MODEL)
  download_model("body (coco)" ${DOWNLOAD_COCO_MODEL} pose/coco/pose_iter_440000.caffemodel
      5156d31f670511fce9b4e28b403f2939)
endif (DOWNLOAD_COCO_MODEL)
  
# Body (MPI)
if (DOWNLOAD_MPI_MODEL)
  download_model("body (mpi)" ${DOWNLOAD_MPI_MODEL} pose/mpi/pose_iter_160000.caffemodel
      2ca0990c7562bd7ae03f3f54afa96e00)
endif (DOWNLOAD_MPI_MODEL)
    
# Face
if (DOWNLOAD_FACE_MODEL)
  download_model("face" ${DOWNLOAD_FACE_MODEL} face/pose_iter_116000.caffemodel
        e747180d728fa4e4418c465828384333)
endif (DOWNLOAD_FACE_MODEL)

# Hand
if (DOWNLOAD_HAND_MODEL)
  download_model("hand" ${DOWNLOAD_HAND_MODEL} hand/pose_iter_102000.caffemodel
        a82cfc3fea7c62f159e11bd3674c1531)
endif (DOWNLOAD_HAND_MODEL)

message(STATUS "Models Downloaded.")
