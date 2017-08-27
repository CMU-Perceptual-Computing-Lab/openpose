project(openpose)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR) # TODO -- Check the lowest version that works

### INSTALL PREREQUISITES

# execute_process(COMMAND ./install_cmake.sh
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)

# add_custom_target(install_pre ALL
#     COMMAND ./install_cmake.sh
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)

### FLAGS

# Turn on C++11
add_definitions(-std=c++11)
# list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # TODO -- why this doesn't work

# Openpose flags
add_definitions(-DUSE_CAFFE)

### PROJECT OPTIONS

SET(DL_FRAMEWORK CAFFE CACHE STRING "Select Deep Learning Framework")
SET_PROPERTY(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE CAFFE2 TENSORFLOW) 

# Suboptions for Caffe DL Framework
include(CMakeDependentOption)
if (${DL_FRAMEWORK} MATCHES "CAFFE")
  CMAKE_DEPENDENT_OPTION(WITH_CAFFE "Build using Caffe" ON
    "DL_FRAMEWORK" ON)
  CMAKE_DEPENDENT_OPTION(BUILD_CAFFE "Build Caffe as part of OpenPose" OFF
    "DL_FRAMEWORK" ON)
else (${DL_FRAMEWORK} MATCHES "CAFFE")
  message("Nothing found")
endif (${DL_FRAMEWORK} MATCHES "CAFFE")

option(CPU_ONLY "Build only for cpu" OFF) # This option is not used right now.
option(BUILD_EXAMPLES "Build OpenPose examples" ON)
option(USE_CUDNN "Build OpenPose with cuDNN library support" ON)
option(BUILD_DOCS "Build OpenPose documentation" OFF)
option(INSTALL_OPENCV "INSTALL_OPENCV" OFF)

# Install OpenCV if said by user
if (INSTALL_OPENCV)
  execute_process(COMMAND sudo apt-get --assume-yes install libopencv-dev)
endif (INSTALL_OPENCV) 

# option(Caffe_INCLUDE_DIRS "Caffe Include directory" NOT_FOUND)
# option(Caffe_LIBS "Caffe Library path" NOT_FOUND)


option(DOWNLOAD_COCO_MODEL "Download coco model" ON)
option(DOWNLOAD_MPI_MODEL "Download mpi model" ON)
option(DOWNLOAD_HAND_MODEL "Download hand model" ON)  
option(DOWNLOAD_FACE_MODEL "Download face model" ON)

### FIND REQUIRED PACKAGES

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(CUDA REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED)

# Set CUDA Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

### CAFFE

# Check if the user specified caffe paths
if ((Caffe_INCLUDE_DIRS AND Caffe_LIBS) AND NOT BUILD_CAFFE)
  message(STATUS "\${Caffe_INCLUDE_DIRS} set by the user" ${Caffe_INCLUDE_DIRS})
  message(STATUS "\${Caffe_LIBS} set by the user" ${Caffe_LIBS})
  set(Caffe_FOUND ON)
endif ((Caffe_INCLUDE_DIRS AND Caffe_LIBS) AND NOT BUILD_CAFFE)

# Check if caffe is installed in known paths
if (NOT Caffe_FOUND AND NOT BUILD_CAFFE) 
  message(STATUS "Looking for caffe around in expected paths.")
  find_package(Caffe)
endif (NOT Caffe_FOUND AND NOT BUILD_CAFFE)

# Else build from scratch
if (NOT Caffe_FOUND OR BUILD_CAFFE)
  message(STATUS "Failed to find caffe, will build from source now.")
  # Build Caffe
  include(ExternalProject)
  set(BUILD_CAFFE ON)
  set(CAFFE_PREFIX caffe)
  set(CAFFE_URL ${CMAKE_SOURCE_DIR}/3rdparty/caffe)

  ExternalProject_Add(${CAFFE_PREFIX}
    SOURCE_DIR ${CAFFE_URL} # Is the url needed to be specified? 
    PREFIX ${CAFFE_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DUSE_CUDNN=${USE_CUDNN})

  ExternalProject_Get_Property(caffe install_dir)

  set(Caffe_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/build/caffe/include)
  set(Caffe_LIBS ${CMAKE_SOURCE_DIR}/build/caffe/lib/libcaffe.so)  
endif ()

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
endif (BUILD_DOCS)

### DOWNLOAD MODELS

# Download the models if flag is set
# TODO -- remove hardcoded ubuntu paths
include(cmake/Utils.cmake)
message(STATUS "Download the models.")

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


