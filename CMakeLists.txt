# CMake version
project(openpose)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Turn on C++11
add_definitions(-std=c++11)

# Project options
option(BUILD_DOCS  "Build documentation." OFF)
option(BUILD_CAFFE "Build using caffe" ON)
option(CPU_ONLY "Build only for cpu" OFF)
option(DOWNLOAD_MODELS "Download the models" OFF)

# Include sub options
include(CMakeDependentOption)
if (DOWNLOAD_MODELS)
  CMAKE_DEPENDENT_OPTION(USE_FOO "Use Foo" ON
    "USE_BAR;NOT USE_ZOT" OFF)

  CMAKE_DEPENDENT_OPTION(DOWNLOAD_COCO_MODEL "Download coco model" OFF
      "DOWNLOAD_MODELS" OFF)
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_MPI_MODEL "Download mpi model" OFF
      "DOWNLOAD_MODELS" OFF)
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_HAND_MODEL "Download hand model" OFF
      "DOWNLOAD_MODELS" OFF)  
  CMAKE_DEPENDENT_OPTION(DOWNLOAD_FACE_MODEL "Download face model" OFF
      "DOWNLOAD_MODELS" OFF)
endif (DOWNLOAD_MODELS)

# Flag for OpenPose Macros
add_definitions(-DUSE_CAFFE)


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

# Make the CMake documentation
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

# Download the models if flag is set
# TODO -- write a function to download the models
# To much clutter right how.
# TODO -- add checksum.
if (DOWNLOAD_MODELS)

  message(STATUS "Download the models.")

  set(OPENPOSE_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/models/")

  # Body (COCO)
  if (DOWNLOAD_COCO_MODEL)
    message(STATUS "Now downloading body (COCO) model")
    set(COCO_MODEL pose/coco/pose_iter_440000.caffemodel)
    set(COCO_FILENAME ${CMAKE_SOURCE_DIR}/models/${COCO_MODEL})
    if (NOT EXISTS ${COCO_FILENAME})
      file(DOWNLOAD ${OPENPOSE_URL}${COCO_MODEL} ${COCO_FILENAME}) # SHOW_PROGRESS)
    else (NOT EXISTS ${COCO_FILENAME})
      message(STATUS "Model already exists.")
    endif (NOT EXISTS ${COCO_FILENAME})
  else (DOWNLOAD_MPI_MODEL)
    message(STATUS "Not downloading body (COCO) model")
  endif (DOWNLOAD_COCO_MODEL)

  # Body (MPI)
  if (DOWNLOAD_MPI_MODEL)
    message(STATUS "Now downloading body (MPI) model")
    set(MPI_MODEL pose/mpi/pose_iter_160000.caffemodel)
    set(MPI_FILENAME ${CMAKE_SOURCE_DIR}/models/${MPI_MODEL})
    if (NOT EXISTS ${MPI_FILENAME})
      file(DOWNLOAD ${OPENPOSE_URL}${MPI_MODEL} ${MPI_FILENAME})
    else (NOT EXISTS ${MPI_FILENAME})
      message(STATUS "Model already exists.")
    endif (NOT EXISTS ${MPI_FILENAME})
  else (DOWNLOAD_MPI_MODEL)
    message(STATUS "Not downloading body (MPI) model")
  endif (DOWNLOAD_MPI_MODEL)

  # Face
  if (DOWNLOAD_FACE_MODEL)
    message(STATUS "Now downloading face model")
    set(FACE_MODEL face/pose_iter_116000.caffemodel)
    set(FACE_FILENAME ${CMAKE_SOURCE_DIR}/models/${FACE_MODEL})
    if (NOT EXISTS ${FACE_FILENAME})
      file(DOWNLOAD ${OPENPOSE_URL}${FACE_MODEL} ${FACE_FILENAME})
    else (NOT EXISTS ${FACE_FILENAME})
      message(STATUS "Model already exists.")
    endif (NOT EXISTS ${FACE_FILENAME})
  else (DOWNLOAD_FACE_MODEL)
    message(STATUS "Not downloading face model")
  endif (DOWNLOAD_FACE_MODEL)

  # Hand
  if (DOWNLOAD_HAND_MODEL) 
    message(STATUS "Now downloading hand model")
    set(HAND_MODEL hand/pose_iter_102000.caffemodel)
    set(HAND_FILENAME ${CMAKE_SOURCE_DIR}/models/${HAND_MODEL})
    if (NOT EXISTS ${HAND_FILENAME})
      file(DOWNLOAD ${OPENPOSE_URL}${HAND_MODEL} ${HAND_FILENAME})
    else (NOT EXISTS ${HAND_FILENAME})
      message(STATUS "Model already exists.")
    endif (NOT EXISTS ${HAND_FILENAME})
  else (DOWNLOAD_HAND_MODEL)
    message(STATUS "Not downloading hand model")
  endif (DOWNLOAD_HAND_MODEL)

  message(STATUS "Completing downloading models.")
  # add_custom_target(download_coco ALL 
  #   DEPENDS ${COCO_MODEL} ${FACE_MODEL} ${HAND_MODEL} ${MPI_MODEL})

endif (DOWNLOAD_MODELS)