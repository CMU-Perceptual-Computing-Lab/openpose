### VERSION INFO
set(OpenPose_VERSION_MAJOR 1)
set(OpenPose_VERSION_MINOR 3)
set(OpenPose_VERSION_PATCH 0)
set(OpenPose_VERSION ${OpenPose_VERSION_MAJOR}.${OpenPose_VERSION_MINOR}.${OpenPose_VERSION_PATCH})


### CMAKE HEADERS
if (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
 cmake_policy(SET CMP0048 NEW)
 project(OpenPose VERSION ${OpenPose_VERSION})
else (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
 project(OpenPose)
endif (${CMAKE_VERSION} VERSION_GREATER 3.0.0)
cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR) # min. cmake version recommended by Caffe


### FIND THE OS
if (WIN32) # OS is Windows
  message(STATUS "Windows OS support currently is experimental.")
  message(STATUS "Feel free to send us any suggestion or bug you encounter.")
elseif (APPLE) # OS is Apple
  message(FATAL_ERROR "Apple OS is not currently supported.")
endif ()


### FLAGS
if (WIN32)
  # TODO -- Makeshift solution -- This prevents rerunning build again
  # https://gitlab.kitware.com/cmake/cmake/issues/16783
  set(CMAKE_SUPPRESS_REGENERATION ON)

  string (REPLACE "/D_WINDOWS" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string (REPLACE "/DWIN32" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  # /GL option causes the code to crash -- fix this
  # sdl flags causes error -- error : unknown attribute \"guard\"

  set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ot /Oi /Gy /Z7")

  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG:incremental /OPT:REF /OPT:ICF")
  
  string(REPLACE "/MDd" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
elseif (UNIX AND NOT APPLE)
  # Turn on C++11
  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif (WIN32)

# C++ additional flags
if (CMAKE_COMPILER_IS_GNUCXX)
  message(STATUS "GCC detected, adding compile flags")
  set(OP_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wpedantic -Wall -Wextra -Wfatal-errors")
endif (CMAKE_COMPILER_IS_GNUCXX)


### PROJECT OPTIONS
# Select the DL Framework
set(DL_FRAMEWORK CAFFE CACHE STRING "Select Deep Learning Framework.")
set_property(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE)
# set_property(CACHE DL_FRAMEWORK PROPERTY STRINGS CAFFE CAFFE2 TENSORFLOW)

# Suboptions for Caffe DL Framework
include(CMakeDependentOption)
if (${DL_FRAMEWORK} MATCHES "CAFFE")
  CMAKE_DEPENDENT_OPTION(BUILD_CAFFE "Build Caffe as part of OpenPose." ON
    "DL_FRAMEWORK" ON)
  # OpenPose flags
  add_definitions(-DUSE_CAFFE)
endif (${DL_FRAMEWORK} MATCHES "CAFFE")

if (WIN32)
  # Select the Enhanced Instruction Set
  set(INSTRUCTION_SET NONE CACHE STRING "Enable Enhanced Instruction Set")
  set_property(CACHE INSTRUCTION_SET PROPERTY STRINGS NONE SSE SSE2 AVX AVX2 IA32)

  # Suboptions for Enhanced Instruction Set
  if (${INSTRUCTION_SET} MATCHES "SSE")
    add_definitions("/arch:SSE")
  elseif (${INSTRUCTION_SET} MATCHES "SSE2")
    add_definitions("/arch:SSE2")
  elseif (${INSTRUCTION_SET} MATCHES "AVX")
    add_definitions("/arch:AVX")
  elseif (${INSTRUCTION_SET} MATCHES "AVX2")
    add_definitions("/arch:AVX2")
  elseif (${INSTRUCTION_SET} MATCHES "IA32")
    add_definitions("/arch:IA32")
  endif (${INSTRUCTION_SET} MATCHES "SSE")
endif (WIN32)

# Set the acceleration library
set(GPU_MODE CUDA CACHE STRING "Select the acceleration GPU library or CPU otherwise.")
# Display proper options to user
# if (CUDA_FOUND AND OpenCL_FOUND)
#   set_property(CACHE GPU_MODE PROPERTY STRINGS CUDA OPENCL CPU_ONLY)
# elseif (CUDA_FOUND)
#   set_property(CACHE GPU_MODE PROPERTY STRINGS CUDA CPU_ONLY)
# elseif (OpenCL_FOUND)
#   set_property(CACHE GPU_MODE PROPERTY STRINGS OPENCL CPU_ONLY)
# else ()
#   set_property(CACHE GPU_MODE PROPERTY STRINGS CPU_ONLY)
# endif ()
set_property(CACHE GPU_MODE PROPERTY STRINGS CUDA CPU_ONLY)

# Look for CUDA
set(CUDA_FOUND FALSE)
if (${GPU_MODE} MATCHES "CUDA")
  find_package(CUDA)
endif (${GPU_MODE} MATCHES "CUDA")
# Look for OpenCL
set(OpenCL_FOUND FALSE)
set(CUDA_VERSION_MAJOR 0)
if (${GPU_MODE} MATCHES "OPENCL")
  find_package(OpenCL)
endif (${GPU_MODE} MATCHES "OPENCL")

# Code to avoid crash at compiling time if OpenCL is not found
if (NOT OpenCL_FOUND)
  set(OpenCL_LIBRARIES "")
endif (NOT OpenCL_FOUND)
# Required for OpenCL in Nvidia graphic cards
if (CUDA_FOUND AND OpenCL_FOUND AND ${CUDA_VERSION_MAJOR} LESS 9)
  add_definitions(-DLOWER_CL_VERSION)
endif ()
# Handle desired GPU mode option
if (${GPU_MODE} MATCHES "CUDA")
  # OpenPose flags
  add_definitions(-DUSE_CUDA)
  message(STATUS "Building with CUDA.")
elseif (${GPU_MODE} MATCHES "CPU_ONLY")
  add_definitions(-DUSE_CPU_ONLY)
  message(STATUS "Building CPU Only.")
  # OpenPose flag for Caffe
  add_definitions(-DCPU_ONLY)
elseif (${GPU_MODE} MATCHES "OPENCL")
  # OpenPose flag for Caffe
  add_definitions(-DUSE_OPENCL)
  add_definitions(-DUSE_GREENTEA)
  message(STATUS "Building with OpenCL.")
endif ()

# Intel branch with MKL Support
if (${GPU_MODE} MATCHES "CPU_ONLY")
  if (UNIX AND NOT APPLE)
    OPTION(USE_MKL "Use MKL Intel Branch." ON)
  endif ()
endif ()

if (${USE_MKL})
  # OpenPose flags
  add_definitions(-DUSE_MKL)
  message(STATUS "Building with MKL support.")
endif (${USE_MKL})

# Set/disable profiler
if (PROFILER_ENABLED)
  add_definitions(-DPROFILER_ENABLED)
endif (PROFILER_ENABLED)

# Suboptions for GPU architectures
if (${GPU_MODE} MATCHES "CUDA")
  set(CUDA_ARCH Auto CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property(CACHE CUDA_ARCH PROPERTY STRINGS Auto All Manual)
endif (${GPU_MODE} MATCHES "CUDA")

# Suboptions for acceleration library
if (${GPU_MODE} MATCHES "CUDA")
  option(USE_CUDNN "Build OpenPose with cuDNN library support." ON)
endif (${GPU_MODE} MATCHES "CUDA")

# Suboptions for OpenPose 3D Reconstruction demo
option(WITH_3D_RENDERER "Add OpenPose 3D renderer module (it requires FreeGLUT library)." OFF)
option(WITH_FLIR_CAMERA "Add FLIR (formerly Point Grey) camera code (requires Spinnaker SDK already installed)." OFF)

# Download the models
option(DOWNLOAD_COCO_MODEL "Download COCO model." ON)
option(DOWNLOAD_MPI_MODEL "Download MPI model." OFF)
option(DOWNLOAD_HAND_MODEL "Download hand model." ON)  
option(DOWNLOAD_FACE_MODEL "Download face model." ON)

# More options
option(BUILD_EXAMPLES "Build OpenPose examples." ON)
option(BUILD_DOCS "Build OpenPose documentation." OFF)

# Build as shared library
option(BUILD_SHARED_LIBS "Build as shared lib" ON)

# Speed profiler
option(PROFILER_ENABLED "If enabled, OpenPose will be able to print out speed information at runtime." OFF)

# Threads
if (${GPU_MODE} MATCHES "OPENCL")
  unset(CMAKE_THREAD_LIBS_INIT CACHE)
  find_package(Threads)
else ()
  set(CMAKE_THREAD_LIBS_INIT "")
endif ()

### FIND REQUIRED PACKAGES
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

if (${GPU_MODE} MATCHES "CUDA")
  find_package(CUDA)
endif ()

# Adding 3D
if (WITH_3D_RENDERER)
  # OpenPose flags
  add_definitions(-DWITH_3D_RENDERER)
endif (WITH_3D_RENDERER)
if (WITH_FLIR_CAMERA)
  # OpenPose flags
  add_definitions(-DWITH_FLIR_CAMERA)
endif (WITH_FLIR_CAMERA)

if (UNIX AND NOT APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    include(cmake/Cuda.cmake)
    find_package(CuDNN)
  endif ()
  find_package(GFlags)
  find_package(Glog)
  if (OpenCV_CONFIG_FILE)
    include (${OpenCV_CONFIG_FILE})
  elseif (OpenCV_INCLUDE_DIRS AND OpenCV_LIBS_DIR)
    file(GLOB_RECURSE OpenCV_LIBS "${OpenCV_LIBS_DIR}*.so")
    set(OpenCV_FOUND 1)
  else ()
    find_package(OpenCV)
  endif (OpenCV_CONFIG_FILE)

  # 3D
  if (WITH_3D_RENDERER)
    # GLUT
    find_package(GLUT REQUIRED)
  endif (WITH_3D_RENDERER)
  if (WITH_FLIR_CAMERA)
    # Spinnaker
    find_package(Spinnaker)
    if (NOT SPINNAKER_FOUND)
      message(FATAL_ERROR "Spinnaker not found. Either turn off the `WITH_FLIR_CAMERA` option or specify the path to
        the Spinnaker includes and libs.")
    endif (NOT SPINNAKER_FOUND)
  # Otherwise error if 3D module enabled, and then disabled (with Spinnaker not installed)
  else (WITH_FLIR_CAMERA)
    set(SPINNAKER_LIB "")
  endif (WITH_FLIR_CAMERA)

  # OpenMP
  if (${GPU_MODE} MATCHES "CPU_ONLY")
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif ()
  endif ()

  if (${GPU_MODE} MATCHES "CUDA")
    # Set CUDA Flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

    if (NOT CUDA_FOUND)
      message(STATUS "CUDA not found.")
      execute_process(COMMAND cat install_cuda.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)
      message(FATAL_ERROR "Install CUDA using the above commands.")
    endif (NOT CUDA_FOUND)

    if (USE_CUDNN AND NOT CUDNN_FOUND)
      message(STATUS "cuDNN not found.")
      execute_process(COMMAND cat install_cudnn.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/ubuntu)
      message(FATAL_ERROR "Install cuDNN using the above commands. or turn off cuDNN by setting USE_CUDNN to OFF.")
    endif (USE_CUDNN AND NOT CUDNN_FOUND)
  endif ()

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

endif (UNIX AND NOT APPLE)

include(cmake/Utils.cmake)

if (WIN32)

  if ("${CMAKE_VERSION}" VERSION_GREATER 3.6.3)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT OpenPoseDemo)
  endif ("${CMAKE_VERSION}" VERSION_GREATER 3.6.3)

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(FIND_LIB_PREFIX ${CMAKE_SOURCE_DIR}/3rdparty/windows)

  # Download Windows 3rd party
  message(STATUS "Downloading windows dependencies...")
  set(OP_WIN_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows")
  set(OP_WIN_DIR "${CMAKE_SOURCE_DIR}/3rdparty/windows")

  #TODO Add hashes
  download_zip("opencv_310.zip" ${OP_WIN_URL} ${FIND_LIB_PREFIX} 1e5240a64b814b3c0b822f136be78ad7)
  download_zip("caffe3rdparty_2018_02_13.zip" ${OP_WIN_URL} ${FIND_LIB_PREFIX} 6653E07E3D2D7226D64FAD2DF0B407FB)
  if (${GPU_MODE} MATCHES "OPENCL")
    download_zip("caffe_opencl_2018_02_13.zip" ${OP_WIN_URL} ${FIND_LIB_PREFIX} 3ac3e1acf5ee6a4e57920be73053067a)
  else ()
    download_zip("caffe_2018_01_18.zip" ${OP_WIN_URL} ${FIND_LIB_PREFIX} 4b8e548cc7ea20abea472950dd5301bd)
  endif ()
  if (WITH_3D_RENDERER)
    download_zip("freeglut_2018_01_14.zip" ${OP_WIN_URL} ${FIND_LIB_PREFIX} BB182187285E06880F0EDE3A39530091)
  endif (WITH_3D_RENDERER)
  message(STATUS "Windows dependencies downloaded.")

  find_library(OpenCV_LIBS opencv_world310 HINTS ${FIND_LIB_PREFIX}/opencv/x64/vc14/lib)
  find_library(GFLAGS_LIBRARY_RELEASE gflags HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
  find_library(GFLAGS_LIBRARY_DEBUG gflagsd HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
  find_library(GLOG_LIBRARY_RELEASE glog HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
  find_library(GLOG_LIBRARY_DEBUG glogd HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
  find_library(OpenCV_LIBS opencv_world310 HINTS ${FIND_LIB_PREFIX}/opencv/x64/vc14/lib)

  if (${GPU_MODE} MATCHES "OPENCL")
    SET(CAFFE_CL_CHECKED True CACHE BOOL "")
    unset(Caffe_LIB CACHE)
    unset(Caffe_Proto_LIB CACHE)
    set(CL "_CL")
    find_library(Caffe_LIB caffe HINTS ${FIND_LIB_PREFIX}/caffe_opencl/lib)
    find_library(Caffe_Proto_LIB caffeproto HINTS ${FIND_LIB_PREFIX}/caffe_opencl/lib)
  else ()
    if (CAFFE_CL_CHECKED)
      unset(Caffe_LIB CACHE)
      unset(Caffe_Proto_LIB CACHE)
    endif ()
    set(CAFFE_CL_CHECKED 0)
    set(CL "")
    find_library(Caffe_LIB caffe HINTS ${FIND_LIB_PREFIX}/caffe/lib)
    find_library(Caffe_Proto_LIB caffeproto HINTS ${FIND_LIB_PREFIX}/caffe/lib)
  endif ()

  if (${GPU_MODE} MATCHES "OPENCL")
    unset(BOOST_SYSTEM_LIB_RELEASE CACHE)
    unset(BOOST_SYSTEM_LIB_DEBUG CACHE)
    find_library(BOOST_SYSTEM_LIB_RELEASE boost_system-vc140-mt-1_61 HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
    find_library(BOOST_SYSTEM_LIB_DEBUG boost_system-vc140-mt-gd-1_61 HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
  else ()
    set(BOOST_SYSTEM_LIB_RELEASE "")
    set(BOOST_SYSTEM_LIB_DEBUG "")
  endif ()

  if (WITH_3D_RENDERER)
    find_library(GLUT_LIBRARY freeglut HINTS ${FIND_LIB_PREFIX}/freeglut/lib)
    message(STATUS "\${GLUT_LIBRARY} = ${GLUT_LIBRARY}")
  endif (WITH_3D_RENDERER)
  if (WITH_FLIR_CAMERA)
    find_library(SPINNAKER_LIB spinnaker_v140 HINTS ${FIND_LIB_PREFIX}/spinnaker/lib)
  endif (WITH_FLIR_CAMERA)
  set(Caffe_LIBS ${Caffe_LIB};${Caffe_Proto_LIB})
  set(OpenCV_INCLUDE_DIRS "3rdparty/windows/opencv/include")
  if (${GPU_MODE} MATCHES "OPENCL")
    SET(CAFFE_CL_CHECKED2 True CACHE BOOL "")
    unset(Caffe_INCLUDE_DIRS CACHE)
    set(Caffe_INCLUDE_DIRS "3rdparty/windows/caffe_opencl/include;3rdparty/windows/caffe_opencl/include2" CACHE FILEPATH "Caffe_INCLUDE_DIRS")
  else ()
    if (CAFFE_CL_CHECKED2)
      unset(Caffe_INCLUDE_DIRS CACHE)
    endif ()
    set(CAFFE_CL_CHECKED2 0)
    set(Caffe_INCLUDE_DIRS "3rdparty/windows/caffe/include;3rdparty/windows/caffe/include2" CACHE FILEPATH "Caffe_INCLUDE_DIRS")
  endif ()
  set(Boost_INCLUDE_DIRS "3rdparty/windows/caffe3rdparty/include/boost-1_61")
  set(WINDOWS_INCLUDE_DIRS "3rdparty/windows/caffe3rdparty/include")
  if (WITH_3D_RENDERER)
    set(GLUT_INCLUDE_DIRS "3rdparty/windows/freeglut/include")
  endif (WITH_3D_RENDERER)
  if (WITH_FLIR_CAMERA)
    set(SPINNAKER_INCLUDE_DIRS "3rdparty/windows/spinnaker/include")
  endif (WITH_FLIR_CAMERA)
  set(Caffe_FOUND 1)
endif (WIN32)


### CAFFE
if (UNIX AND NOT APPLE)

  if (${DL_FRAMEWORK} MATCHES "CAFFE")

    # Check if the user specified caffe paths
    if (Caffe_INCLUDE_DIRS AND Caffe_LIBS AND NOT BUILD_CAFFE)
      message(STATUS "\${Caffe_INCLUDE_DIRS} set by the user to " ${Caffe_INCLUDE_DIRS})
      message(STATUS "\${Caffe_LIBS} set by the user to " ${Caffe_LIBS})
      set(Caffe_FOUND 1)
    endif (Caffe_INCLUDE_DIRS AND Caffe_LIBS AND NOT BUILD_CAFFE)

    # Else build from scratch
    if (BUILD_CAFFE)

      # Download Caffe
      message(STATUS "Caffe will be downloaded from source now. NOTE: This process might take several minutes depending
        on your internet connection.")

      # Check if pulled
      file(GLOB CAFFE_DIR_VALID ${CMAKE_SOURCE_DIR}/3rdparty/caffe/*)
      list(LENGTH CAFFE_DIR_VALID CAFFE_DIR_VALID_LENGTH)
      if (CAFFE_DIR_VALID_LENGTH EQUAL 0)
        execute_process(COMMAND git submodule update --init --recursive)
      else()
        message(STATUS "Caffe has already been downloaded.")
      endif ()

      # Build Process
      set(CAFFE_CPU_ONLY OFF)
      if (${GPU_MODE} MATCHES "CUDA")
        execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/caffe)
      elseif (${GPU_MODE} MATCHES "CPU_ONLY")
        if (USE_MKL)
          execute_process(COMMAND git checkout intel WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/caffe)
          execute_process(COMMAND sh prepare_mkl.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/caffe/external/mkl
              OUTPUT_VARIABLE rv)
          set( MLIST ${rv} )
          separate_arguments(MLIST)
          list(GET MLIST 0 MKL_PATH)
          message(STATUS ${MKL_PATH})
          file(GLOB MKL_SO
            "${MKL_PATH}lib/*"
            )
          file(COPY ${MKL_SO} DESTINATION ${CMAKE_BINARY_DIR}/caffe)

          set(MKL_LIBS
            "${CMAKE_BINARY_DIR}/caffe/libiomp5.so"
            "${CMAKE_BINARY_DIR}/caffe/libmklml_intel.so"
            "${CMAKE_BINARY_DIR}/caffe/lib/libmkldnn.so.0"
            "${CMAKE_BINARY_DIR}/caffe/lib/libmkldnn.so"
            "${CMAKE_BINARY_DIR}/caffe/lib/libmkldnn.so.0.12.0"
            )
        else ()
          execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/caffe)
        endif ()
        set(CAFFE_CPU_ONLY ON)
        set(USE_CUDNN OFF)
      elseif (${GPU_MODE} MATCHES "OPENCL")
        execute_process(COMMAND git checkout opencl WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/caffe)
        set(USE_CUDNN OFF)
      endif ()

      # Build Caffe
      message(STATUS "Caffe will be built from source now.")
      find_package(Caffe)
      include(ExternalProject)
      set(CAFFE_PREFIX caffe)
      set(CAFFE_URL ${CMAKE_SOURCE_DIR}/3rdparty/caffe)

      # One for Intel Branch and one for Master
      if (USE_MKL)
        ExternalProject_Add(openpose_caffe
          SOURCE_DIR ${CAFFE_URL}
          PREFIX ${CAFFE_PREFIX}
          CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DMKLDNN_INSTALL_DIR:PATH=<INSTALL_DIR>
          -DUSE_CUDNN=${USE_CUDNN}
          -DBUILD_python=OFF
          -DCPU_ONLY=${CAFFE_CPU_ONLY}
          -DUSE_MKL2017_AS_DEFAULT_ENGINE=${CAFFE_CPU_ONLY}
          -DCMAKE_BUILD_TYPE=Release
          -DBUILD_docs=OFF
          -DBUILD_python=OFF
          -DBUILD_python_layer=OFF
          -DUSE_LEVELDB=OFF
          -DUSE_LMDB=OFF
          -DUSE_OPENCV=OFF)
          # -DOpenCV_DIR=${OpenCV_DIR})
      else ()
        ExternalProject_Add(openpose_caffe
          SOURCE_DIR ${CAFFE_URL}
          PREFIX ${CAFFE_PREFIX}
          CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DUSE_CUDNN=${USE_CUDNN}
          -DCPU_ONLY=${CAFFE_CPU_ONLY}
          -DCMAKE_BUILD_TYPE=Release
          -DBUILD_docs=OFF
          -DBUILD_python=OFF
          -DBUILD_python_layer=OFF
          -DUSE_LEVELDB=OFF
          -DUSE_LMDB=OFF
          -DUSE_OPENCV=OFF)
          # -DOpenCV_DIR=${OpenCV_DIR})
      endif ()

      ExternalProject_Get_Property(openpose_caffe install_dir)

      if (NOT Caffe_FOUND)
        add_custom_command(TARGET openpose_caffe
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} 
            COMMAND $(MAKE)
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Rerunning cmake after building Caffe submodule")
      endif (NOT Caffe_FOUND)

    endif (BUILD_CAFFE)

    if (NOT Caffe_FOUND AND NOT BUILD_CAFFE)
      message(FATAL_ERROR "Caffe not found. Either turn on the BUILD_CAFFE option or specify the path of Caffe includes
        and libs using -DCaffe_INCLUDE_DIRS and -DCaffe_LIBS.")
    endif (NOT Caffe_FOUND AND NOT BUILD_CAFFE)

  endif (${DL_FRAMEWORK} MATCHES "CAFFE")

endif (UNIX AND NOT APPLE)

### PROJECT INCLUDES
# Specify the include directories
include_directories(
  include
  ${GFLAGS_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS})

if(USE_MKL)
 include_directories(
     "${MKL_PATH}/include/")
endif (USE_MKL)

if (Caffe_FOUND)
  include_directories(
      ${Caffe_INCLUDE_DIRS})
endif (Caffe_FOUND)

if (${GPU_MODE} MATCHES "CUDA")
  include_directories(
      ${CUDA_INCLUDE_DIRS})
endif ()
if (${GPU_MODE} MATCHES "OPENCL")
  include_directories(
    ${OpenCL_INCLUDE_DIRS})
endif ()
# 3D
if (WITH_3D_RENDERER)
  include_directories(${GLUT_INCLUDE_DIRS})
endif (WITH_3D_RENDERER)
if (WITH_FLIR_CAMERA)
  include_directories(${SPINNAKER_INCLUDE_DIRS})
endif (WITH_FLIR_CAMERA)
# Windows includes
if (WIN32)
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${WINDOWS_INCLUDE_DIRS})
endif (WIN32)


### ADD SUBDIRECTORIES
if (Caffe_FOUND)
  add_subdirectory(src)
  if (BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif (BUILD_EXAMPLES)
endif (Caffe_FOUND)


### DOWNLOAD MODELS
# Download the models if flag is set
message(STATUS "Download the models.")

# URL to the models
set(OPENPOSE_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/models/")

download_model("body (COCO)" ${DOWNLOAD_COCO_MODEL} pose/coco/pose_iter_440000.caffemodel
  5156d31f670511fce9b4e28b403f2939) # Body (COCO)
download_model("body (MPI)" ${DOWNLOAD_MPI_MODEL} pose/mpi/pose_iter_160000.caffemodel
  2ca0990c7562bd7ae03f3f54afa96e00) # Body (MPI)
download_model("face" ${DOWNLOAD_FACE_MODEL} face/pose_iter_116000.caffemodel
  e747180d728fa4e4418c465828384333) # Face
download_model("hand" ${DOWNLOAD_HAND_MODEL} hand/pose_iter_102000.caffemodel
  a82cfc3fea7c62f159e11bd3674c1531) # Hand

message(STATUS "Models Downloaded.")


### GENERATE DOCUMENTATION
if (UNIX AND NOT APPLE)

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
        VERBATIM)
    else (DOXYGEN_FOUND)
      message(FATAL_ERROR "Doxygen need to be installed to generate the doxygen documentation.")
    endif (DOXYGEN_FOUND)
  endif (BUILD_DOCS)

endif (UNIX AND NOT APPLE)


### INSTALL
if (UNIX AND NOT APPLE)
  if (Caffe_FOUND)
    # Install the headers
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/openpose DESTINATION include)
    install(EXPORT OpenPose DESTINATION lib/OpenPose)
    if (BUILD_CAFFE)
      install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe/include/caffe DESTINATION include)
      install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe/lib/ DESTINATION lib)
    endif (BUILD_CAFFE)

    # Compute installation prefix relative to this file
    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/OpenPoseConfig.cmake.in
      ${CMAKE_BINARY_DIR}/cmake/OpenPoseConfig.cmake @ONLY)

    install(FILES ${CMAKE_BINARY_DIR}/cmake/OpenPoseConfig.cmake
      DESTINATION lib/OpenPose)

    # Uninstall target
    configure_file(
      "${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
      "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
      IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif (Caffe_FOUND)
endif (UNIX AND NOT APPLE)
