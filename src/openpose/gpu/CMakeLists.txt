set(SOURCES_OP_GPU
    cuda.cpp
    cuda.cu
    gpu.cpp
    opencl.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_GPU_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_GPU})
set(SOURCES_OP_GPU_WITH_CP ${SOURCES_OP_GPU_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_GPU_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
  if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose_gpu ${SOURCES_OP_GPU})
  else ()
    add_library(openpose_gpu ${SOURCES_OP_GPU})
  endif ()

  target_link_libraries(openpose_gpu openpose_core)

  install(TARGETS openpose_gpu
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
