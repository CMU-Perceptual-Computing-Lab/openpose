set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")
set(CUDNN_LIB_NAME "libcudnn.so")

find_path(CUDNN_INCLUDE cudnn.h
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
    DOC "Path to cuDNN include directory." )

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib 
    DOC "Path to cuDNN library.")

message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library:     ${CUDNN_LIBRARY})")

if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
  set(CUDNN_FOUND TRUE)
endif(CUDNN_INCLUDE AND CUDNN_LIBRARY)
