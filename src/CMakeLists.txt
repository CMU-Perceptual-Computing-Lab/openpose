# if (NOT CUDA_FOUND)
# 	messsage(FATAL_ERROR "CUDA not found")
# 	return ()
# endif (NOT CUDA_FOUND)

# if (NOT CAFFE_FOUND)
# 	messsage(FATAL_ERROR "Caffe not found")
# 	return ()
# endif (NOT CAFFE_FOUND)

add_subdirectory(openpose)
