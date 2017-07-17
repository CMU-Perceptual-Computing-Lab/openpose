#ifndef OPENPOSE_UTILITIES_CUDA_HPP
#define OPENPOSE_UTILITIES_CUDA_HPP

#include <utility> // std::pair
#include <cuda.h>
#include <cuda_runtime.h>
#include <openpose/core/common.hpp>

namespace op
{
    const auto CUDA_NUM_THREADS = 512u;

    OP_API void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    OP_API int getGpuNumber();

    inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }

    OP_API dim3 getNumberCudaBlocks(const Point<int>& frameSize, const dim3 numberCudaThreads = dim3{ CUDA_NUM_THREADS, CUDA_NUM_THREADS, 1 });

    OP_API std::pair<dim3, dim3> getNumberCudaThreadsAndBlocks(const Point<int>& frameSize);
}

#endif // OPENPOSE_UTILITIES_CUDA_HPP
