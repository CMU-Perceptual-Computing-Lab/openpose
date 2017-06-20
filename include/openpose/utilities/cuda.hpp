#ifndef OPENPOSE_UTILITIES_CUDA_HPP
#define OPENPOSE_UTILITIES_CUDA_HPP

#include <string>
#include <utility> // std::pair
#include <cuda.h>
#include <cuda_runtime.h>
#include <openpose/core/point.hpp>

namespace op
{
    const auto CUDA_NUM_THREADS = 512u;

    void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    int getGpuNumber();

    inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }

    dim3 getNumberCudaBlocks(const Point<int>& frameSize, const dim3 numberCudaThreads = dim3{ CUDA_NUM_THREADS, CUDA_NUM_THREADS, 1 });

    std::pair<dim3, dim3> getNumberCudaThreadsAndBlocks(const Point<int>& frameSize);
}

#endif // OPENPOSE_UTILITIES_CUDA_HPP
