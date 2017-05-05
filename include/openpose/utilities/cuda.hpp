#ifndef OPENPOSE__UTILITIES_CUDA_HPP
#define OPENPOSE__UTILITIES_CUDA_HPP

#include <string>
#include <utility> // std::pair
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

namespace op
{
    const auto CUDA_NUM_THREADS = 512u;

    void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }

    inline dim3 getNumberCudaBlocks(const cv::Size& frameSize, const dim3 numberCudaThreads = dim3{CUDA_NUM_THREADS, CUDA_NUM_THREADS, 1})
    {
        return dim3{getNumberCudaBlocks(frameSize.width, numberCudaThreads.x),
                    getNumberCudaBlocks(frameSize.height, numberCudaThreads.y),
                    numberCudaThreads.z};
    }

    std::pair<dim3, dim3> getNumberCudaThreadsAndBlocks(const cv::Size& frameSize);
}

#endif // OPENPOSE__UTILITIES_CUDA_HPP
