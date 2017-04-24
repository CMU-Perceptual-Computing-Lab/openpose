#ifndef OPENPOSE__UTILITIES_CUDA_HPP
#define OPENPOSE__UTILITIES_CUDA_HPP

#include <string>

namespace op
{
    void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }
}

#endif // OPENPOSE__UTILITIES_CUDA_HPP
