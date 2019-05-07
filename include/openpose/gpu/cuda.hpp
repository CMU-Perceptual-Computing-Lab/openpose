#ifndef OPENPOSE_GPU_CUDA_HPP
#define OPENPOSE_GPU_CUDA_HPP

#include <utility> // std::pair
#include <openpose/core/common.hpp>

namespace op
{
    const auto CUDA_NUM_THREADS = 512u;

    OP_API void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    OP_API int getCudaGpuNumber();

    inline unsigned int getNumberCudaBlocks(
        const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }

    OP_API void getNumberCudaThreadsAndBlocks(
        dim3& numberCudaThreads, dim3& numberCudaBlocks, const Point<int>& frameSize);

    template <typename T>
    void reorderAndNormalize(
        T* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);

    template <typename T>
    void uCharImageCast(unsigned char* targetPtr, const T* const srcPtr, const int volume);
}

#endif // OPENPOSE_GPU_CUDA_HPP
