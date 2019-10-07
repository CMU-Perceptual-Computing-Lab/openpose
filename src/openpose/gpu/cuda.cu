#include <openpose/gpu/cuda.hpp>
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <openpose_private/gpu/cuda.hu>
#endif

namespace op
{
    template <typename T>
    __global__ void reorderAndNormalizeKernel(
        T* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto c = (blockIdx.z * blockDim.z) + threadIdx.z;
        if (x < width && y < height)
        {
            const auto originFramePtrOffsetY = y * width;
            const auto channelOffset = c * width * height;
            const auto targetIndex = channelOffset + y * width + x;
            const auto srcIndex = (originFramePtrOffsetY + x) * channels + c;
            targetPtr[targetIndex] =  T(srcPtr[srcIndex]) * T(1/256.f) - T(0.5f);
        }
    }

    template <typename T>
    __global__ void uCharImageCastKernel(
        unsigned char* targetPtr, const T* const srcPtr, const int volume)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (x < volume)
            targetPtr[x] =  (unsigned char)(fastTruncateCuda(srcPtr[x], T(0), T(255)));
    }

    template <typename T>
    void reorderAndNormalize(
        T* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels)
    {
        try
        {
            const dim3 threadsPerBlock{32, 1, 1};
            const dim3 numBlocks{
                getNumberCudaBlocks(width, threadsPerBlock.x),
                getNumberCudaBlocks(height, threadsPerBlock.y),
                getNumberCudaBlocks(channels, threadsPerBlock.z)};
            reorderAndNormalizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr, srcPtr, width, height, channels);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void uCharImageCast(unsigned char* targetPtr, const T* const srcPtr, const int volume)
    {
        try
        {
            const dim3 threadsPerBlock{32, 1, 1};
            const dim3 numBlocks{
                getNumberCudaBlocks(volume, threadsPerBlock.x)};
            uCharImageCastKernel<<<numBlocks, threadsPerBlock>>>(targetPtr, srcPtr, volume);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void reorderAndNormalize(
        float* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);
    template void reorderAndNormalize(
        double* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);

    template void uCharImageCast(
        unsigned char* targetPtr, const float* const srcPtr, const int volume);
    template void uCharImageCast(
        unsigned char* targetPtr, const double* const srcPtr, const int volume);
}
