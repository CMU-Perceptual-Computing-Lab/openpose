#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/cuda.hu>
#include <openpose/core/resizeAndMergeBase.hpp>

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;

    template <typename T>
    __global__ void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight, const T invScaleWidth, const T invScaleHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const T xSource = (x + 0.5f) * invScaleWidth - 0.5f;
            const T ySource = (y + 0.5f) * invScaleHeight - 0.5f;

            targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeKernelAndMerge(T* targetPtr, const T* const sourcePtr, const int sourceNumOffset, const int num, const T* scaleRatios,
                                         const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
               
        const auto currentWidth = sourceWidth;
        const auto currentHeight = sourceHeight;

        const auto scaleWidth = targetWidth / currentWidth;
        const auto scaleHeight = targetHeight / currentHeight;

 
        if (x < targetWidth && y < targetHeight)
        {
            auto& targetPixel = targetPtr[y*targetWidth+x];
            targetPixel = 0.f; // For average
            // targetPixel = -1000.f; // For fastMax
            for (auto n = 0; n < num; n++)
            {
                const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
                const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

                const T* const sourcePtrN = sourcePtr + n * sourceNumOffset;
                const auto interpolated = bicubicInterpolate(sourcePtrN, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
                targetPixel += interpolated;
                // targetPixel = fastMax(targetPixel, interpolated);
            }
            targetPixel /= num;
        }
    }

    template <typename T>
    void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                           const std::array<int, 4>& sourceSize, const std::vector<T>& scaleRatios)
    {
        try
        {
            const auto num = sourceSize[0];
            const auto channels = sourceSize[1];
            const auto sourceHeight = sourceSize[2];
            const auto sourceWidth = sourceSize[3];
            const auto targetHeight = targetSize[2];
            const auto targetWidth = targetSize[3];

            const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
            const dim3 numBlocks{getNumberCudaBlocks(targetWidth, threadsPerBlock.x), getNumberCudaBlocks(targetHeight, threadsPerBlock.y)};
            const auto sourceChannelOffset = sourceHeight * sourceWidth;
            const auto targetChannelOffset = targetWidth * targetHeight;
            const auto scaleWidth = sourceWidth/T(targetWidth);
            const auto scaleHeight = sourceHeight/T(targetHeight);
            // No multi-scale merging
            /*if (targetSize[0] > 1)
            {
                for (auto n = 0; n < num; n++)
                {*/
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        resizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr + c * targetChannelOffset,
                                                                     sourcePtr + c * sourceChannelOffset,
                                                                     sourceWidth, sourceHeight, targetWidth, targetHeight, scaleWidth, scaleHeight);
                    }
/*
                }
            }
            // Multi-scale merging
            else
            {
                const auto currentWidth = sourceWidth;
                const auto currentHeight = sourceHeight;

                const auto scaleWidth = targetWidth / currentWidth;
                const auto scaleHeight = targetHeight / currentHeight;

                // Perform resize + merging
                const auto sourceNumOffset = channels * sourceChannelOffset;
                for (auto c = 0 ; c < channels ; c++)
                    resizeKernelAndMerge<<<numBlocks, threadsPerBlock>>>(targetPtr + c * targetChannelOffset,
                                                                         sourcePtr + c * sourceChannelOffset, sourceNumOffset,
                                                                         num, scaleRatiosGpuPtr, sourceWidth, sourceHeight, targetWidth, targetHeight);
            }*/

            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeGpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<float>& scaleRatios);
    template void resizeAndMergeGpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<double>& scaleRatios);
}
