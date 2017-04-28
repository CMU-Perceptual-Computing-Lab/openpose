#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/cuda.hu"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/resizeAndMergeBase.hpp"

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;

    template <typename T>
    __global__ void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const auto scaleWidth = targetWidth / T(sourceWidth);
            const auto scaleHeight = targetHeight / T(sourceHeight);
            const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
            const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

            targetPtr[y*targetWidth+x] = cubicResize(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeKernelAndMerge(T* targetPtr, const T* const sourcePtr, const int sourceNumOffset, const int num, const T scaleGap,
                                         const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            auto& targetPixel = targetPtr[y*targetWidth+x];
            targetPixel = 0.f; // For average
            // targetPixel = -1000.f; // For fastMax
            for (auto n = 0; n < num; n++)
            {
                const auto numberScale = 1 - n * scaleGap;
                const auto widthPaddedSource = int(sourceWidth * numberScale);
                const auto heightPaddedSource = int(sourceHeight * numberScale);

                const auto scaleWidth = targetWidth / T(widthPaddedSource);
                const auto scaleHeight = targetHeight / T(heightPaddedSource);
                const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
                const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

                const T* const sourcePtrN = sourcePtr + n * sourceNumOffset;
                const auto interpolated = cubicResize(sourcePtrN, xSource, ySource, widthPaddedSource, heightPaddedSource, sourceWidth);
                targetPixel += interpolated;
                // targetPixel = fastMax(targetPixel, interpolated);
            }
            targetPixel /= num;
        }
    }

    template <typename T>
    void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const T scaleGap)
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

            if (targetSize[0] > 1)
            {
                for (auto n = 0; n < num; n++)
                    for (auto c = 0; c < channels; c++)
                        resizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr + (n*channels + c) * targetChannelOffset, sourcePtr + (n*channels + c) * sourceChannelOffset,
                                                                     sourceWidth, sourceHeight, targetWidth, targetHeight);
            }
            else
            {
                if (scaleGap <= 0.f && num != targetSize[0])
                    error("The scale gap must be greater than 0.", __LINE__, __FUNCTION__, __FILE__);
                const auto sourceNumOffset = channels * sourceChannelOffset;
                for (auto c = 0; c < channels; c++)
                    resizeKernelAndMerge<<<numBlocks, threadsPerBlock>>>(targetPtr + c * targetChannelOffset, sourcePtr + c * sourceChannelOffset, sourceNumOffset,
                                                                         num, scaleGap, sourceWidth, sourceHeight, targetWidth, targetHeight);
            }

            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeGpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const float scaleGap);
    template void resizeAndMergeGpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const double scaleGap);
}
