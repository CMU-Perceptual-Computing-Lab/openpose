#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/cuda.hu>
#include <openpose/core/resizeAndMergeBase.hpp>

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;

    template <typename T>
    __global__ void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
                                 const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const auto scaleWidth = targetWidth / T(sourceWidth);
            const auto scaleHeight = targetHeight / T(sourceHeight);
            const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
            const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

            targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeKernelAndMerge(T* targetPtr, const T* const sourcePtr, const int sourceNumOffset, const int num, const T* scaleRatios,
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
                const auto currentWidth = sourceWidth * scaleRatios[n];
                const auto currentHeight = sourceHeight * scaleRatios[n];

                const auto scaleWidth = targetWidth / currentWidth;
                const auto scaleHeight = targetHeight / currentHeight;
                const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
                const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

                const T* const sourcePtrN = sourcePtr + n * sourceNumOffset;
                const auto interpolated = bicubicInterpolate(sourcePtrN, xSource, ySource, intRound(currentWidth),
                                                             intRound(currentHeight), sourceWidth);
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

            // No multi-scale merging
            if (targetSize[0] > 1)
            {
                for (auto n = 0; n < num; n++)
                {
                    const auto offsetBase = n*channels;
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        const auto offset = offsetBase + c;
                        resizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr + offset * targetChannelOffset,
                                                                     sourcePtr + offset * sourceChannelOffset,
                                                                     sourceWidth, sourceHeight, targetWidth, targetHeight);
                    }
                }
            }
            // Multi-scale merging
            else
            {
                // If scale_number > 1 --> scaleRatios must be set
                if (scaleRatios.size() != num)
                    error("The scale ratios size must be equal than the number of scales.", __LINE__, __FUNCTION__, __FILE__);
                const auto maxScales = 10;
                if (scaleRatios.size() > maxScales)
                    error("The maximum number of scales is " + std::to_string(maxScales) + ".", __LINE__, __FUNCTION__, __FILE__);
                // Copy scaleRatios
                T* scaleRatiosGpuPtr;
                cudaMalloc((void**)&scaleRatiosGpuPtr, maxScales * sizeof(T));
                cudaMemcpy(scaleRatiosGpuPtr, scaleRatios.data(), scaleRatios.size() * sizeof(T), cudaMemcpyHostToDevice);
                // Perform resize + merging
                const auto sourceNumOffset = channels * sourceChannelOffset;
                for (auto c = 0 ; c < channels ; c++)
                    resizeKernelAndMerge<<<numBlocks, threadsPerBlock>>>(targetPtr + c * targetChannelOffset,
                                                                         sourcePtr + c * sourceChannelOffset, sourceNumOffset,
                                                                         num, scaleRatiosGpuPtr, sourceWidth, sourceHeight, targetWidth, targetHeight);
                // Free memory
                cudaFree(scaleRatiosGpuPtr);
            }

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
