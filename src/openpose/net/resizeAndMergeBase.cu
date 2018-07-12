#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
#include <openpose/net/resizeAndMergeBase.hpp>

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;

    template <typename T>
    __global__ void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight,
                                 const int targetWidth, const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const T xSource = (x + T(0.5f)) * sourceWidth / T(targetWidth) - T(0.5f);
            const T ySource = (y + T(0.5f)) * sourceHeight / T(targetHeight) - T(0.5f);
            targetPtr[y*targetWidth+x] = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight,
                                                            sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeKernelAndAdd(T* targetPtr, const T* const sourcePtr, const T scaleWidth,
                                       const T scaleHeight, const int sourceWidth, const int sourceHeight,
                                       const int targetWidth, const int targetHeight)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const T xSource = (x + T(0.5f)) / scaleWidth - T(0.5f);
            const T ySource = (y + T(0.5f)) / scaleHeight - T(0.5f);
            targetPtr[y*targetWidth+x] += bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight,
                                                             sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeKernelAndAverage(T* targetPtr, const T* const sourcePtr, const T scaleWidth,
                                           const T scaleHeight, const int sourceWidth, const int sourceHeight,
                                           const int targetWidth, const int targetHeight, const int counter)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const T xSource = (x + T(0.5f)) / scaleWidth - T(0.5f);
            const T ySource = (y + T(0.5f)) / scaleHeight - T(0.5f);
            const auto interpolated = bicubicInterpolate(sourcePtr, xSource, ySource, sourceWidth, sourceHeight,
                                                         sourceWidth);
            auto& targetPixel = targetPtr[y*targetWidth+x];
            targetPixel = (targetPixel + interpolated) / T(counter);
        }
    }

    template <typename T>
    void resizeAndMergeGpu(T* targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            // Security checks
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (sourcePtrs.size() != sourceSizes.size() || sourceSizes.size() != scaleInputToNetInputs.size())
                error("Size(sourcePtrs) must match size(sourceSizes) and size(scaleInputToNetInputs). Currently: "
                      + std::to_string(sourcePtrs.size()) + " vs. " + std::to_string(sourceSizes.size()) + " vs. "
                      + std::to_string(scaleInputToNetInputs.size()) + ".", __LINE__, __FUNCTION__, __FILE__);

            // Parameters
            const auto channels = targetSize[1];
            const auto targetHeight = targetSize[2];
            const auto targetWidth = targetSize[3];
            const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
            const dim3 numBlocks{getNumberCudaBlocks(targetWidth, threadsPerBlock.x),
                                 getNumberCudaBlocks(targetHeight, threadsPerBlock.y)};
            const auto& sourceSize = sourceSizes[0];
            const auto sourceHeight = sourceSize[2];
            const auto sourceWidth = sourceSize[3];

            // No multi-scale merging or no merging required
            if (sourceSizes.size() == 1)
            {
                const auto num = sourceSize[0];
                if (targetSize[0] > 1 || num == 1)
                {
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;
                    const auto targetChannelOffset = targetWidth * targetHeight;
                    for (auto n = 0; n < num; n++)
                    {
                        const auto offsetBase = n*channels;
                        for (auto c = 0 ; c < channels ; c++)
                        {
                            const auto offset = offsetBase + c;
                            resizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr + offset * targetChannelOffset,
                                                                         sourcePtrs.at(0) + offset * sourceChannelOffset,
                                                                         sourceWidth, sourceHeight, targetWidth,
                                                                         targetHeight);
                        }
                    }
                }
                // Old inefficient multi-scale merging
                else
                    error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Multi-scaling merging
            else
            {
                const auto targetChannelOffset = targetWidth * targetHeight;
                cudaMemset(targetPtr, 0.f, channels*targetChannelOffset * sizeof(T));
                const auto scaleToMainScaleWidth = targetWidth / T(sourceWidth);
                const auto scaleToMainScaleHeight = targetHeight / T(sourceHeight);

                for (auto i = 0u ; i < sourceSizes.size(); i++)
                {
                    const auto& currentSize = sourceSizes.at(i);
                    const auto currentHeight = currentSize[2];
                    const auto currentWidth = currentSize[3];
                    const auto sourceChannelOffset = currentHeight * currentWidth;
                    const auto scaleInputToNet = scaleInputToNetInputs[i] / scaleInputToNetInputs[0];
                    const auto scaleWidth = scaleToMainScaleWidth / scaleInputToNet;
                    const auto scaleHeight = scaleToMainScaleHeight / scaleInputToNet;
                    // All but last image --> add
                    if (i < sourceSizes.size() - 1)
                    {
                        for (auto c = 0 ; c < channels ; c++)
                        {
                            resizeKernelAndAdd<<<numBlocks, threadsPerBlock>>>(
                                targetPtr + c * targetChannelOffset, sourcePtrs[i] + c * sourceChannelOffset,
                                scaleWidth, scaleHeight, currentWidth, currentHeight, targetWidth,
                                targetHeight
                            );
                        }
                    }
                    // Last image --> average all
                    else
                    {
                        for (auto c = 0 ; c < channels ; c++)
                        {
                            resizeKernelAndAverage<<<numBlocks, threadsPerBlock>>>(
                                targetPtr + c * targetChannelOffset, sourcePtrs[i] + c * sourceChannelOffset,
                                scaleWidth, scaleHeight, currentWidth, currentHeight, targetWidth,
                                targetHeight, sourceSizes.size()
                            );
                        }
                    }
                }
            }

            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeGpu(float* targetPtr, const std::vector<const float*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<float>& scaleInputToNetInputs);
    template void resizeAndMergeGpu(double* targetPtr, const std::vector<const double*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<double>& scaleInputToNetInputs);
}
