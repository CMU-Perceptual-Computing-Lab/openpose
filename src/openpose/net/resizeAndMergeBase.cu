#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
#include <openpose/net/resizeAndMergeBase.hpp>

namespace op
{
    const auto THREADS_PER_BLOCK_1D = 16u;

    // template <typename T>
    // __global__ void resizeKernelOld(
    //     T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
    //     const int targetHeight)
    // {
    //     const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    //     const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //     if (x < targetWidth && y < targetHeight)
    //     {
    //         const T xSource = (x + T(0.5f)) * sourceWidth / T(targetWidth) - T(0.5f);
    //         const T ySource = (y + T(0.5f)) * sourceHeight / T(targetHeight) - T(0.5f);
    //         targetPtr[y*targetWidth+x] = bicubicInterpolate(
    //             sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
    //     }
    // }

    template <typename T>
    __global__ void resizeKernel(
        T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
        const int targetHeight, const int channels)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

        const auto sourceArea = sourceWidth * sourceHeight;
        const auto targetArea = targetWidth * targetHeight;

        if (x < targetWidth && y < targetHeight && channel < channels)
        {
            const T xSource = (x + T(0.5f)) * sourceWidth / T(targetWidth) - T(0.5f);
            const T ySource = (y + T(0.5f)) * sourceHeight / T(targetHeight) - T(0.5f);
            const T* sourcePtrChannel = sourcePtr + channel * sourceArea;
            targetPtr[channel * targetArea + y*targetWidth+x] = bicubicInterpolate(
                sourcePtrChannel, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
        }
    }

    template <typename T>
    __global__ void resizeAndPadKernel(
        T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
        const int targetHeight, const float rescaleFactor, const int channels)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

        const auto sourceArea = sourceWidth * sourceHeight;
        const auto targetArea = targetWidth * targetHeight;

        if (x < targetWidth && y < targetHeight && channel < channels)
        {
            const T xSource = (x + T(0.5f)) * 1.0 / T(rescaleFactor) - T(0.5f);
            const T ySource = (y + T(0.5f)) * 1.0 / T(rescaleFactor) - T(0.5f);
            const T* sourcePtrChannel = sourcePtr + channel * sourceArea;
            if (x < sourceWidth * rescaleFactor && y < sourceHeight * rescaleFactor)
                targetPtr[channel * targetArea + y*targetWidth+x] = bicubicInterpolate(
                    sourcePtrChannel, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
            else
                targetPtr[channel * targetArea + y*targetWidth+x] = 0;
        }
    }


        template <typename T>
    __global__ void resize8TimesKernel(
        T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
        const int targetHeight, const unsigned int rescaleFactor)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

        // Load shared memory
        // If resize >= 5, then #threads per block >= # elements of shared memory
        __shared__ T sourcePtrShared[25];
        const auto sharedLoadId = threadIdx.x + rescaleFactor*threadIdx.y;
        if (sharedLoadId < 25)
        {
            const auto minTargetX = blockIdx.x * rescaleFactor;
            const auto minSourceX = (minTargetX + T(0.5f)) * sourceWidth / T(targetWidth) - T(0.5f);
            const auto minSourceXInt_1 = int(minSourceX+ 1e-5);
            const auto minSourceXInt = minSourceXInt_1 - 1;
            const auto minTargetY = blockIdx.y * rescaleFactor;
            const auto minSourceY = (minTargetY + T(0.5f)) * sourceHeight / T(targetHeight) - T(0.5f);
            const auto minSourceYInt_1 = int(minSourceY  + 1e-5);
            const auto minSourceYInt = minSourceYInt_1 - 1;

            const auto yClean = fastTruncateCuda(int(minSourceYInt+sharedLoadId/5 + 1e-5), 0, sourceHeight - 1);
            const auto xClean = fastTruncateCuda(int(minSourceXInt+sharedLoadId%5 + 1e-5), 0, sourceWidth - 1);
            const auto sourceArea = sourceWidth * sourceHeight;
            const T* sourcePtrChannel = sourcePtr + channel * sourceArea;
            sourcePtrShared[sharedLoadId] = sourcePtrChannel[yClean * sourceWidth + xClean];
        }
        // Wait here until shared memory has been loaded
        __syncthreads();

        if (x < targetWidth && y < targetHeight)
        {
            const auto targetArea = targetWidth * targetHeight;
            const T xSource = (x + T(0.5f)) * sourceWidth / T(targetWidth) - T(0.5f);
            const T ySource = (y + T(0.5f)) * sourceHeight / T(targetHeight) - T(0.5f);
            targetPtr[channel * targetArea + y*targetWidth+x] = bicubicInterpolate8Times(
                sourcePtrShared, xSource, ySource, sourceWidth, sourceHeight, sourceWidth, threadIdx.x, threadIdx.y);
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

    __global__ void reorderAndCastKernel(
        float* targetPtr, const unsigned char* const srcPtr, const int width, const int height)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto c = (blockIdx.z * blockDim.z) + threadIdx.z;
        if (x < width && y < height)
        {
            const auto channels = 3;
            const auto originFramePtrOffsetY = y * width;
            const auto channelOffset = c * width * height;
            const auto targetIndex = channelOffset + y * width + x;
            const auto srcIndex = (originFramePtrOffsetY + x) * channels + c;
            targetPtr[targetIndex] =  float(srcPtr[srcIndex]) * (1/256.f) - 0.5f;
        }
    }

    void reorderAndCast(float* targetPtr, const unsigned char* const srcPtr, const int width, const int height)
    {
        const dim3 threadsPerBlock{32, 1, 1};
        const dim3 numBlocks{
            getNumberCudaBlocks(width, threadsPerBlock.x),
            getNumberCudaBlocks(height, threadsPerBlock.y),
            getNumberCudaBlocks(3, threadsPerBlock.z)};
        reorderAndCastKernel<<<numBlocks, threadsPerBlock>>>(targetPtr, srcPtr, width, height);
    }

    void resizeAndMergeRGBGPU(
        float* targetPtr, const float* const srcPtr, const int sourceWidth, const int sourceHeight,
        const int targetWidth, const int targetHeight, const float scaleFactor)

    {
        const auto channels = 3;
        const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
        const dim3 numBlocks{
            getNumberCudaBlocks(targetWidth, threadsPerBlock.x),
            getNumberCudaBlocks(targetHeight, threadsPerBlock.y),
            getNumberCudaBlocks(channels, threadsPerBlock.z)};

        resizeAndPadKernel<<<numBlocks, threadsPerBlock>>>(
            targetPtr, srcPtr, sourceWidth, sourceHeight, targetWidth, targetHeight, scaleFactor, channels);
    }

    template <typename T>
    void resizeAndMergeGpu(
        T* targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            // Sanity checks
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
                    // // Profiling code
                    // const auto REPS = 250;
                    // double timeNormalize0 = 0.;
                    // double timeNormalize1 = 0.;
                    // double timeNormalize2 = 0.;
                    // double timeNormalize3 = 0.;
                    // // Non-optimized function
                    // OP_CUDA_PROFILE_INIT(REPS);
                    // const auto sourceChannelOffset = sourceHeight * sourceWidth;
                    // const auto targetChannelOffset = targetWidth * targetHeight;
                    // for (auto n = 0; n < num; n++)
                    // {
                    //     const auto offsetBase = n*channels;
                    //     for (auto c = 0 ; c < channels ; c++)
                    //     {
                    //         const auto offset = offsetBase + c;
                    //         resizeKernelOld<<<numBlocks, threadsPerBlock>>>(
                    //             targetPtr + offset * targetChannelOffset,
                    //             sourcePtrs.at(0) + offset * sourceChannelOffset,
                    //             sourceWidth, sourceHeight, targetWidth, targetHeight);
                    //     }
                    // }
                    // OP_CUDA_PROFILE_END(timeNormalize1, 1e3, REPS);

                    // Optimized function for any resize size (suboptimal for 8x resize)
                    // OP_CUDA_PROFILE_INIT(REPS);
                    const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
                    const dim3 numBlocks{getNumberCudaBlocks(targetWidth, threadsPerBlock.x),
                                         getNumberCudaBlocks(targetHeight, threadsPerBlock.y),
                                         getNumberCudaBlocks(num * channels, threadsPerBlock.z)};
                    resizeKernel<<<numBlocks, threadsPerBlock>>>(
                        targetPtr, sourcePtrs.at(0), sourceWidth, sourceHeight, targetWidth, targetHeight,
                        num * channels);
                    // OP_CUDA_PROFILE_END(timeNormalize2, 1e3, REPS);

                    // // Optimized function for 8x resize
                    // OP_CUDA_PROFILE_INIT(REPS);
                    // if (targetWidth / sourceWidth != 8 || targetHeight / sourceHeight != 8)
                    //     error("Kernel only implemented for 8x resize. Notify us if this error appears.",
                    //         __LINE__, __FUNCTION__, __FILE__);
                    // const auto rescaleFactor = (unsigned int) std::ceil((float)(targetHeight) / (float)(sourceHeight));

                    // const dim3 threadsPerBlock{rescaleFactor, rescaleFactor, 1};
                    // const dim3 numBlocks{getNumberCudaBlocks(targetWidth, threadsPerBlock.x),
                    //                      getNumberCudaBlocks(targetHeight, threadsPerBlock.y),
                    //                      getNumberCudaBlocks(num * channels, threadsPerBlock.z)};
                    // resize8TimesKernel<<<numBlocks, threadsPerBlock>>>(
                    //     targetPtr, sourcePtrs.at(0), sourceWidth, sourceHeight, targetWidth, targetHeight, rescaleFactor);
                    // OP_CUDA_PROFILE_END(timeNormalize3, 1e3, REPS);

                    // Profiling code
                    // log("  Res(ori)=" + std::to_string(timeNormalize1) + "ms");
                    // log("  Res(new)=" + std::to_string(timeNormalize2) + "ms");
                    // log("  Res(new8x)=" + std::to_string(timeNormalize3) + "ms");
                }
                // Old inefficient multi-scale merging
                else
                    error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Multi-scaling merging
            else
            {
                const auto targetChannelOffset = targetWidth * targetHeight;
                cudaMemset(targetPtr, 0, channels*targetChannelOffset * sizeof(T));
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
                                targetHeight, (int)sourceSizes.size()
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

    template void resizeAndMergeGpu(
        float* targetPtr, const std::vector<const float*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<float>& scaleInputToNetInputs);
    template void resizeAndMergeGpu(
        double* targetPtr, const std::vector<const double*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<double>& scaleInputToNetInputs);
}
