#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/maximumBase.hpp>

namespace op
{
    template <typename T>
    __global__ void fillTargetPtrPart(T* targetPtrOffsetted, const T* sourcePtrOffsetted, const int sourceIndex,
                                      const int x, const int y)
    {
        targetPtrOffsetted[0] = x;
        targetPtrOffsetted[1] = y;
        targetPtrOffsetted[2] = sourcePtrOffsetted[sourceIndex];
    }

    // template <typename T>
    // __global__ void fillTargetPtrChannel(T* targetPtrOffsetted, const T* sourcePtrOffsetted, const int width,
    //                                      const int imageOffset)
    // {
    //     const auto sourceThrustPtr = thrust::device_pointer_cast(sourcePtrOffsetted);
    //     // Ideal option (not working for CUDA < 8)
    //     // const auto sourceIndexIterator = thrust::max_element(
    //     //     thrust::device, sourceThrustPtr, sourceThrustPtr + imageOffset);
    //     // Workaround to make it work for CUDA 7.5
    //     const auto sourceIndexIterator = thrust::max_element(sourceThrustPtr, sourceThrustPtr + imageOffset);
    //     const auto sourceIndex = (int)(sourceIndexIterator - sourceThrustPtr);
    //     targetPtrOffsetted[0] = sourceIndex % width;
    //     targetPtrOffsetted[1] = sourceIndex / width;
    //     targetPtrOffsetted[2] = sourcePtrOffsetted[sourceIndex];
    // }

    // template <typename T>
    // __global__ void fillTargetPtr(T* targetPtr, const T* sourcePtr, const int width, const int imageOffset,
    //                               const int numberSubparts, const int offsetChannel)
    // {
    //     // get pixel location (x,y)
    //     const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    //     const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
    //     const auto part = y*width + x;

    //     // if (0 < x && x < (w-1) && 0 < y && y < (h-1))
    //     {
    //         auto* targetPtrOffsetted = targetPtr + (offsetChannel + part) * numberSubparts;
    //         const auto* const sourcePtrOffsetted = sourcePtr + (offsetChannel + part) * imageOffset;
    //         auto sourceThrustPtr = thrust::device_pointer_cast(sourcePtrOffsetted);
    //         const auto sourceIndexIterator = thrust::max_element(thrust::device, sourceThrustPtr,
    //                                                              sourceThrustPtr + imageOffset);
    //         // Ideal option (not working for CUDA < 8)
    //         // const auto sourceIndexIterator = thrust::max_element(
    //         //     thrust::device, sourceThrustPtr, sourceThrustPtr + imageOffset);
    //         // Workaround to make it work for CUDA 7.5
    //         const auto sourceIndexIterator = thrust::max_element(sourceThrustPtr, sourceThrustPtr + imageOffset);
    //         const auto sourceIndex = (int)(sourceIndexIterator - sourceThrustPtr);
    //         targetPtrOffsetted[0] = sourceIndex % width;
    //         targetPtrOffsetted[1] = sourceIndex / width;
    //         targetPtrOffsetted[2] = sourcePtrOffsetted[sourceIndex];
    //     }
    // }

    template <typename T>
    void maximumGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                    const std::array<int, 4>& sourceSize)
    {
        try
        {
            const auto height = sourceSize[2];
            const auto width = sourceSize[3];
            const auto imageOffset = height * width;
            const auto num = targetSize[0];
            const auto channels = targetSize[1];
            const auto numberParts = targetSize[2];
            const auto numberSubparts = targetSize[3];

            // log("sourceSize[0]: " + std::to_string(sourceSize[0]));  // = 1
            // log("sourceSize[1]: " + std::to_string(sourceSize[1]));  // = #BodyParts + bkg = 22 (hands) or 71 (face)
            // log("sourceSize[2]: " + std::to_string(sourceSize[2]));  // = 368 = height
            // log("sourceSize[3]: " + std::to_string(sourceSize[3]));  // = 368 = width
            // log("targetSize[0]: " + std::to_string(targetSize[0]));  // = 1
            // log("targetSize[1]: " + std::to_string(targetSize[1]));  // = 1
            // log("targetSize[2]: " + std::to_string(targetSize[2]));  // = 21(hands) or 70 (face)
            // log("targetSize[3]: " + std::to_string(targetSize[3]));  // = 3 = [x, y, score]
            // log(" ");
            for (auto n = 0; n < num; n++)
            {
                for (auto c = 0; c < channels; c++)
                {
                    // // Parameters
                    const auto offsetChannel = (n * channels + c);
                    for (auto part = 0; part < numberParts; part++)
                    {
                        auto* targetPtrOffsetted = targetPtr + (offsetChannel + part) * numberSubparts;
                        const auto* const sourcePtrOffsetted = sourcePtr + (offsetChannel + part) * imageOffset;
                        // Option a - 6.3 fps
                        const auto sourceThrustPtr = thrust::device_pointer_cast(sourcePtrOffsetted);
                        // Ideal option (not working for CUDA < 8)
                        // const auto sourceIndexIterator = thrust::max_element(
                        //     thrust::device, sourceThrustPtr, sourceThrustPtr + imageOffset);
                        // Workaround to make it work for CUDA 7.5
                        const auto sourceIndexIterator = thrust::max_element(
                            sourceThrustPtr, sourceThrustPtr + imageOffset);
                        const auto sourceIndex = (int)(sourceIndexIterator - sourceThrustPtr);
                        fillTargetPtrPart<<<1, 1>>>(targetPtrOffsetted, sourcePtrOffsetted, sourceIndex,
                                                    sourceIndex % width, sourceIndex / width);
                        // // Option b - <1 fps
                        // fillTargetPtrChannel<<<1, 1>>>(targetPtrOffsetted, sourcePtrOffsetted, width, imageOffset);
                    }
                    // Option c - 4.9 fps
                    // fillTargetPtr<<<1, numberParts>>>(targetPtr, sourcePtr, width, imageOffset, numberSubparts,
                    //                                   offsetChannel);
                }
            }
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void maximumGpu(
        float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize,
        const std::array<int, 4>& sourceSize);
    template void maximumGpu(
        double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize,
        const std::array<int, 4>& sourceSize);
}
