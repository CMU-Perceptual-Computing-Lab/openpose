// #include <thrust/extrema.h>
#include <openpose/core/maximumBase.hpp>

namespace op
{
    template <typename T>
    void maximumCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
    {
        try
        {
            UNUSED(targetPtr);
            UNUSED(kernelPtr);
            UNUSED(sourcePtr);
            UNUSED(targetSize);
            UNUSED(sourceSize);
            error("CPU version not completely implemented.", __LINE__, __FUNCTION__, __FILE__);

            // // TODO: ideally done, try, debug & compare to *.cu
            // TODO: (maybe): remove thrust dependencies for computers without CUDA?
            // const auto height = sourceSize[2];
            // const auto width = sourceSize[3];
            // const auto imageOffset = height * width;
            // const auto num = targetSize[0];
            // const auto channels = targetSize[1];
            // const auto numberParts = targetSize[2];
            // const auto numberSubparts = targetSize[3];

            // // log("sourceSize[0]: " + std::to_string(sourceSize[0]));  // = 1
            // // log("sourceSize[1]: " + std::to_string(sourceSize[1]));  // = #body parts + bck = 22 (hands) or 71 (face) 
            // // log("sourceSize[2]: " + std::to_string(sourceSize[2]));  // = 368 = height
            // // log("sourceSize[3]: " + std::to_string(sourceSize[3]));  // = 368 = width
            // // log("targetSize[0]: " + std::to_string(targetSize[0]));  // = 1
            // // log("targetSize[1]: " + std::to_string(targetSize[1]));  // = 1
            // // log("targetSize[2]: " + std::to_string(targetSize[2]));  // = 21(hands) or 70 (face)
            // // log("targetSize[3]: " + std::to_string(targetSize[3]));  // = 3 = [x, y, score]
            // // log(" ");
            // for (auto n = 0; n < num; n++)
            // {
            //     for (auto c = 0; c < channels; c++)
            //     {
            //         // // Parameters
            //         const auto offsetChannel = (n * channels + c);
            //         for (auto part = 0; part < numberParts; part++)
            //         {
            //             auto* targetPtrOffsetted = targetPtr + (offsetChannel + part) * numberSubparts;
            //             const auto* const sourcePtrOffsetted = sourcePtr + (offsetChannel + part) * imageOffset;
            //             // Option a - 6.3 fps
            //             const auto sourceIndexIterator = thrust::max_element(thrust::host, sourcePtrOffsetted, sourcePtrOffsetted + imageOffset);
            //             const auto sourceIndex = (int)(sourceIndexIterator - sourcePtrOffsetted);
            //             targetPtrOffsetted[0] = sourceIndex % width;
            //             targetPtrOffsetted[1] = sourceIndex / width;
            //             targetPtrOffsetted[2] = sourcePtrOffsetted[sourceIndex];
            //         }
            //     }
            // }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void maximumCpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
    template void maximumCpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}
