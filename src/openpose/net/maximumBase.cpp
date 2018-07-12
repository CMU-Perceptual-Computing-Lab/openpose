// #include <thrust/extrema.h>
#include <openpose/net/maximumBase.hpp>

namespace op
{
    template <typename T>
    void maximumCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                    const std::array<int, 4>& sourceSize)
    {
        try
        {
            // // TODO: ideally done, try, debug & compare to *.cu
            const auto height = sourceSize[2];
            const auto width = sourceSize[3];
            const auto imageOffset = height * width;
            const auto num = targetSize[0];
            const auto channels = targetSize[1];
            const auto numberParts = targetSize[2];
            const auto numberSubparts = targetSize[3];

            // log("sourceSize[0]: " + std::to_string(sourceSize[0])); // = 1
            // log("sourceSize[1]: " + std::to_string(sourceSize[1])); // = #body_parts+bck=22(hands) or 71(face)
            // log("sourceSize[2]: " + std::to_string(sourceSize[2])); // = 368 = height
            // log("sourceSize[3]: " + std::to_string(sourceSize[3])); // = 368 = width
            // log("targetSize[0]: " + std::to_string(targetSize[0])); // = 1
            // log("targetSize[1]: " + std::to_string(targetSize[1])); // = 1
            // log("targetSize[2]: " + std::to_string(targetSize[2])); // = 21(hands) or 70 (face)
            // log("targetSize[3]: " + std::to_string(targetSize[3])); // = 3 = [x, y, score]
            // log(" ");

            for (auto n = 0; n < num; n++)
            {
                for (auto c = 0; c < channels; c++)
                {
                    // Parameters
                    const auto offsetChannel = (n * channels + c);
                    for (auto part = 0; part < numberParts; part++)
                    {
                        auto* targetPtrOffsetted = targetPtr + (offsetChannel + part) * numberSubparts;
                        const auto* const sourcePtrOffsetted = sourcePtr + (offsetChannel + part) * imageOffset;
                        cv::Mat source(cv::Size(width, height), CV_32FC1, const_cast<T*>(sourcePtrOffsetted));
                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc;
                        cv::minMaxLoc(source, &minVal, &maxVal, &minLoc, &maxLoc);
                        targetPtrOffsetted[0] = maxLoc.x;
                        targetPtrOffsetted[1] = maxLoc.y;
                        targetPtrOffsetted[2] = maxVal;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void maximumCpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize,
                             const std::array<int, 4>& sourceSize);
    template void maximumCpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize,
                             const std::array<int, 4>& sourceSize);
}
