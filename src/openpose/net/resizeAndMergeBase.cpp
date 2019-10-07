#include <openpose/net/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const std::vector<const T*>& sourcePtrs,
                           const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            // Scale used in CUDA/CL to know scale ratio between input and output
            // CPU directly uses sourceWidth/Height and targetWidth/Height
            UNUSED(scaleInputToNetInputs);

            // Sanity check
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);

            // Params
            const auto nums = (signed)sourceSizes.size();
            const auto channels = targetSize[1]; // 57
            const auto targetHeight = targetSize[2]; // 368
            const auto targetWidth = targetSize[3]; // 496
            const auto targetChannelOffset = targetWidth * targetHeight;

            // No multi-scale merging or no merging required
            if (sourceSizes.size() == 1)
            {
                // Params
                const auto& sourceSize = sourceSizes[0];
                const auto sourceHeight = sourceSize[2]; // 368/8 ..
                const auto sourceWidth = sourceSize[3]; // 496/8 ..
                const auto sourceChannelOffset = sourceHeight * sourceWidth;
                if (sourceSize[0] != 1)
                    error("It should never reache this point. Notify us otherwise.",
                          __LINE__, __FUNCTION__, __FILE__);

                // Per channel resize
                const T* sourcePtr = sourcePtrs[0];
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                   const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                   (&targetPtr[c*targetChannelOffset]));
                    cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
                }
            }
            // Multi-scale merging
            else
            {
                // Construct temp targets. We resuse targetPtr to store first scale
                std::vector<std::unique_ptr<T>> tempTargetPtrs;
                for (auto n = 1; n < nums; n++){
                    tempTargetPtrs.emplace_back(std::unique_ptr<T>(new T[targetChannelOffset * channels]()));
                }

                // Resize and sum
                for (auto n = 0; n < nums; n++){

                    // Params
                    const auto& sourceSize = sourceSizes[n];
                    const auto sourceHeight = sourceSize[2]; // 368/6 ..
                    const auto sourceWidth = sourceSize[3]; // 496/8 ..
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;

                    // Access pointers
                    const T* sourcePtr = sourcePtrs[n];
                    T* tempTargetPtr;
                    if (n != 0)
                        tempTargetPtr = tempTargetPtrs[n-1].get();
                    else
                        tempTargetPtr = targetPtr;

                    T* firstTempTargetPtr = targetPtr;
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        // Resize
                        cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                       const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                        cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                       (&tempTargetPtr[c*targetChannelOffset]));
                        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);

                        // Add
                        if (n != 0)
                        {
                            cv::Mat addTarget(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                              (&firstTempTargetPtr[c*targetChannelOffset]));
                            cv::add(target, addTarget, addTarget);
                        }
                    }
                }

                // Average
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1, (&targetPtr[c*targetChannelOffset]));
                    target /= (float)nums;
                }

            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template OP_API void resizeAndMergeCpu(
        float* targetPtr, const std::vector<const float*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<float>& scaleInputToNetInputs);
    template OP_API void resizeAndMergeCpu(
        double* targetPtr, const std::vector<const double*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<double>& scaleInputToNetInputs);
}
