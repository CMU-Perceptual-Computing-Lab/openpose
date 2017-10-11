#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>


namespace op
{
    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                           const std::array<int, 4>& sourceSize, const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            const int num = sourceSize[0];
            const int channels = sourceSize[1];
            const int sourceHeight = sourceSize[2];
            const int sourceWidth = sourceSize[3];
            const int targetHeight = targetSize[2];
            const int targetWidth = targetSize[3];
            
            const auto sourceChannelOffset = sourceHeight * sourceWidth;
            const auto targetChannelOffset = targetWidth * targetHeight;
             
            // Perform resize + merging
            const auto sourceNumOffset = channels * sourceChannelOffset;
            for (auto c = 0 ; c < channels ; c++) {
                cv::Mat target (targetHeight, targetWidth, CV_32F, (void*)(targetPtr + c * targetChannelOffset));
                cv::multiply(target, 0.f, target);
                cv::Mat t;
                for (auto n = 0; n < num; n++) {
                    cv::Mat source(std::rint(sourceHeight * scaleRatios[n]), std::rint(sourceWidth * scaleRatios[n]), CV_32F, (void*)(sourcePtr + c * sourceChannelOffset + n * sourceNumOffset));
                    cv::resize(source, t, cv::Size(targetWidth, targetHeight), 0., 0., cv::INTER_CUBIC);
                    cv::add(target, t, target);
                }
                cv::divide(target, (float)num, target);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeCpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<float>& scaleInputToNetInputs);
    template void resizeAndMergeCpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<double>& scaleInputToNetInputs);
}
