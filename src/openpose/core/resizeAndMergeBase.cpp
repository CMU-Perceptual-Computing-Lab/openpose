// #include <opencv2/imgproc/imgproc.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>

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
            UNUSED(targetPtr);
            UNUSED(sourcePtrs);
            UNUSED(scaleInputToNetInputs);
            UNUSED(targetSize);
            UNUSED(sourceSizes);
            error("CPU version not completely implemented.", __LINE__, __FUNCTION__, __FILE__);

            // TODO: THIS CODE IS WORKING, BUT IT DOES NOT CONSIDER THE SCALES (I.E. SCALE NUMBER, START AND GAP) 
            // const int num = bottom->shape(0);
            // const int channel = bottom->shape(1);
            // const int sourceHeight = bottom->shape(2);
            // const int sourceWidth = bottom->shape(3);
            // const int targetHeight = top->shape(2);
            // const int targetWidth = top->shape(3);

            // //stupid method
            // for (int n = 0; n < num; n++)
            // {
            //     for (int c = 0; c < channel; c++)
            //     {
            //         //fill source
            //         cv::Mat source(sourceWidth, sourceHeight, CV_32FC1);
            //         const auto sourceOffsetChannel = sourceHeight * sourceWidth;
            //         const auto sourceOffsetNum = sourceOffsetChannel * channel;
            //         const auto sourceOffset = n*sourceOffsetNum + c*sourceOffsetChannel;
            //         const T* const sourcePtrs = bottom->cpu_data();
            //         for (int y = 0; y < sourceHeight; y++)
            //             for (int x = 0; x < sourceWidth; x++)
            //                 source.at<T>(x,y) = sourcePtrs[sourceOffset + y*sourceWidth + x];

            //         // spatial resize
            //         cv::Mat target;
            //         cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);

            //         //fill top
            //         const auto targetOffsetChannel = targetHeight * targetWidth;
            //         const auto targetOffsetNum = targetOffsetChannel * channel;
            //         const auto targetOffset = n*targetOffsetNum + c*targetOffsetChannel;
            //         T* targetPtr = top->mutable_cpu_data();
            //         for (int y = 0; y < targetHeight; y++)
            //             for (int x = 0; x < targetWidth; x++)
            //                 targetPtr[targetOffset + y*targetWidth + x] = target.at<T>(x,y);
            //     }
            // }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeCpu(float* targetPtr, const std::vector<const float*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<float>& scaleInputToNetInputs);
    template void resizeAndMergeCpu(double* targetPtr, const std::vector<const double*>& sourcePtrs,
                                    const std::array<int, 4>& targetSize,
                                    const std::vector<std::array<int, 4>>& sourceSizes,
                                    const std::vector<double>& scaleInputToNetInputs);
}
