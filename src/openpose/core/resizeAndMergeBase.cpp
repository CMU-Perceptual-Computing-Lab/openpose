#include <opencv2/highgui/highgui.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>

// FOR RAAJ DEBUGGING
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
using namespace std;
using namespace std::chrono;

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
            // Security checks
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (sourcePtrs.size() != sourceSizes.size() || sourceSizes.size() != scaleInputToNetInputs.size())
                error("Size(sourcePtrs) must match size(sourceSizes) and size(scaleInputToNetInputs). Currently: "
                      + std::to_string(sourcePtrs.size()) + " vs. " + std::to_string(sourceSizes.size()) + " vs. "
                      + std::to_string(scaleInputToNetInputs.size()) + ".", __LINE__, __FUNCTION__, __FILE__);


            // Parameters
            const auto channels = targetSize[1]; // 57
            const auto targetHeight = targetSize[2]; // 368
            const auto targetWidth = targetSize[3]; // 496
            const auto& sourceSize = sourceSizes[0];
            const auto sourceHeight = sourceSize[2]; // 368/6 ..
            const auto sourceWidth = sourceSize[3]; // 496/8 ..

            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // No multi-scale merging or no merging required
            if(sourceSizes.size() == 1)
            {
                const auto num = sourceSize[0]; // 1 always
                if(num == 1)
                {
                    const T* sourcePtr = sourcePtrs[0];
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;
                    const auto targetChannelOffset = targetWidth * targetHeight;
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        // OpenCV Method (There should be some way to avoid the copys below but it doesnt work)
                        cv::Mat source(cv::Size(sourceWidth, sourceHeight) , CV_32FC1, sourcePtr[c*sourceChannelOffset]);
                        for (int y = 0; y < sourceHeight; y++){
                            T* Mi = source.ptr<T>(y);
                            for (int x = 0; x < sourceWidth; x++)
                                Mi[x] = sourcePtr[(c*sourceChannelOffset) + y*sourceWidth + x];
                        }
                        cv::Mat target;
                        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
                        //targetPtr[(c*targetChannelOffset)] = *target.ptr<T>(0); // Cant do this target will go out of scope
                        for (int y = 0; y < targetHeight; y++){
                            T* Mi = target.ptr<T>(y);
                            for (int x = 0; x < targetWidth; x++)
                                targetPtr[(c*targetChannelOffset) + y*targetWidth + x] = Mi[x];
                        }
                    }
                }
                else
                    error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);

                //error("CPU version - No multiscale not completely implemented.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Multi-scale merging
            else
            {
                error("CPU version - Multiscale not completely implemented.", __LINE__, __FUNCTION__, __FILE__);
            }

            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            cout << duration_cast<milliseconds>( t2 - t1 ).count()  << endl;


//            // //stupid method
//             for (int n = 0; n < num; n++)
//             {
//                 for (int c = 0; c < channel; c++)
//                 {
//                     //fill source
//                     cv::Mat source(sourceWidth, sourceHeight, CV_32FC1);
//                     const auto sourceOffsetChannel = sourceHeight * sourceWidth;
//                     const auto sourceOffsetNum = sourceOffsetChannel * channel;
//                     const auto sourceOffset = n*sourceOffsetNum + c*sourceOffsetChannel;
//                     const T* const sourcePtrs = bottom->cpu_data();
//                     for (int y = 0; y < sourceHeight; y++)
//                         for (int x = 0; x < sourceWidth; x++)
//                             source.at<T>(x,y) = sourcePtrs[sourceOffset + y*sourceWidth + x];

//                     // spatial resize
//                     cv::Mat target;
//                     cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);

//                     //fill top
//                     const auto targetOffsetChannel = targetHeight * targetWidth;
//                     const auto targetOffsetNum = targetOffsetChannel * channel;
//                     const auto targetOffset = n*targetOffsetNum + c*targetOffsetChannel;
//                     T* targetPtr = top->mutable_cpu_data();
//                     for (int y = 0; y < targetHeight; y++)
//                         for (int x = 0; x < targetWidth; x++)
//                             targetPtr[targetOffset + y*targetWidth + x] = target.at<T>(x,y);
//                 }
//             }
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
