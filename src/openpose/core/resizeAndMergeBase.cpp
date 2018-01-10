#include <opencv2/highgui/highgui.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <openpose/utilities/openCv.hpp>

//#include <iostream>
//#include <chrono>
//#include <sys/time.h>
//using namespace std;
//using namespace std::chrono;

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

            // TEST WITH SCALEINPUTTONETINPUTS

            // Params
            const auto nums = (signed)sourceSizes.size();
            const auto channels = targetSize[1]; // 57
            const auto targetHeight = targetSize[2]; // 368
            const auto targetWidth = targetSize[3]; // 496
            const auto targetChannelOffset = targetWidth * targetHeight;
            //high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // No multi-scale merging or no merging required
            if(sourceSizes.size() == 1)
            {
                // Params
                const auto& sourceSize = sourceSizes[0];
                const auto sourceHeight = sourceSize[2]; // 368/6 ..
                const auto sourceWidth = sourceSize[3]; // 496/8 ..
                const auto sourceChannelOffset = sourceHeight * sourceWidth;
                if(sourceSize[0] != 1) error("It should never reache this point. Notify us otherwise.", __LINE__, __FUNCTION__, __FILE__);

                // Warp Matrix
                auto scaleFactor = op::resizeGetScaleFactor(op::Point<int>(sourceWidth, sourceHeight),op::Point<int>(targetWidth, targetHeight));
                cv::Mat M = cv::Mat::eye(2,3,CV_64F);
                M.at<double>(0,0) = scaleFactor;
                M.at<double>(1,1) = scaleFactor;

                // Per channel resize
                const T* sourcePtr = sourcePtrs[0];
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1, const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1, (&targetPtr[c*targetChannelOffset]));
                    cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
                    //cv::warpAffine(source, target, M, cv::Size(targetWidth, targetHeight),(scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC));
                }
            }
            // Multi-scale merging
            else
            {
                // Construct temp targets. Could have a way of storing this somewhere if reusing same frame
                std::vector<T*> tempTargetPtrs;
                for(auto n = 0; n < nums; n++){
                    if(n==0) tempTargetPtrs.emplace_back(targetPtr);
                    else tempTargetPtrs.emplace_back(new T[targetChannelOffset * channels]());
                }

                // Resize and sum
                for(auto n = 0; n < nums; n++){

                    // Params
                    const auto& sourceSize = sourceSizes[n];
                    const auto sourceHeight = sourceSize[2]; // 368/6 ..
                    const auto sourceWidth = sourceSize[3]; // 496/8 ..
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;

                    // Warp Matrix
                    auto scaleFactor = op::resizeGetScaleFactor(op::Point<int>(sourceWidth, sourceHeight),op::Point<int>(targetWidth, targetHeight));
                    cv::Mat M = cv::Mat::eye(2,3,CV_64F);
                    M.at<double>(0,0) = scaleFactor;
                    M.at<double>(1,1) = scaleFactor;

                    const T* sourcePtr = sourcePtrs[n];
                    T* tempTargetPtr = tempTargetPtrs[n];
                    T* firstTempTargetPtr = tempTargetPtrs[0];
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        // Resize
                        cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1, const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                        cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1, (&tempTargetPtr[c*targetChannelOffset]));
                        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
log(scaleFactor);
                        //cv::warpAffine(source, target, M, cv::Size(targetWidth, targetHeight),(scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC));

                        // Add
                        if(n != 0){
                            cv::Mat addTarget(cv::Size(targetWidth, targetHeight), CV_32FC1, (&firstTempTargetPtr[c*targetChannelOffset]));
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

                // Delete tempTargetPtrs later
                for(auto n = 0; n < nums; n++){
                    if(n!=0){
                        T* tempTargetPtr = tempTargetPtrs[n];
                        delete tempTargetPtr;
                    }
                }
            }
            //high_resolution_clock::time_point t2 = high_resolution_clock::now();
            //cout << duration_cast<milliseconds>( t2 - t1 ).count()  << endl;
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
