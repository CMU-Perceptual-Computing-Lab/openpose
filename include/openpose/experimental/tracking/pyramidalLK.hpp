#ifndef OPENPOSE_TRACKING_LKPYRAMIDAL_HPP
#define OPENPOSE_TRACKING_LKPYRAMIDAL_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API void pyramidalLKCpu(std::vector<cv::Point2f>& coordI,
                               std::vector<cv::Point2f>& coordJ,
                               std::vector<cv::Mat>& pyramidImagesPrevious,
                               std::vector<cv::Mat>& pyramidImagesCurrent,
                               std::vector<char>& status,
                               const cv::Mat& imagePrevious,
                               const cv::Mat& imageCurrent,
                               const int levels,
                               const int patchSize = 5);
    OP_API int pyramidalLKGpu(std::vector<cv::Point2f>& ptsI,
                              std::vector<cv::Point2f>& ptsJ,
                              std::vector<char>& status,
                              const cv::Mat& imagePrevious,
                              const cv::Mat& imageCurrent,
                              const int levels, const int patchSize); 
}

#endif // OPENPOSE_TRACKING_LKPYRAMIDAL_HPP
