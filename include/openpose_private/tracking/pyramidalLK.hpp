#ifndef OPENPOSE_PRIVATE_TRACKING_LKPYRAMIDAL_HPP
#define OPENPOSE_PRIVATE_TRACKING_LKPYRAMIDAL_HPP

#include <opencv2/core/core.hpp> // cv::Mat, cv::Point2f
#include <openpose/core/common.hpp>

namespace op
{
    void pyramidalLKCpu(
        std::vector<cv::Point2f>& coordI, std::vector<cv::Point2f>& coordJ,
        std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
        std::vector<char>& status, const cv::Mat& imagePrevious, const cv::Mat& imageCurrent,
        const int levels = 3, const int patchSize = 21);

    int pyramidalLKGpu(
        std::vector<cv::Point2f>& ptsI, std::vector<cv::Point2f>& ptsJ,
        std::vector<char>& status, const cv::Mat& imagePrevious, const cv::Mat& imageCurrent,
        const int levels = 3, const int patchSize = 21);

    void pyramidalLKOcv(
        std::vector<cv::Point2f>& coordI, std::vector<cv::Point2f>& coordJ,
        std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
        std::vector<char>& status, const cv::Mat& imagePrevious, const cv::Mat& imageCurrent,
        const int levels = 3, const int patchSize = 21, const bool initFlow = false);
}

#endif // OPENPOSE_PRIVATE_TRACKING_LKPYRAMIDAL_HPP
