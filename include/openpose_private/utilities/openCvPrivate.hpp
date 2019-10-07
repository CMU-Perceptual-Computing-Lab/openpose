#ifndef OPENPOSE_PRIVATE_UTILITIES_OPEN_CV_PRIVATE_HPP
#define OPENPOSE_PRIVATE_UTILITIES_OPEN_CV_PRIVATE_HPP

#include <opencv2/core/core.hpp> // cv::Mat, cv::Rect, cv::Scalar
#include <opencv2/imgproc/imgproc.hpp> // cv::BORDER_CONSTANT
#include <openpose/core/common.hpp>

namespace op
{
    void putTextOnCvMat(
        cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position,
        const cv::Scalar& color, const bool normalizeWidth, const int imageWidth);

    void resizeFixedAspectRatio(
        cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize,
        const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar{0,0,0});
}

#endif // OPENPOSE_PRIVATE_UTILITIES_OPEN_CV_PRIVATE_HPP
