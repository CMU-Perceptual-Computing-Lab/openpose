#ifndef OPENPOSE_UTILITIES_OPEN_CV_HPP
#define OPENPOSE_UTILITIES_OPEN_CV_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgproc/imgproc.hpp> // cv::warpAffine, cv::BORDER_CONSTANT
#include <openpose/core/point.hpp> // Point<int>

namespace op
{
    void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position, const cv::Scalar& color, const bool normalizeWidth);

    void floatPtrToUCharCvMat(cv::Mat& cvMat, const float* const floatImage, const Point<int>& resolutionSize, const int resolutionChannels);

    void unrollArrayToUCharCvMat(cv::Mat& cvMatResult, const Array<float>& array);

    void uCharCvMatToFloatPtr(float* floatImage, const cv::Mat& cvImage, const bool normalize);

    double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize);

    cv::Mat resizeFixedAspectRatio(const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize, const int borderMode = cv::BORDER_CONSTANT,
                                   const cv::Scalar& borderValue = cv::Scalar{0,0,0});
}

#endif // OPENPOSE_UTILITIES_OPEN_CV_HPP
