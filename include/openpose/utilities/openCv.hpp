#ifndef OPENPOSE_UTILITIES_OPEN_CV_HPP
#define OPENPOSE_UTILITIES_OPEN_CV_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgproc/imgproc.hpp> // cv::warpAffine, cv::BORDER_CONSTANT
#include <openpose/core/common.hpp>

namespace op
{
    OP_API void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position,
                               const cv::Scalar& color, const bool normalizeWidth, const int imageWidth);

    OP_API void unrollArrayToUCharCvMat(cv::Mat& cvMatResult, const Array<float>& array);

    OP_API void uCharCvMatToFloatPtr(float* floatPtrImage, const cv::Mat& cvImage, const int normalize);

    OP_API double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize);

    OP_API void resizeFixedAspectRatio(
        cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize,
        const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar{0,0,0});

    OP_API void keepRoiInside(cv::Rect& roi, const int imageWidth, const int imageHeight);
}

#endif // OPENPOSE_UTILITIES_OPEN_CV_HPP
