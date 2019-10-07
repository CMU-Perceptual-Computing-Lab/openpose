#include <openpose_private/utilities/openCvPrivate.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose_private/utilities/avx.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position,
                        const cv::Scalar& color, const bool normalizeWidth, const int imageWidth)
    {
        try
        {
            const auto font = cv::FONT_HERSHEY_SIMPLEX;
            const auto ratio = imageWidth/1280.;
            // const auto fontScale = 0.75;
            const auto fontScale = 0.8 * ratio;
            const auto fontThickness = std::max(1, positiveIntRound(2*ratio));
            const auto shadowOffset = std::max(1, positiveIntRound(2*ratio));
            int baseline = 0;
            const auto textSize = cv::getTextSize(textToDisplay, font, fontScale, fontThickness, &baseline);
            const cv::Size finalPosition{position.x - (normalizeWidth ? textSize.width : 0),
                                         position.y + textSize.height/2};
            cv::putText(cvMat, textToDisplay,
                        cv::Size{finalPosition.width + shadowOffset, finalPosition.height + shadowOffset},
                        font, fontScale, cv::Scalar{0,0,0}, fontThickness);
            cv::putText(cvMat, textToDisplay, finalPosition, font, fontScale, color, fontThickness);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void resizeFixedAspectRatio(cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor,
                                const Point<int>& targetSize, const int borderMode, const cv::Scalar& borderValue)
    {
        try
        {
            const cv::Size cvTargetSize{targetSize.x, targetSize.y};
            cv::Mat M = cv::Mat::eye(2,3,CV_64F);
            M.at<double>(0,0) = scaleFactor;
            M.at<double>(1,1) = scaleFactor;
            if (scaleFactor != 1. || cvTargetSize != cvMat.size())
                cv::warpAffine(cvMat, resizedCvMat, M, cvTargetSize,
                               (scaleFactor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA), borderMode, borderValue);
            else
                cvMat.copyTo(resizedCvMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
