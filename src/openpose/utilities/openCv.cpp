#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>

namespace op
{
    void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position, const cv::Scalar& color, const bool normalizeWidth)
    {
        try
        {
            const auto font = cv::FONT_HERSHEY_SIMPLEX;
            // const auto fontScale = 0.75;
            const auto fontScale = 0.8;
            const auto fontThickness = 2;
            const auto shadowOffset = 2;
            int baseline = 0;
            const auto textSize = cv::getTextSize(textToDisplay, font, fontScale, fontThickness, &baseline);
            const cv::Size finalPosition{position.x - (normalizeWidth ? textSize.width : 0), position.y + textSize.height/2};
            cv::putText(cvMat, textToDisplay, cv::Size{finalPosition.width + shadowOffset, finalPosition.height + shadowOffset}, font, fontScale, cv::Scalar{0,0,0}, fontThickness);
            cv::putText(cvMat, textToDisplay, finalPosition, font, fontScale, color, fontThickness);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void floatPtrToUCharCvMat(cv::Mat& cvMat, const float* const floatImage, const Point<int>& resolutionSize, const int resolutionChannels)
    {
        try
        {
            // float* (deep net format): C x H x W
            // cv::Mat (OpenCV format): H x W x C
            if (cvMat.rows != resolutionSize.y || cvMat.cols != resolutionSize.x || cvMat.type() != CV_8UC3)
                cvMat = cv::Mat{resolutionSize.y, resolutionSize.x, CV_8UC3};
            const auto offsetBetweenChannels = resolutionSize.x * resolutionSize.y;
            for (auto c = 0; c < resolutionChannels; c++)
            {
                const auto offsetChannelC = c*offsetBetweenChannels;
                for (auto y = 0; y < resolutionSize.y; y++)
                {
                    const auto floatImageOffsetY = offsetChannelC + y*resolutionSize.x;
                    for (auto x = 0; x < resolutionSize.x; x++)
                    {
                        const auto value = uchar(   fastTruncate(intRound(floatImage[floatImageOffsetY + x]), 0, 255)   );
                        *(cvMat.ptr<uchar>(y) + x*resolutionChannels + c) = value;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void unrollArrayToUCharCvMat(cv::Mat& cvMatResult, const Array<float>& array)
    {
        try
        {
            if (array.getNumberDimensions() != 3)
                error("Only implemented for array.getNumberDimensions() == 3 so far.", __LINE__, __FUNCTION__, __FILE__);

            if (!array.empty())
            {
                const auto channels = array.getSize(0);
                const auto height = array.getSize(1);
                const auto width = array.getSize(2);
                const auto areaInput = height * width;
                const auto areaOutput = channels * width;

                // Allocate cv::Mat
                if (cvMatResult.cols != channels * width || cvMatResult.rows != height)
                    cvMatResult = cv::Mat{height, areaOutput, CV_8UC1};

                // Fill cv::Mat
                for (auto channel = 0 ; channel < channels ; channel++)
                {
                    // Get memory to be modified
                    cv::Mat cvMatROI{cvMatResult, cv::Rect{channel * width, 0, width, height}};

                    // Modify memory
                    const auto* arrayPtr = array.getConstPtr() + channel * areaInput;
                    for (auto y = 0 ; y < height ; y++)
                    {
                        auto* cvMatROIPtr = cvMatROI.ptr<uchar>(y);
                        const auto offsetHeight = y * width;
                        for (auto x = 0 ; x < width ; x++)
                        {
                            const auto value = uchar(   fastTruncate(intRound(arrayPtr[offsetHeight + x]), 0, 255)   );
                            cvMatROIPtr[x] = (unsigned char)(value);
                        }
                    }
                }
            }
            else
                cvMatResult = cv::Mat{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void uCharCvMatToFloatPtr(float* floatImage, const cv::Mat& cvImage, const bool normalize)
    {
        try
        {
            // float* (deep net format): C x H x W
            // cv::Mat (OpenCV format): H x W x C 
            const int width = cvImage.cols;
            const int height = cvImage.rows;
            const int channels = cvImage.channels();

            const auto* const originFramePtr = cvImage.data;    // cv::Mat.data is always uchar
            for (auto c = 0; c < channels; c++)
            {
                const auto floatImageOffsetC = c * height;
                for (auto y = 0; y < height; y++)
                {
                    const auto floatImageOffsetY = (floatImageOffsetC + y) * width;
                    const auto originFramePtrOffsetY = y * width;
                    for (auto x = 0; x < width; x++)
                        floatImage[floatImageOffsetY + x] = float(originFramePtr[(originFramePtrOffsetY + x) * channels + c]);
                }
            }
            // Normalizing if desired
                // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and other optimized instruction sets)
                // In addition, the following if statement does not copy the pointer to a cv::Mat, just wrapps it
            if (normalize)
            {
                cv::Mat floatImageCvWrapper{height, width, CV_32FC3, floatImage};
                floatImageCvWrapper = floatImageCvWrapper/256.f - 0.5f;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize)
    {
        try
        {
            const auto ratioWidth = targetSize.x / (double)initialSize.x;
            const auto ratioHeight = targetSize.y / (double)initialSize.y;
            return fastMin(ratioWidth, ratioHeight);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    cv::Mat resizeFixedAspectRatio(const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize, const int borderMode, const cv::Scalar& borderValue)
    {
        try
        {
            const cv::Size cvTargetSize{targetSize.x, targetSize.y};
            cv::Mat resultingCvMat;
            cv::Mat M = cv::Mat::eye(2,3,CV_64F);
            M.at<double>(0,0) = scaleFactor;
            M.at<double>(1,1) = scaleFactor;
            if (scaleFactor != 1. || cvTargetSize != cvMat.size())
                cv::warpAffine(cvMat, resultingCvMat, M, cvTargetSize, (scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC), borderMode, borderValue);
            else
                resultingCvMat = cvMat.clone();
            return resultingCvMat;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat{};
        }
    }
}
