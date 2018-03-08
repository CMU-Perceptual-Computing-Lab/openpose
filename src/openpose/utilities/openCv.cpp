#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>

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
            const auto fontScale = 0.8 * std::sqrt(ratio);
            const auto fontThickness = std::max(1, intRound(2*ratio));
            const auto shadowOffset = std::max(1, intRound(2*ratio));
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

    void floatPtrToUCharCvMat(cv::Mat& uCharCvMat, const float* const floatPtrImage,
                              const std::array<int, 3> resolutionSize)
    {
        try
        {
            // Info:
                // float* (deep net format): C x H x W
                // cv::Mat (OpenCV format): H x W x C
            // Allocate cv::Mat if it was not initialized yet
            if (uCharCvMat.empty() || uCharCvMat.rows != resolutionSize[1]
                || uCharCvMat.cols != resolutionSize[0] || uCharCvMat.type() != CV_8UC3)
                uCharCvMat = cv::Mat(resolutionSize[1], resolutionSize[0], CV_8UC3);
            // Fill uCharCvMat from floatPtrImage
            auto* uCharPtrCvMat = (unsigned char*)(uCharCvMat.data);
            const auto offsetBetweenChannels = resolutionSize[0] * resolutionSize[1];
            const auto stepSize = uCharCvMat.step; // step = cols * channels
            for (auto c = 0; c < resolutionSize[2]; c++)
            {
                const auto offsetChannelC = c*offsetBetweenChannels;
                for (auto y = 0; y < resolutionSize[1]; y++)
                {
                    const auto yOffset = y * stepSize;
                    const auto floatPtrImageOffsetY = offsetChannelC + y*resolutionSize[0];
                    for (auto x = 0; x < resolutionSize[0]; x++)
                    {
                        const auto value = uchar(
                            fastTruncate(intRound(floatPtrImage[floatPtrImageOffsetY + x]), 0, 255)
                        );
                        uCharPtrCvMat[yOffset + x * resolutionSize[2] + c] = value;
                        // *(uCharCvMat.ptr<uchar>(y, x) + c) = value; // Slower but safer and cleaner equivalent
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
                error("Only implemented for array.getNumberDimensions() == 3 so far.",
                      __LINE__, __FUNCTION__, __FILE__);

            if (!array.empty())
            {
                const auto channels = array.getSize(0);
                const auto height = array.getSize(1);
                const auto width = array.getSize(2);
                const auto areaInput = height * width;
                const auto areaOutput = channels * width;
                // Allocate cv::Mat if it was not initialized yet
                if (cvMatResult.empty() || cvMatResult.cols != channels * width || cvMatResult.rows != height)
                    cvMatResult = cv::Mat(height, areaOutput, CV_8UC1);
                // Fill cvMatResult from array
                for (auto channel = 0 ; channel < channels ; channel++)
                {
                    // Get memory to be modified
                    cv::Mat cvMatROI(cvMatResult, cv::Rect{channel * width, 0, width, height});
                    // Modify memory
                    const auto* arrayPtr = array.getConstPtr() + channel * areaInput;
                    for (auto y = 0 ; y < height ; y++)
                    {
                        auto* cvMatROIPtr = cvMatROI.ptr<uchar>(y);
                        const auto offsetHeight = y * width;
                        for (auto x = 0 ; x < width ; x++)
                        {
                            const auto value = uchar( fastTruncate(intRound(arrayPtr[offsetHeight + x]), 0, 255) );
                            cvMatROIPtr[x] = (unsigned char)(value);
                        }
                    }
                }
            }
            else
                cvMatResult = cv::Mat();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void uCharCvMatToFloatPtr(float* floatPtrImage, const cv::Mat& cvImage, const int normalize)
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
                const auto floatPtrImageOffsetC = c * height;
                for (auto y = 0; y < height; y++)
                {
                    const auto floatPtrImageOffsetY = (floatPtrImageOffsetC + y) * width;
                    const auto originFramePtrOffsetY = y * width;
                    for (auto x = 0; x < width; x++)
                        floatPtrImage[floatPtrImageOffsetY + x] = float(originFramePtr[(originFramePtrOffsetY + x)
                                                                        * channels + c]);
                }
            }
            // Normalizing if desired
            // floatPtrImage wrapped as cv::Mat
                // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and
                // other optimized instruction sets).
                // In addition, the following if statement does not copy the pointer to a cv::Mat, just wrapps it.
            if (normalize == 1)
            {
                cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
                floatPtrImageCvWrapper = floatPtrImageCvWrapper/256.f - 0.5f;
            }
            else if (normalize == 2)
            {
                const int imageArea = width * height;
                const std::array<float,3> means{102.9801, 115.9465, 122.7717};
                for (auto i = 0 ; i < 3 ; i++)
                {
                    cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i*imageArea);
                    floatPtrImageCvWrapper = floatPtrImageCvWrapper - means[i];
                }
            }
            else if (normalize != 0)
                error("Unknown normalization value (" + std::to_string(normalize) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
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
            const auto ratioWidth = (targetSize.x - 1) / (double)(initialSize.x - 1);
            const auto ratioHeight = (targetSize.y - 1) / (double)(initialSize.y - 1);
            return fastMin(ratioWidth, ratioHeight);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    cv::Mat resizeFixedAspectRatio(const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize,
                                   const int borderMode, const cv::Scalar& borderValue)
    {
        try
        {
            const cv::Size cvTargetSize{targetSize.x, targetSize.y};
            cv::Mat resultingCvMat;
            cv::Mat M = cv::Mat::eye(2,3,CV_64F);
            M.at<double>(0,0) = scaleFactor;
            M.at<double>(1,1) = scaleFactor;
            if (scaleFactor != 1. || cvTargetSize != cvMat.size())
                cv::warpAffine(cvMat, resultingCvMat, M, cvTargetSize,
                               (scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC), borderMode, borderValue);
            else
                resultingCvMat = cvMat.clone();
            return resultingCvMat;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }
}
