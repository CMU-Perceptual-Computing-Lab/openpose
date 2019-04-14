#include <openpose/utilities/avx.hpp>
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
                            const auto value = uchar(
                                fastTruncate(positiveIntRound(arrayPtr[offsetHeight + x]), 0, 255));
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
                        floatPtrImage[floatPtrImageOffsetY + x] = float(
                            originFramePtr[(originFramePtrOffsetY + x) * channels + c]);
                }
            }
            // Normalizing if desired
            // VGG
            if (normalize == 1)
            {
                #ifdef WITH_AVX
                    // // C++ code
                    // const auto ratio = 1.f/256.f;
                    // for (auto pixel = 0 ; pixel < width*height*channels ; ++pixel)
                    //     floatPtrImage[pixel] = floatPtrImage[pixel]*ratio - 0.5f;
                    // AVX code
                    const auto volume = width*height*channels;
                    int pixel;
                    const __m256 mmRatio = _mm256_set1_ps(1.f/256.f);
                    const __m256 mmBias = _mm256_set1_ps(-0.5f);
                    for (pixel = 0 ; pixel < volume-7 ; pixel += 8)
                    {
                        const __m256 input = _mm256_load_ps(&floatPtrImage[pixel]);
                        // const __m256 input = _mm256_loadu_ps(&floatPtrImage[pixel]); // If non-aligned pointer
                        const __m256 output = _mm256_fmadd_ps(input, mmRatio, mmBias);
                        _mm256_store_ps(&floatPtrImage[pixel], output);
                        // _mm256_storeu_ps(&floatPtrImage[pixel], output); // If non-aligned pointer
                    }
                    const auto ratio = 1.f/256.f;
                    for (; pixel < volume ; ++pixel)
                        floatPtrImage[pixel] = floatPtrImage[pixel]*ratio - 0.5f;
                // Non optimized code
                #else
                    // floatPtrImage wrapped as cv::Mat
                        // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and
                        // other optimized instruction sets).
                        // In addition, the following if statement does not copy the pointer to a cv::Mat, just wrapps it.
                    cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
                    floatPtrImageCvWrapper = floatPtrImageCvWrapper*(1/256.f) - 0.5f;
                #endif
            }
            // // ResNet
            // else if (normalize == 2)
            // {
            //     const int imageArea = width * height;
            //     const std::array<float,3> means{102.9801, 115.9465, 122.7717};
            //     for (auto i = 0 ; i < 3 ; i++)
            //     {
            //         cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i*imageArea);
            //         floatPtrImageCvWrapper = floatPtrImageCvWrapper - means[i];
            //     }
            // }
            // DenseNet
            else if (normalize == 2)
            {
                const auto scaleDenseNet = 0.017;
                const int imageArea = width * height;
                const std::array<float,3> means{103.94f, 116.78f, 123.68f};
                for (auto i = 0 ; i < 3 ; i++)
                {
                    cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i*imageArea);
                    floatPtrImageCvWrapper = scaleDenseNet*(floatPtrImageCvWrapper - means[i]);
                }
            }
            // Unknown
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

    void keepRoiInside(cv::Rect& roi, const int imageWidth, const int imageHeight)
    {
        try
        {
            // x,y < 0
            if (roi.x < 0)
            {
                roi.width += roi.x;
                roi.x = 0;
            }
            if (roi.y < 0)
            {
                roi.height += roi.y;
                roi.y = 0;
            }
            // Bigger than image
            if (roi.width + roi.x >= imageWidth)
                roi.width = imageWidth - 1 - roi.x;
            if (roi.height + roi.y >= imageHeight)
                roi.height = imageHeight - 1 - roi.y;
            // Width/height negative
            roi.width = fastMax(0, roi.width);
            roi.height = fastMax(0, roi.height);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void rotateAndFlipFrame(cv::Mat& frame, const double rotationAngle, const bool flipFrame)
    {
        try
        {
            if (!frame.empty())
            {
                const auto rotationAngleInt = (int)std::round(rotationAngle) % 360;
                if (rotationAngleInt == 0 || rotationAngleInt == 360)
                {
                    if (flipFrame)
                        cv::flip(frame, frame, 1);
                }
                else if (rotationAngleInt == 90 || rotationAngleInt == -270)
                {
                    cv::transpose(frame, frame);
                    if (!flipFrame)
                        cv::flip(frame, frame, 0);
                }
                else if (rotationAngleInt == 180 || rotationAngleInt == -180)
                {
                    if (flipFrame)
                        cv::flip(frame, frame, 0);
                    else
                        cv::flip(frame, frame, -1);
                }
                else if (rotationAngleInt == 270 || rotationAngleInt == -90)
                {
                    cv::transpose(frame, frame);
                    if (flipFrame)
                        cv::flip(frame, frame, -1);
                    else
                        cv::flip(frame, frame, 1);
                }
                else
                    error("Rotation angle = " + std::to_string(rotationAngleInt)
                          + " != {0, 90, 180, 270} degrees.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
