#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose_private/utilities/avx.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    void unrollArrayToUCharCvMat(Matrix& matResult, const Array<float>& array)
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
                cv::Mat cvMatResult = OP_OP2CVMAT(matResult);
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
                matResult = OP_CV2OPMAT(cvMatResult);
            }
            else
                matResult = Matrix();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void uCharCvMatToFloatPtr(float* floatPtrImage, const Matrix& matImage, const int normalize)
    {
        try
        {
            const cv::Mat cvImage = OP_OP2CVCONSTMAT(matImage);
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
                    // // To check results are the same (norm(x1-x2) = 0)
                    // cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
                    // cv::Mat floatPtrImageCvWrapperTest = floatPtrImageCvWrapper*(1/256.f) - 0.5f;
                    // // Speed profiling
                    // const auto REPS = 2000;
                    // double timeNormalize0 = 0.;
                    // double timeNormalize1 = 0.;
                    // double timeNormalize2 = 0.;
                    // double timeNormalize3 = 0.;
                    // // OpenCV wrapper
                    // OP_PROFILE_INIT(REPS);
                    // cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
                    // floatPtrImageCvWrapper = floatPtrImageCvWrapper*(1/256.f) - 0.5f;
                    // OP_PROFILE_END(timeNormalize0, 1e6, REPS);
                    // // C++ sequential code
                    // OP_PROFILE_INIT(REPS);
                    // const auto ratio = 1.f/256.f;
                    // for (auto pixel = 0 ; pixel < width*height*channels ; ++pixel)
                    //     floatPtrImage[pixel] = floatPtrImage[pixel]*ratio - 0.5f;
                    // OP_PROFILE_END(timeNormalize1, 1e6, REPS);
                    // // OpenCV wrapper
                    // OP_PROFILE_INIT(REPS);
                    // cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
                    // floatPtrImageCvWrapper = floatPtrImageCvWrapper*(1/256.f) - 0.5f;
                    // OP_PROFILE_END(timeNormalize2, 1e6, REPS);
                    // OP_PROFILE_INIT(REPS);
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
                    // OP_PROFILE_END(timeNormalize3, 1e6, REPS);
                    // std::cout
                    //     << "TN1: " << timeNormalize0 << " us\n"
                    //     << "TN1: " << timeNormalize1 << " us\n"
                    //     << "TN2: " << timeNormalize2 << " us\n"
                    //     << "TN3: " << timeNormalize3 << " us\n"
                    //     << std::endl;
                    // std::cout
                    //     << "Norm: " << cv::norm(floatPtrImageCvWrapper-floatPtrImageCvWrapperTest) << "\n"
                    //     << std::endl;
                // Non optimized code
                #else
                    // floatPtrImage wrapped as cv::Mat
                        // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and
                        // other optimized instruction sets).
                        // In addition, the following if statement does not copy the pointer to a cv::Mat, just wraps it.
                    cv::Mat floatPtrImageCvWrapper(height*width*3, 1, CV_32FC1, floatPtrImage); // CV_32FC3 warns about https://github.com/opencv/opencv/issues/16739
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

    void keepRoiInside(Rectangle<int>& roi, const int imageWidth, const int imageHeight)
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

    void transpose(Matrix& matrix)
    {
        cv::Mat cvMatrix = OP_OP2CVMAT(matrix);
        Matrix matrixFinal(matrix.cols(), matrix.rows(), matrix.type());
        cv::Mat cvMatrixFinal = OP_OP2CVMAT(matrixFinal);
        cv::transpose(cvMatrix, cvMatrixFinal);
        std::swap(matrix, matrixFinal);
    }

    void rotateAndFlipFrame(Matrix& frame, const double rotationAngle, const bool flipFrame)
    {
        try
        {
            // cv::flip() does not moidify the memory location of the cv::Mat, but cv::transpose does
            if (!frame.empty())
            {
                const auto rotationAngleInt = (int)std::round(rotationAngle) % 360;
                // Transposing
                if (rotationAngleInt == 90 || rotationAngleInt == 270 || rotationAngleInt == -90 || rotationAngleInt == -270)
                    transpose(frame);
                // Mirroring (flipping)
                cv::Mat cvMatFrame = OP_OP2CVMAT(frame);
                if (rotationAngleInt == 0 || rotationAngleInt == 360)
                {
                    if (flipFrame)
                        cv::flip(cvMatFrame, cvMatFrame, 1);
                }
                else if (rotationAngleInt == 90 || rotationAngleInt == -270)
                {
                    if (!flipFrame)
                        cv::flip(cvMatFrame, cvMatFrame, 0);
                }
                else if (rotationAngleInt == 180 || rotationAngleInt == -180)
                {
                    if (flipFrame)
                        cv::flip(cvMatFrame, cvMatFrame, 0);
                    else
                        cv::flip(cvMatFrame, cvMatFrame, -1);
                }
                else if (rotationAngleInt == 270 || rotationAngleInt == -90)
                {
                    if (flipFrame)
                        cv::flip(cvMatFrame, cvMatFrame, -1);
                    else
                        cv::flip(cvMatFrame, cvMatFrame, 1);
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

    int getCvCapPropFrameCount()
    {
        try
        {
            return CV_CAP_PROP_FRAME_COUNT;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvCapPropFrameFps()
    {
        try
        {
            return CV_CAP_PROP_FPS;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvCapPropFrameWidth()
    {
        try
        {
            return CV_CAP_PROP_FRAME_WIDTH;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvCapPropFrameHeight()
    {
        try
        {
            return CV_CAP_PROP_FRAME_HEIGHT;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvFourcc(const char c1, const char c2, const char c3, const char c4)
    {
        try
        {
            return CV_FOURCC(c1,c2,c3,c4);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvImwriteJpegQuality()
    {
        try
        {
            return CV_IMWRITE_JPEG_QUALITY;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvImwritePngCompression()
    {
        try
        {
            return CV_IMWRITE_PNG_COMPRESSION;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvLoadImageAnydepth()
    {
        try
        {
            return CV_LOAD_IMAGE_ANYDEPTH;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int getCvLoadImageGrayScale()
    {
        try
        {
            return CV_LOAD_IMAGE_GRAYSCALE;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }
}
