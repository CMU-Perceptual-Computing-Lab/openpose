#include <openpose/producer/videoReader.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    VideoReader::VideoReader(const std::string& videoPath, const std::string& cameraParameterPath,
                             const bool undistortImage, const int numberViews) :
        VideoCaptureReader{videoPath, ProducerType::Video, cameraParameterPath, undistortImage, numberViews},
        mPathName{getFileNameNoExtension(videoPath)}
    {
    }

    VideoReader::~VideoReader()
    {
    }

    std::string VideoReader::getNextFrameName()
    {
        try
        {
            return mPathName + "_" + VideoCaptureReader::getNextFrameName();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    double VideoReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                return VideoCaptureReader::get(capProperty)
                    / positiveIntRound(Producer::get(ProducerProperty::NumberViews));
            else
                return VideoCaptureReader::get(capProperty);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void VideoReader::set(const int capProperty, const double value)
    {
        try
        {
            VideoCaptureReader::set(capProperty, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix VideoReader::getRawFrame()
    {
        try
        {
            return VideoCaptureReader::getRawFrame();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    std::vector<Matrix> VideoReader::getRawFrames()
    {
        try
        {
            const auto numberViews = positiveIntRound(Producer::get(ProducerProperty::NumberViews));
            auto cvMats = VideoCaptureReader::getRawFrames();
            // Split image
            if (cvMats.size() == 1 && numberViews > 1)
            {
                Matrix opMatConcatenated = cvMats.at(0);
                cv::Mat matConcatenated = OP_OP2CVMAT(opMatConcatenated);
                cvMats.clear();
                const auto individualWidth = matConcatenated.cols/numberViews;
                for (auto i = 0 ; i < numberViews ; i++)
                {
                    cv::Mat cvMat(
                        matConcatenated,
                        cv::Rect{
                            (int)(i*individualWidth), 0,
                            (int)individualWidth, (int)matConcatenated.rows });
                    const Matrix opMat = OP_CV2OPMAT(cvMat);
                    cvMats.emplace_back(opMat);
                }
            }
            // Sanity check
            else if (cvMats.size() != 1u && numberViews > 1)
                error("Unexpected error. Notify us (" + std::to_string(numberViews) + " vs. "
                      + std::to_string(numberViews) + ").", __LINE__, __FUNCTION__, __FILE__);
            // Return images
            return cvMats;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
