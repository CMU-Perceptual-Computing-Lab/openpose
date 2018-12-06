#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/producer/videoReader.hpp>

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
                return VideoCaptureReader::get(capProperty) / intRound(Producer::get(ProducerProperty::NumberViews));
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

    cv::Mat VideoReader::getRawFrame()
    {
        try
        {
            return VideoCaptureReader::getRawFrame();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    std::vector<cv::Mat> VideoReader::getRawFrames()
    {
        try
        {
            const auto numberViews = intRound(Producer::get(ProducerProperty::NumberViews));
            auto cvMats = VideoCaptureReader::getRawFrames();
            // Split image
            if (cvMats.size() == 1 && numberViews > 1)
            {
                cv::Mat cvMatConcatenated = cvMats.at(0);
                cvMats.clear();
                const auto individualWidth = cvMatConcatenated.cols/numberViews;
                for (auto i = 0 ; i < numberViews ; i++)
                    cvMats.emplace_back(
                        cv::Mat(cvMatConcatenated,
                                cv::Rect{(int)(i*individualWidth), 0,
                                         (int)individualWidth,
                                         (int)cvMatConcatenated.rows}));
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
