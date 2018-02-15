#include <openpose/utilities/fileSystem.hpp>
#include <openpose/producer/videoReader.hpp>

namespace op
{
    VideoReader::VideoReader(const std::string & videoPath) :
        VideoCaptureReader{videoPath, ProducerType::Video},
        mPathName{getFileNameNoExtension(videoPath)}
    {
    }

    std::vector<cv::Mat> VideoReader::getCameraMatrices()
    {
        try
        {
            return {};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
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
            return VideoCaptureReader::getRawFrames();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
