#include <openpose/utilities/fileSystem.hpp>
#include <openpose/producer/videoReader.hpp>

namespace op
{
    VideoReader::VideoReader(const std::string & videoPath) :
        VideoCaptureReader{videoPath},
        mPathName{getFileNameNoExtension(videoPath)}
    {
    }

    std::string VideoReader::getFrameName()
    {
        try
        {
            return mPathName + "_" + VideoCaptureReader::getFrameName();
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
}
