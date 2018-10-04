#include <openpose/producer/ipCameraReader.hpp>

namespace op
{
    // Public IP cameras for testing (add ?x.mjpeg):
    // http://iris.not.iac.es/axis-cgi/mjpg/video.cgi?resolution=320x240?x.mjpeg
    // http://www.webcamxp.com/publicipcams.aspx

    IpCameraReader::IpCameraReader(const std::string & cameraPath) :
        VideoCaptureReader{cameraPath, ProducerType::IPCamera},
        mPathName{cameraPath}
    {
    }

    IpCameraReader::~IpCameraReader()
    {
    }

    std::vector<cv::Mat> IpCameraReader::getCameraMatrices()
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

    std::vector<cv::Mat> IpCameraReader::getCameraExtrinsics()
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

    std::vector<cv::Mat> IpCameraReader::getCameraIntrinsics()
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

    std::string IpCameraReader::getNextFrameName()
    {
        try
        {
            return VideoCaptureReader::getNextFrameName();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    cv::Mat IpCameraReader::getRawFrame()
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

    std::vector<cv::Mat> IpCameraReader::getRawFrames()
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
