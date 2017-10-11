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

    std::string IpCameraReader::getFrameName()
    {
        try
        {
            return VideoCaptureReader::getFrameName();
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
}
