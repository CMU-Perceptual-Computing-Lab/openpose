#include <iostream>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose/producer/videoCaptureReader.hpp>

namespace op
{
    VideoCaptureReader::VideoCaptureReader(const int index, const bool throwExceptionIfNoOpened,
                                           const std::string& cameraParameterPath, const bool undistortImage,
                                           const int numberViews) :
        Producer{ProducerType::Webcam, cameraParameterPath, undistortImage, numberViews}
    {
        try
        {
            resetWebcam(index, throwExceptionIfNoOpened);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    VideoCaptureReader::VideoCaptureReader(const std::string& path, const ProducerType producerType,
                                           const std::string& cameraParameterPath, const bool undistortImage,
                                           const int numberViews) :
        Producer{producerType, cameraParameterPath, undistortImage, numberViews},
        mVideoCapture{path}
    {
        try
        {
            // Make sure only video or IP camera
            if (producerType != ProducerType::IPCamera && producerType != ProducerType::Video)
                error("VideoCapture with an input path must be IP camera or video.",
                      __LINE__, __FUNCTION__, __FILE__);
            // Make sure video capture was opened
            if (!isOpened())
                error("VideoCapture (IP camera/video) could not be opened for path: '" + path + "'. If"
                      " it is a video path, is the path correct?", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    VideoCaptureReader::~VideoCaptureReader()
    {
        try
        {
            release();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::string VideoCaptureReader::getNextFrameName()
    {
        try
        {
            const auto stringLength = 12u;
            return toFixedLengthString(   fastMax(0ull, uLongLongRound(get(CV_CAP_PROP_POS_FRAMES))),   stringLength);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    bool VideoCaptureReader::isOpened() const
    {
        try
        {
            return mVideoCapture.isOpened();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    cv::Mat VideoCaptureReader::getRawFrame()
    {
        try
        {
            // Get frame
            cv::Mat frame;
            mVideoCapture >> frame;
            // Skip frames if frame step > 1
            const auto frameStep = Producer::get(ProducerProperty::FrameStep);
            if (frameStep > 1 && !frame.empty() && get(CV_CAP_PROP_POS_FRAMES) < get(CV_CAP_PROP_FRAME_COUNT)-1)
            {
                // Close if end of video
                if (get(CV_CAP_PROP_POS_FRAMES) + frameStep-1 >= get(CV_CAP_PROP_FRAME_COUNT))
                    mVideoCapture.release();
                // Frame step usually more efficient if just reading sequentially
                else if (frameStep < 51)
                    for (auto i = 1 ; i < frameStep ; i++)
                        mVideoCapture >> frame;
                // Using set(CV_CAP_PROP_POS_FRAMES, value) is efficient only if step is big
                else
                    set(CV_CAP_PROP_POS_FRAMES, get(CV_CAP_PROP_POS_FRAMES) + frameStep-1);
            }
            // Return frame
            return frame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    std::vector<cv::Mat> VideoCaptureReader::getRawFrames()
    {
        try
        {
            return std::vector<cv::Mat>{getRawFrame()};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void VideoCaptureReader::resetWebcam(const int index, const bool throwExceptionIfNoOpened)
    {
        try
        {
            // Open webcam
            mVideoCapture = cv::VideoCapture{index};
            // Make sure video capture was opened
            if (throwExceptionIfNoOpened && !isOpened())
                error("VideoCapture (webcam) could not be opened.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void VideoCaptureReader::release()
    {
        try
        {
            if (mVideoCapture.isOpened())
            {
                mVideoCapture.release();
                log("cv::VideoCapture released.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    double VideoCaptureReader::get(const int capProperty)
    {
        try
        {
            // Specific cases
            // If rotated 90 or 270 degrees, then width and height is exchanged
            if ((capProperty == CV_CAP_PROP_FRAME_WIDTH || capProperty == CV_CAP_PROP_FRAME_HEIGHT)
                && (Producer::get(ProducerProperty::Rotation) != 0.
                    && Producer::get(ProducerProperty::Rotation) != 180.))
            {
                if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                    return mVideoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
                else
                    return mVideoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
            }

            // Generic cases
            return mVideoCapture.get(capProperty);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void VideoCaptureReader::set(const int capProperty, const double value)
    {
        try
        {
            mVideoCapture.set(capProperty, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
