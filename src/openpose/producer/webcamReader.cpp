#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/producer/webcamReader.hpp>

namespace op
{
    WebcamReader::WebcamReader(const int webcamIndex, const Point<int>& webcamResolution, const double fps) :
        VideoCaptureReader{webcamIndex},
        mFps{fps},
        mFrameNameCounter{-1}
    {
        try
        {
            if (webcamResolution != Point<int>{})
            {
                set(CV_CAP_PROP_FRAME_WIDTH, webcamResolution.x);
                set(CV_CAP_PROP_FRAME_HEIGHT, webcamResolution.y);
                if ((int)get(CV_CAP_PROP_FRAME_WIDTH) != webcamResolution.x || (int)get(CV_CAP_PROP_FRAME_HEIGHT) != webcamResolution.y)
                {
                    const std::string logMessage{ "Desired webcam resolution " + std::to_string(webcamResolution.x) + "x" + std::to_string(webcamResolution.y)
                                                + " could not being set. Final resolution: " + std::to_string(intRound(get(CV_CAP_PROP_FRAME_WIDTH))) + "x"
                                                + std::to_string(intRound(get(CV_CAP_PROP_FRAME_HEIGHT))) };
                    log(logMessage, Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                }

                // Start buffering thread
                if (isOpened())
                    mThread = std::thread{&WebcamReader::bufferingThread, this};
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    WebcamReader::~WebcamReader()
    {
        try
        {
            // Close and join thread
            mCloseThread = true;
            mThread.join();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::string WebcamReader::getFrameName()
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

    double WebcamReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_POS_FRAMES)
                return (double)mFrameNameCounter;
            else if (capProperty == CV_CAP_PROP_FPS)
                return mFps;
            else
                return VideoCaptureReader::get(capProperty);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void WebcamReader::set(const int capProperty, const double value)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FPS)
                mFps = value;
            else
                VideoCaptureReader::set(capProperty, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    cv::Mat WebcamReader::getRawFrame()
    {
        try
        {
            mFrameNameCounter++; // Simple counter: 0,1,2,3,...

            // Retrieve frame from buffer
            cv::Mat cvMat;
            auto cvMatRetrieved = false;
            while (!cvMatRetrieved)
            {
                // Retrieve frame
                std::unique_lock<std::mutex> lock{mBufferMutex};
                if (!mBuffer.empty())
                {
                    std::swap(cvMat, mBuffer);
                    cvMatRetrieved = true;
                }
                // No frames available -> sleep & wait
                else
                {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds{5});
                }
            }
            return cvMat;

            // Naive implementation - No flashing buffers
            // return VideoCaptureReader::getRawFrame();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat{};
        }
    }

    void WebcamReader::bufferingThread()
    {
        mCloseThread = false;
        while (!mCloseThread)
        {
            // Get frame
            auto cvMat = VideoCaptureReader::getRawFrame();
            // Move to buffer
            if (!cvMat.empty())
            {
                const std::lock_guard<std::mutex> lock{mBufferMutex};
                std::swap(mBuffer, cvMat);
            }
        }
    }
}
