#include <opencv2/highgui/highgui.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/producer/webcamReader.hpp>

namespace op
{
    WebcamReader::WebcamReader(const int webcamIndex, const Point<int>& webcamResolution, const double fps,
                               const bool throwExceptionIfNoOpened) :
        VideoCaptureReader{webcamIndex, throwExceptionIfNoOpened},
        mIndex{webcamIndex},
        mFps{fps},
        mFrameNameCounter{-1},
        mThreadOpened{std::atomic<bool>{false}},
        mResolution{webcamResolution}
    {
        try
        {
            if (isOpened())
            {
                mFrameNameCounter = 0;
                if (mResolution != Point<int>{})
                {
                    set(CV_CAP_PROP_FRAME_WIDTH, mResolution.x);
                    set(CV_CAP_PROP_FRAME_HEIGHT, mResolution.y);
                    if ((int)get(CV_CAP_PROP_FRAME_WIDTH) != mResolution.x
                        || (int)get(CV_CAP_PROP_FRAME_HEIGHT) != mResolution.y)
                    {
                        const std::string logMessage{
                            "Desired webcam resolution " + std::to_string(mResolution.x) + "x"
                            + std::to_string(mResolution.y) + " could not being set. Final resolution: "
                            + std::to_string(intRound(get(CV_CAP_PROP_FRAME_WIDTH))) + "x"
                            + std::to_string(intRound(get(CV_CAP_PROP_FRAME_HEIGHT))) };
                        log(logMessage, Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                    }
                }
                // Set resolution
                mResolution = Point<int>{
                    intRound(get(CV_CAP_PROP_FRAME_WIDTH)),
                    intRound(get(CV_CAP_PROP_FRAME_HEIGHT))};
                // Start buffering thread
                mThreadOpened = true;
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
            if (mThreadOpened)
            {
                mCloseThread = true;
                mThread.join();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<cv::Mat> WebcamReader::getCameraMatrices()
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

    std::vector<cv::Mat> WebcamReader::getCameraExtrinsics()
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

    std::vector<cv::Mat> WebcamReader::getCameraIntrinsics()
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

    std::string WebcamReader::getNextFrameName()
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

    bool WebcamReader::isOpened() const
    {
        try
        {
            return (VideoCaptureReader::isOpened() || mDisconnectedCounter > 0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
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
            return cv::Mat();
        }
    }

    std::vector<cv::Mat> WebcamReader::getRawFrames()
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

    const auto DISCONNETED_THRESHOLD = 100;
    void WebcamReader::bufferingThread()
    {
        try
        {
            mCloseThread = false;
            while (!mCloseThread)
            {
                // Reset camera if disconnected
                bool cameraConnected = true;
                if (mDisconnectedCounter > DISCONNETED_THRESHOLD)
                    cameraConnected = reset();
                // Get frame
                auto cvMat = VideoCaptureReader::getRawFrame();
                // Detect whether camera is connected
                const auto newNorm = (
                    cvMat.empty() ? mLastNorm : cv::norm(cvMat.row(cvMat.rows/2)));
                if (mLastNorm == newNorm)
                    mDisconnectedCounter++;
                else
                {
                    mLastNorm = newNorm;
                    mDisconnectedCounter = 0;
                }
                // Camera disconnected: black image
                if (!cameraConnected || cvMat.empty())
                {
                    cvMat = cv::Mat(mResolution.y, mResolution.x, CV_8UC3, cv::Scalar{0,0,0});
                    putTextOnCvMat(cvMat, "Camera disconnected, reconnecting...", {cvMat.cols/16, cvMat.rows/2},
                                   cv::Scalar{255, 255, 255}, false, intRound(2.3*cvMat.cols));
                }
                // Move to buffer
                if (!cvMat.empty())
                {
                    const std::lock_guard<std::mutex> lock{mBufferMutex};
                    std::swap(mBuffer, cvMat);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool WebcamReader::reset()
    {
        try
        {
            // If unplugged
            log("Webcam was unplugged, trying to reconnect it.", Priority::Max,
                __LINE__, __FUNCTION__, __FUNCTION__);
            // Sleep
            std::this_thread::sleep_for(std::chrono::milliseconds{1000});
            // Reset camera
            VideoCaptureReader::resetWebcam(mIndex, false);
            // Re-set resolution
            if (isOpened())
            {
                set(CV_CAP_PROP_FRAME_WIDTH, mResolution.x);
                set(CV_CAP_PROP_FRAME_HEIGHT, mResolution.y);
            }
            // Camera replugged?
            return (!isOpened()
                    && (mResolution.x != intRound(get(CV_CAP_PROP_FRAME_WIDTH))
                        || mResolution.y != intRound(get(CV_CAP_PROP_FRAME_HEIGHT))));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
}
