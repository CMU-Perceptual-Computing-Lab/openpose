#include <openpose/producer/webcamReader.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>
#include <openpose_private/utilities/openCvPrivate.hpp>

namespace op
{
    WebcamReader::WebcamReader(const int webcamIndex, const Point<int>& webcamResolution,
                               const bool throwExceptionIfNoOpened, const std::string& cameraParameterPath,
                               const bool undistortImage) :
        VideoCaptureReader{webcamIndex, throwExceptionIfNoOpened, cameraParameterPath, undistortImage, 1},
        mIndex{webcamIndex},
        mWebcamStarted{VideoCaptureReader::isOpened()},
        mFrameNameCounter{-1},
        mThreadOpened{std::atomic<bool>{false}},
        mDisconnectedCounter{0},
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
                            + std::to_string(positiveIntRound(get(CV_CAP_PROP_FRAME_WIDTH))) + "x"
                            + std::to_string(positiveIntRound(get(CV_CAP_PROP_FRAME_HEIGHT))) };
                        opLog(logMessage, Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                    }
                }
                // Set resolution
                mResolution = Point<int>{
                    positiveIntRound(get(CV_CAP_PROP_FRAME_WIDTH)),
                    positiveIntRound(get(CV_CAP_PROP_FRAME_HEIGHT))};
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
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
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
            return (VideoCaptureReader::isOpened() || (mDisconnectedCounter > 0 && mWebcamStarted));
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
            VideoCaptureReader::set(capProperty, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix WebcamReader::getRawFrame()
    {
        try
        {
            mFrameNameCounter++; // Simple counter: 0,1,2,3,...

            // Retrieve frame from buffer
            Matrix opMat;
            auto cvMatRetrieved = false;
            while (!cvMatRetrieved)
            {
                // Retrieve frame
                std::unique_lock<std::mutex> lock{mBufferMutex};
                if (!mBuffer.empty())
                {
                    std::swap(opMat, mBuffer);
                    cvMatRetrieved = true;
                }
                // No frames available -> sleep & wait
                else
                {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds{5});
                }
            }
            return opMat;

            // Naive implementation - No flashing buffers
            // return VideoCaptureReader::getRawFrame();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    std::vector<Matrix> WebcamReader::getRawFrames()
    {
        try
        {
            return std::vector<Matrix>{getRawFrame()};
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
                auto opMat = VideoCaptureReader::getRawFrame();
                // Detect whether camera is connected
                // Equivalent code:
                // const auto newNorm = (
                //     opMat.empty() ? mLastNorm : cv::norm(opMat.row(opMat.rows() / 2)));
                double newNorm;
                if (opMat.empty())
                    newNorm = mLastNorm;
                else
                {
                    cv::Mat rowMat;
                    OP_CONST_MAT_RETURN_FUNCTION(rowMat, opMat, row(opMat.rows() / 2));
                    newNorm = cv::norm(rowMat);
                }
                if (mLastNorm == newNorm)
                {
                    mDisconnectedCounter++;
                    if (mDisconnectedCounter > 1 && opMat.empty())
                        opLog("Camera frame empty (it has occurred for the last " + std::to_string(mDisconnectedCounter)
                            + " consecutive frames).", Priority::Max);
                }
                else
                {
                    mLastNorm = newNorm;
                    mDisconnectedCounter = 0;
                }
                // If camera disconnected: black image
                if (!cameraConnected)
                {
                    cv::Mat cvMat(mResolution.y, mResolution.x, CV_8UC3, cv::Scalar{0,0,0});
                    putTextOnCvMat(cvMat, "Camera disconnected, reconnecting...", {cvMat.cols/16, cvMat.rows/2},
                                   cv::Scalar{255, 255, 255}, false, positiveIntRound(2.3*cvMat.cols));
                    // Anti flip + anti rotate frame (so it is balanced with the final flip + rotate)
                    auto rotationAngle = -Producer::get(ProducerProperty::Rotation);
                    // Not using 0 or 180 might provoke a row/col dimension swap, thus an OP error
                    if (int(std::round(rotationAngle)) % 180 != 0.)
                        rotationAngle = 0;
                    const auto flipFrame = ((unsigned char)Producer::get(ProducerProperty::Flip) == 1.);
                    opMat = OP_CV2OPMAT(cvMat);
                    rotateAndFlipFrame(opMat, rotationAngle, flipFrame);
                }
                // Move to buffer
                if (!opMat.empty())
                {
                    const std::lock_guard<std::mutex> lock{mBufferMutex};
                    std::swap(mBuffer, opMat);
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
            opLog("Webcam was unplugged, trying to reconnect it.", Priority::Max);
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
                    && (mResolution.x != positiveIntRound(get(CV_CAP_PROP_FRAME_WIDTH))
                        || mResolution.y != positiveIntRound(get(CV_CAP_PROP_FRAME_HEIGHT))));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
}
