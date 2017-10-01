#include <thread>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    void reset(unsigned int& numberEmptyFrames, bool& trackingFps)
    {
        try
        {
            // Reset number empty frames
            numberEmptyFrames = 0;
            // Reset keepDesiredFrameRate
            trackingFps = false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Producer::Producer(const ProducerType type) :
        mType{type},
        mProducerFpsMode{ProducerFpsMode::RetrievalFps},
        mNumberEmptyFrames{0},
        mTrackingFps{false}
    {
        mProperties[(unsigned char)ProducerProperty::AutoRepeat] = (double) false;
        mProperties[(unsigned char)ProducerProperty::Flip] = (double) false;
        mProperties[(unsigned char)ProducerProperty::Rotation] = 0.;
    }

    Producer::~Producer(){}

    cv::Mat Producer::getFrame()
    {
        try
        {
            cv::Mat frame;

            if (isOpened())
            {
                // If ProducerFpsMode::OriginalFps, then force producer to keep the frame rate of the frames producer
                // sources (e.g. a video)
                keepDesiredFrameRate();
                // Get frame
                frame = getRawFrame();
                // Flip + rotate frame
                flipAndRotate(frame);
                // Check frame integrity
                checkFrameIntegrity(frame);
                // Check if video capture did finish and close/restart it
                ifEndedResetOrRelease();
            }
            return frame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    void Producer::setProducerFpsMode(const ProducerFpsMode fpsMode)
    {
        try
        {
            check(fpsMode == ProducerFpsMode::RetrievalFps || fpsMode == ProducerFpsMode::OriginalFps,
                  "Unknown ProducerFpsMode.", __LINE__, __FUNCTION__, __FILE__);
            // For webcam, ProducerFpsMode::OriginalFps == ProducerFpsMode::RetrievalFps, since the internal webcam
            // cache will overwrite frames after it gets full
            if (mType == ProducerType::Webcam)
            {
                mProducerFpsMode = {ProducerFpsMode::RetrievalFps};
                if (fpsMode == ProducerFpsMode::OriginalFps)
                    log("The producer fps mode set to `OriginalFps` (flag `process_real_time` on the demo) is not"
                        " necessary, it is already assumed for webcam.",
                        Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            }
            // If no webcam
            else
            {
                check(fpsMode == ProducerFpsMode::RetrievalFps || get(CV_CAP_PROP_FPS) > 0,
                      "Selected to keep the source fps but get(CV_CAP_PROP_FPS) <= 0, i.e. the source did not set"
                      " its fps property.", __LINE__, __FUNCTION__, __FILE__);
                mProducerFpsMode = {fpsMode};
            }
            reset(mNumberEmptyFrames, mTrackingFps);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    double Producer::get(const ProducerProperty property)
    {
        try
        {
            if (property < ProducerProperty::Size)
                return mProperties[(unsigned char)property];
            else
            {
                error("Unkown ProducerProperty.", __LINE__, __FUNCTION__, __FILE__);
                return 0.;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void Producer::set(const ProducerProperty property, double value)
    {
        try
        {
            if (property < ProducerProperty::Size)
            {
                // Individual checks
                if (property == ProducerProperty::AutoRepeat)
                {
                    check(value != 1. || (mType == ProducerType::ImageDirectory || mType == ProducerType::Video),
                          "ProducerProperty::AutoRepeat only implemented for ProducerType::ImageDirectory and"
                          " Video.", __LINE__, __FUNCTION__, __FILE__);
                }
                else if (property == ProducerProperty::Rotation)
                {
                    check(value == 0. || value == 90. || value == 180. || value == 270.,
                          "ProducerProperty::Rotation only implemented for {0, 90, 180, 270} degrees.",
                          __LINE__, __FUNCTION__, __FILE__);
                }

                // Common operation
                mProperties[(unsigned char)property] = value;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Producer::checkFrameIntegrity(cv::Mat& frame)
    {
        try
        {
            // Process wrong frames
            if (frame.empty())
            {
                log("Empty frame detected, frame number " + std::to_string((int)get(CV_CAP_PROP_POS_FRAMES))
                    + " of " + std::to_string((int)get(CV_CAP_PROP_FRAME_COUNT)) + ".",
                    Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                mNumberEmptyFrames++;
            }
            else
            {
                mNumberEmptyFrames = 0;

                if (mType != ProducerType::ImageDirectory
                      && (frame.cols != get(CV_CAP_PROP_FRAME_WIDTH) || frame.rows != get(CV_CAP_PROP_FRAME_HEIGHT)))
                {
                    log("Frame size changed. Returning empty frame.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                    frame = cv::Mat();
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Producer::flipAndRotate(cv::Mat& frame) const
    {
        try
        {
            if (!frame.empty())
            {
                // Rotate it if desired
                const auto rotationAngle = mProperties[(unsigned char)ProducerProperty::Rotation];
                const auto flipFrame = (mProperties[(unsigned char)ProducerProperty::Flip] == 1.);
                if (rotationAngle == 0.)
                {
                    if (flipFrame)
                        cv::flip(frame, frame, 1);
                }
                else if (rotationAngle == 90.)
                {
                    cv::transpose(frame, frame);
                    if (!flipFrame)
                        cv::flip(frame, frame, 0);
                }
                else if (rotationAngle == 180.)
                {
                    if (flipFrame)
                        cv::flip(frame, frame, 0);
                    else
                        cv::flip(frame, frame, -1);
                }
                else if (rotationAngle == 270.)
                {
                    cv::transpose(frame, frame);
                    if (flipFrame)
                        cv::flip(frame, frame, -1);
                    else
                        cv::flip(frame, frame, 1);
                }
                else
                    error("Rotation angle != {0, 90, 180, 270} degrees.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Producer::ifEndedResetOrRelease()
    {
        try
        {
            if (isOpened())
            {
                // OpenCV closing issue: OpenCV goes in the range [1, get(CV_CAP_PROP_FRAME_COUNT) - 1] in some
                // videos (i.e. there is a frame missing), mNumberEmptyFrames allows the program to be properly
                // closed keeping the 0-index frame counting
                if (mNumberEmptyFrames > 2
                    || (mType != ProducerType::IPCamera && mType != ProducerType::Webcam
                        && get(CV_CAP_PROP_POS_FRAMES) >= get(CV_CAP_PROP_FRAME_COUNT)))
                {
                    // Repeat video
                    if (mProperties[(unsigned char)ProducerProperty::AutoRepeat])
                        set(CV_CAP_PROP_POS_FRAMES, 0);

                    // Warning + release mVideoCapture
                    else
                        release();
                    reset(mNumberEmptyFrames, mTrackingFps);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Producer::keepDesiredFrameRate()
    {
        try
        {
            if (isOpened())
            {
                // Be sure fps is not slower than desired
                if (mProducerFpsMode == ProducerFpsMode::OriginalFps)
                {
                    if (mTrackingFps)
                    {
                        mNumberFramesTrackingFps++;
                        // Current #frames
                        const auto currentFrames = get(CV_CAP_PROP_POS_FRAMES) - mFirstFrameTrackingFps;
                        // Expected #frames
                        const auto nsPerFrame = 1e9/get(CV_CAP_PROP_FPS);
                        const auto timeNs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now()-mClockTrackingFps
                        ).count();
                        const auto expectedFrames = timeNs / nsPerFrame;

                        const auto difference = expectedFrames - currentFrames;
                        const auto numberSetPositionThreshold = 3u;
                        // Speed up frame extraction
                        if (difference > 1)
                        {
                            if (difference > 15)
                            {
                                set(CV_CAP_PROP_POS_FRAMES, std::floor(expectedFrames) + mFirstFrameTrackingFps);
                                mNumberSetPositionTrackingFps = fastMin(mNumberSetPositionTrackingFps+1,
                                                                        numberSetPositionThreshold);
                            }
                            else
                            {
                                cv::Mat frame;
                                for (auto i = 0 ; i < std::floor(difference) ; i++)
                                    frame = getRawFrame();
                            }
                        }
                        // Low down frame extraction - sleep thread unless it is too slow in most frames (using
                        // set(frames, X) sets to frame X+delta, due to codecs issues)
                        else if (difference < -0.45 && mNumberSetPositionTrackingFps < numberSetPositionThreshold)
                        {
                            const auto sleepMs = intRound( (-difference*nsPerFrame*1e-6)*0.99 );
                            std::this_thread::sleep_for(std::chrono::milliseconds{sleepMs});
                        }
                    }
                    else
                    {
                        mTrackingFps = true;
                        mFirstFrameTrackingFps = 0;
                        mNumberFramesTrackingFps = 0;
                        mNumberSetPositionTrackingFps = 0;
                        mClockTrackingFps = std::chrono::high_resolution_clock::now();
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
