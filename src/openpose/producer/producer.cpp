#include <openpose/producer/producer.hpp>
#include <openpose/producer/headers.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

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

    Producer::Producer(const ProducerType type, const std::string& cameraParameterPath, const bool undistortImage,
                       const int numberViews) :
        mType{type},
        mProducerFpsMode{ProducerFpsMode::RetrievalFps},
        mNumberEmptyFrames{0},
        mTrackingFps{false}
    {
        try
        {
            // Basic properties
            mProperties[(unsigned int)ProducerProperty::AutoRepeat] = (double) false;
            mProperties[(unsigned int)ProducerProperty::Flip] = (double) false;
            mProperties[(unsigned int)ProducerProperty::Rotation] = 0.;
            mProperties[(unsigned int)ProducerProperty::NumberViews] = numberViews;
            auto& mNumberViews = mProperties[(unsigned int)ProducerProperty::NumberViews];
            // Camera (distortion, intrinsic, and extrinsic) parameters
            if (mType != ProducerType::FlirCamera)
            {
                // Undistort image?
                mCameraParameterReader.setUndistortImage(undistortImage);
                // If no stereo --> Set to 1
                if (mNumberViews <= 0)
                    mNumberViews = 1;
                // Get camera parameters
                if (mNumberViews > 1 || undistortImage)
                {
                    const auto extension = getFileExtension(cameraParameterPath);
                    // Get camera parameters
                    if (extension == "xml" || extension == "XML")
                        mCameraParameterReader.readParameters(
                            getFileParentFolderPath(cameraParameterPath), getFileNameNoExtension(cameraParameterPath));
                    else // if (mNumberViews > 1)
                    {
                        const auto cameraParameterPathCleaned = formatAsDirectory(cameraParameterPath);
                        // Read camera parameters from SN
                        auto serialNumbers = getFilesOnDirectory(cameraParameterPathCleaned, ".xml");
                        // Get serial numbers
                        for (auto& serialNumber : serialNumbers)
                            serialNumber = getFileNameNoExtension(serialNumber);
                        // Get camera parameters
                        mCameraParameterReader.readParameters(cameraParameterPathCleaned, serialNumbers);
                    }
                    // Sanity check
                    if ((int)mCameraParameterReader.getNumberCameras() != mNumberViews)
                        error("Found a different number of camera parameter files than the number indicated by"
                              " `--3d_views` ("
                              + std::to_string(mCameraParameterReader.getNumberCameras()) + " vs. "
                              + std::to_string(mNumberViews) + "). Make sure they are the same number of files and/or"
                              + " set `--frame_undistort` to false.",
                              __LINE__, __FUNCTION__, __FILE__);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Producer::~Producer(){}

    Matrix Producer::getFrame()
    {
        try
        {
            // Return first element from getFrames (if any)
            const auto frames = getFrames();
            return (frames.empty() ? Matrix() : frames[0]);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    std::vector<Matrix> Producer::getFrames()
    {
        try
        {
            std::vector<Matrix> frames;

            if (isOpened())
            {
                // If ProducerFpsMode::OriginalFps, then force producer to keep the frame rate of the frames producer
                // sources (e.g., a video)
                keepDesiredFrameRate();
                // Get frame
                frames = getRawFrames();
                // Undistort frames
                // TODO: Multi-thread if > 1 frame
                for (auto i = 0u ; i < frames.size() ; i++)
                    if (!frames[i].empty() && mCameraParameterReader.getUndistortImage())
                        mCameraParameterReader.undistort(frames[i], i);
                // Post-process frames
                for (auto& frame : frames)
                {
                    // Flip + rotate frame
                    const auto rotationAngle = mProperties[(unsigned char)ProducerProperty::Rotation];
                    const auto flipFrame = (mProperties[(unsigned char)ProducerProperty::Flip] == 1.);
                    rotateAndFlipFrame(frame, rotationAngle, flipFrame);
                    // Check frame integrity
                    checkFrameIntegrity(frame);
                    // If any frame invalid --> exit
                    if (frame.empty())
                    {
                        frames.clear();
                        break;
                    }
                }
                // Check if video capture did finish and close/restart it
                ifEndedResetOrRelease();
            }
            return frames;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> Producer::getCameraMatrices()
    {
        try
        {
            return mCameraParameterReader.getCameraMatrices();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> Producer::getCameraExtrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraExtrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> Producer::getCameraIntrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraIntrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void Producer::setProducerFpsMode(const ProducerFpsMode fpsMode)
    {
        try
        {
            checkBool(
                fpsMode == ProducerFpsMode::RetrievalFps || fpsMode == ProducerFpsMode::OriginalFps,
                "Unknown ProducerFpsMode.", __LINE__, __FUNCTION__, __FILE__);
            // For webcam, ProducerFpsMode::OriginalFps == ProducerFpsMode::RetrievalFps, since the internal webcam
            // cache will overwrite frames after it gets full
            if (mType == ProducerType::Webcam)
            {
                mProducerFpsMode = {ProducerFpsMode::RetrievalFps};
                if (fpsMode == ProducerFpsMode::OriginalFps)
                    opLog("The producer fps mode set to `OriginalFps` (flag `process_real_time` on the demo) is not"
                        " necessary, it is already assumed for webcam.",
                        Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            }
            // If no webcam
            else
            {
                checkBool(
                    fpsMode == ProducerFpsMode::RetrievalFps || get(CV_CAP_PROP_FPS) > 0,
                    "Selected to keep the source fps but get(CV_CAP_PROP_FPS) <= 0, i.e., the source did not set"
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
                error("Unknown ProducerProperty.", __LINE__, __FUNCTION__, __FILE__);
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
                    checkBool(
                        value != 1. || (mType == ProducerType::ImageDirectory || mType == ProducerType::Video),
                        "ProducerProperty::AutoRepeat only implemented for ProducerType::ImageDirectory and"
                        " Video.", __LINE__, __FUNCTION__, __FILE__);
                }
                else if (property == ProducerProperty::Rotation)
                {
                    checkBool(
                        value == 0. || value == 90. || value == 180. || value == 270.,
                        "ProducerProperty::Rotation only implemented for {0, 90, 180, 270} degrees.",
                        __LINE__, __FUNCTION__, __FILE__);
                }
                else if (property == ProducerProperty::FrameStep)
                {
                    // Sanity check
                    if (value < 1)
                    {
                        const auto message = "The frame step must be greater than 0 (`--frame_step`). Use 1 by default.";
                        error(message, __LINE__, __FUNCTION__, __FILE__);
                    }
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

    void Producer::checkFrameIntegrity(Matrix& frame)
    {
        try
        {
            // Process wrong frames
            if (frame.empty())
            {
                opLog("Empty frame detected, frame number " + std::to_string((int)get(CV_CAP_PROP_POS_FRAMES))
                    + " of " + std::to_string((int)get(CV_CAP_PROP_FRAME_COUNT)) + ".",
                    Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                mNumberEmptyFrames++;
            }
            else
            {
                mNumberEmptyFrames = 0;

                if (mType != ProducerType::ImageDirectory
                      && ((frame.cols() != get(CV_CAP_PROP_FRAME_WIDTH) && get(CV_CAP_PROP_FRAME_WIDTH) > 0)
                          || (frame.rows() != get(CV_CAP_PROP_FRAME_HEIGHT) && get(CV_CAP_PROP_FRAME_HEIGHT) > 0)))
                {
                    opLog("Frame size changed. Returning empty frame.\nExpected vs. received sizes: "
                        + std::to_string(positiveIntRound(get(CV_CAP_PROP_FRAME_WIDTH)))
                        + "x" + std::to_string(positiveIntRound(get(CV_CAP_PROP_FRAME_HEIGHT)))
                        + " vs. " + std::to_string(frame.cols()) + "x" + std::to_string(frame.rows()),
                        Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                    frame = Matrix();
                }
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
                // videos (i.e., there is a frame missing), mNumberEmptyFrames allows the program to be properly
                // closed keeping the 0-index frame counting
                if (mNumberEmptyFrames > 2
                    || (mType != ProducerType::FlirCamera && mType != ProducerType::IPCamera
                        && mType != ProducerType::Webcam
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
                                std::vector<Matrix> frames;
                                for (auto i = 0 ; i < std::floor(difference) ; i++)
                                    frames = getRawFrames();
                            }
                        }
                        // Low down frame extraction - sleep thread unless it is too slow in most frames (using
                        // set(frames, X) sets to frame X+delta, due to codecs issues)
                        else if (difference < -0.45 && mNumberSetPositionTrackingFps < numberSetPositionThreshold)
                        {
                            const auto sleepMs = positiveIntRound( (-difference*nsPerFrame*1e-6)*0.99 );
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

    std::shared_ptr<Producer> createProducer(
        const ProducerType producerType, const std::string& producerString, const Point<int>& cameraResolution,
        const std::string& cameraParameterPath, const bool undistortImage, const int numberViews)
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Directory of images
            if (producerType == ProducerType::ImageDirectory)
                return std::make_shared<ImageDirectoryReader>(
                    producerString, cameraParameterPath, undistortImage, numberViews);
            // Video
            else if (producerType == ProducerType::Video)
                return std::make_shared<VideoReader>(
                    producerString, cameraParameterPath, undistortImage, numberViews);
            // IP camera
            else if (producerType == ProducerType::IPCamera)
                return std::make_shared<IpCameraReader>(producerString, cameraParameterPath, undistortImage);
            // Flir camera
            else if (producerType == ProducerType::FlirCamera)
                return std::make_shared<FlirReader>(
                    cameraParameterPath, cameraResolution, undistortImage, std::stoi(producerString));
            // Webcam
            else if (producerType == ProducerType::Webcam)
            {
                const auto webcamIndex = std::stoi(producerString);
                auto cameraResolutionFinal = cameraResolution;
                if (cameraResolutionFinal.x < 0 || cameraResolutionFinal.y < 0)
                    cameraResolutionFinal = Point<int>{1280,720};
                if (webcamIndex >= 0)
                {
                    const auto throwExceptionIfNoOpened = true;
                    return std::make_shared<WebcamReader>(
                        webcamIndex, cameraResolutionFinal, throwExceptionIfNoOpened, cameraParameterPath,
                        undistortImage);
                }
                else
                {
                    const auto throwExceptionIfNoOpened = false;
                    std::shared_ptr<WebcamReader> webcamReader;
                    for (auto index = 0 ; index < 10 ; index++)
                    {
                        webcamReader = std::make_shared<WebcamReader>(
                            index, cameraResolutionFinal, throwExceptionIfNoOpened, cameraParameterPath,
                            undistortImage);
                        if (webcamReader->isOpened())
                        {
                            opLog("Auto-detecting camera index... Detected and opened camera " + std::to_string(index)
                                + ".", Priority::High);
                            return webcamReader;
                        }
                    }
                    error("No camera found.", __LINE__, __FUNCTION__, __FILE__);
                }
            }
            // Unknown
            else if (producerType != ProducerType::None)
                error("Undefined Producer selected.", __LINE__, __FUNCTION__, __FILE__);
            // None
            return nullptr;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
