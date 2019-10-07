#include <openpose/gui/gui.hpp>
#include <opencv2/highgui/highgui.hpp> // cv::waitKey
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/check.hpp>

namespace op
{
    inline void showGuiHelp()
    {
        try
        {
            const auto helpCvMat = loadImage("./doc/GUI_help/GUI_help.png");

            if (!helpCvMat.empty())
            {
                const auto fullScreen = false;
                FrameDisplayer frameDisplayer{OPEN_POSE_NAME_AND_VERSION + " - GUI Help",
                                              Point<int>{helpCvMat.cols(), helpCvMat.rows()}, fullScreen};
                frameDisplayer.displayFrame(helpCvMat, 33);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void handleWaitKey(bool& guiPaused, FrameDisplayer& frameDisplayer,
                       std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets,
                       std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets,
                       std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets,
                       std::vector<std::shared_ptr<Renderer>>& renderers,
                       std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                       std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr,
                       DisplayMode& displayMode, DisplayMode& displayModeOriginal)
    {
        try
        {
            const int key = cv::waitKey(1);
            if (key != -1)
            {
                // Some OpenCV versions has a problem and key must be casted to char
                const auto castedKey = (char)std::tolower((char)key);
                // ------------------------- General Commands ------------------------- //
                // Exit program
                if (castedKey==27)
                {
                    if (isRunningSharedPtr != nullptr)
                    {
                        *isRunningSharedPtr = false;
                        guiPaused = false;
                    }
                }
                // Help
                else if (castedKey=='h')
                    showGuiHelp();
                // Switch full screen - normal screen
                else if (castedKey=='f')
                    frameDisplayer.switchFullScreenMode();
                // ------------------------- Producer-Related ------------------------- //
                // Pause
                else if (castedKey==' ')
                    guiPaused = !guiPaused;
                // Fake pause
                else if (castedKey=='m')
                {
                    if (videoSeekSharedPtr != nullptr)
                        videoSeekSharedPtr->first = !videoSeekSharedPtr->first;
                }
                // Seeking in video
                else if (castedKey=='l' || castedKey=='k')
                {
                    if (videoSeekSharedPtr != nullptr)
                    {
                        // Normal case, +-30 frames
                        if (!videoSeekSharedPtr->first)
                            videoSeekSharedPtr->second += 30 * (castedKey=='k' ? -2 : 1);
                        // Frame by frame (if forced paused)
                        else
                            videoSeekSharedPtr->second += (castedKey=='k' ? -1 : 1);
                    }
                }
                // Enable/disable blending
                else if (castedKey=='b')
                {
                    for (auto& renderer : renderers)
                        renderer->setBlendOriginalFrame(!renderer->getBlendOriginalFrame());
                }
                // ------------------------- OpenPose-Related ------------------------- //
                // Modifying thresholds
                else if (castedKey=='-' || castedKey=='=')
                    for (auto& poseExtractorNet : poseExtractorNets)
                        poseExtractorNet->increase(PoseProperty::NMSThreshold, 0.005f * (castedKey=='-' ? -1 : 1));
                else if (castedKey=='_' || castedKey=='+')
                    for (auto& poseExtractorNet : poseExtractorNets)
                        poseExtractorNet->increase(PoseProperty::ConnectMinSubsetScore,
                                                0.005f * (castedKey=='_' ? -1 : 1));
                else if (castedKey=='[' || castedKey==']')
                    for (auto& poseExtractorNet : poseExtractorNets)
                        poseExtractorNet->increase(PoseProperty::ConnectInterThreshold,
                                                0.005f * (castedKey=='[' ? -1 : 1));
                else if (castedKey=='{' || castedKey=='}')
                    for (auto& poseExtractorNet : poseExtractorNets)
                        poseExtractorNet->increase(PoseProperty::ConnectInterMinAboveThreshold,
                                                (castedKey=='{' ? -0.1f : 0.1f));
                else if (castedKey==';' || castedKey=='\'')
                    for (auto& poseExtractorNet : poseExtractorNets)
                        poseExtractorNet->increase(PoseProperty::ConnectMinSubsetCnt, (castedKey==';' ? -1 : 1));
                // ------------------------- Face/hands-Related ------------------------- //
                // Enable/disable face
                else if (castedKey=='z')
                {
                    for (auto& faceExtractorNet : faceExtractorNets)
                        faceExtractorNet->setEnabled(!faceExtractorNet->getEnabled());
                    // Warning if not enabled
                    if (faceExtractorNets.empty())
                        opLog("OpenPose must be run with face keypoint estimation enabled (`--face` flag).",
                            Priority::High);
                }
                // Enable/disable hands
                else if (castedKey=='x')
                {
                    for (auto& handExtractorNet : handExtractorNets)
                        handExtractorNet->setEnabled(!handExtractorNet->getEnabled());
                    // Warning if not enabled
                    if (handExtractorNets.empty())
                        opLog("OpenPose must be run with face keypoint estimation enabled (`--hand` flag).",
                            Priority::High);
                }
                // Enable/disable extra rendering (3D/Adam), while keeping 2D rendering
                else if (castedKey=='c')
                {
                    displayMode = (displayMode == displayModeOriginal
                                   ? DisplayMode::Display2D : displayModeOriginal);
                }
                // ------------------------- Miscellaneous ------------------------- //
                // Show googly eyes
                else if (castedKey=='g')
                    for (auto& renderer : renderers)
                        renderer->setShowGooglyEyes(!renderer->getShowGooglyEyes());
                // ------------------------- OpenPose-Related ------------------------- //
                else if (castedKey==',' || castedKey=='.')
                {
                    const auto increment = (castedKey=='.' ? 1 : -1);
                    for (auto& renderer : renderers)
                        renderer->increaseElementToRender(increment);
                }
                else
                {
                    // Skeleton / Background / Add keypoints / Add PAFs
                    const std::string key2part = "1234";
                    const auto keyPressed = key2part.find(castedKey);
                    if (keyPressed != std::string::npos)
                    {
                        ElementToRender elementToRender;
                        if (castedKey=='1')
                            elementToRender = ElementToRender::Skeleton;
                        else if (castedKey=='2')
                            elementToRender = ElementToRender::Background;
                        else if (castedKey=='3')
                            elementToRender = ElementToRender::AddKeypoints;
                        else if (castedKey=='4')
                            elementToRender = ElementToRender::AddPAFs;
                        else
                        {
                            error("Unknown ElementToRender value.", __LINE__, __FUNCTION__, __FILE__);
                            elementToRender = ElementToRender::Skeleton;
                        }
                        for (auto& renderer : renderers)
                            renderer->setElementToRender(elementToRender);
                    }

                    // Heatmap visualization
                    else
                    {
                        // Other rendering options
                        // const std::string key2partHeatmaps = "0123456789qwertyuiopasd";
                        const std::string key2partHeatmaps = "567890";
                        const auto newElementToRender = key2partHeatmaps.find(castedKey);
                        if (newElementToRender != std::string::npos)
                            for (auto& renderer : renderers)
                                renderer->setElementToRender(int(newElementToRender+key2part.size()));
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void handleUserInput(FrameDisplayer& frameDisplayer,
                         std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets,
                         std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets,
                         std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets,
                         std::vector<std::shared_ptr<Renderer>>& renderers,
                         std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                         std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr,
                         DisplayMode& displayMode, DisplayMode& displayModeOriginal)
    {
        try
        {
            // The handleUserInput must be always performed, even if no tDatum is detected
            bool guiPaused = false;
            handleWaitKey(guiPaused, frameDisplayer, poseExtractorNets, faceExtractorNets, handExtractorNets,
                          renderers, isRunningSharedPtr, videoSeekSharedPtr, displayMode, displayModeOriginal);
            while (guiPaused)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                handleWaitKey(guiPaused, frameDisplayer, poseExtractorNets, faceExtractorNets, handExtractorNets,
                              renderers, isRunningSharedPtr, videoSeekSharedPtr, displayMode, displayModeOriginal);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Gui::Gui(const Point<int>& outputSize, const bool fullScreen,
             const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
             const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr,
             const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets,
             const std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets,
             const std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets,
             const std::vector<std::shared_ptr<Renderer>>& renderers,
             const DisplayMode displayMode) :
        spIsRunning{isRunningSharedPtr},
        mDisplayMode{displayMode},
        mDisplayModeOriginal{displayMode},
        mFrameDisplayer{OPEN_POSE_NAME_AND_VERSION, outputSize, fullScreen},
        mPoseExtractorNets{poseExtractorNets},
        mFaceExtractorNets{faceExtractorNets},
        mHandExtractorNets{handExtractorNets},
        mRenderers{renderers},
        spVideoSeek{videoSeekSharedPtr}
    {
    }

    Gui::~Gui()
    {
    }

    void Gui::initializationOnThread()
    {
        mFrameDisplayer.initializationOnThread();
    }

    void Gui::setImage(const Matrix& cvMatOutput)
    {
        try
        {
            setImage(std::vector<Matrix>{cvMatOutput});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Gui::setImage(const std::vector<Matrix>& cvMatOutputs)
    {
        try
        {
            // Check tDatum integrity
            bool returnedIsValidFrame = ((spIsRunning == nullptr || *spIsRunning) && !cvMatOutputs.empty());
            for (const auto& cvMatOutput : cvMatOutputs)
            {
                if (cvMatOutput.empty())
                {
                    returnedIsValidFrame = false;
                    break;
                }
            }

            // Display
            if (returnedIsValidFrame)
                mFrameDisplayer.displayFrame(cvMatOutputs, -1);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Gui::update()
    {
        try
        {
            // Handle user input
            handleUserInput(mFrameDisplayer, mPoseExtractorNets, mFaceExtractorNets, mHandExtractorNets,
                            mRenderers, spIsRunning, spVideoSeek, mDisplayMode, mDisplayModeOriginal);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
