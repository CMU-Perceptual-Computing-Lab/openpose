#include <chrono>
#include <thread>
#include <opencv2/highgui/highgui.hpp> // cv::waitKey
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/gui/gui.hpp>

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
                                              Point<int>{helpCvMat.cols, helpCvMat.rows}, fullScreen};
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
                       std::vector<std::shared_ptr<Renderer>>& renderers,
                       std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                       std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr)
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
                    const std::string key2part = "0123456789qwertyuiopasd";
                    const auto newElementToRender = key2part.find(castedKey);
                    if (newElementToRender != std::string::npos)
                        for (auto& renderer : renderers)
                            renderer->setElementToRender((int)newElementToRender);
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
                         std::vector<std::shared_ptr<Renderer>>& renderers,
                         std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                         std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr)
    {
        try
        {
            // The handleUserInput must be always performed, even if no tDatum is detected
            bool guiPaused = false;
            handleWaitKey(guiPaused, frameDisplayer, poseExtractorNets, renderers, isRunningSharedPtr,
                          videoSeekSharedPtr);
            while (guiPaused)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                handleWaitKey(guiPaused, frameDisplayer, poseExtractorNets, renderers, isRunningSharedPtr,
                              videoSeekSharedPtr);
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
             const std::vector<std::shared_ptr<Renderer>>& renderers) :
        spIsRunning{isRunningSharedPtr},
        mFrameDisplayer{OPEN_POSE_NAME_AND_VERSION, outputSize, fullScreen},
        mPoseExtractorNets{poseExtractorNets},
        mRenderers{renderers},
        spVideoSeek{videoSeekSharedPtr}
    {
    }

    void Gui::initializationOnThread()
    {
        mFrameDisplayer.initializationOnThread();
    }

    void Gui::setImage(const cv::Mat& cvMatOutput)
    {
        try
        {
            setImage(std::vector<cv::Mat>{cvMatOutput});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Gui::setImage(const std::vector<cv::Mat>& cvMatOutputs)
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
            handleUserInput(mFrameDisplayer, mPoseExtractorNets, mRenderers, spIsRunning, spVideoSeek);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
