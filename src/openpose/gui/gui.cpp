#include <chrono>
#include <thread>
#include <opencv2/highgui/highgui.hpp> // cv::waitKey
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/errorAndLog.hpp>
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
                FrameDisplayer frameDisplayer{Point<int>{helpCvMat.cols, helpCvMat.rows}, "OpenPose - GUI Help", fullScreen};
                frameDisplayer.displayFrame(helpCvMat, 33);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void handleWaitKey(bool& guiPaused, FrameDisplayer& mFrameDisplayer, std::vector<std::shared_ptr<PoseExtractor>>& mPoseExtractors,
                       std::vector<std::shared_ptr<PoseRenderer>>& mPoseRenderers, std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                       std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& spVideoSeek)
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
                    mFrameDisplayer.switchGuiDisplayMode();
                // ------------------------- Producer-Related ------------------------- //
                // Pause
                else if (castedKey==' ')
                    guiPaused = !guiPaused;
                // Fake pause
                else if (castedKey=='m')
                {
                    if (spVideoSeek != nullptr)
                        spVideoSeek->first = !spVideoSeek->first;
                }
                // Seeking in video
                else if (castedKey=='l' || castedKey=='k')
                {
                    if (spVideoSeek != nullptr)
                    {
                        // Normal case, +-30 frames
                        if (!spVideoSeek->first)
                            spVideoSeek->second += 30 * (castedKey=='l' ? -2 : 1);
                        // Frame by frame (if forced paused)
                        else
                            spVideoSeek->second += (castedKey=='l' ? -1 : 1);
                    }
                }
                // Enable/disable blending
                else if (castedKey=='b')
                {
                    for (auto& poseRenderer : mPoseRenderers)
                        poseRenderer->setBlendOriginalFrame(!poseRenderer->getBlendOriginalFrame());
                }
                // ------------------------- OpenPose-Related ------------------------- //
                // Modifying thresholds
                else if (castedKey=='-' || castedKey=='=')
                    for (auto& poseExtractor : mPoseExtractors)
                        poseExtractor->increase(PoseProperty::NMSThreshold, 0.005f * (castedKey=='-' ? -1 : 1));
                else if (castedKey=='_' || castedKey=='+')
                    for (auto& poseExtractor : mPoseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectMinSubsetScore, 0.005f * (castedKey=='_' ? -1 : 1));
                else if (castedKey=='[' || castedKey==']')
                    for (auto& poseExtractor : mPoseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectInterThreshold, 0.005f * (castedKey=='[' ? -1 : 1));
                else if (castedKey=='{' || castedKey=='}')
                    for (auto& poseExtractor : mPoseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectInterMinAboveThreshold, (castedKey=='{' ? -1 : 1));
                else if (castedKey==';' || castedKey=='\'')
                    for (auto& poseExtractor : mPoseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectMinSubsetCnt, (castedKey==';' ? -1 : 1));
                // ------------------------- Miscellaneous ------------------------- //
                // Show googly eyes
                else if (castedKey=='g')
                    for (auto& poseRenderer : mPoseRenderers)
                        poseRenderer->setShowGooglyEyes(!poseRenderer->getShowGooglyEyes());
                // ------------------------- OpenPose-Related ------------------------- //
                else if (castedKey==',' || castedKey=='.')
                {
                    const auto increment = (castedKey=='.' ? 1 : -1);
                    for (auto& poseRenderer : mPoseRenderers)
                        poseRenderer->increaseElementToRender(increment);
                }
                else
                {
                    const std::string key2part = "0123456789qwertyuiopasd";
                    const auto newElementToRender = key2part.find(castedKey);
                    if (newElementToRender != std::string::npos)
                        for (auto& poseRenderer : mPoseRenderers)
                            poseRenderer->setElementToRender((int)newElementToRender);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void handleUserInput(FrameDisplayer& mFrameDisplayer, std::vector<std::shared_ptr<PoseExtractor>>& mPoseExtractors,
                         std::vector<std::shared_ptr<PoseRenderer>>& mPoseRenderers, std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                         std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& spVideoSeek)
    {
        try
        {
            // The handleUserInput must be always performed, even if no tDatum is detected
            bool guiPaused = false;
            handleWaitKey(guiPaused, mFrameDisplayer, mPoseExtractors, mPoseRenderers, isRunningSharedPtr, spVideoSeek);
            while (guiPaused)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                handleWaitKey(guiPaused, mFrameDisplayer, mPoseExtractors, mPoseRenderers, isRunningSharedPtr, spVideoSeek);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Gui::Gui(const bool fullScreen, const Point<int>& outputSize, const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr, 
             const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr,
             const std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors, const std::vector<std::shared_ptr<PoseRenderer>>& poseRenderers) :
        mFrameDisplayer{outputSize, "OpenPose - GUI", fullScreen},
        mPoseExtractors{poseExtractors},
        mPoseRenderers{poseRenderers},
        spIsRunning{isRunningSharedPtr},
        spVideoSeek{videoSeekSharedPtr}
    {
    }

    void Gui::initializationOnThread()
    {
        mFrameDisplayer.initializationOnThread();
    }

    void Gui::update(const cv::Mat& cvOutputData)
    {
        try
        {
            // Check tDatum integrity
            const bool returnedIsValidFrame = ((spIsRunning == nullptr || *spIsRunning) && !cvOutputData.empty());

            // Display
            if (returnedIsValidFrame)
                mFrameDisplayer.displayFrame(cvOutputData, -1);

            // Handle user input
            handleUserInput(mFrameDisplayer, mPoseExtractors, mPoseRenderers, spIsRunning, spVideoSeek);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
