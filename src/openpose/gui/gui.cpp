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
                       std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors,
                       std::vector<std::shared_ptr<Renderer>>& renderers,
                       std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
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
                    frameDisplayer.switchFullScreenMode();
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
                    for (auto& renderer : renderers)
                        renderer->setBlendOriginalFrame(!renderer->getBlendOriginalFrame());
                }
                // ------------------------- OpenPose-Related ------------------------- //
                // Modifying thresholds
                else if (castedKey=='-' || castedKey=='=')
                    for (auto& poseExtractor : poseExtractors)
                        poseExtractor->increase(PoseProperty::NMSThreshold, 0.005f * (castedKey=='-' ? -1 : 1));
                else if (castedKey=='_' || castedKey=='+')
                    for (auto& poseExtractor : poseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectMinSubsetScore,
                                                0.005f * (castedKey=='_' ? -1 : 1));
                else if (castedKey=='[' || castedKey==']')
                    for (auto& poseExtractor : poseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectInterThreshold,
                                                0.005f * (castedKey=='[' ? -1 : 1));
                else if (castedKey=='{' || castedKey=='}')
                    for (auto& poseExtractor : poseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectInterMinAboveThreshold,
                                                (castedKey=='{' ? -0.1f : 0.1f));
                else if (castedKey==';' || castedKey=='\'')
                    for (auto& poseExtractor : poseExtractors)
                        poseExtractor->increase(PoseProperty::ConnectMinSubsetCnt, (castedKey==';' ? -1 : 1));
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

    void handleUserInput(FrameDisplayer& frameDisplayer, std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors,
                         std::vector<std::shared_ptr<Renderer>>& renderers,
                         std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                         std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& spVideoSeek)
    {
        try
        {
            // The handleUserInput must be always performed, even if no tDatum is detected
            bool guiPaused = false;
            handleWaitKey(guiPaused, frameDisplayer, poseExtractors, renderers, isRunningSharedPtr, spVideoSeek);
            while (guiPaused)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                handleWaitKey(guiPaused, frameDisplayer, poseExtractors, renderers, isRunningSharedPtr, spVideoSeek);
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
             const std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors,
             const std::vector<std::shared_ptr<Renderer>>& renderers) :
        spIsRunning{isRunningSharedPtr},
        mFrameDisplayer{OPEN_POSE_NAME_AND_VERSION, outputSize, fullScreen},
        mPoseExtractors{poseExtractors},
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
            // Check tDatum integrity
            const bool returnedIsValidFrame = ((spIsRunning == nullptr || *spIsRunning) && !cvMatOutput.empty());

            // Display
            if (returnedIsValidFrame)
                mFrameDisplayer.displayFrame(cvMatOutput, -1);
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
            // 0 image
            if (cvMatOutputs.empty())
                setImage(cvMatOutputs[0]);
            // 1 image
            else if (cvMatOutputs.size() == 1)
                setImage(cvMatOutputs[0]);
            // > 1 image
            else
            {
                // Check tDatum integrity
                bool returnedIsValidFrame = ((spIsRunning == nullptr || *spIsRunning) && !cvMatOutputs.empty());
                if (returnedIsValidFrame)
                {
                    // Security checks
                    for (const auto& cvMatOutput : cvMatOutputs)
                        if (cvMatOutput.empty())
                            returnedIsValidFrame = false;
                    // Prepare final cvMat
                    if (returnedIsValidFrame)
                    {
                        // Concat (0)
                        cv::Mat cvMat = cvMatOutputs[0].clone();
                        // Concat (1,size()-1)
                        for (auto i = 1u; i < cvMatOutputs.size(); i++)
                            cv::hconcat(cvMat, cvMatOutputs[i], cvMat);
                        // Display
                        mFrameDisplayer.displayFrame(cvMat, -1);
                    }
                }
            }
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
            handleUserInput(mFrameDisplayer, mPoseExtractors, mRenderers, spIsRunning, spVideoSeek);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
