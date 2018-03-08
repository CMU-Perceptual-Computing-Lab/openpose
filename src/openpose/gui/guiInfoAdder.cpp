#include <chrono>
#include <cstdio> // std::snprintf
#include <limits> // std::numeric_limits
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/gui/guiInfoAdder.hpp>

namespace op
{
    void updateFps(unsigned long long& lastId, double& fps, unsigned int& fpsCounter,
                   std::queue<std::chrono::high_resolution_clock::time_point>& fpsQueue,
                   const unsigned long long id, const int numberGpus)
    {
        try
        {
            // If only 1 GPU -> update fps every frame.
            // If > 1 GPU:
                // We updated fps every (3*numberGpus) frames. This is due to the variability introduced by
                // using > 1 GPU at the same time.
                // However, we update every frame during the first few frames to have an initial estimator.
            // In any of the previous cases, the fps value is estimated during the last several frames.
            // In this way, a sudden fps drop will be quickly visually identified.
            if (lastId != id)
            {
                lastId = id;
                fpsQueue.emplace(std::chrono::high_resolution_clock::now());
                bool updatePrintedFps = true;
                if (fpsQueue.size() > 5)
                {
                    const auto factor = (numberGpus > 1 ? 25u : 15u);
                    updatePrintedFps = (fpsCounter % factor == 0);
                    // updatePrintedFps = (numberGpus == 1 ? true : fpsCounter % (3*numberGpus) == 0);
                    fpsCounter++;
                    if (fpsQueue.size() > factor)
                        fpsQueue.pop();
                }
                if (updatePrintedFps)
                {
                    const auto timeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        fpsQueue.back()-fpsQueue.front()
                    ).count() * 1e-9;
                    fps = (fpsQueue.size()-1) / (timeSec != 0. ? timeSec : 1.);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    GuiInfoAdder::GuiInfoAdder(const int numberGpus, const bool guiEnabled) :
        mNumberGpus{numberGpus},
        mGuiEnabled{guiEnabled},
        mFpsCounter{0u},
        mLastElementRenderedCounter{std::numeric_limits<int>::max()},
        mLastId{std::numeric_limits<unsigned long long>::max()}
    {
    }

    void GuiInfoAdder::addInfo(cv::Mat& cvOutputData, const int numberPeople, const unsigned long long id,
                               const std::string& elementRenderedName, const unsigned long long frameNumber)
    {
        try
        {
            // Security checks
            if (cvOutputData.empty())
                error("Wrong input element (empty cvOutputData).", __LINE__, __FUNCTION__, __FILE__);
            // Size
            const auto borderMargin = intRound(fastMax(cvOutputData.cols, cvOutputData.rows) * 0.025);
            // Update fps
            updateFps(mLastId, mFps, mFpsCounter, mFpsQueue, id, mNumberGpus);
            // Used colors
            const cv::Scalar white{255, 255, 255};
            // Fps or s/gpu
            char charArrayAux[15];
            std::snprintf(charArrayAux, 15, "%4.1f fps", mFps);
            // Recording inverse: sec/gpu
            // std::snprintf(charArrayAux, 15, "%4.2f s/gpu", (mFps != 0. ? mNumberGpus/mFps : 0.));
            putTextOnCvMat(cvOutputData, charArrayAux, {intRound(cvOutputData.cols - borderMargin), borderMargin},
                           white, true, cvOutputData.cols);
            // Part to show
            // Allowing some buffer when changing the part to show (if >= 2 GPUs)
            // I.e. one GPU might return a previous part after the other GPU returns the new desired part, it looks
            // like a mini-bug on screen
            // Difference between Titan X (~110 ms) & 1050 Ti (~290ms)
            if (mNumberGpus == 1 || (elementRenderedName != mLastElementRenderedName
                                     && mLastElementRenderedCounter > 4))
            {
                mLastElementRenderedName = elementRenderedName;
                mLastElementRenderedCounter = 0;
            }
            mLastElementRenderedCounter = fastMin(mLastElementRenderedCounter, std::numeric_limits<int>::max() - 5);
            mLastElementRenderedCounter++;
            // OpenPose name as well as help or part to show
            putTextOnCvMat(cvOutputData, "OpenPose - " +
                           (!mLastElementRenderedName.empty() ?
                                mLastElementRenderedName : (mGuiEnabled ? "'h' for help" : "")),
                           {borderMargin, borderMargin}, white, false, cvOutputData.cols);
            // Frame number
            putTextOnCvMat(cvOutputData, "Frame: " + std::to_string(frameNumber),
                           {borderMargin, (int)(cvOutputData.rows - borderMargin)}, white, false, cvOutputData.cols);
            // Number people
            putTextOnCvMat(cvOutputData, "People: " + std::to_string(numberPeople),
                           {(int)(cvOutputData.cols - borderMargin), (int)(cvOutputData.rows - borderMargin)},
                           white, true, cvOutputData.cols);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
