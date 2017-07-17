#ifndef OPENPOSE_GUI_ADD_GUI_INFO_HPP
#define OPENPOSE_GUI_ADD_GUI_INFO_HPP

#include <queue>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API GuiInfoAdder
    {
    public:
        GuiInfoAdder(const Point<int>& outputSize, const int numberGpus, const bool guiEnabled = false);

        void addInfo(cv::Mat& cvOutputData, const Array<float>& poseKeypoints, const unsigned long long id, const std::string& elementRenderedName);

    private:
        // Const variables
        const Point<int> mOutputSize;
        const int mBorderMargin;
        const int mNumberGpus;
        const bool mGuiEnabled;
        // Other variables
        std::queue<std::chrono::high_resolution_clock::time_point> mFpsQueue;
        double mFps;
        unsigned int mFpsCounter;
        std::string mLastElementRenderedName;
        int mLastElementRenderedCounter;
        unsigned long long mLastId;
    };
}

#endif // OPENPOSE_GUI_ADD_GUI_INFO_HPP
