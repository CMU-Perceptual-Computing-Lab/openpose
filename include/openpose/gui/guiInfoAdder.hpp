#ifndef OPENPOSE__GUI__ADD_GUI_INFO_HPP
#define OPENPOSE__GUI__ADD_GUI_INFO_HPP

#include <queue>
#include <opencv2/core/core.hpp>
#include "../core/array.hpp"

namespace op
{
    class GuiInfoAdder
    {
    public:
        GuiInfoAdder(const cv::Size& outputSize, const int numberGpus);

        void addInfo(cv::Mat& cvOutputData, const Array<float>& poseKeyPoints, const unsigned long long id, const std::string& elementRenderedName);

    private:
        // Const variables
        const cv::Size mOutputSize;
        const int mBorderMargin;
        const int mNumberGpus;
        // Other variables
        std::queue<std::chrono::high_resolution_clock::time_point> mFpsQueue;
        double mFps;
        unsigned int mFpsCounter;
        std::string mLastElementRenderedName;
        int mLastElementRenderedCounter;
    };
}

#endif // OPENPOSE__GUI__ADD_GUI_INFO_HPP
