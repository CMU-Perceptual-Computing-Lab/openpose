#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CvMatToOpOutput
    {
    public:
        CvMatToOpOutput(const Point<int>& outputResolution, const bool generateOutput = true);

        std::tuple<double, Array<float>> format(const cv::Mat& cvInputData) const;

    private:
        const bool mGenerateOutput;
        const std::vector<int> mOutputSize3D;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
