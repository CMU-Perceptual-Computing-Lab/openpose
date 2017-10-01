#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CvMatToOpOutput
    {
    public:
        // Use outputResolution <= {0,0} to keep input resolution
        CvMatToOpOutput(const Point<int>& outputResolution = Point<int>{0, 0}, const bool generateOutput = true);

        std::tuple<double, Array<float>> format(const cv::Mat& cvInputData) const;

    private:
        const bool mGenerateOutput;
        const std::vector<int> mOutputSize3D;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
