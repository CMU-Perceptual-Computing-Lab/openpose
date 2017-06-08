#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP

#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include "array.hpp"
#include "point.hpp"

namespace op
{
    class CvMatToOpOutput
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
