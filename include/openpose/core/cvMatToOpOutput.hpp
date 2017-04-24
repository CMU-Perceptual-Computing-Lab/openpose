#ifndef OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include "array.hpp"

namespace op
{
    class CvMatToOpOutput
    {
    public:
        CvMatToOpOutput(const cv::Size& outputResolution, const bool generateOutput = true);

        std::tuple<double, Array<float>> format(const cv::Mat& cvInputData) const;

    private:
        const bool mGenerateOutput;
        const std::vector<int> mOutputSize3D;
    };
}

#endif // OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP
