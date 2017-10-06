#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CvMatToOpOutput
    {
    public:
        Array<float> createArray(const cv::Mat& cvInputData, const double scaleInputToOutput,
                                 const Point<int>& outputResolution) const;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
