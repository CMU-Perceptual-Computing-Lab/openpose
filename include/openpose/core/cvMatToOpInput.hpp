#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CvMatToOpInput
    {
    public:
        std::vector<Array<float>> createArray(const cv::Mat& cvInputData,
                                              const std::vector<double>& scaleInputToNetInputs,
                                              const std::vector<Point<int>>& netInputSizes) const;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
