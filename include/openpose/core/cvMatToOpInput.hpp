#ifndef OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include "array.hpp"

namespace op
{
    class CvMatToOpInput
    {
    public:
        CvMatToOpInput(const cv::Size& netInputResolution, const int scaleNumber = 1, const float scaleGap = 0.25);

        Array<float> format(const cv::Mat& cvInputData) const;

    private:
        const int mScaleNumber;
        const float mScaleGap;
        const std::vector<int> mInputNetSize4D;
    };
}

#endif // OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP
