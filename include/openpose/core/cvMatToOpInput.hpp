#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP

#include <utility> // std::pair
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include "array.hpp"
#include "point.hpp"

namespace op
{
    class CvMatToOpInput
    {
    public:
        CvMatToOpInput(const Point<int>& netInputResolution, const int scaleNumber = 1, const float scaleGap = 0.25);

        std::pair<Array<float>, std::vector<float>> format(const cv::Mat& cvInputData) const;

    private:
        const int mScaleNumber;
        const float mScaleGap;
        const std::vector<int> mInputNetSize4D;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
