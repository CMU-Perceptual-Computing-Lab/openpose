#ifndef OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API OpOutputToCvMat
    {
    public:
        explicit OpOutputToCvMat(const Point<int>& outputResolution);

        cv::Mat formatToCvMat(const Array<float>& outputData) const;

    private:
        const Point<int> mOutputResolution;
    };
}

#endif // OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
