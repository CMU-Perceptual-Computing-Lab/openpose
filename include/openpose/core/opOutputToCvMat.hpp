#ifndef OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP

#include <opencv2/core/core.hpp>
#include "array.hpp"

namespace op
{
    class OpOutputToCvMat
    {
    public:
        explicit OpOutputToCvMat(const cv::Size& outputResolution);

        cv::Mat formatToCvMat(const Array<float>& outputData) const;

    private:
        const cv::Size mOutputResolution;
    };
}

#endif // OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP
