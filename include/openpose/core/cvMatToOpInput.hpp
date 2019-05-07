#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    class OP_API CvMatToOpInput
    {
    public:
        CvMatToOpInput(const PoseModel poseModel = PoseModel::BODY_25, const bool gpuResize = false);

        virtual ~CvMatToOpInput();

        std::vector<Array<float>> createArray(
            const cv::Mat& cvInputData, const std::vector<double>& scaleInputToNetInputs,
            const std::vector<Point<int>>& netInputSizes);

    private:
        const PoseModel mPoseModel;
        const bool mGpuResize;
        unsigned char* pInputImageCuda;
        float* pInputImageReorderedCuda;
        float* pOutputImageCuda;
        unsigned long long pInputMaxSize;
        unsigned long long pOutputMaxSize;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
