#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CvMatToOpOutput
    {
    public:
        CvMatToOpOutput(const bool gpuResize = false);

        virtual ~CvMatToOpOutput();

        std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<unsigned long long>>
            getSharedParameters();

        Array<float> createArray(
            const Matrix& inputData, const double scaleInputToOutput, const Point<int>& outputResolution);

    private:
        const bool mGpuResize;
        unsigned char* pInputImageCuda;
        std::shared_ptr<float*> spOutputImageCuda;
        unsigned long long pInputMaxSize;
        std::shared_ptr<unsigned long long> spOutputMaxSize;
        std::shared_ptr<bool> spGpuMemoryAllocated;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_OUTPUT_HPP
