#ifndef OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP

#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API OpOutputToCvMat
    {
    public:
        OpOutputToCvMat(const bool gpuResize = false);

        virtual ~OpOutputToCvMat();

        void setSharedParameters(
            const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<unsigned long long>>& tuple);

        cv::Mat formatToCvMat(const Array<float>& outputData);

    private:
        const bool mGpuResize;
        // Shared variables
        std::shared_ptr<float*> spOutputImageFloatCuda;
        std::shared_ptr<unsigned long long> spOutputMaxSize;
        std::shared_ptr<bool> spGpuMemoryAllocated;
        // Local variables
        unsigned char* pOutputImageUCharCuda;
        unsigned long long mOutputMaxSizeUChar;
    };
}

#endif // OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
