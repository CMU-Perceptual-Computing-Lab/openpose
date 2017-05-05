#include <cuda.h>
#include <cuda_runtime.h>
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/cuda.hpp"

namespace op
{
    const dim3 THREADS_PER_BLOCK_SMALL{32, 32, 1};
    const dim3 THREADS_PER_BLOCK_MEDIUM{128, 128, 1};
    const dim3 THREADS_PER_BLOCK_BIG{256, 256, 1};

    void cudaCheck(const int line, const std::string& function, const std::string& file)
    {
        const auto errorCode = cudaPeekAtLastError();
        if(errorCode != cudaSuccess)
            error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "): " + cudaGetErrorString(errorCode), line, function, file);
    }

    std::pair<dim3, dim3> getNumberCudaThreadsAndBlocks(const cv::Size& frameSize)
    {
        try
        {
            // Image <= 1280x720    --> THREADS_PER_BLOCK_SMALL
            // Image < 16K          --> THREADS_PER_BLOCK_MEDIUM
            // Image > 16K          --> THREADS_PER_BLOCK_BIG
            const auto maxValue = fastMax(frameSize.width, frameSize.height);
            const auto threadsPerBlock = (maxValue < 1281 ? THREADS_PER_BLOCK_SMALL
                                          : (maxValue < 16384 ? THREADS_PER_BLOCK_MEDIUM
                                             : THREADS_PER_BLOCK_BIG));
            return std::make_pair(threadsPerBlock, getNumberCudaBlocks(frameSize, threadsPerBlock));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(dim3{1,1,1}, dim3{1,1,1});
        }
    }
}
