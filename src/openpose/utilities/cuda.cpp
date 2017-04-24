#include <cuda.h>
#include <cuda_runtime.h>
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/cuda.hpp"

namespace op
{
    void cudaCheck(const int line, const std::string& function, const std::string& file)
    {
        const auto errorCode = cudaPeekAtLastError();
        if(errorCode != cudaSuccess)
        	error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "):" + cudaGetErrorString(errorCode), line, function, file);
    }
}
