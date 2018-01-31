#ifdef USE_CUDA
    #include <openpose/utilities/cuda.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/core/clManager.hpp>
#endif
#include <openpose/utilities/gpu.hpp>

namespace op
{
    int getGpuNumber()
    {
        try
        {
            int totalGpuNumber = -1;
            #ifdef USE_CUDA
                totalGpuNumber = cudaGetGpuNumber();
            #elif USE_OPENCL
                totalGpuNumber = op::CLManager::getTotalGPU();
            #else
                error("OpenPose must be compiled with the `USE_CUDA` or `USE_OPENCL` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            return totalGpuNumber;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }
}
