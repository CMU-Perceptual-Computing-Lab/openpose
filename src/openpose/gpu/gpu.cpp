#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
#endif
#include <openpose/gpu/gpu.hpp>

namespace op
{
    int getGpuNumber()
    {
        try
        {
            #ifdef USE_CUDA
                return getCudaGpuNumber();
            #elif defined USE_OPENCL
                return OpenCL::getTotalGPU();
            #else
                error("OpenPose must be compiled with the `USE_CUDA` or `USE_OPENCL` macro definition in order to use"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    GpuMode getGpuMode()
    {
        try
        {
            #ifdef USE_CUDA
                return GpuMode::Cuda;
            #elif defined USE_OPENCL
                return GpuMode::OpenCL;
            #else
                return GpuMode::NoGpu;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return GpuMode::NoGpu;
        }
    }
}
