#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <openpose/utilities/fastMath.hpp>
#endif
#include <openpose/utilities/cuda.hpp>

namespace op
{
    #ifdef USE_CUDA
        const dim3 THREADS_PER_BLOCK_TINY{32, 32, 1};
        const dim3 THREADS_PER_BLOCK_SMALL{64, 64, 1};
        const dim3 THREADS_PER_BLOCK_MEDIUM{128, 128, 1};
        const dim3 THREADS_PER_BLOCK_BIG{256, 256, 1};
    #endif

    void cudaCheck(const int line, const std::string& function, const std::string& file)
    {
        try
        {
            #ifdef USE_CUDA
                const auto errorCode = cudaPeekAtLastError();
                if(errorCode != cudaSuccess)
                    error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "): "
                          + cudaGetErrorString(errorCode), line, function, file);
            #else
                UNUSED(line);
                UNUSED(function);
                UNUSED(file);
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    int getGpuNumber()
    {
        try
        {
            #ifdef USE_CUDA
                int gpuNumber;
                cudaGetDeviceCount(&gpuNumber);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                return gpuNumber;
            #else
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    void getNumberCudaThreadsAndBlocks(dim3& numberCudaThreads, dim3& numberCudaBlocks, const Point<int>& frameSize)
    {
        try
        {
            #ifdef USE_CUDA
                // numberCudaThreads
                // Image <= 480p    --> THREADS_PER_BLOCK_TINY
                // Image <= 720p    --> THREADS_PER_BLOCK_SMALL
                // Image <= 16K     --> THREADS_PER_BLOCK_MEDIUM
                // Image > 16K      --> THREADS_PER_BLOCK_BIG
                const auto maxValue = fastMax(frameSize.x, frameSize.y);
                if (maxValue >= 16384)
                    numberCudaThreads = THREADS_PER_BLOCK_BIG;
                else if (maxValue > 1280)
                    numberCudaThreads = THREADS_PER_BLOCK_MEDIUM;
                else if (maxValue > 640)
                    numberCudaThreads = THREADS_PER_BLOCK_SMALL;
                else
                    numberCudaThreads = THREADS_PER_BLOCK_TINY;
                // numberCudaBlocks
                numberCudaBlocks = dim3{getNumberCudaBlocks((unsigned int)frameSize.x, numberCudaThreads.x),
                                        getNumberCudaBlocks((unsigned int)frameSize.y, numberCudaThreads.y),
                                        numberCudaThreads.z};
            #else
                UNUSED(numberCudaThreads);
                UNUSED(numberCudaBlocks);
                UNUSED(frameSize);
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
