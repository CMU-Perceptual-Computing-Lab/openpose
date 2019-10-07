#include <openpose/gpu/cuda.hpp>
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <openpose/utilities/fastMath.hpp>
#endif

namespace op
{
    #ifdef USE_CUDA
        #ifdef DNDEBUG
            #define base 32
        #else
            #define base 64
        #endif
        const dim3 THREADS_PER_BLOCK_TINY{base, base, 1};       // 32 |64
        const dim3 THREADS_PER_BLOCK_SMALL{2*base, 2*base, 1};  // 64 |128
        const dim3 THREADS_PER_BLOCK_MEDIUM{4*base, 4*base, 1}; // 128|256
        const dim3 THREADS_PER_BLOCK_BIG{8*base, 8*base, 1};    // 256|512
        const dim3 THREADS_PER_BLOCK_HUGE{16*base, 16*base, 1}; // 512|1024
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

    int getCudaGpuNumber()
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
                // Image <= 1080p   --> THREADS_PER_BLOCK_MEDIUM
                // Image <= 4k      --> THREADS_PER_BLOCK_BIG
                // Image >  4K      --> THREADS_PER_BLOCK_HUGE
                const auto maxValue = fastMax(frameSize.x, frameSize.y);
                // > 4K
                if (maxValue > 3840)
                    numberCudaThreads = THREADS_PER_BLOCK_HUGE;
                // 4K
                else if (maxValue > 1980)
                    numberCudaThreads = THREADS_PER_BLOCK_BIG;
                // FullHD
                else if (maxValue > 1280)
                    numberCudaThreads = THREADS_PER_BLOCK_MEDIUM;
                // HD
                else if (maxValue > 640)
                    numberCudaThreads = THREADS_PER_BLOCK_SMALL;
                // VGA
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
