#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
    #include <openpose/gpu/cuda.hu>
#endif
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/opOutputToCvMat.hpp>

namespace op
{
    OpOutputToCvMat::OpOutputToCvMat(const bool gpuResize) :
        mGpuResize{gpuResize},
        spOutputImageFloatCuda{std::make_shared<float*>()},
        spOutputMaxSize{std::make_shared<unsigned long long>(0ull)},
        spGpuMemoryAllocated{std::make_shared<bool>(false)},
        pOutputImageUCharCuda{nullptr},
        mOutputMaxSizeUChar{0ull}
    {
        try
        {
            #ifndef USE_CUDA
                if (mGpuResize)
                    error("You need to compile OpenPose with CUDA support in order to use GPU resize.",
                        __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    OpOutputToCvMat::~OpOutputToCvMat()
    {
        try
        {
            #ifdef USE_CUDA
                if (mGpuResize)
                {
                    // Free temporary memory
                    cudaFree(*spOutputImageFloatCuda);
                    cudaFree(pOutputImageUCharCuda);
                }
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void OpOutputToCvMat::setSharedParameters(
            const std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<unsigned long long>>& tuple)
    {
        try
        {
            spOutputImageFloatCuda = std::get<0>(tuple);
            spGpuMemoryAllocated = std::get<1>(tuple);
            spOutputMaxSize = std::get<2>(tuple);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    cv::Mat OpOutputToCvMat::formatToCvMat(const Array<float>& outputData)
    {
        try
        {
            // Sanity check
            if (outputData.empty())
                error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);
            // Final result
            cv::Mat cvMat;
            // CPU version
            if (!mGpuResize)
            {
                // outputData to cvMat
                outputData.getConstCvMat().convertTo(cvMat, CV_8UC3);
            }
            // CUDA version
            else
            {
                #ifdef USE_CUDA
                    // (Free and re-)Allocate temporary memory
                    if (mOutputMaxSizeUChar < *spOutputMaxSize)
                    {
                        mOutputMaxSizeUChar = *spOutputMaxSize;
                        cudaFree(pOutputImageUCharCuda);
                        cudaMalloc((void**)&pOutputImageUCharCuda, sizeof(unsigned char) * mOutputMaxSizeUChar);
                    }
                    // Float ptr --> unsigned char ptr
                    const auto volume = (int)outputData.getVolume();
                    uCharImageCast(pOutputImageUCharCuda, *spOutputImageFloatCuda, volume);
                    // Allocate cvMat
                    cvMat = cv::Mat(outputData.getSize(0), outputData.getSize(1), CV_8UC3);
                    // CUDA --> CPU: Copy output image back to CPU
                    cudaMemcpy(
                        cvMat.data, pOutputImageUCharCuda, sizeof(unsigned char) * volume,
                        cudaMemcpyDeviceToHost);
                    // Indicate memory was copied out
                    *spGpuMemoryAllocated = false;
                #else
                    error("You need to compile OpenPose with CUDA support in order to use GPU resize.",
                        __LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            // Return cvMat
            return cvMat;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }
}
