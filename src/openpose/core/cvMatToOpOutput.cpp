#include <openpose/core/cvMatToOpOutput.hpp>
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
    #include <openpose/net/resizeAndMergeBase.hpp>
    #include <openpose_private/gpu/cuda.hu>
#endif
#include <openpose_private/utilities/openCvPrivate.hpp>

namespace op
{
    CvMatToOpOutput::CvMatToOpOutput(const bool gpuResize) :
        mGpuResize{gpuResize},
        pInputImageCuda{nullptr},
        spOutputImageCuda{std::make_shared<float*>()},
        pInputMaxSize{0ull},
        spOutputMaxSize{std::make_shared<unsigned long long>(0ull)},
        spGpuMemoryAllocated{std::make_shared<bool>(false)}
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

    CvMatToOpOutput::~CvMatToOpOutput()
    {
        try
        {
            #ifdef USE_CUDA
                if (mGpuResize)
                {
                    // Free temporary memory
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    if (pInputImageCuda != nullptr)
                    {
                        cudaFree(pInputImageCuda);
                        pInputImageCuda = nullptr;
                    }
                    if (*spOutputImageCuda != nullptr)
                    {
                        cudaFree(*spOutputImageCuda);
                        *spOutputImageCuda = nullptr;
                    }
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::tuple<std::shared_ptr<float*>, std::shared_ptr<bool>, std::shared_ptr<unsigned long long>>
        CvMatToOpOutput::getSharedParameters()
    {
        try
        {
            return std::make_tuple(spOutputImageCuda, spGpuMemoryAllocated, spOutputMaxSize);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(nullptr, nullptr, nullptr);
        }
    }

    Array<float> CvMatToOpOutput::createArray(
         const Matrix& inputData, const double scaleInputToOutput, const Point<int>& outputResolution)
    {
        try
        {
            cv::Mat cvInputData = OP_OP2CVCONSTMAT(inputData);
            // Sanity checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.channels() != 3)
                error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.cols <= 0 || cvInputData.rows <= 0)
                error("Input images has 0 area.", __LINE__, __FUNCTION__, __FILE__);
            if (outputResolution.x <= 0 || outputResolution.y <= 0)
                error("Output resolution has 0 area.", __LINE__, __FUNCTION__, __FILE__);
            // outputData - Reescale keeping aspect ratio and transform to float the output image
            Array<float> outputData({outputResolution.y, outputResolution.x, 3}); // This size is used everywhere
            // CPU version (faster if #Gpus <= 3 and relatively small images)
            if (!mGpuResize)
            {
                cv::Mat frameWithOutputSize;
                resizeFixedAspectRatio(frameWithOutputSize, cvInputData, scaleInputToOutput, outputResolution);
                // Equivalent: frameWithOutputSize.convertTo(outputData.getCvMat(), CV_32FC3);
                cv::Mat cvOutputData = OP_OP2CVMAT(outputData.getCvMat());
                frameWithOutputSize.convertTo(cvOutputData, CV_32FC3);
            }
            // CUDA version (if #Gpus > 3)
            else
            {
                #ifdef USE_CUDA
                    // Input image can be shared between this one and cvMatToOpInput.hpp
                    // However, that version reduces the global accuracy a bit
                    // (Free and re-)Allocate temporary memory
                    const unsigned int inputImageSize = 3 * cvInputData.rows * cvInputData.cols;
                    if (pInputMaxSize < inputImageSize)
                    {
                        pInputMaxSize = inputImageSize;
                        cudaFree(pInputImageCuda);
                        cudaMalloc((void**)&pInputImageCuda, sizeof(unsigned char) * inputImageSize);
                    }
                    // (Free and re-)Allocate temporary memory
                    const unsigned int outputImageSize = 3 * outputResolution.x * outputResolution.y;
                    if (*spOutputMaxSize < outputImageSize)
                    {
                        *spOutputMaxSize = outputImageSize;
                        cudaFree(*spOutputImageCuda);
                        cudaMalloc((void**)spOutputImageCuda.get(), sizeof(float) * outputImageSize);
                    }
                    // Copy original image to GPU
                    cudaMemcpy(
                        pInputImageCuda, cvInputData.data, sizeof(unsigned char) * inputImageSize, cudaMemcpyHostToDevice);
                    // Resize output image on GPU
                    resizeAndPadRbgGpu(
                        *spOutputImageCuda, pInputImageCuda, cvInputData.cols, cvInputData.rows, outputResolution.x,
                        outputResolution.y, (float)scaleInputToOutput);
                    *spGpuMemoryAllocated = true;
                    // // No need to copy output image back to CPU
                    // cudaMemcpy(
                    //     outputData.getPtr(), *spOutputImageCuda, sizeof(float) * outputImageSize,
                    //     cudaMemcpyDeviceToHost);
                #else
                    error("You need to compile OpenPose with CUDA support in order to use GPU resize.",
                        __LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            // Return result
            return outputData;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }
}
