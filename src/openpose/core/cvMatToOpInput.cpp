// #include <opencv2/opencv.hpp>
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
    #include <openpose/gpu/cuda.hu>
    #include <openpose/net/resizeAndMergeBase.hpp>
#endif
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpInput.hpp>

namespace op
{
    CvMatToOpInput::CvMatToOpInput(const PoseModel poseModel, const bool gpuResize) :
        mPoseModel{poseModel},
        mGpuResize{gpuResize},
        pInputImageCuda{nullptr},
        pInputImageReorderedCuda{nullptr},
        pOutputImageCuda{nullptr},
        pInputMaxSize{0ull},
        pOutputMaxSize{0ull}
    {
        #ifndef USE_CUDA
            if (mGpuResize)
                error("You need to compile OpenPose with CUDA support in order to use GPU resize.",
                    __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    CvMatToOpInput::~CvMatToOpInput()
    {
        try
        {
            #ifdef USE_CUDA
                if (mGpuResize)
                {
                    // Free temporary memory
                    cudaFree(pInputImageCuda);
                    cudaFree(pOutputImageCuda);
                    cudaFree(pInputImageReorderedCuda);
                }
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Array<float>> CvMatToOpInput::createArray(
        const cv::Mat& cvInputData, const std::vector<double>& scaleInputToNetInputs,
        const std::vector<Point<int>>& netInputSizes)
    {
        try
        {
            // Sanity checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.channels() != 3)
                error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
            if (scaleInputToNetInputs.size() != netInputSizes.size())
                error("scaleInputToNetInputs.size() != netInputSizes.size().", __LINE__, __FUNCTION__, __FILE__);
            // inputNetData - Reescale keeping aspect ratio and transform to float the input deep net image
            const auto numberScales = (int)scaleInputToNetInputs.size();
            std::vector<Array<float>> inputNetData(numberScales);
            for (auto i = 0u ; i < inputNetData.size() ; i++)
            {
                // CPU version (faster if #Gpus <= 3 and relatively small images)
                if (!mGpuResize)
                {
                    cv::Mat frameWithNetSize;
                    resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
                    // Fill inputNetData[i]
                    inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                    uCharCvMatToFloatPtr(
                        inputNetData[i].getPtr(), frameWithNetSize, (mPoseModel == PoseModel::BODY_19N ? 2 : 1));

                    // // OpenCV equivalent
                    // const auto scale = 1/255.;
                    // const cv::Scalar mean{128,128,128};
                    // const cv::Size outputSize{netInputSizes[i].x, netInputSizes[i].y};
                    // // cv::Mat cvMat;
                    // cv::dnn::blobFromImage(
                    //     // frameWithNetSize, cvMat, scale, outputSize, mean);
                    //     frameWithNetSize, inputNetData[i].getCvMat(), scale, outputSize, mean);
                    // // log(cv::norm(cvMat - inputNetData[i].getCvMat())); // ~0.25
                }
                // CUDA version (if #Gpus > n)
                else
                {
                    // Note: This version reduces the global accuracy about 0.1%, so it is disabled for now
                    error("This version reduces the global accuracy about 0.1%, so it is disabled for now.",
                        __LINE__, __FUNCTION__, __FILE__);
                    #ifdef USE_CUDA
                        // (Re)Allocate temporary memory
                        const unsigned int inputImageSize = 3 * cvInputData.rows * cvInputData.cols;
                        const unsigned int outputImageSize = 3 * netInputSizes[i].x * netInputSizes[i].y;
                        if (pInputMaxSize < inputImageSize)
                        {
                            pInputMaxSize = inputImageSize;
                            // Free temporary memory
                            cudaFree(pInputImageCuda);
                            cudaFree(pInputImageReorderedCuda);
                            // Re-allocate memory
                            cudaMalloc((void**)&pInputImageCuda, sizeof(unsigned char) * inputImageSize);
                            cudaMalloc((void**)&pInputImageReorderedCuda, sizeof(float) * inputImageSize);
                        }
                        if (pOutputMaxSize < outputImageSize)
                        {
                            pOutputMaxSize = outputImageSize;
                            // Free temporary memory
                            cudaFree(pOutputImageCuda);
                            // Re-allocate memory
                            cudaMalloc((void**)&pOutputImageCuda, sizeof(float) * outputImageSize);
                        }
                        // Copy image to GPU
                        cudaMemcpy(
                            pInputImageCuda, cvInputData.data, sizeof(unsigned char) * inputImageSize,
                            cudaMemcpyHostToDevice);
                        // Resize image on GPU
                        reorderAndNormalize(
                            pInputImageReorderedCuda, pInputImageCuda, cvInputData.cols, cvInputData.rows, 3);
                        resizeAndPadRbgGpu(
                            pOutputImageCuda, pInputImageReorderedCuda, cvInputData.cols, cvInputData.rows,
                            netInputSizes[i].x, netInputSizes[i].y, (float)scaleInputToNetInputs[i]);
                        // Copy back to CPU
                        inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                        cudaMemcpy(
                            inputNetData[i].getPtr(), pOutputImageCuda, sizeof(float) * outputImageSize,
                            cudaMemcpyDeviceToHost);
                    #else
                        error("You need to compile OpenPose with CUDA support in order to use GPU resize.",
                            __LINE__, __FUNCTION__, __FILE__);
                    #endif
                }
            }
            return inputNetData;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
