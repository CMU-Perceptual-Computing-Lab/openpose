// #include <opencv2/opencv.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
// #include <nppi_geometry_transforms.h>
// #include <nppdefs.h>
#include <openpose/net/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpInput.hpp>

namespace op
{
    CvMatToOpInput::CvMatToOpInput(const PoseModel poseModel) :
        mPoseModel{poseModel}
    {
    }

    CvMatToOpInput::~CvMatToOpInput()
    {
    }

    std::vector<Array<float>> CvMatToOpInput::createArray(
        const cv::Mat& cvInputData, const std::vector<double>& scaleInputToNetInputs,
        const std::vector<Point<int>>& netInputSizes) const
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
                // const auto REPS = 1;
                // double timeNormalize0 = 0.;
                // double timeNormalize1 = 0.;
                // double timeNormalize2 = 0.;
                // //// warm up code /////
                // OP_PROFILE_INIT(1);
                // cv::Mat frameWithNetSize;
                // resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
                // // Fill inputNetData[i]
                // inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                // uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize,
                //                      (mPoseModel == PoseModel::BODY_19N ? 2 : 1));
                // OP_PROFILE_END(timeNormalize0, 1e3, 5);
                //// warm up code /////

                // OP_PROFILE_INIT(REPS);
                cv::Mat frameWithNetSize;
                resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
                // Fill inputNetData[i]
                inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                uCharCvMatToFloatPtr(
                    inputNetData[i].getPtr(), frameWithNetSize, (mPoseModel == PoseModel::BODY_19N ? 2 : 1));
                // OP_PROFILE_END(timeNormalize1, 1e3, REPS);


                //resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);

                /* 1) Allocate memory on GPU
                   2) copy cvINPUTData to GPU
                   3) resize image on GPU using Nvidia performance primatives
                   4) Copy back to CPU
                 */
                // // allocate memory on gpu
                // unsigned char *CUDA_input_image = 0;
                // float *CUDA_input_image_reord = 0;
                // float *CUDA_output_image = 0;
                // float *output_image;
                // int input_image_size = cvInputData.rows * cvInputData.cols;
                // int output_image_size = netInputSizes[i].x * netInputSizes[i].y;

                // //output_image = (float *) malloc(sizeof(unsigned char) * output_image_size * 3);
                // cudaMalloc((void**)&CUDA_input_image, sizeof(unsigned char) * input_image_size * 3);
                // cudaMalloc((void**)&CUDA_input_image_reord, sizeof(float) * input_image_size * 3);
                // cudaMalloc((void**)&CUDA_output_image, sizeof(float) * output_image_size * 3);

                // // copy image to GPU
                // //inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                // cudaMemcpy(CUDA_input_image, cvInputData.data,
                //             sizeof(unsigned char) * 3 * input_image_size,
                //             cudaMemcpyHostToDevice);

                // reorderAndCast(CUDA_input_image, CUDA_input_image_reord, cvInputData.cols, cvInputData.rows);

                // resizeAndMergeRGBGPU(CUDA_input_image_reord, CUDA_output_image, cvInputData.cols, cvInputData.rows, netInputSizes[i].x, netInputSizes[i].y, scaleInputToNetInputs[i]);
                // // copy back to CPU
                // inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});

                // cudaMemcpy(inputNetData[i].getPtr(),
                //            CUDA_output_image,
                //            sizeof(float) * 3 * output_image_size,
                //            cudaMemcpyDeviceToHost);
                // cudaFree(CUDA_input_image);
                // cudaFree(CUDA_output_image);
                // cudaFree(CUDA_input_image_reord);

                // log("  Res_Original=" + std::to_string(timeNormalize1) + "ms");



                // Fill inputNetData[i]
                // inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                // uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize,
                //                      (mPoseModel == PoseModel::BODY_19N ? 2 : 1));
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
            return inputNetData;
        }

        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
