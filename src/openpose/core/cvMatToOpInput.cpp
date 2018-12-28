// #include <opencv2/opencv.hpp>
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

    std::vector<Array<float>> CvMatToOpInput::createArray(const cv::Mat& cvInputData,
                                                          const std::vector<double>& scaleInputToNetInputs,
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
                cv::Mat frameWithNetSize;
                resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
                // Fill inputNetData[i]
                inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize,
                                     (mPoseModel == PoseModel::BODY_19N ? 2 : 1));
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
