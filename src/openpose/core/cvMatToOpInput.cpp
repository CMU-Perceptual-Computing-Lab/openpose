#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpInput.hpp>

namespace op
{
    CvMatToOpInput::CvMatToOpInput(const PoseModel poseModel) :
        mPoseModel{poseModel}
    {
    }

    std::vector<Array<float>> CvMatToOpInput::createArray(const cv::Mat& cvInputData,
                                                          const std::vector<double>& scaleInputToNetInputs,
                                                          const std::vector<Point<int>>& netInputSizes) const
    {
        try
        {
            // Security checks
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
                inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
                std::vector<double> scaleRatios(numberScales, 1.f);
                const cv::Mat frameWithNetSize = resizeFixedAspectRatio(cvInputData, scaleInputToNetInputs[i],
                                                                        netInputSizes[i]);
                // Fill inputNetData[i]
                uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize, (mPoseModel == PoseModel::BODY_19N ? 2 : 1));
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
