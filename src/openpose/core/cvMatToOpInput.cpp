#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpInput.hpp>

namespace op
{
    Array<float> CvMatToOpInput::createArray(const cv::Mat& cvInputData,
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
            Array<float> inputNetData{{numberScales, 3, netInputSizes.at(0).y, netInputSizes.at(0).x}};
            std::vector<double> scaleRatios(numberScales, 1.f);
            const auto inputNetDataOffset = inputNetData.getVolume(1, 3);
            for (auto i = 0; i < numberScales; i++)
            {
                const cv::Mat frameWithNetSize = resizeFixedAspectRatio(cvInputData, scaleInputToNetInputs[i],
                                                                        netInputSizes[i]);
                // Fill inputNetData
                uCharCvMatToFloatPtr(inputNetData.getPtr() + i * inputNetDataOffset, frameWithNetSize, true);
            }
            return inputNetData;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }
}
