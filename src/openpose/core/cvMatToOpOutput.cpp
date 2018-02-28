#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpOutput.hpp>

namespace op
{
    Array<float> CvMatToOpOutput::createArray(const cv::Mat& cvInputData, const double scaleInputToOutput, const Point<int>& outputResolution) const
    {
        try
        {
            // Security checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.channels() != 3)
                error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.cols <= 0 || cvInputData.rows <= 0)
                error("Input images has 0 area.", __LINE__, __FUNCTION__, __FILE__);
            if (outputResolution.x <= 0 || outputResolution.y <= 0)
                error("Output resolution has 0 area.", __LINE__, __FUNCTION__, __FILE__);
            // outputData - Reescale keeping aspect ratio and transform to float the output image
            const cv::Mat frameWithOutputSize = resizeFixedAspectRatio(cvInputData, scaleInputToOutput,
                                                                       outputResolution);
            Array<float> outputData({3, outputResolution.y, outputResolution.x});
            uCharCvMatToFloatPtr(outputData.getPtr(), frameWithOutputSize, false);
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
