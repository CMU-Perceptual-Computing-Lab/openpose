#include <openpose/utilities/openCv.hpp>
#include <openpose/core/opOutputToCvMat.hpp>

namespace op
{
    cv::Mat OpOutputToCvMat::formatToCvMat(const Array<float>& outputData) const
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);
            // outputData to cvMat
            cv::Mat cvMat;
            const std::array<int, 3> outputResolution{outputData.getSize(2), outputData.getSize(1),
                                                      outputData.getSize(0)};
            floatPtrToUCharCvMat(cvMat, outputData.getConstPtr(), outputResolution);
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
