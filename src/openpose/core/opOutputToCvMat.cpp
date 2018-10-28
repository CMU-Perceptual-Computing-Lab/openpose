#include <openpose/utilities/openCv.hpp>
#include <openpose/core/opOutputToCvMat.hpp>

namespace op
{
    cv::Mat OpOutputToCvMat::formatToCvMat(const Array<float>& outputData) const
    {
        try
        {
            // Sanity check
            if (outputData.empty())
                error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);
            // outputData to cvMat
            cv::Mat cvMat;
            outputData.getConstCvMat().convertTo(cvMat, CV_8UC3);
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
