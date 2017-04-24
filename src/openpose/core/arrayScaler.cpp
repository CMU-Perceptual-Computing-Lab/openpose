#include "openpose/core/scalePose.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/arrayScaler.hpp"

namespace op
{
    ArrayScaler::ArrayScaler(const ScaleMode scalePose) :
        mScaleMode{scalePose}
    {
    }

    void ArrayScaler::scale(Array<float>& array, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const
    {
        try
        {
            std::vector<Array<float>> arrays{array};
            scale(arrays, scaleInputToOutput, scaleNetToOutput, producerSize);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void ArrayScaler::scale(std::vector<Array<float>>& arrays, const double scaleInputToOutput, const double scaleNetToOutput, const cv::Size& producerSize) const
    {
        try
        {
            if (mScaleMode != ScaleMode::OutputResolution)
            {
                // InputResolution
                if (mScaleMode == ScaleMode::InputResolution)
                    for (auto& array : arrays)
                        scalePose(array, 1./scaleInputToOutput);
                // NetOutputResolution
                else if (mScaleMode == ScaleMode::NetOutputResolution)
                    for (auto& array : arrays)
                        scalePose(array, 1./scaleNetToOutput);
                // [0,1]
                else if (mScaleMode == ScaleMode::ZeroToOne)
                {
                    const auto scale = 1./scaleInputToOutput;
                    const auto scaleX = scale / ((double)producerSize.width - 1.);
                    const auto scaleY = scale / ((double)producerSize.height - 1.);
                    for (auto& array : arrays)
                        scalePose(array, scaleX, scaleY);
                }
                // [-1,1]
                else if (mScaleMode == ScaleMode::PlusMinusOne)
                {
                    const auto scale = 2./scaleInputToOutput;
                    const auto scaleX = (scale / ((double)producerSize.width - 1.));
                    const auto scaleY = (scale / ((double)producerSize.height - 1.));
                    const auto offset = -1.;
                    for (auto& array : arrays)
                        scalePose(array, scaleX, scaleY, offset, offset);
                }
                // Unknown
                else
                    error("Unknown ScaleMode selected.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
