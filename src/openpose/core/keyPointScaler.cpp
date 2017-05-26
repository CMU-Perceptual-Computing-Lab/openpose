#include <openpose/core/scaleKeyPoints.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/keyPointScaler.hpp>

namespace op
{
    KeyPointScaler::KeyPointScaler(const ScaleMode scaleMode) :
        mScaleMode{scaleMode}
    {
    }

    void KeyPointScaler::scale(Array<float>& arrayToScale, const float scaleInputToOutput, const float scaleNetToOutput, const cv::Size& producerSize) const
    {
        try
        {
            std::vector<Array<float>> arrayToScalesToScale{arrayToScale};
            scale(arrayToScalesToScale, scaleInputToOutput, scaleNetToOutput, producerSize);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void KeyPointScaler::scale(std::vector<Array<float>>& arrayToScalesToScale, const float scaleInputToOutput, const float scaleNetToOutput, const cv::Size& producerSize) const
    {
        try
        {
            if (mScaleMode != ScaleMode::OutputResolution)
            {
                // InputResolution
                if (mScaleMode == ScaleMode::InputResolution)
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeyPoints(arrayToScale, 1.f/scaleInputToOutput);
                // NetOutputResolution
                else if (mScaleMode == ScaleMode::NetOutputResolution)
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeyPoints(arrayToScale, 1.f/scaleNetToOutput);
                // [0,1]
                else if (mScaleMode == ScaleMode::ZeroToOne)
                {
                    const auto scale = 1.f/scaleInputToOutput;
                    const auto scaleX = scale / ((float)producerSize.width - 1.f);
                    const auto scaleY = scale / ((float)producerSize.height - 1.f);
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeyPoints(arrayToScale, scaleX, scaleY);
                }
                // [-1,1]
                else if (mScaleMode == ScaleMode::PlusMinusOne)
                {
                    const auto scale = 2.f/scaleInputToOutput;
                    const auto scaleX = (scale / ((float)producerSize.width - 1.f));
                    const auto scaleY = (scale / ((float)producerSize.height - 1.f));
                    const auto offset = -1.f;
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeyPoints(arrayToScale, scaleX, scaleY, offset, offset);
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
