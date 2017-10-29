#include <openpose/utilities/keypoint.hpp>
#include <openpose/core/keypointScaler.hpp>

namespace op
{
    KeypointScaler::KeypointScaler(const ScaleMode scaleMode) :
        mScaleMode{scaleMode}
    {
    }

    void KeypointScaler::scale(Array<float>& arrayToScale, const double scaleInputToOutput,
                               const double scaleNetToOutput, const Point<int>& producerSize) const
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

    void KeypointScaler::scale(std::vector<Array<float>>& arrayToScalesToScale, const double scaleInputToOutput,
                               const double scaleNetToOutput, const Point<int>& producerSize) const
    {
        try
        {
            if (mScaleMode != ScaleMode::InputResolution)
            {
                // OutputResolution
                if (mScaleMode == ScaleMode::OutputResolution)
                {
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints(arrayToScale, float(scaleInputToOutput));
                }
                // NetOutputResolution
                else if (mScaleMode == ScaleMode::NetOutputResolution)
                {
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints(arrayToScale, float(1./scaleNetToOutput));
                }
                // [0,1]
                else if (mScaleMode == ScaleMode::ZeroToOne)
                {
                    const auto scaleX = 1.f / ((float)producerSize.x - 1.f);
                    const auto scaleY = 1.f / ((float)producerSize.y - 1.f);
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints(arrayToScale, scaleX, scaleY);
                }
                // [-1,1]
                else if (mScaleMode == ScaleMode::PlusMinusOne)
                {
                    const auto scaleX = (2.f / ((float)producerSize.x - 1.f));
                    const auto scaleY = (2.f / ((float)producerSize.y - 1.f));
                    const auto offset = -1.f;
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints(arrayToScale, scaleX, scaleY, offset, offset);
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
