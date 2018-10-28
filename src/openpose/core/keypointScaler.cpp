#include <openpose/utilities/keypoint.hpp>
#include <openpose/core/keypointScaler.hpp>

namespace op
{
    Rectangle<float> getScaleAndOffset(const ScaleMode scaleMode, const double scaleInputToOutput,
                                       const double scaleNetToOutput, const Point<int>& producerSize)
    {
        try
        {
            // OutputResolution
            if (scaleMode == ScaleMode::OutputResolution)
                return Rectangle<float>{0.f, 0.f, float(scaleInputToOutput), float(scaleInputToOutput)};
            // NetOutputResolution
            else if (scaleMode == ScaleMode::NetOutputResolution)
                return Rectangle<float>{0.f, 0.f, float(1./scaleNetToOutput),
                                        float(1./scaleNetToOutput)};
            // [0,1]
            else if (scaleMode == ScaleMode::ZeroToOne)
                return Rectangle<float>{0.f, 0.f, 1.f / ((float)producerSize.x - 1.f),
                                        1.f / ((float)producerSize.y - 1.f)};
            // [-1,1]
            else if (scaleMode == ScaleMode::PlusMinusOne)
                return Rectangle<float>{-1.f, -1.f, 2.f / ((float)producerSize.x - 1.f),
                                        2.f / ((float)producerSize.y - 1.f)};
            // InputResolution
            else if (scaleMode == ScaleMode::InputResolution)
                return Rectangle<float>{0.f, 0.f, 1.f, 1.f};
            // Unknown
            error("Unknown ScaleMode selected.", __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    KeypointScaler::KeypointScaler(const ScaleMode scaleMode) :
        mScaleMode{scaleMode}
    {
    }


    KeypointScaler::~KeypointScaler()
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
                // Get scale and offset
                const auto scaleAndOffset = getScaleAndOffset(mScaleMode, scaleInputToOutput, scaleNetToOutput,
                                                              producerSize);
                // Only scaling
                if (scaleAndOffset.x == 0 && scaleAndOffset.y == 0)
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints2d(arrayToScale, scaleAndOffset.width, scaleAndOffset.height);
                // Scaling + offset
                else
                    for (auto& arrayToScale : arrayToScalesToScale)
                        scaleKeypoints2d(arrayToScale, scaleAndOffset.width, scaleAndOffset.height,
                                         scaleAndOffset.x, scaleAndOffset.y);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void KeypointScaler::scale(std::vector<std::vector<std::array<float,3>>>& poseCandidates,
                               const double scaleInputToOutput, const double scaleNetToOutput,
                               const Point<int>& producerSize) const
    {
        try
        {
            if (mScaleMode != ScaleMode::InputResolution)
            {
                // Get scale and offset
                const auto scaleAndOffset = getScaleAndOffset(mScaleMode, scaleInputToOutput, scaleNetToOutput,
                                                              producerSize);
                // Only scaling
                if (scaleAndOffset.x == 0 && scaleAndOffset.y == 0)
                {
                    for (auto& partCandidates : poseCandidates)
                    {
                        for (auto& candidate : partCandidates)
                        {
                            candidate[0] *= scaleAndOffset.width;
                            candidate[1] *= scaleAndOffset.height;
                        }
                    }
                }
                // Scaling + offset
                else
                {
                    for (auto& partCandidates : poseCandidates)
                    {
                        for (auto& candidate : partCandidates)
                        {
                            candidate[0] = candidate[0]*scaleAndOffset.width + scaleAndOffset.x;
                            candidate[1] = candidate[1]*scaleAndOffset.height + scaleAndOffset.y;
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
