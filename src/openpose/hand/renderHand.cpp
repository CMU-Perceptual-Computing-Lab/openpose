#include <openpose/hand/handParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/hand/renderHand.hpp>

namespace op
{
    void renderHandKeypointsCpu(Array<float>& frameArray, const std::array<Array<float>, 2>& handKeypoints,
                                const float renderThreshold)
    {
        try
        {
            // Parameters
            const auto thicknessCircleRatio = 1.f/50.f;
            const auto thicknessLineRatioWRTCircle = 0.75f;
            const auto& pairs = HAND_PAIRS_RENDER;
            const auto& scales = HAND_SCALES_RENDER;
            // Render keypoints
            if (!frameArray.empty())
                renderKeypointsCpu(frameArray, handKeypoints[0], pairs, HAND_COLORS_RENDER, thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, scales, renderThreshold);
            if (!frameArray.empty())
                renderKeypointsCpu(frameArray, handKeypoints[1], pairs, HAND_COLORS_RENDER, thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, scales, renderThreshold);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
