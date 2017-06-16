#include <openpose/experimental/hand/handParameters.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/experimental/hand/renderHand.hpp>

namespace op
{
    const std::vector<float> COLORS{HAND_COLORS_RENDER};

    void renderHandKeypointsCpu(Array<float>& frameArray, const Array<float>& handKeypoints)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Parameters
                const auto thicknessCircleRatio = 1.f/200.f;
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = HAND_PAIRS_RENDER;

                // Render keypoints
                renderKeypointsCpu(frameArray, handKeypoints, pairs, COLORS, thicknessCircleRatio, thicknessLineRatioWRTCircle);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
