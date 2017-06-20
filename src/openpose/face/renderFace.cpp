#include <openpose/face/faceParameters.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/face/renderFace.hpp>

namespace op
{
    const std::vector<float> COLORS{FACE_COLORS_RENDER};

    void renderFaceKeypointsCpu(Array<float>& frameArray, const Array<float>& faceKeypoints)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.334f;
                const auto& pairs = FACE_PAIRS_RENDER;

                // Render keypoints
                renderKeypointsCpu(frameArray, faceKeypoints, pairs, COLORS, thicknessCircleRatio, thicknessLineRatioWRTCircle, FACE_RENDER_THRESHOLD);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
