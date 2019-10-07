#include <openpose/face/renderFace.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    void renderFaceKeypointsCpu(Array<float>& frameArray, const Array<float>& faceKeypoints,
                                const float renderThreshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.334f;
                const auto& pairs = FACE_PAIRS_RENDER;
                const auto& scales = FACE_SCALES_RENDER;

                // Render keypoints
                renderKeypointsCpu(frameArray, faceKeypoints, pairs, FACE_COLORS_RENDER, thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, scales, renderThreshold);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
