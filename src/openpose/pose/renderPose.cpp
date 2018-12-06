#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/renderPose.hpp>

namespace op
{
    void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel,
                                const float renderThreshold, const bool blendOriginalFrame)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Background
                if (!blendOriginalFrame)
                    frameArray.getCvMat().setTo(0.f); // [0-255]

                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = getPoseBodyPartPairsRender(poseModel);
                const auto& poseScales = getPoseScales(poseModel);

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, getPoseColors(poseModel), thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, poseScales, renderThreshold);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
