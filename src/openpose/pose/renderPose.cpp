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
                // Array<float> --> cv::Mat
                auto frame = frameArray.getCvMat();

                // Background
                if (!blendOriginalFrame)
                    frame.setTo(0.f); // [0-255]

                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = POSE_BODY_PART_PAIRS_RENDER[(int)poseModel];

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, POSE_COLORS[(int)poseModel], thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, renderThreshold);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
