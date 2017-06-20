#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/renderPose.hpp>

namespace op
{
    const std::vector<float> COCO_COLORS{POSE_COCO_COLORS_RENDER};
    const std::vector<float> MPI_COLORS{POSE_MPI_COLORS_RENDER};

    void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel, const bool blendOriginalFrame)
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
                const auto& colors = (poseModel == PoseModel::COCO_18 ? COCO_COLORS : MPI_COLORS);

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, colors, thicknessCircleRatio, thicknessLineRatioWRTCircle, POSE_RENDER_THRESHOLD);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
