#include <openpose/pose/renderPose.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/poseCpuRenderer.hpp>

namespace op
{
    PoseCpuRenderer::PoseCpuRenderer(const PoseModel poseModel, const float renderThreshold,
                                     const bool blendOriginalFrame, const float alphaKeypoint,
                                     const float alphaHeatMap, const unsigned int elementToRender) :
        Renderer{renderThreshold, alphaKeypoint, alphaHeatMap, blendOriginalFrame, elementToRender,
                 getNumberElementsToRender(poseModel)}, // mNumberElementsToRender
        PoseRenderer{poseModel}
    {
    }

    std::pair<int, std::string> PoseCpuRenderer::renderPose(Array<float>& outputData,
                                                            const Array<float>& poseKeypoints,
                                                            const float scaleInputToOutput,
                                                            const float scaleNetToOutput)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            // CPU rendering
            const auto elementRendered = spElementToRender->load();
            std::string elementRenderedName;
            // Draw poseKeypoints
            if (elementRendered == 0)
            {
                // Rescale keypoints to output size
                auto poseKeypointsRescaled = poseKeypoints.clone();
                scaleKeypoints(poseKeypointsRescaled, scaleInputToOutput);
                // Render keypoints
                renderPoseKeypointsCpu(outputData, poseKeypointsRescaled, mPoseModel, mRenderThreshold,
                                       mBlendOriginalFrame);
            }
            // Draw heat maps / PAFs
            else
            {
                UNUSED(scaleNetToOutput);
                error("CPU rendering only available for drawing keypoints, no heat maps nor PAFs.",
                      __LINE__, __FUNCTION__, __FILE__);
            }
            // Return result
            return std::make_pair(elementRendered, elementRenderedName);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(-1, "");
        }
    }
}
