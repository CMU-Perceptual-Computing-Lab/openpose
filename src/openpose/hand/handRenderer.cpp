#include <openpose/utilities/keypoint.hpp>
#include <openpose/hand/handRenderer.hpp>

namespace op
{
    void HandRenderer::renderHand(Array<float>& outputData,
                                  const std::array<Array<float>, 2>& handKeypoints,
                                  const float scaleInputToOutput)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            if (handKeypoints[0].getSize(0) != handKeypoints[1].getSize(0))
                error("Wrong hand format: handKeypoints.getSize(0) != handKeypoints.getSize(1).",
                      __LINE__, __FUNCTION__, __FILE__);
            // Rescale keypoints to output size
            auto leftHandKeypointsRescaled = handKeypoints[0].clone();
            scaleKeypoints(leftHandKeypointsRescaled, scaleInputToOutput);
            auto rightHandKeypointsRescaled = handKeypoints[1].clone();
            scaleKeypoints(rightHandKeypointsRescaled, scaleInputToOutput);
            // CPU/GPU rendering
            renderHandInherited(
                outputData,
                std::array<Array<float>, 2>{leftHandKeypointsRescaled, rightHandKeypointsRescaled}
            );
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
