#include <openpose/face/faceRenderer.hpp>
#include <openpose/face/renderFace.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    void FaceRenderer::renderFace(Array<float>& outputData, const Array<float>& faceKeypoints,
                                  const float scaleInputToOutput)
    {
        try
        {
            // Sanity check
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            // Rescale keypoints to output size
            auto faceKeypointsRescaled = faceKeypoints.clone();
            scaleKeypoints(faceKeypointsRescaled, scaleInputToOutput);
            // CPU/GPU rendering
            renderFaceInherited(outputData, faceKeypointsRescaled);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
