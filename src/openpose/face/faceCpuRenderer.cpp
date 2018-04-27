#include <openpose/face/renderFace.hpp>
#include <openpose/face/faceCpuRenderer.hpp>

namespace op
{
    FaceCpuRenderer::FaceCpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        Renderer{renderThreshold, alphaKeypoint, alphaHeatMap}
    {
    }

    void FaceCpuRenderer::renderFaceInherited(Array<float>& outputData, const Array<float>& faceKeypoints)
    {
        try
        {
            // CPU rendering
            renderFaceKeypointsCpu(outputData, faceKeypoints, mRenderThreshold);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
