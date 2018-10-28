#include <openpose/hand/renderHand.hpp>
#include <openpose/hand/handCpuRenderer.hpp>

namespace op
{
    HandCpuRenderer::HandCpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        Renderer{renderThreshold, alphaKeypoint, alphaHeatMap}
    {
    }

    HandCpuRenderer::~HandCpuRenderer()
    {
    }

    void HandCpuRenderer::renderHandInherited(Array<float>& outputData,
                                              const std::array<Array<float>, 2>& handKeypoints)
    {
        try
        {
            // CPU rendering
            renderHandKeypointsCpu(
                outputData,
                handKeypoints,
                mRenderThreshold
            );
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
