#include <openpose/hand/renderHand.hpp>
#include <openpose/hand/handCpuRenderer.hpp>

namespace op
{
    HandCpuRenderer::HandCpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        Renderer{renderThreshold, alphaKeypoint, alphaHeatMap}
    {
    }

    void HandCpuRenderer::renderHand(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            if (handKeypoints[0].getSize(0) != handKeypoints[1].getSize(0))
                error("Wrong hand format: handKeypoints.getSize(0) != handKeypoints.getSize(1).",
                      __LINE__, __FUNCTION__, __FILE__);
            // CPU rendering
            renderHandKeypointsCpu(outputData, handKeypoints, mRenderThreshold);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
