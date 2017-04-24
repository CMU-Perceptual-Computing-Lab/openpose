#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/experimental/hands/handsRenderGpu.hpp"

namespace op
{
    void renderHandsGpu(float* framePtr, const cv::Size& frameSize, const float* const handsPtr, const int numberHands, const float alphaColorToAdd)
    {
        try
        {
            error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                  __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
