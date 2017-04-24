#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/experimental/hands/handsRenderer.hpp"

namespace op
{
    namespace experimental
    {
        HandsRenderer::HandsRenderer(const cv::Size& frameSize) :
            Renderer{(unsigned long long)(frameSize.area() * 3)},
            mFrameSize{frameSize}
        {
            error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                  __LINE__, __FUNCTION__, __FILE__);
        }

        HandsRenderer::~HandsRenderer()
        {
        }

        void HandsRenderer::initializationOnThread()
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

        void HandsRenderer::renderHands(Array<float>& outputData, const Array<float>& hands)
        {
            try
            {
                UNUSED(outputData);
                UNUSED(hands);
                error("Hands code is not ready yet. A first beta version will be included in around 1-2 months. Please, set extractAndRenderHands = false in the OpenPose wrapper.",
                      __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    }
}
