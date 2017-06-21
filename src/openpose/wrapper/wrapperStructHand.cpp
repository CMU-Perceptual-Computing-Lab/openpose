#include <openpose/wrapper/wrapperStructHand.hpp>

namespace op
{
    WrapperStructHand::WrapperStructHand(const bool enable_, const Point<int>& netInputSize_, const DetectionMode detectionMode_,
                                         const RenderMode renderMode_, const float alphaKeypoint_, const float alphaHeatMap_) :
        enable{enable_},
        netInputSize{netInputSize_},
        detectionMode{detectionMode_},
        renderMode{renderMode_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_}
    {
    }
}
