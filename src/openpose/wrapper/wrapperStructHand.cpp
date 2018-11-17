#include <openpose/wrapper/wrapperStructHand.hpp>

namespace op
{
    WrapperStructHand::WrapperStructHand(
        const bool enable_, const Point<int>& netInputSize_, const int scalesNumber_, const float scaleRange_,
        const bool tracking_, const RenderMode renderMode_, const float alphaKeypoint_, const float alphaHeatMap_,
        const float renderThreshold_) :
        enable{enable_},
        netInputSize{netInputSize_},
        scalesNumber{scalesNumber_},
        scaleRange{scaleRange_},
        tracking{tracking_},
        renderMode{renderMode_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_},
        renderThreshold{renderThreshold_}
    {
    }
}
