#include <openpose/wrapper/wrapperStructFace.hpp>

namespace op
{
    WrapperStructFace::WrapperStructFace(const bool enable_, const Point<int>& netInputSize_, const bool renderOutput_, const float alphaKeypoint_,
                          				 const float alphaHeatMap_) :
        enable{enable_},
        netInputSize{netInputSize_},
        renderOutput{renderOutput_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_}
    {
    }
}
