#include <openpose/wrapper/wrapperStructFace.hpp>

namespace op
{
    WrapperStructFace::WrapperStructFace(
        const bool enable_, const Detector detector_, const Point<int>& netInputSize_, const RenderMode renderMode_,
        const float alphaKeypoint_, const float alphaHeatMap_, const float renderThreshold_) :
        enable{enable_},
        detector{detector_},
        netInputSize{netInputSize_},
        renderMode{renderMode_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_},
        renderThreshold{renderThreshold_}
    {
    }

#ifdef USE_CUDA
    const RenderMode WrapperStructFace::FACE_DEFAULT_RENDER_MODE = RenderMode::Gpu;
#else
    const RenderMode WrapperStructFace::FACE_DEFAULT_RENDER_MODE = RenderMode::Cpu;
#endif
}
