#ifndef OPENPOSE_FACE_FACE_CPU_RENDERER_HPP
#define OPENPOSE_FACE_FACE_CPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/face/faceRenderer.hpp>

namespace op
{
    class OP_API FaceCpuRenderer : public Renderer, public FaceRenderer
    {
    public:
        FaceCpuRenderer(const float renderThreshold, const float alphaKeypoint = FACE_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = FACE_DEFAULT_ALPHA_HEAT_MAP);

        virtual ~FaceCpuRenderer();

        void renderFaceInherited(Array<float>& outputData, const Array<float>& faceKeypoints);

        DELETE_COPY(FaceCpuRenderer);
    };
}

#endif // OPENPOSE_FACE_FACE_CPU_RENDERER_HPP
