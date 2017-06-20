#ifndef OPENPOSE_FACE_FACE_RENDERER_HPP
#define OPENPOSE_FACE_FACE_RENDERER_HPP

#include <openpose/core/array.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/thread/worker.hpp>
#include "faceParameters.hpp"

namespace op
{
    class FaceRenderer : public Renderer
    {
    public:
        explicit FaceRenderer(const Point<int>& frameSize, const float alphaKeypoint = FACE_DEFAULT_ALPHA_KEYPOINT,
                              const float alphaHeatMap = FACE_DEFAULT_ALPHA_HEAT_MAP, const RenderMode renderMode = RenderMode::Cpu);

        ~FaceRenderer();

        void initializationOnThread();

        void renderFace(Array<float>& outputData, const Array<float>& faceKeypoints);

    private:
        const Point<int> mFrameSize;
        const RenderMode mRenderMode;
        float* pGpuFace; // GPU aux memory

        void renderFaceCpu(Array<float>& outputData, const Array<float>& faceKeypoints);

        void renderFaceGpu(Array<float>& outputData, const Array<float>& faceKeypoints);

        DELETE_COPY(FaceRenderer);
    };
}

#endif // OPENPOSE_FACE_FACE_RENDERER_HPP
