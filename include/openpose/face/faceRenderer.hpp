#ifndef OPENPOSE_FACE_FACE_RENDERER_HPP
#define OPENPOSE_FACE_FACE_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    class OP_API FaceRenderer : public Renderer
    {
    public:
        FaceRenderer(const Point<int>& frameSize, const float renderThreshold,
                     const float alphaKeypoint = FACE_DEFAULT_ALPHA_KEYPOINT,
                     const float alphaHeatMap = FACE_DEFAULT_ALPHA_HEAT_MAP,
                     const RenderMode renderMode = RenderMode::Cpu);

        ~FaceRenderer();

        void initializationOnThread();

        void renderFace(Array<float>& outputData, const Array<float>& faceKeypoints);

    private:
        const float mRenderThreshold;
        const Point<int> mFrameSize;
        const RenderMode mRenderMode;
        float* pGpuFace; // GPU aux memory

        void renderFaceCpu(Array<float>& outputData, const Array<float>& faceKeypoints);

        void renderFaceGpu(Array<float>& outputData, const Array<float>& faceKeypoints);

        DELETE_COPY(FaceRenderer);
    };
}

#endif // OPENPOSE_FACE_FACE_RENDERER_HPP
