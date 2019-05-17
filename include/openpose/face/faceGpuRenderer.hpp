#ifndef OPENPOSE_FACE_FACE_GPU_RENDERER_HPP
#define OPENPOSE_FACE_FACE_GPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/gpuRenderer.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/face/faceRenderer.hpp>

namespace op
{
    class OP_API FaceGpuRenderer : public GpuRenderer, public FaceRenderer
    {
    public:
        FaceGpuRenderer(const float renderThreshold,
                        const float alphaKeypoint = FACE_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = FACE_DEFAULT_ALPHA_HEAT_MAP);

        virtual ~FaceGpuRenderer();

        void initializationOnThread();

        void renderFaceInherited(Array<float>& outputData, const Array<float>& faceKeypoints);

    private:
        float* pGpuFace; // GPU aux memory
        float* pMaxPtr; // GPU aux memory
        float* pMinPtr; // GPU aux memory
        float* pScalePtr; // GPU aux memory

        DELETE_COPY(FaceGpuRenderer);
    };
}

#endif // OPENPOSE_FACE_FACE_GPU_RENDERER_HPP
