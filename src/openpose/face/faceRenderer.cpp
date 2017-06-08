#include <cuda.h>
#include <cuda_runtime_api.h>
#include <openpose/face/faceParameters.hpp>
#include <openpose/face/faceRenderGpu.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/face/faceRenderer.hpp>

namespace op
{
    FaceRenderer::FaceRenderer(const Point<int>& frameSize, const float alphaKeypoint, const float alphaHeatMap) :
        Renderer{(unsigned long long)(frameSize.area() * 3), alphaKeypoint, alphaHeatMap},
        mFrameSize{frameSize}
    {
    }

    FaceRenderer::~FaceRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e. nullptr), no operation is performed.
            cudaFree(pGpuFace);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceRenderer::initializationOnThread()
    {
        try
        {
            Renderer::initializationOnThread();
            cudaMalloc((void**)(&pGpuFace), POSE_MAX_PEOPLE * FACE_NUMBER_PARTS * 3 * sizeof(float));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceRenderer::renderFace(Array<float>& outputData, const Array<float>& faceKeypoints)
    {
        try
        {
            const auto elementRendered = spElementToRender->load(); // I prefer std::round(T&) over intRound(T) for std::atomic
            const auto numberPeople = faceKeypoints.getSize(0);
            // GPU rendering
            if (numberPeople > 0 && elementRendered == 0)
            {
                cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr());
                // Draw faceKeypoints
                cudaMemcpy(pGpuFace, faceKeypoints.getConstPtr(), faceKeypoints.getSize(0) * FACE_NUMBER_PARTS * 3 * sizeof(float), cudaMemcpyHostToDevice);
                renderFaceGpu(*spGpuMemoryPtr, mFrameSize, pGpuFace, faceKeypoints.getSize(0), getAlphaKeypoint());
                // CUDA check
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
            // GPU memory to CPU if last renderer
            gpuToCpuMemoryIfLastRenderer(outputData.getPtr());
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
