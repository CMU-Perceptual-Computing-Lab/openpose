#include <openpose/face/faceGpuRenderer.hpp>
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/face/renderFace.hpp>
#include <openpose/gpu/cuda.hpp>

namespace op
{
    FaceGpuRenderer::FaceGpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        GpuRenderer{renderThreshold, alphaKeypoint, alphaHeatMap},
        pGpuFace{nullptr},
        pMaxPtr{nullptr},
        pMinPtr{nullptr},
        pScalePtr{nullptr}
    {
    }

    FaceGpuRenderer::~FaceGpuRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e., nullptr), no operation is performed.
            #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                if (pGpuFace != nullptr)
                {
                    cudaFree(pGpuFace);
                    pGpuFace = nullptr;
                }
                if (pMaxPtr != nullptr)
                {
                    cudaFree(pMaxPtr);
                    pMaxPtr = nullptr;
                }
                if (pMinPtr != nullptr)
                {
                    cudaFree(pMinPtr);
                    pMinPtr = nullptr;
                }
                if (pScalePtr != nullptr)
                {
                    cudaFree(pScalePtr);
                    pScalePtr = nullptr;
                }
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceGpuRenderer::initializationOnThread()
    {
        try
        {
            opLog("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // GPU memory allocation for rendering
            #ifdef USE_CUDA
                cudaMalloc((void**)(&pGpuFace), POSE_MAX_PEOPLE * FACE_NUMBER_PARTS * 3 * sizeof(float));
                cudaMalloc((void**)&pMaxPtr, sizeof(float) * 2 * FACE_NUMBER_PARTS);
                cudaMalloc((void**)&pMinPtr, sizeof(float) * 2 * FACE_NUMBER_PARTS);
                cudaMalloc((void**)&pScalePtr, sizeof(float) * FACE_NUMBER_PARTS);
            #endif
            opLog("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceGpuRenderer::renderFaceInherited(Array<float>& outputData, const Array<float>& faceKeypoints)
    {
        try
        {
            // GPU rendering
            #ifdef USE_CUDA
                // I prefer std::round(T&) over positiveIntRound(T) for std::atomic
                const auto elementRendered = spElementToRender->load();
                const auto numberPeople = faceKeypoints.getSize(0);
                const Point<int> frameSize{outputData.getSize(1), outputData.getSize(0)};
                if (numberPeople > 0 && elementRendered == 0)
                {
                    // Draw faceKeypoints
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr(), outputData.getVolume());
                    cudaMemcpy(pGpuFace, faceKeypoints.getConstPtr(),
                               faceKeypoints.getSize(0) * FACE_NUMBER_PARTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice);
                    renderFaceKeypointsGpu(
                        *spGpuMemory, pMaxPtr, pMinPtr, pScalePtr, frameSize, pGpuFace, faceKeypoints.getSize(0),
                        mRenderThreshold, getAlphaKeypoint());
                    // CUDA check
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                }
                // GPU memory to CPU if last renderer
                gpuToCpuMemoryIfLastRenderer(outputData.getPtr(), outputData.getVolume());
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(outputData);
                UNUSED(faceKeypoints);
                error("OpenPose must be compiled with the `USE_CUDA` macro definitions in order to run this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
