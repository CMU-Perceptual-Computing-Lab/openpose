#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <openpose/face/renderFace.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/face/faceGpuRenderer.hpp>

namespace op
{
    FaceGpuRenderer::FaceGpuRenderer(const float renderThreshold, const float alphaKeypoint,
                                     const float alphaHeatMap) :
        GpuRenderer{renderThreshold, alphaKeypoint, alphaHeatMap}
    {
    }

    FaceGpuRenderer::~FaceGpuRenderer()
    {
        try
        {
            // Free CUDA pointers - Note that if pointers are 0 (i.e. nullptr), no operation is performed.
            #ifdef USE_CUDA
                cudaFree(pGpuFace);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceGpuRenderer::initializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // GPU memory allocation for rendering
            #ifdef USE_CUDA
                cudaMalloc((void**)(&pGpuFace), POSE_MAX_PEOPLE * FACE_NUMBER_PARTS * 3 * sizeof(float));
            #endif
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void FaceGpuRenderer::renderFace(Array<float>& outputData, const Array<float>& faceKeypoints)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            // GPU rendering
            #ifdef USE_CUDA
                const auto elementRendered = spElementToRender->load(); // I prefer std::round(T&) over intRound(T) for std::atomic
                const auto numberPeople = faceKeypoints.getSize(0);
                const Point<int> frameSize{outputData.getSize(2), outputData.getSize(1)};
                if (numberPeople > 0 && elementRendered == 0)
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr(), outputData.getVolume());
                    // Draw faceKeypoints
                    cudaMemcpy(pGpuFace, faceKeypoints.getConstPtr(),
                               faceKeypoints.getSize(0) * FACE_NUMBER_PARTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice);
                    renderFaceKeypointsGpu(*spGpuMemory, frameSize, pGpuFace, faceKeypoints.getSize(0),
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
