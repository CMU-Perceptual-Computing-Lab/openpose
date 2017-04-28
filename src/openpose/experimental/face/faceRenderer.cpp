#include <cuda.h>
#include <cuda_runtime_api.h>
#include "openpose/experimental/face/faceParameters.hpp"
#include "openpose/experimental/face/faceRenderGpu.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/experimental/face/faceRenderer.hpp"

namespace op
{
    namespace experimental
    {
        FaceRenderer::FaceRenderer(const cv::Size& frameSize) :
            Renderer{(unsigned long long)(frameSize.area() * 3)},
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
                cudaMalloc((void**)(&pGpuFace), FACE_NUMBER_PARTS * 3 * sizeof(float) );
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void FaceRenderer::renderFace(Array<float>& outputData, const Array<float>& faceKeyPoints)
        {
            try
            {
                if (!faceKeyPoints.empty())
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr());
                    cudaMemcpy(pGpuFace, faceKeyPoints.getConstPtr(), FACE_NUMBER_PARTS*3 * sizeof(float), cudaMemcpyHostToDevice);
                    renderFaceGpu(*spGpuMemoryPtr, mFrameSize, pGpuFace, faceKeyPoints.getSize(0));
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
}
