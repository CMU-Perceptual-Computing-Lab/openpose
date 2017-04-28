#include <cuda.h>
#include <cuda_runtime_api.h>
#include "openpose/experimental/hand/handParameters.hpp"
#include "openpose/experimental/hand/handRenderGpu.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/experimental/hand/handRenderer.hpp"

namespace op
{
    namespace experimental
    {
        HandRenderer::HandRenderer(const cv::Size& frameSize) :
            Renderer{(unsigned long long)(frameSize.area() * 3)},
            mFrameSize{frameSize}
        {
        }

        HandRenderer::~HandRenderer()
        {
            try
            {
                // Free CUDA pointers - Note that if pointers are 0 (i.e. nullptr), no operation is performed.
                cudaFree(pGpuHands);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void HandRenderer::initializationOnThread()
        {
            try
            {
                Renderer::initializationOnThread();
                cudaMalloc((void**)(&pGpuHands), 2*HAND_NUMBER_PARTS * 3 * sizeof(float) );
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void HandRenderer::renderHands(Array<float>& outputData, const Array<float>& handKeyPoints)
        {
            try
            {
                if (!handKeyPoints.empty())
                {
                    cpuToGpuMemoryIfNotCopiedYet(outputData.getPtr());
                    cudaMemcpy(pGpuHands, handKeyPoints.getConstPtr(), 2*HAND_NUMBER_PARTS*3 * sizeof(float), cudaMemcpyHostToDevice);
                    renderHandsGpu(*spGpuMemoryPtr, mFrameSize, pGpuHands, handKeyPoints.getSize(0));
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
