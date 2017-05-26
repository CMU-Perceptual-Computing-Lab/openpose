#ifndef OPENPOSE__FACE__FACE_RENDERER_HPP
#define OPENPOSE__FACE__FACE_RENDERER_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/array.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    namespace experimental
    {
        class FaceRenderer : public Renderer
        {
        public:
            explicit FaceRenderer(const cv::Size& frameSize);

            ~FaceRenderer();

            void initializationOnThread();

            void renderFace(Array<float>& outputData, const Array<float>& faceKeyPoints);

        private:
            const cv::Size mFrameSize;
            float* pGpuFace;           // GPU aux memory

            DELETE_COPY(FaceRenderer);
        };
    }
}

#endif // OPENPOSE__FACE__FACE_RENDERER_HPP
