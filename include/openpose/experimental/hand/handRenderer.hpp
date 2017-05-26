#ifndef OPENPOSE__HAND__HAND_RENDERER_HPP
#define OPENPOSE__HAND__HAND_RENDERER_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/array.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    namespace experimental
    {
        class HandRenderer : public Renderer
        {
        public:
            explicit HandRenderer(const cv::Size& frameSize);

            ~HandRenderer();

            void initializationOnThread();

            void renderHands(Array<float>& outputData, const Array<float>& handKeyPoints);

        private:
            const cv::Size mFrameSize;
            float* pGpuHands;           // GPU aux memory

            DELETE_COPY(HandRenderer);
        };
    }
}

#endif // OPENPOSE__HAND__HAND_RENDERER_HPP
