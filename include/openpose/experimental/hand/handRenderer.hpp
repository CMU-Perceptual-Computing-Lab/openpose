#ifndef OPENPOSE_HAND_HAND_RENDERER_HPP
#define OPENPOSE_HAND_HAND_RENDERER_HPP

#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    namespace experimental
    {
        class HandRenderer : public Renderer
        {
        public:
            explicit HandRenderer(const Point<int>& frameSize);

            ~HandRenderer();

            void initializationOnThread();

            void renderHands(Array<float>& outputData, const Array<float>& handKeypoints);

        private:
            const Point<int> mFrameSize;
            float* pGpuHands;           // GPU aux memory

            DELETE_COPY(HandRenderer);
        };
    }
}

#endif // OPENPOSE_HAND_HAND_RENDERER_HPP
