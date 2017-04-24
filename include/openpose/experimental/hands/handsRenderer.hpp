#ifndef OPENPOSE__HANDS__HANDS_RENDERER_HPP
#define OPENPOSE__HANDS__HANDS_RENDERER_HPP

#include <opencv2/core/core.hpp>
#include "../../core/array.hpp"
#include "../../core/renderer.hpp"
#include "../../thread/worker.hpp"

namespace op
{
    namespace experimental
    {
        class HandsRenderer : public Renderer
        {
        public:
            explicit HandsRenderer(const cv::Size& frameSize);

            ~HandsRenderer();

            void initializationOnThread();

            void renderHands(Array<float>& outputData, const Array<float>& hands);

        private:
            const cv::Size mFrameSize;
            float* pGpuHands;           // GPU aux memory

            DELETE_COPY(HandsRenderer);
        };
    }
}

#endif // OPENPOSE__HANDS__HANDS_RENDERER_HPP
