#ifndef OPENPOSE_HAND_HAND_RENDERER_HPP
#define OPENPOSE_HAND_HAND_RENDERER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API HandRenderer
    {
    public:
        virtual void initializationOnThread(){};

        virtual void renderHand(Array<float>& outputData, const std::array<Array<float>, 2>& handKeypoints) = 0;
    };
}

#endif // OPENPOSE_HAND_HAND_RENDERER_HPP
