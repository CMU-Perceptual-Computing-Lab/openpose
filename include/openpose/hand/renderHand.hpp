#ifndef OPENPOSE_HAND_GPU_HAND_RENDER_HPP
#define OPENPOSE_HAND_GPU_HAND_RENDER_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handParameters.hpp>

namespace op
{
    OP_API void renderHandKeypointsCpu(
        Array<float>& frameArray, const std::array<Array<float>, 2>& handKeypoints, const float renderThreshold);

    void renderHandKeypointsGpu(
        float* framePtr, float* maxPtr, float* minPtr, float* scalePtr, const Point<unsigned int>& frameSize,
        const float* const handsPtr, const int numberHands, const float renderThreshold,
        const float alphaColorToAdd = HAND_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_HAND_GPU_HAND_RENDER_HPP
