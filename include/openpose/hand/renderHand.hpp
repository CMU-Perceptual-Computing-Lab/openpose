#ifndef OPENPOSE_HAND_GPU_HAND_RENDER_HPP
#define OPENPOSE_HAND_GPU_HAND_RENDER_HPP

#include <array>
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include "handParameters.hpp"

namespace op
{
    void renderHandKeypointsCpu(Array<float>& frameArray, const std::array<Array<float>, 2>& handKeypoints);

    void renderHandKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const handsPtr, const int numberHands,
                                const float alphaColorToAdd = HAND_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_HAND_GPU_HAND_RENDER_HPP
