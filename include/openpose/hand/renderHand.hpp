#ifndef OPENPOSE_HAND_GPU_HAND_RENDER_HPP
#define OPENPOSE_HAND_GPU_HAND_RENDER_HPP

#include <array>
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/macros.hpp>
#include "handParameters.hpp"

namespace op
{
    OP_API void renderHandKeypointsCpu(Array<float>& frameArray, const std::array<Array<float>, 2>& handKeypoints,
                                       const float renderThreshold);

    OP_API void renderHandKeypointsGpu(float* framePtr, const Point<int>& frameSize, const float* const handsPtr,
                                       const int numberHands, const float renderThreshold,
                                       const float alphaColorToAdd = HAND_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_HAND_GPU_HAND_RENDER_HPP
