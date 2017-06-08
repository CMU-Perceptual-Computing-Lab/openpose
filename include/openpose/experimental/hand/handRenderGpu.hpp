#ifndef OPENPOSE_HAND_GPU_HAND_RENDER_HPP
#define OPENPOSE_HAND_GPU_HAND_RENDER_HPP

#include <openpose/core/point.hpp>
#include "handParameters.hpp"

namespace op
{
	void renderHandsGpu(float* framePtr, const Point<int>& frameSize, const float* const handsPtr, const int numberHands, const float alphaColorToAdd = HAND_DEFAULT_ALPHA_KEYPOINT);
}

#endif // OPENPOSE_HAND_GPU_HAND_RENDER_HPP
