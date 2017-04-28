#ifndef OPENPOSE__HAND__GPU_HAND_RENDER_HPP
#define OPENPOSE__HAND__GPU_HAND_RENDER_HPP

#include <opencv2/core/core.hpp>
#include "handParameters.hpp"

namespace op
{
	void renderHandsGpu(float* framePtr, const cv::Size& frameSize, const float* const handsPtr, const int numberHands, const float alphaColorToAdd = HAND_DEFAULT_ALPHA_HANDS);
}

#endif // OPENPOSE__HAND__GPU_HAND_RENDER_HPP
