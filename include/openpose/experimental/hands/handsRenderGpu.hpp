#ifndef OPENPOSE__HANDS__GPU_HANDS_RENDER_HPP
#define OPENPOSE__HANDS__GPU_HANDS_RENDER_HPP

#include <opencv2/core/core.hpp>
#include "handsParameters.hpp"

namespace op
{
	void renderHandsGpu(float* framePtr, const cv::Size& frameSize, const float* const handsPtr, const int numberHands, const float alphaColorToAdd = HANDS_DEFAULT_ALPHA_HANDS);
}

#endif // OPENPOSE__HANDS__GPU_HANDS_RENDER_HPP
