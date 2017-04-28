#ifndef OPENPOSE__FACE__GPU_FACE_RENDER_HPP
#define OPENPOSE__FACE__GPU_FACE_RENDER_HPP

#include <opencv2/core/core.hpp>
#include "faceParameters.hpp"

namespace op
{
	void renderFaceGpu(float* framePtr, const cv::Size& frameSize, const float* const facePtr, const int numberFace, const float alphaColorToAdd = FACE_DEFAULT_ALPHA_FACE);
}

#endif // OPENPOSE__FACE__GPU_FACE_RENDER_HPP
