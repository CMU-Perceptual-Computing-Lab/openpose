#ifndef OPENPOSE_3D_POSE_TRIANGULATION_HPP
#define OPENPOSE_3D_POSE_TRIANGULATION_HPP

#include <opencv2/core/core.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    OP_API void reconstructArray(Array<float>& keypoints3D, const std::vector<Array<float>>& keypointsVector,
                                 const std::vector<cv::Mat>& matrixEachCamera);
}

#endif // OPENPOSE_3D_POSE_TRIANGULATION_HPP
