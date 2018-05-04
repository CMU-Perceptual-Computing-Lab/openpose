#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API float getDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB);

    OP_API void averageKeypoints(Array<float>& keypointsA, const Array<float>& keypointsB, const int personA);

    OP_API void scaleKeypoints(Array<float>& keypoints, const float scale);

    OP_API void scaleKeypoints2d(Array<float>& keypoints, const float scaleX, const float scaleY);

    OP_API void scaleKeypoints2d(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX,
                                 const float offsetY);

    OP_API void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
                                   const std::vector<unsigned int>& pairs, const std::vector<float> colors,
                                   const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                                   const std::vector<float>& poseScales, const float threshold);

    OP_API Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person,
                                                  const float threshold);

    OP_API float getAverageScore(const Array<float>& keypoints, const int person);

    OP_API float getKeypointsArea(const Array<float>& keypoints, const int person, const float threshold);

    OP_API int getBiggestPerson(const Array<float>& keypoints, const float threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
