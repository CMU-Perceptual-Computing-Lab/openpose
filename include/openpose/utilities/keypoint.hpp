#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    OP_API T getDistance(const Array<T>& keypoints, const int person, const int elementA, const int elementB);

    template <typename T>
    OP_API void averageKeypoints(Array<T>& keypointsA, const Array<T>& keypointsB, const int personA);

    template <typename T>
    OP_API void scaleKeypoints(Array<T>& keypoints, const T scale);

    template <typename T>
    OP_API void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY);

    template <typename T>
    OP_API void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY, const T offsetX,
                                 const T offsetY);

    template <typename T>
    OP_API void renderKeypointsCpu(Array<T>& frameArray, const Array<T>& keypoints,
                                   const std::vector<unsigned int>& pairs, const std::vector<T> colors,
                                   const T thicknessCircleRatio, const T thicknessLineRatioWRTCircle,
                                   const std::vector<T>& poseScales, const T threshold);

    template <typename T>
    OP_API Rectangle<T> getKeypointsRectangle(const Array<T>& keypoints, const int person,
                                              const T threshold);

    template <typename T>
    OP_API T getAverageScore(const Array<T>& keypoints, const int person);

    template <typename T>
    OP_API T getKeypointsArea(const Array<T>& keypoints, const int person, const T threshold);

    template <typename T>
    OP_API int getBiggestPerson(const Array<T>& keypoints, const T threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
