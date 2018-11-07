#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    T getDistance(const Array<T>& keypoints, const int person, const int elementA, const int elementB);

    template <typename T>
    void averageKeypoints(Array<T>& keypointsA, const Array<T>& keypointsB, const int personA);

    template <typename T>
    void scaleKeypoints(Array<T>& keypoints, const T scale);

    template <typename T>
    void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY);

    template <typename T>
    void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY, const T offsetX, const T offsetY);

    template <typename T>
    void renderKeypointsCpu(Array<T>& frameArray, const Array<T>& keypoints, const std::vector<unsigned int>& pairs,
                            const std::vector<T> colors, const T thicknessCircleRatio,
                            const T thicknessLineRatioWRTCircle, const std::vector<T>& poseScales, const T threshold);

    template <typename T>
    Rectangle<T> getKeypointsRectangle(const Array<T>& keypoints, const int person, const T threshold);

    template <typename T>
    T getAverageScore(const Array<T>& keypoints, const int person);

    template <typename T>
    T getKeypointsArea(const Array<T>& keypoints, const int person, const T threshold);

    template <typename T>
    int getBiggestPerson(const Array<T>& keypoints, const T threshold);

    template <typename T>
    int getNonZeroKeypoints(const Array<T>& keypoints, const int person, const T threshold);

    template <typename T>
    T getDistanceAverage(const Array<T>& keypoints, const int personA, const int personB, const T threshold);

    template <typename T>
    T getDistanceAverage(const Array<T>& keypointsA, const int personA, const Array<T>& keypointsB, const int personB,
                         const T threshold);

    template <typename T>
    float getKeypointsROI(const Array<T>& keypoints, const int personA, const int personB, const T threshold);

    template <typename T>
    float getKeypointsROI(const Array<T>& keypointsA, const int personA, const Array<T>& keypointsB, const int personB,
                          const T threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
