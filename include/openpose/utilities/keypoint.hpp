#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/core/rectangle.hpp>

namespace op
{
    float getDistance(const float* keypointPtr, const int elementA, const int elementB);

    void scaleKeypoints(Array<float>& keypoints, const float scale);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY);

    void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints, const std::vector<unsigned int>& pairs,
                            const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle);

    Rectangle<unsigned int> getKeypointsRectangle(const float* keypointPtr, const int numberBodyParts, const float threshold);

    int getKeypointsArea(const float* keypointPtr, const int numberBodyParts, const float threshold);

    int getBiggestPerson(const Array<float>& keypoints, const float threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
