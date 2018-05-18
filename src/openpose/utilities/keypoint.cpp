#include <limits> // std::numeric_limits
#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    const std::string errorMessage = "The Array<float> is not a RGB image or 3-channel keypoint array. This function"
                                     " is only for array of dimension: [sizeA x sizeB x 3].";

    float getDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB)
    {
        try
        {
            const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
            const auto pixelX = keypointPtr[elementA*3] - keypointPtr[elementB*3];
            const auto pixelY = keypointPtr[elementA*3+1] - keypointPtr[elementB*3+1];
            return std::sqrt(pixelX*pixelX+pixelY*pixelY);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.f;
        }
    }

    void averageKeypoints(Array<float>& keypointsA, const Array<float>& keypointsB, const int personA)
    {
        try
        {
            // Security checks
            if (keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions())
                error("keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions().",
                      __LINE__, __FUNCTION__, __FILE__);
            for (auto dimension = 1u ; dimension < keypointsA.getNumberDimensions() ; dimension++)
                if (keypointsA.getSize(dimension) != keypointsB.getSize(dimension))
                    error("keypointsA.getSize() != keypointsB.getSize().", __LINE__, __FUNCTION__, __FILE__);
            // For each body part
            const auto numberParts = keypointsA.getSize(1);
            for (auto part = 0 ; part < numberParts ; part++)
            {
                const auto finalIndexA = keypointsA.getSize(2)*(personA*numberParts + part);
                const auto finalIndexB = keypointsA.getSize(2)*part;
                if (keypointsB[finalIndexB+2] - keypointsA[finalIndexA+2] > 0.05f)
                {
                    keypointsA[finalIndexA] = keypointsB[finalIndexB];
                    keypointsA[finalIndexA+1] = keypointsB[finalIndexB+1];
                    keypointsA[finalIndexA+2] = keypointsB[finalIndexB+2];
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scale)
    {
        try
        {
            if (!keypoints.empty() && scale != 1.f)
            {
                // Error check
                if (keypoints.getSize(2) != 3 && keypoints.getSize(2) != 4)
                    error("The Array<float> is not a (x,y,score) or (x,y,z,score) format array. This"
                          " function is only for those 2 dimensions: [sizeA x sizeB x 3or4].",
                          __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                const auto xyzChannels = keypoints.getSize(2);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = xyzChannels*(person*numberParts + part);
                        for (auto xyz = 0 ; xyz < xyzChannels-1 ; xyz++)
                            keypoints[finalIndex+xyz] *= scale;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints2d(Array<float>& keypoints, const float scaleX, const float scaleY)
    {
        try
        {
            if (!keypoints.empty() && scaleX != 1.f && scaleY != 1.f)
            {
                // Error check
                if (keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        keypoints[finalIndex] *= scaleX;
                        keypoints[finalIndex+1] *= scaleY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints2d(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX,
                          const float offsetY)
    {
        try
        {
            if (!keypoints.empty() && scaleX != 1.f && scaleY != 1.f)
            {
                // Error check
                if (keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = keypoints.getSize(2)*(person*numberParts + part);
                        keypoints[finalIndex] = keypoints[finalIndex] * scaleX + offsetX;
                        keypoints[finalIndex+1] = keypoints[finalIndex+1] * scaleY + offsetY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
                            const std::vector<unsigned int>& pairs, const std::vector<float> colors,
                            const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                            const std::vector<float>& poseScales, const float threshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Array<float> --> cv::Mat
                auto frame = frameArray.getCvMat();

                // Security check
                if (frame.dims != 3 || frame.size[0] != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

                // Get frame channels
                const auto width = frame.size[2];
                const auto height = frame.size[1];
                const auto area = width * height;
                const auto channelOffset = area * sizeof(float) / sizeof(uchar);
                cv::Mat frameB(height, width, CV_32FC1, &frame.data[0]);
                cv::Mat frameG(height, width, CV_32FC1, &frame.data[channelOffset]);
                cv::Mat frameR(height, width, CV_32FC1, &frame.data[2 * channelOffset]);

                // Parameters
                const auto lineType = 8;
                const auto shift = 0;
                const auto numberColors = colors.size();
                const auto numberScales = poseScales.size();
                const auto thresholdRectangle = 0.1f;
                const auto numberKeypoints = keypoints.getSize(1);

                // Keypoints
                for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
                {
                    const auto personRectangle = getKeypointsRectangle(keypoints, person, thresholdRectangle);
                    if (personRectangle.area() > 0)
                    {
                        const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width/(float)width,
                                                                     personRectangle.height/(float)height));
                        // Size-dependent variables
                        const auto thicknessRatio = fastMax(intRound(std::sqrt(area)
                                                                     * thicknessCircleRatio * ratioAreas), 2);
                        // Negative thickness in cv::circle means that a filled circle is to be drawn.
                        const auto thicknessCircle = fastMax(1, (ratioAreas > 0.05f ? thicknessRatio : -1));
                        const auto thicknessLine = fastMax(1, intRound(thicknessRatio * thicknessLineRatioWRTCircle));
                        const auto radius = thicknessRatio / 2;

                        // Draw lines
                        for (auto pair = 0u ; pair < pairs.size() ; pair+=2)
                        {
                            const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                            const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);
                            if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                            {
                                const auto thicknessLineScaled = thicknessLine
                                                               * poseScales[pairs[pair+1] % numberScales];
                                const auto colorIndex = pairs[pair+1]*3; // Before: colorIndex = pair/2*3;
                                const cv::Scalar color{colors[colorIndex % numberColors],
                                                       colors[(colorIndex+1) % numberColors],
                                                       colors[(colorIndex+2) % numberColors]};
                                const cv::Point keypoint1{intRound(keypoints[index1]), intRound(keypoints[index1+1])};
                                const cv::Point keypoint2{intRound(keypoints[index2]), intRound(keypoints[index2+1])};
                                cv::line(frameR, keypoint1, keypoint2, color[0], thicknessLineScaled, lineType, shift);
                                cv::line(frameG, keypoint1, keypoint2, color[1], thicknessLineScaled, lineType, shift);
                                cv::line(frameB, keypoint1, keypoint2, color[2], thicknessLineScaled, lineType, shift);
                            }
                        }

                        // Draw circles
                        for (auto part = 0 ; part < numberKeypoints ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            if (keypoints[faceIndex+2] > threshold)
                            {
                                const auto radiusScaled = radius * poseScales[part % numberScales];
                                const auto thicknessCircleScaled = thicknessCircle * poseScales[part % numberScales];
                                const auto colorIndex = part*3;
                                const cv::Scalar color{colors[colorIndex % numberColors],
                                                       colors[(colorIndex+1) % numberColors],
                                                       colors[(colorIndex+2) % numberColors]};
                                const cv::Point center{intRound(keypoints[faceIndex]),
                                                       intRound(keypoints[faceIndex+1])};
                                cv::circle(frameR, center, radiusScaled, color[0], thicknessCircleScaled, lineType,
                                           shift);
                                cv::circle(frameG, center, radiusScaled, color[1], thicknessCircleScaled, lineType,
                                           shift);
                                cv::circle(frameB, center, radiusScaled, color[2], thicknessCircleScaled, lineType,
                                           shift);
                            }
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person, const float threshold)
    {
        try
        {
            const auto numberKeypoints = keypoints.getSize(1);
            // Security checks
            if (numberKeypoints < 1)
                error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);
            // Define keypointPtr
            const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
            float minX = std::numeric_limits<float>::max();
            float maxX = 0.f;
            float minY = minX;
            float maxY = maxX;
            for (auto part = 0 ; part < numberKeypoints ; part++)
            {
                const auto score = keypointPtr[3*part + 2];
                if (score > threshold)
                {
                    const auto x = keypointPtr[3*part];
                    const auto y = keypointPtr[3*part + 1];
                    // Set X
                    if (maxX < x)
                        maxX = x;
                    if (minX > x)
                        minX = x;
                    // Set Y
                    if (maxY < y)
                        maxY = y;
                    if (minY > y)
                        minY = y;
                }
            }
            if (maxX >= minX && maxY >= minY)
                return Rectangle<float>{minX, minY, maxX-minX, maxY-minY};
            else
                return Rectangle<float>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    float getAverageScore(const Array<float>& keypoints, const int person)
    {
        try
        {
            // Security checks
            if (person >= keypoints.getSize(0))
                error("Person index out of bounds.", __LINE__, __FUNCTION__, __FILE__);
            // Get average score
            auto score = 0.f;
            const auto numberKeypoints = keypoints.getSize(1);
            const auto area = numberKeypoints * keypoints.getSize(2);
            const auto personOffset = person * area;
            for (auto part = 0 ; part < numberKeypoints ; part++)
                score += keypoints[personOffset + part*keypoints.getSize(2) + 2];
            return score / numberKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getKeypointsArea(const Array<float>& keypoints, const int person, const float threshold)
    {
        try
        {
            return getKeypointsRectangle(keypoints, person, threshold).area();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    int getBiggestPerson(const Array<float>& keypoints, const float threshold)
    {
        try
        {
            if (!keypoints.empty())
            {
                const auto numberPeople = keypoints.getSize(0);
                auto biggestPoseIndex = -1;
                auto biggestArea = -1.f;
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    const auto newPersonArea = getKeypointsArea(keypoints, person, threshold);
                    if (newPersonArea > biggestArea)
                    {
                        biggestArea = newPersonArea;
                        biggestPoseIndex = person;
                    }
                }
                return biggestPoseIndex;
            }
            else
                return -1;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }
}
