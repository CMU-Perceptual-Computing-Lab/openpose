#include <limits> // std::numeric_limits
#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    const std::string errorMessage = "The Array<T> is not a RGB image or 3-channel keypoint array. This function"
                                     " is only for array of dimension: [sizeA x sizeB x 3].";

    template <typename T>
    T getDistance(const Array<T>& keypoints, const int person, const int elementA, const int elementB)
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
            return T(-1);
        }
    }
    template OP_API float getDistance(
        const Array<float>& keypoints, const int person, const int elementA, const int elementB);
    template OP_API double getDistance(
        const Array<double>& keypoints, const int person, const int elementA, const int elementB);

    template <typename T>
    void averageKeypoints(Array<T>& keypointsA, const Array<T>& keypointsB, const int personA)
    {
        try
        {
            // Sanity checks
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
                if (keypointsB[finalIndexB+2] - keypointsA[finalIndexA+2] > T(0.05))
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
    template OP_API void averageKeypoints(
        Array<float>& keypointsA, const Array<float>& keypointsB, const int personA);
    template OP_API void averageKeypoints(
        Array<double>& keypointsA, const Array<double>& keypointsB, const int personA);

    template <typename T>
    void scaleKeypoints(Array<T>& keypoints, const T scale)
    {
        try
        {
            if (!keypoints.empty() && scale != T(1))
            {
                // Error check
                if (keypoints.getSize(2) != 3 && keypoints.getSize(2) != 4)
                    error("The Array<T> is not a (x,y,score) or (x,y,z,score) format array. This"
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
    template OP_API void scaleKeypoints(Array<float>& keypoints, const float scale);
    template OP_API void scaleKeypoints(Array<double>& keypoints, const double scale);

    template <typename T>
    void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY)
    {
        try
        {
            if (!keypoints.empty() && (scaleX != T(1) || scaleY != T(1)))
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
    template OP_API void scaleKeypoints2d(Array<float>& keypoints, const float scaleX, const float scaleY);
    template OP_API void scaleKeypoints2d(Array<double>& keypoints, const double scaleX, const double scaleY);

    template <typename T>
    void scaleKeypoints2d(Array<T>& keypoints, const T scaleX, const T scaleY, const T offsetX, const T offsetY)
    {
        try
        {
            if (!keypoints.empty() && (scaleX != T(1) || scaleY != T(1) || offsetX != T(0) || offsetY != T(0)))
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
    template OP_API void scaleKeypoints2d(
        Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY);
    template OP_API void scaleKeypoints2d(
        Array<double>& keypoints, const double scaleX, const double scaleY, const double offsetX,
        const double offsetY);

    template <typename T>
    void renderKeypointsCpu(Array<T>& frameArray, const Array<T>& keypoints, const std::vector<unsigned int>& pairs,
                            const std::vector<T> colors, const T thicknessCircleRatio,
                            const T thicknessLineRatioWRTCircle, const std::vector<T>& poseScales, const T threshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Array<T> --> cv::Mat
                auto frame = frameArray.getCvMat();

                // Sanity check
                if (frame.channels() != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

                // Get frame channels
                const auto width = frame.size[1];
                const auto height = frame.size[0];
                const auto area = width * height;
                cv::Mat frameBGR(height, width, CV_32FC3, frame.data);

                // Parameters
                const auto lineType = 8;
                const auto shift = 0;
                const auto numberColors = colors.size();
                const auto numberScales = poseScales.size();
                const auto thresholdRectangle = T(0.1);
                const auto numberKeypoints = keypoints.getSize(1);

                // Keypoints
                for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
                {
                    const auto personRectangle = getKeypointsRectangle(keypoints, person, thresholdRectangle);
                    if (personRectangle.area() > 0)
                    {
                        const auto ratioAreas = fastMin(T(1), fastMax(personRectangle.width/(T)width,
                                                                     personRectangle.height/(T)height));
                        // Size-dependent variables
                        const auto thicknessRatio = fastMax(
                            positiveIntRound(std::sqrt(area)* thicknessCircleRatio * ratioAreas), 2);
                        // Negative thickness in cv::circle means that a filled circle is to be drawn.
                        const auto thicknessCircle = fastMax(1, (ratioAreas > T(0.05) ? thicknessRatio : -1));
                        const auto thicknessLine = fastMax(
                            1, positiveIntRound(thicknessRatio * thicknessLineRatioWRTCircle));
                        const auto radius = thicknessRatio / 2;

                        // Draw lines
                        for (auto pair = 0u ; pair < pairs.size() ; pair+=2)
                        {
                            const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                            const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);
                            if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                            {
                                const auto thicknessLineScaled = positiveIntRound(
                                    thicknessLine * poseScales[pairs[pair+1] % numberScales]);
                                const auto colorIndex = pairs[pair+1]*3; // Before: colorIndex = pair/2*3;
                                const cv::Scalar color{
                                    colors[(colorIndex+2) % numberColors],
                                    colors[(colorIndex+1) % numberColors],
                                    colors[colorIndex % numberColors]
                                };
                                const cv::Point keypoint1{
                                    positiveIntRound(keypoints[index1]), positiveIntRound(keypoints[index1+1])};
                                const cv::Point keypoint2{
                                    positiveIntRound(keypoints[index2]), positiveIntRound(keypoints[index2+1])};
                                cv::line(frameBGR, keypoint1, keypoint2, color, thicknessLineScaled, lineType, shift);
                            }
                        }

                        // Draw circles
                        for (auto part = 0 ; part < numberKeypoints ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            if (keypoints[faceIndex+2] > threshold)
                            {
                                const auto radiusScaled = positiveIntRound(radius * poseScales[part % numberScales]);
                                const auto thicknessCircleScaled = positiveIntRound(
                                    thicknessCircle * poseScales[part % numberScales]);
                                const auto colorIndex = part*3;
                                const cv::Scalar color{
                                    colors[(colorIndex+2) % numberColors],
                                    colors[(colorIndex+1) % numberColors],
                                    colors[colorIndex % numberColors]
                                };
                                const cv::Point center{positiveIntRound(keypoints[faceIndex]),
                                                       positiveIntRound(keypoints[faceIndex+1])};
                                cv::circle(frameBGR, center, radiusScaled, color, thicknessCircleScaled, lineType,
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
    template OP_API void renderKeypointsCpu(
        Array<float>& frameArray, const Array<float>& keypoints, const std::vector<unsigned int>& pairs,
        const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
        const std::vector<float>& poseScales, const float threshold);
    template OP_API void renderKeypointsCpu(
        Array<double>& frameArray, const Array<double>& keypoints, const std::vector<unsigned int>& pairs,
        const std::vector<double> colors, const double thicknessCircleRatio, const double thicknessLineRatioWRTCircle,
        const std::vector<double>& poseScales, const double threshold);

    template <typename T>
    Rectangle<T> getKeypointsRectangle(const Array<T>& keypoints, const int person, const T threshold)
    {
        try
        {
            const auto numberKeypoints = keypoints.getSize(1);
            // Sanity check
            if (numberKeypoints < 1)
                error("Number body parts must be > 0.", __LINE__, __FUNCTION__, __FILE__);
            // Define keypointPtr
            const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
            T minX = std::numeric_limits<T>::max();
            T maxX = std::numeric_limits<T>::lowest();
            T minY = minX;
            T maxY = maxX;
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
                return Rectangle<T>{minX, minY, maxX-minX, maxY-minY};
            else
                return Rectangle<T>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<T>{};
        }
    }
    template OP_API Rectangle<float> getKeypointsRectangle(
        const Array<float>& keypoints, const int person, const float threshold);
    template OP_API Rectangle<double> getKeypointsRectangle(
        const Array<double>& keypoints, const int person, const double threshold);

    template <typename T>
    T getAverageScore(const Array<T>& keypoints, const int person)
    {
        try
        {
            // Sanity check
            if (person >= keypoints.getSize(0))
                error("Person index out of bounds.", __LINE__, __FUNCTION__, __FILE__);
            // Get average score
            T score = T(0);
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
            return T(0);
        }
    }
    template OP_API float getAverageScore(const Array<float>& keypoints, const int person);
    template OP_API double getAverageScore(const Array<double>& keypoints, const int person);

    template <typename T>
    T getKeypointsArea(const Array<T>& keypoints, const int person, const T threshold)
    {
        try
        {
            return getKeypointsRectangle(keypoints, person, threshold).area();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }
    template OP_API float getKeypointsArea(const Array<float>& keypoints, const int person, const float threshold);
    template OP_API double getKeypointsArea(const Array<double>& keypoints, const int person, const double threshold);

    template <typename T>
    int getBiggestPerson(const Array<T>& keypoints, const T threshold)
    {
        try
        {
            if (!keypoints.empty())
            {
                const auto numberPeople = keypoints.getSize(0);
                auto biggestPoseIndex = -1;
                auto biggestArea = T(-1);
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
    template OP_API int getBiggestPerson(const Array<float>& keypoints, const float threshold);
    template OP_API int getBiggestPerson(const Array<double>& keypoints, const double threshold);

    template <typename T>
    int getNonZeroKeypoints(const Array<T>& keypoints, const int person, const T threshold)
    {
        try
        {
            if (!keypoints.empty())
            {
                // Sanity check
                if (keypoints.getSize(0) <= person)
                    error("Person index out of range.", __LINE__, __FUNCTION__, __FILE__);
                // Count keypoints
                auto nonZeroCounter = 0;
                const auto baseIndex = person * keypoints.getVolume(1,2);
                for (auto part = 0 ; part < keypoints.getSize(1) ; part++)
                    if (keypoints[baseIndex + 3*part + 2] >= threshold)
                        nonZeroCounter++;
                return nonZeroCounter;
            }
            else
                return 0;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }
    template OP_API int getNonZeroKeypoints(const Array<float>& keypoints, const int person, const float threshold);
    template OP_API int getNonZeroKeypoints(const Array<double>& keypoints, const int person, const double threshold);

    template <typename T>
    T getDistanceAverage(const Array<T>& keypoints, const int personA, const int personB, const T threshold)
    {
        try
        {
            return getDistanceAverage(keypoints, personA, keypoints, personB, threshold);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }
    template OP_API float getDistanceAverage(
        const Array<float>& keypoints, const int personA, const int personB, const float threshold);
    template OP_API double getDistanceAverage(
        const Array<double>& keypoints, const int personA, const int personB, const double threshold);

    template <typename T>
    T getDistanceAverage(const Array<T>& keypointsA, const int personA, const Array<T>& keypointsB, const int personB,
                         const T threshold)
    {
        try
        {
            // Sanity checks
            if (keypointsA.getSize(0) <= personA)
                error("PersonA index out of range.", __LINE__, __FUNCTION__, __FILE__);
            if (keypointsB.getSize(0) <= personB)
                error("PersonB index out of range.", __LINE__, __FUNCTION__, __FILE__);
            if (keypointsA.getSize(1) != keypointsB.getSize(1))
                error("Keypoints should have the same number of keypoints.", __LINE__, __FUNCTION__, __FILE__);
            // Get total distance
            T totalDistance = 0;
            int nonZeroCounter = 0;
            const auto baseIndexA = personA * keypointsA.getVolume(1,2);
            const auto baseIndexB = personB * keypointsB.getVolume(1,2);
            for (auto part = 0 ; part < keypointsA.getSize(1) ; part++)
            {
                if (keypointsA[baseIndexA+3*part+2] >= threshold && keypointsB[baseIndexB+3*part+2] >= threshold)
                {
                    const auto x = keypointsA[baseIndexA+3*part] - keypointsB[baseIndexB+3*part];
                    const auto y = keypointsA[baseIndexA+3*part+1] - keypointsB[baseIndexB+3*part+1];
                    totalDistance += T(std::sqrt(x*x+y*y));
                    nonZeroCounter++;
                }
            }
            // Get distance average
            return totalDistance / nonZeroCounter;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }
    template OP_API float getDistanceAverage(
        const Array<float>& keypointsA, const int personA, const Array<float>& keypointsB, const int personB,
        const float threshold);
    template OP_API double getDistanceAverage(
        const Array<double>& keypointsA, const int personA, const Array<double>& keypointsB, const int personB,
        const double threshold);

    template <typename T>
    Array<T> getKeypointsPerson(const Array<T>& keypoints, const int person, const bool noCopy)
    {
        try
        {
            return Array<T>(keypoints, person, noCopy);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<T>{};
        }
    }
    template OP_API Array<float> getKeypointsPerson(
        const Array<float>& keypoints, const int person, const bool noCopy);
    template OP_API Array<double> getKeypointsPerson(
        const Array<double>& keypoints, const int person, const bool noCopy);

    template <typename T>
    float getKeypointsRoi(const Array<T>& keypoints, const int personA, const int personB, const T threshold)
    {
        try
        {
            return getKeypointsRoi(keypoints, personA, keypoints, personB, threshold);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }
    template OP_API float getKeypointsRoi(
        const Array<float>& keypoints, const int personA, const int personB, const float threshold);
    template OP_API float getKeypointsRoi(
        const Array<double>& keypoints, const int personA, const int personB, const double threshold);

    template <typename T>
    float getKeypointsRoi(const Array<T>& keypointsA, const int personA, const Array<T>& keypointsB, const int personB,
                          const T threshold)
    {
        try
        {
            // Sanity checks
            if (keypointsA.getSize(0) <= personA)
                error("PersonA index out of range.", __LINE__, __FUNCTION__, __FILE__);
            if (keypointsB.getSize(0) <= personB)
                error("PersonB index out of range.", __LINE__, __FUNCTION__, __FILE__);
            if (keypointsA.getSize(1) != keypointsB.getSize(1))
                error("Keypoints should have the same number of keypoints.", __LINE__, __FUNCTION__, __FILE__);
            // Get ROI
            const auto rectangleA = getKeypointsRectangle(keypointsA, personA, threshold);
            const auto rectangleB = getKeypointsRectangle(keypointsB, personB, threshold);
            const Point<T> pointAIntersection{
                fastMax(rectangleA.x, rectangleB.x),
                fastMax(rectangleA.y, rectangleB.y)
            };
            const Point<T> pointBIntersection{
                fastMin(rectangleA.x+rectangleA.width, rectangleB.x+rectangleB.width),
                fastMin(rectangleA.y+rectangleA.height, rectangleB.y+rectangleB.height)
            };
            // Make sure there is overlap
            if (pointAIntersection.x < pointBIntersection.x && pointAIntersection.y < pointBIntersection.y)
            {
                const Rectangle<T> rectangleIntersection{
                    pointAIntersection.x,
                    pointAIntersection.y,
                    pointBIntersection.x-pointAIntersection.x,
                    pointBIntersection.y-pointAIntersection.y
                };
                const auto areaA = rectangleA.area();
                const auto areaB = rectangleB.area();
                const auto intersection = rectangleIntersection.area();
                return float(intersection) / float(areaA + areaB - intersection);
            }
            // If non overlap --> Return 0
            return 0.f;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }
    template OP_API float getKeypointsRoi(
        const Array<float>& keypointsA, const int personA, const Array<float>& keypointsB, const int personB,
        const float threshold);
    template OP_API float getKeypointsRoi(
        const Array<double>& keypointsA, const int personA, const Array<double>& keypointsB, const int personB,
        const double threshold);
}
