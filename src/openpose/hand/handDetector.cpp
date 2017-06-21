#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/hand/handDetector.hpp>
 
namespace op
{
    inline Rectangle<float> getHandFromPoseIndexes(const Array<float>& poseKeypoints, const unsigned int person, const unsigned int wrist,
                                                   const unsigned int elbow, const unsigned int shoulder, const float threshold)
    {
        try
        {
            Rectangle<float> handRectangle;
            // Parameters
            const auto* posePtr = &poseKeypoints.at(person*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
            const auto wristScoreAbove = (posePtr[wrist*3+2] > threshold);
            const auto elbowScoreAbove = (posePtr[elbow*3+2] > threshold);
            const auto shoulderScoreAbove = (posePtr[shoulder*3+2] > threshold);
            const auto ratioWristElbow = 0.33f;
            // Hand
            if (wristScoreAbove && elbowScoreAbove && shoulderScoreAbove)
            {
                // pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
                handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
                handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
                const auto distanceWristElbow = getDistance(posePtr, wrist, elbow);
                const auto distanceElbowShoulder = getDistance(posePtr, elbow, shoulder);
                handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            }
            // height = width
            handRectangle.height = handRectangle.width;
            // x-y refers to the center --> offset to topLeft point
            handRectangle.x -= handRectangle.width / 2.f;
            handRectangle.y -= handRectangle.height / 2.f;
            // Return result
            return handRectangle;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    inline std::array<Rectangle<float>, 2> getHandFromPoseIndexes(const Array<float>& poseKeypoints, const unsigned int person,
                                                                  const unsigned int lWrist, const unsigned int lElbow, const unsigned int lShoulder,
                                                                  const unsigned int rWrist, const unsigned int rElbow, const unsigned int rShoulder,
                                                                  const float threshold)
    {
        try
        {
            return {getHandFromPoseIndexes(poseKeypoints, person, lWrist, lElbow, lShoulder, threshold),
                    getHandFromPoseIndexes(poseKeypoints, person, rWrist, rElbow, rShoulder, threshold)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<Rectangle<float>, 2>(); // Parentheses instead of braces to avoid error in GCC 4.8
        }
    }

    float getAreaRatio(const Rectangle<float>& rectangleA, const Rectangle<float>& rectangleB)
    {
        try
        {
            // https://stackoverflow.com/a/22613463
            const auto sA = rectangleA.area();
            const auto sB = rectangleB.area();
            const auto bottomRightA = rectangleA.bottomRight();
            const auto bottomRightB = rectangleB.bottomRight();
            const auto sI = fastMax(0.f, 1.f + fastMin(bottomRightA.x, bottomRightB.x) - fastMax(rectangleA.x, rectangleB.x))
                          * fastMax(0.f, 1.f + fastMin(bottomRightA.y, bottomRightB.y) - fastMax(rectangleA.y, rectangleB.y));
            // // Option a - areaRatio = 1.f only if both Rectangle has same size and location
            // const auto sU = sA + sB - sI;
            // return sI / (float)sU;
            // Option b - areaRatio = 1.f if at least one Rectangle is contained in the other
            const auto sU = fastMin(sA, sB);
            return fastMin(1.f, sI / (float)sU);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void trackHand(Rectangle<float>& currentRectangle, const std::vector<Rectangle<float>>& previousHands)
    {
        try
        {
            if (currentRectangle.area() > 0 && previousHands.size() > 0)
            {
                // Find closest previous rectangle
                auto maxIndex = -1;
                auto maxValue = 0.f;
                for (auto previous = 0 ; previous < previousHands.size() ; previous++)
                {
                    const auto areaRatio = getAreaRatio(currentRectangle, previousHands[previous]);
                    if (maxValue < areaRatio)
                    {
                        maxValue = areaRatio;
                        maxIndex = previous;
                    }
                }
                // Update current rectangle with closest previous rectangle
                if (maxIndex > -1)
                {
                    const auto& prevRectangle = previousHands[maxIndex];
                    const auto ratio = 2.f;
                    const auto newWidth = fastMax((currentRectangle.width * ratio + prevRectangle.width) * 0.5f,
                                                  (currentRectangle.height * ratio + prevRectangle.height) * 0.5f);
                    currentRectangle.x = 0.5f * (currentRectangle.x + prevRectangle.x + 0.5f * (currentRectangle.width + prevRectangle.width) - newWidth);
                    currentRectangle.y = 0.5f * (currentRectangle.y + prevRectangle.y + 0.5f * (currentRectangle.height + prevRectangle.height) - newWidth);
                    currentRectangle.width = newWidth;
                    currentRectangle.height = newWidth;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    HandDetector::HandDetector(const PoseModel poseModel) :
        // Parentheses instead of braces to avoid error in GCC 4.8
        mPoseIndexes(getPoseKeypoints(poseModel, {"LWrist", "LElbow", "LShoulder", "RWrist", "RElbow", "RShoulder"})),
        mCurrentId{0}
    {
    }

    std::vector<std::array<Rectangle<float>, 2>> HandDetector::detectHands(const Array<float>& poseKeypoints, const float scaleInputToOutput) const
    {
        try
        {
            const auto numberPeople = poseKeypoints.getSize(0);
            std::vector<std::array<Rectangle<float>, 2>> handRectangles(numberPeople);
            const auto threshold = 0.03f;
            // If no poseKeypoints detected -> no way to detect hand location
            // Otherwise, get hand position(s)
            if (!poseKeypoints.empty())
            {
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    handRectangles.at(person) = getHandFromPoseIndexes(
                        poseKeypoints, person, mPoseIndexes[(int)PosePart::LWrist], mPoseIndexes[(int)PosePart::LElbow],
                        mPoseIndexes[(int)PosePart::LShoulder], mPoseIndexes[(int)PosePart::RWrist],
                        mPoseIndexes[(int)PosePart::RElbow], mPoseIndexes[(int)PosePart::RShoulder], threshold
                    );
                    handRectangles.at(person).at(0) /= scaleInputToOutput;
                    handRectangles.at(person).at(1) /= scaleInputToOutput;
                }
            }
            return handRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::array<Rectangle<float>, 2>>{};
        }
    }

    std::vector<std::array<Rectangle<float>, 2>> HandDetector::trackHands(const Array<float>& poseKeypoints, const float scaleInputToOutput)
    {
        try
        {
            std::lock_guard<std::mutex> lock{mMutex};
            // Baseline detectHands
            auto handRectangles = detectHands(poseKeypoints, scaleInputToOutput);
            // If previous hands saved
            // for (auto current = 0 ; current < handRectangles.size() ; current++)
            for (auto& handRectangle : handRectangles)
            {
                // trackHand(handRectangles[current][0], mHandLeftPrevious);
                // trackHand(handRectangles[current][1], mHandRightPrevious);
                trackHand(handRectangle[0], mHandLeftPrevious);
                trackHand(handRectangle[1], mHandRightPrevious);
            }
            // Return result
            return handRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::array<Rectangle<float>, 2>>{};
        }
    }

    void HandDetector::updateTracker(const Array<float>& poseKeypoints, const std::array<Array<float>, 2>& handKeypoints)
    {
        try
        {
            std::lock_guard<std::mutex> lock{mMutex};
            // Security checks
            if (poseKeypoints.getSize(0) != handKeypoints[0].getSize(0) || poseKeypoints.getSize(0) != handKeypoints[1].getSize(0))
                error("Number people on poseKeypoints different than in handKeypoints.", __LINE__, __FUNCTION__, __FILE__);
            // Parameters
            const auto numberPeople = poseKeypoints.getSize(0);
            // const auto poseNumberParts = poseKeypoints.getSize(1);
            const auto handNumberParts = handKeypoints[0].getSize(1);
            const auto numberChannels = poseKeypoints.getSize(2);
            const auto thresholdRectangle = 0.25f;
            // Update pose keypoints and hand rectangles
            mPoseTrack.resize(numberPeople);
            mHandLeftPrevious.clear();
            mHandRightPrevious.clear();
            for (auto person = 0 ; person < mPoseTrack.size() ; person++)
            {
                const auto offset = person * handNumberParts * numberChannels;
                // Left hand
                const auto* handLeftPtr = handKeypoints[0].getConstPtr() + offset;
                const auto handLeftRectangle = getKeypointsRectangle(handLeftPtr, handNumberParts, thresholdRectangle);
                if (handLeftRectangle.area() > 0)
                    mHandLeftPrevious.emplace_back(handLeftRectangle);
                const auto* handRightPtr = handKeypoints[1].getConstPtr() + offset;
                // Right hand
                const auto handRightRectangle = getKeypointsRectangle(handRightPtr, handNumberParts, thresholdRectangle);
                if (handRightRectangle.area() > 0)
                    mHandRightPrevious.emplace_back(handRightRectangle);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::array<unsigned int, (int)HandDetector::PosePart::Size> HandDetector::getPoseKeypoints(
        const PoseModel poseModel, const std::array<std::string, (int)HandDetector::PosePart::Size>& poseStrings
    ) const
    {
        std::array<unsigned int, (int)PosePart::Size> poseKeypoints;
        for (auto i = 0 ; i < poseKeypoints.size() ; i++)
            poseKeypoints.at(i) = poseBodyPartMapStringToKey(poseModel, poseStrings.at(i));
        return poseKeypoints;
    }
}
