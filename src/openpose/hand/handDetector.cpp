#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/hand/handDetector.hpp>
 
namespace op
{
    inline std::array<Rectangle<float>, 2> getHandFromPoseIndexes(const Array<float>& poseKeypoints, const unsigned int personIndex, const unsigned int lWrist,
                                                                  const unsigned int lElbow, const unsigned int lShoulder, const unsigned int rWrist,
                                                                  const unsigned int rElbow, const unsigned int rShoulder, const float threshold)
    {
        try
        {
            std::array<Rectangle<float>, 2> handRectangle;

            const auto* posePtr = &poseKeypoints.at(personIndex*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
            const auto lWristScoreAbove = (posePtr[lWrist*3+2] > threshold);
            const auto lElbowScoreAbove = (posePtr[lElbow*3+2] > threshold);
            const auto lShoulderScoreAbove = (posePtr[lShoulder*3+2] > threshold);
            const auto rWristScoreAbove = (posePtr[rWrist*3+2] > threshold);
            const auto rElbowScoreAbove = (posePtr[rElbow*3+2] > threshold);
            const auto rShoulderScoreAbove = (posePtr[rShoulder*3+2] > threshold);
            // const auto neckScoreAbove = (posePtr[neck*3+2] > threshold);
            // const auto headNoseScoreAbove = (posePtr[headNose*3+2] > threshold);

            const auto ratio = 0.33f;
            auto& handLeftRectangle = handRectangle.at(0);
            auto& handRightRectangle = handRectangle.at(1);
            // Left hand
            if (lWristScoreAbove && lElbowScoreAbove && lShoulderScoreAbove)
            {
                handLeftRectangle.x = posePtr[lWrist*3] + ratio * (posePtr[lWrist*3] - posePtr[lElbow*3]);
                handLeftRectangle.y = posePtr[lWrist*3+1] + ratio * (posePtr[lWrist*3+1] - posePtr[lElbow*3+1]);
                const auto distanceWristElbow = getDistance(posePtr, lWrist, lElbow);
                const auto distanceElbowShoulder = getDistance(posePtr, lElbow, lShoulder);
                // const auto distanceWristShoulder = getDistance(posePtr, lWrist, lShoulder);
                // if (distanceWristElbow / distanceElbowShoulder > 0.85)
                    handLeftRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
                // else
                    // handLeftRectangle.width = 1.5f * 0.9f * distanceElbowShoulder * fastMin(distanceElbowShoulder / distanceWristElbow, 3.f);
                // somehow --> if distanceWristShoulder ~ distanceElbowShoulder --> do zoom in
            }
            // Right hand
            if (rWristScoreAbove && rElbowScoreAbove && rShoulderScoreAbove)
            {
                handRightRectangle.x = posePtr[rWrist*3] + ratio * (posePtr[rWrist*3] - posePtr[rElbow*3]);
                handRightRectangle.y = posePtr[rWrist*3+1] + ratio * (posePtr[rWrist*3+1] - posePtr[rElbow*3+1]);
                handRightRectangle.width = 1.5f * fastMax(getDistance(posePtr, rWrist, rElbow), 0.9f * getDistance(posePtr, rElbow, rShoulder));
            }
            handLeftRectangle.height = handLeftRectangle.width;
            handLeftRectangle.x -= handLeftRectangle.width / 2.f;
            handLeftRectangle.y -= handLeftRectangle.height / 2.f;
            handRightRectangle.height = handRightRectangle.width;
            handRightRectangle.x -= handRightRectangle.width / 2.f;
            handRightRectangle.y -= handRightRectangle.height / 2.f;
            return handRectangle;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<Rectangle<float>, 2>{};
        }
    }

    HandDetector::HandDetector(const PoseModel poseModel) :
        mPoseIndexes{getPoseKeypoints(poseModel, {"LWrist", "LElbow", "LShoulder", "RWrist", "RElbow", "RShoulder"})}
    {
    }

    std::vector<std::array<Rectangle<float>, 2>> HandDetector::detectHands(const Array<float>& poseKeypoints, const float scaleInputToOutput) const
    {
        try
        {
            const auto numberPeople = poseKeypoints.getSize(0);
            std::vector<std::array<Rectangle<float>, 2>> handRectangles(numberPeople);
            const auto threshold = 0.25f;
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
            auto handRectangles = detectHands(poseKeypoints, scaleInputToOutput);
            return handRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::array<Rectangle<float>, 2>>{};
        }
    }

    void HandDetector::updateTracker(const Array<float>& poseKeypoints, const Array<float>& handKeypoints)
    {
        try
        {
            // Security checks
            if (poseKeypoints.getSize(0) != handKeypoints.getSize(0))
                error("Number people on poseKeypoints different than in handKeypoints.", __LINE__, __FUNCTION__, __FILE__);
            // Parameters
            const auto numberPeople = poseKeypoints.getSize(0);
            const auto numberParts = poseKeypoints.getSize(1);
            const auto numberChannels = poseKeypoints.getSize(2);
            // Update pose keypoints and hand rectangles
            mPoseTrack.resize(numberPeople);
            mHandTrack.resize(numberPeople);
            for (auto personIndex = 0 ; personIndex < mPoseTrack.size() ; personIndex++)
            {
                // Update pose keypoints
                const auto* posePtr = &poseKeypoints.at(personIndex * numberParts * numberChannels);
                for (auto j = 0 ; j < mPoseIndexes.size() ; j++)
                    mPoseTrack[personIndex][j] = Point<float>{posePtr[numberChannels*mPoseIndexes[j]], posePtr[numberChannels*mPoseIndexes[j+1]]};
                // Update hand rectangles
                // for (auto j = 0 ; j < mPoseIndexes.size() ; j++)
                    // mHandTrack[personIndex] = XXXXXXXXXXXXXXXXXx;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::array<unsigned int, (int)HandDetector::PosePart::Size> HandDetector::getPoseKeypoints(const PoseModel poseModel,
                                                                                               const std::array<std::string,
                                                                                                                (int)HandDetector::PosePart::Size>& poseStrings
    )
    {
        std::array<unsigned int, (int)PosePart::Size> poseKeypoints;
        for (auto i = 0 ; i < poseKeypoints.size() ; i++)
        {
            poseKeypoints.at(i) = poseBodyPartMapStringToKey(poseModel, poseStrings.at(i));
        }
        return poseKeypoints;
    }
}
