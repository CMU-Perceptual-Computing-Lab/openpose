#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/face/faceDetector.hpp>
 
namespace op
{
    FaceDetector::FaceDetector(const PoseModel poseModel) :
        mNeck{poseBodyPartMapStringToKey(poseModel, "Neck")},
        mNose{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"Nose", "Head"})},
        mLEar{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"LEar", "Head"})},
        mREar{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"REar", "Head"})},
        mLEye{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"LEye", "Head"})},
        mREye{poseBodyPartMapStringToKey(poseModel, std::vector<std::string>{"REye", "Head"})}
    {
    }

    inline Rectangle<float> getFaceFromPoseKeypoints(const Array<float>& poseKeypoints, const unsigned int personIndex,
                                                     const unsigned int neck, const unsigned int headNose,
                                                     const unsigned int lEar, const unsigned int rEar,
                                                     const unsigned int lEye, const unsigned int rEye,
                                                     const float threshold)
    {
        try
        {
            Point<float> pointTopLeft{0.f, 0.f};
            auto faceSize = 0.f;

            const auto* posePtr = &poseKeypoints.at(personIndex*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
            const auto neckScoreAbove = (posePtr[neck*3+2] > threshold);
            const auto headNoseScoreAbove = (posePtr[headNose*3+2] > threshold);
            const auto lEarScoreAbove = (posePtr[lEar*3+2] > threshold);
            const auto rEarScoreAbove = (posePtr[rEar*3+2] > threshold);
            const auto lEyeScoreAbove = (posePtr[lEye*3+2] > threshold);
            const auto rEyeScoreAbove = (posePtr[rEye*3+2] > threshold);

            auto counter = 0;
            // Face and neck given (e.g. MPI)
            if (headNose == lEar && lEar == rEar)
            {
                if (neckScoreAbove && headNoseScoreAbove)
                {
                    pointTopLeft.x = posePtr[headNose*3];
                    pointTopLeft.y = posePtr[headNose*3+1];
                    faceSize = 1.33f * getDistance(poseKeypoints, personIndex, neck, headNose);
                }
            }
            // Face as average between different body keypoints (e.g. COCO)
            else
            {
                // factor * dist(neck, headNose)
                if (neckScoreAbove && headNoseScoreAbove)
                {
                    // If profile (i.e. only 1 eye and ear visible) --> avg(headNose, eye & ear position)
                    if ((lEyeScoreAbove) == (lEarScoreAbove)
                        && (rEyeScoreAbove) == (rEarScoreAbove)
                        && (lEyeScoreAbove) != (rEyeScoreAbove))
                    {
                        if (lEyeScoreAbove)
                        {
                            pointTopLeft.x += (posePtr[lEye*3] + posePtr[lEar*3] + posePtr[headNose*3]) / 3.f;
                            pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[lEar*3+1] + posePtr[headNose*3+1]) / 3.f;
                            faceSize += 0.85f * (getDistance(poseKeypoints, personIndex, headNose, lEye)
                                                 + getDistance(poseKeypoints, personIndex, headNose, lEar)
                                                 + getDistance(poseKeypoints, personIndex, neck, headNose));
                        }
                        else // if(lEyeScoreAbove)
                        {
                            pointTopLeft.x += (posePtr[rEye*3] + posePtr[rEar*3] + posePtr[headNose*3]) / 3.f;
                            pointTopLeft.y += (posePtr[rEye*3+1] + posePtr[rEar*3+1] + posePtr[headNose*3+1]) / 3.f;
                            faceSize += 0.85f * (getDistance(poseKeypoints, personIndex, headNose, rEye)
                                                 + getDistance(poseKeypoints, personIndex, headNose, rEar)
                                                 + getDistance(poseKeypoints, personIndex, neck, headNose));
                        }
                    }
                    // else --> 2 * dist(neck, headNose)
                    else
                    {
                        pointTopLeft.x += (posePtr[neck*3] + posePtr[headNose*3]) / 2.f;
                        pointTopLeft.y += (posePtr[neck*3+1] + posePtr[headNose*3+1]) / 2.f;
                        faceSize += 2.f * getDistance(poseKeypoints, personIndex, neck, headNose);
                    }
                    counter++;
                }
                // 3 * dist(lEye, rEye)
                if (lEyeScoreAbove && rEyeScoreAbove)
                {
                    pointTopLeft.x += (posePtr[lEye*3] + posePtr[rEye*3]) / 2.f;
                    pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[rEye*3+1]) / 2.f;
                    faceSize += 3.f * getDistance(poseKeypoints, personIndex, lEye, rEye);
                    counter++;
                }
                // 2 * dist(lEar, rEar)
                if (lEarScoreAbove && rEarScoreAbove)
                {
                    pointTopLeft.x += (posePtr[lEar*3] + posePtr[rEar*3]) / 2.f;
                    pointTopLeft.y += (posePtr[lEar*3+1] + posePtr[rEar*3+1]) / 2.f;
                    faceSize += 2.f * getDistance(poseKeypoints, personIndex, lEar, rEar);
                    counter++;
                }
                // Average (if counter > 0)
                if (counter > 0)
                {
                    pointTopLeft /= (float)counter;
                    faceSize /= counter;
                }
            }
            return Rectangle<float>{pointTopLeft.x - faceSize / 2, pointTopLeft.y - faceSize / 2, faceSize, faceSize};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    std::vector<Rectangle<float>> FaceDetector::detectFaces(const Array<float>& poseKeypoints) const
    {
        try
        {
            const auto numberPeople = poseKeypoints.getSize(0);
            std::vector<Rectangle<float>> faceRectangles(numberPeople);
            const auto threshold = 0.25f;
            // If no poseKeypoints detected -> no way to detect face location
            // Otherwise, get face position(s)
            if (!poseKeypoints.empty())
                for (auto person = 0 ; person < numberPeople ; person++)
                    faceRectangles.at(person) = getFaceFromPoseKeypoints(
                        poseKeypoints, person, mNeck, mNose, mLEar, mREar, mLEye, mREye, threshold);
            return faceRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
