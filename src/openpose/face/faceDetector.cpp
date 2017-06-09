#include <opencv2/opencv.hpp> // CV_WARP_INVERSE_MAP, CV_INTER_LINEAR
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/errorAndLog.hpp>
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

    float getDistance(const float* posePtr, const int elementA, const int elementB)
    {
        try
        {
            const auto pixelX = posePtr[elementA*3] - posePtr[elementB*3];
            const auto pixelY = posePtr[elementA*3+1] - posePtr[elementB*3+1];
            return std::sqrt(pixelX*pixelX+pixelY*pixelY);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.f;
        }
    }

    inline Rectangle<float> getFaceFromPoseKeypoints(const Array<float>& poseKeypoints, const unsigned int personIndex, const unsigned int neck,
                                                     const unsigned int nose, const unsigned int lEar, const unsigned int rEar,
                                                     const unsigned int lEye, const unsigned int rEye, const float threshold)
    {
        try
        {
            Point<float> pointTopLeft{0.f, 0.f};
            auto faceSize = 0.f;

            const auto* posePtr = &poseKeypoints.at(personIndex*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
            const auto neckScoreAbove = (posePtr[neck*3+2] > threshold);
            const auto noseScoreAbove = (posePtr[nose*3+2] > threshold);
            const auto lEarScoreAbove = (posePtr[lEar*3+2] > threshold);
            const auto rEarScoreAbove = (posePtr[rEar*3+2] > threshold);
            const auto lEyeScoreAbove = (posePtr[lEye*3+2] > threshold);
            const auto rEyeScoreAbove = (posePtr[rEye*3+2] > threshold);

            auto counter = 0;
            // Face and neck given (e.g. MPI)
            if (nose == lEar && lEar == rEar)
            {
                if (neckScoreAbove && noseScoreAbove)
                {
                    pointTopLeft.x = posePtr[nose*3];
                    pointTopLeft.y = posePtr[nose*3+1];
                    faceSize = 1.33f * getDistance(posePtr, neck, nose);
                }
            }
            // Face as average between different body keypoints (e.g. COCO)
            else
            {
                // factor * dist(neck, nose)
                if (neckScoreAbove && noseScoreAbove)
                {
                    // If profile (i.e. only 1 eye and ear visible) --> avg(nose, eye & ear position)
                    if ((lEyeScoreAbove) == (lEarScoreAbove)
                        && (rEyeScoreAbove) == (rEarScoreAbove)
                        && (lEyeScoreAbove) != (rEyeScoreAbove))
                    {
                        if (lEyeScoreAbove)
                        {
                            pointTopLeft.x += (posePtr[lEye*3] + posePtr[lEar*3] + posePtr[nose*3]) / 3.f;
                            pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[lEar*3+1] + posePtr[nose*3+1]) / 3.f;
                            faceSize += 0.85 * (getDistance(posePtr, nose, lEye) + getDistance(posePtr, nose, lEar) + getDistance(posePtr, neck, nose));
                        }
                        else // if(lEyeScoreAbove)
                        {
                            pointTopLeft.x += (posePtr[rEye*3] + posePtr[rEar*3] + posePtr[nose*3]) / 3.f;
                            pointTopLeft.y += (posePtr[rEye*3+1] + posePtr[rEar*3+1] + posePtr[nose*3+1]) / 3.f;
                            faceSize += 0.85 * (getDistance(posePtr, nose, rEye) + getDistance(posePtr, nose, rEar) + getDistance(posePtr, neck, nose));
                        }
                    }
                    // else --> 2 * dist(neck, nose)
                    else
                    {
                        pointTopLeft.x += (posePtr[neck*3] + posePtr[nose*3]) / 2.f;
                        pointTopLeft.y += (posePtr[neck*3+1] + posePtr[nose*3+1]) / 2.f;
                        faceSize += 2.f * getDistance(posePtr, neck, nose);
                    }
                    counter++;
                }
                // 3 * dist(lEye, rEye)
                if (lEyeScoreAbove && rEyeScoreAbove)
                {
                    pointTopLeft.x += (posePtr[lEye*3] + posePtr[rEye*3]) / 2.f;
                    pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[rEye*3+1]) / 2.f;
                    faceSize += 3.f * getDistance(posePtr, lEye, rEye);
                    counter++;
                }
                // 2 * dist(lEar, rEar)
                if (lEarScoreAbove && rEarScoreAbove)
                {
                    pointTopLeft.x += (posePtr[lEar*3] + posePtr[rEar*3]) / 2.f;
                    pointTopLeft.y += (posePtr[lEar*3+1] + posePtr[rEar*3+1]) / 2.f;
                    faceSize += 2.f * getDistance(posePtr, lEar, rEar);
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

    std::vector<Rectangle<float>> FaceDetector::detectFaces(const Array<float>& poseKeypoints, const float scaleInputToOutput)
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
                    faceRectangles.at(person) = getFaceFromPoseKeypoints(poseKeypoints, person, mNeck, mNose, mLEar, mREar, mLEye, mREye, threshold) / scaleInputToOutput;
            return faceRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
