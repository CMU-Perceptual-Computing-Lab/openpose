#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/scaleKeypoints.hpp>

namespace op
{
    const std::string errorMessage = "This function is only for array of dimension: [sizeA x sizeB x 3].";

    void scaleKeypoints(Array<float>& keypoints, const float scale)
    {
        try
        {
            scaleKeypoints(keypoints, scale, scale);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
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

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
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
}
