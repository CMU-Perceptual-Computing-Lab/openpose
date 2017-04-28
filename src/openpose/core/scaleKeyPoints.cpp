#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/scaleKeyPoints.hpp"

namespace op
{
    const std::string errorMessage = "This function is only for array of dimension: [sizeA x sizeB x 3].";

    void scaleKeyPoints(Array<float>& keyPoints, const double scale)
    {
        try
        {
            scaleKeyPoints(keyPoints, scale, scale);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeyPoints(Array<float>& keyPoints, const double scaleX, const double scaleY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keyPoints.empty() && keyPoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keyPoints.getSize(0);
                const auto numberParts = keyPoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        keyPoints[finalIndex] *= scaleX;
                        keyPoints[finalIndex+1] *= scaleY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeyPoints(Array<float>& keyPoints, const double scaleX, const double scaleY, const double offsetX, const double offsetY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keyPoints.empty() && keyPoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keyPoints.getSize(0);
                const auto numberParts = keyPoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        keyPoints[finalIndex] = keyPoints[finalIndex] * scaleX + offsetX;
                        keyPoints[finalIndex+1] = keyPoints[finalIndex+1] * scaleY + offsetY;
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
