#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/scalePose.hpp"

namespace op
{
    const std::string errorMessage = "This function is only for array of dimension: [sizeA x sizeB x 3].";

    void scalePose(Array<float>& pose, const double scale)
    {
        try
        {
            scalePose(pose, scale, scale);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scalePose(Array<float>& pose, const double scaleX, const double scaleY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!pose.empty() && pose.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = pose.getSize(0);
                const auto numberParts = pose.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        pose[finalIndex] *= scaleX;
                        pose[finalIndex+1] *= scaleY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scalePose(Array<float>& pose, const double scaleX, const double scaleY, const double offsetX, const double offsetY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!pose.empty() && pose.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = pose.getSize(0);
                const auto numberParts = pose.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        pose[finalIndex] = pose[finalIndex] * scaleX + offsetX;
                        pose[finalIndex+1] = pose[finalIndex+1] * scaleY + offsetY;
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
