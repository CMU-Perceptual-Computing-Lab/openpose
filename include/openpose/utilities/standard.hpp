#ifndef OPENPOSE_UTILITIES_STANDARD_HPP
#define OPENPOSE_UTILITIES_STANDARD_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    bool vectorsAreEqual(const std::vector<T>& vectorA, const std::vector<T>& vectorB)
    {
        try
        {
            if (vectorA.size() != vectorB.size())
                return false;
            else
            {
                for (auto i = 0u ; i < vectorA.size() ; i++)
                    if (vectorA[i] != vectorB[i])
                        return false;
                return true;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
}

#endif // OPENPOSE_UTILITIES_STANDARD_HPP
