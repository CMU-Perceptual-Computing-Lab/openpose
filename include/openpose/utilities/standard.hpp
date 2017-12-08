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

    /**
     * std::vector<T> concatenator.
     * Auxiliary function that concatenate std::vectors of any class type T.
     * It assumes basic copy (ideal for smart pointers, pointers, etc.), so note that the copy still shares the same
     * internal data. It will not work for element that cannot be copied.
     * @param vectorA First std::shared_ptr<T> element to be concatenated.
     * @param vectorB Second std::shared_ptr<T> element to be concatenated.
     * @return Concatenated std::vector<T> of both vectorA and vectorB.
     */
    template <typename T>
    std::vector<T> mergeVectors(const std::vector<T>& vectorA, const std::vector<T>& vectorB)
    {
        try
        {
            auto vectorToReturn(vectorA);
            for (auto& tElement : vectorB)
                vectorToReturn.emplace_back(tElement);
            return vectorToReturn;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<T>{};
        }
    }
}

#endif // OPENPOSE_UTILITIES_STANDARD_HPP
