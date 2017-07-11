#ifndef OPENPOSE_UTILITIES_STRING_HPP
#define OPENPOSE_UTILITIES_STRING_HPP

#include <string>
#include <vector>
#include <openpose/core/macros.hpp>

namespace op
{
    /**
     * This template function turns an integer number into a fixed-length std::string.
     * @param number T integer corresponding to the integer to be formatted.
     * @param stringLength unsigned long long indicating the final length. If 0, the
     * final length is the original number length.
     * @return std::string with the formatted value.
     */
    template<typename T>
    OP_API std::string toFixedLengthString(const T number, const unsigned long long stringLength = 0);

    OP_API std::vector<std::string> splitString(const std::string& stringToSplit, const std::string& delimiter);
}

#endif // OPENPOSE_UTILITIES_STRING_HPP
