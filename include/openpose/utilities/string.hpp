#ifndef OPENPOSE_UTILITIES_STRING_HPP
#define OPENPOSE_UTILITIES_STRING_HPP

#include <string>

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
    std::string toFixedLengthString(const T number, const unsigned long long stringLength = 0);
}

#endif // OPENPOSE_UTILITIES_STRING_HPP
