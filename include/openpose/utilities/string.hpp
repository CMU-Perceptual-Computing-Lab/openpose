#ifndef OPENPOSE_UTILITIES_STRING_HPP
#define OPENPOSE_UTILITIES_STRING_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API unsigned long long getLastNumber(const std::string& string);

    /**
     * This template function turns an integer number into a fixed-length std::string.
     * @param number T integer corresponding to the integer to be formatted.
     * @param stringLength unsigned long long indicating the final length. If 0, the
     * final length is the original number length.
     * @return std::string with the formatted value.
     */
    template<typename T>
    std::string toFixedLengthString(const T number, const unsigned long long stringLength = 0);

    OP_API std::vector<std::string> splitString(const std::string& stringToSplit, const std::string& delimiter);

    OP_API std::string toLower(const std::string& string);

    OP_API std::string toUpper(const std::string& string);

    OP_API std::string remove0sFromString(const std::string& string);

    OP_API std::string getFirstNumberOnString(const std::string& string);
}

#endif // OPENPOSE_UTILITIES_STRING_HPP
