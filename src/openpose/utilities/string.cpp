#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/string.hpp>

namespace op
{
    template<typename T>
    std::string toFixedLengthString(const T number, const unsigned long long stringLength)
    {
        try
        {
            const auto numberAsString = std::to_string(number);
            if (stringLength > 0)
            {
                if (number < 0)
                    error("toFixedLengthString: number cannot be <= 0, in this case it is: " + numberAsString + ".", __LINE__, __FUNCTION__, __FILE__);

                const auto zerosToAdd = stringLength - numberAsString.size();
                if (zerosToAdd < 0)
                {
                    const auto errorMessage = "toFixedLengthString: number greater than maximum number of digits (stringLength): "
                                            + numberAsString + " vs. " + std::to_string(stringLength) + ".";
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                }

                return { std::string(zerosToAdd, '0') + numberAsString};
            }
            else
                return numberAsString;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    // Signed
    template std::string toFixedLengthString<char>(const char number, const unsigned long long stringLength);
    template std::string toFixedLengthString<signed char>(const signed char number, const unsigned long long stringLength);
    template std::string toFixedLengthString<short>(const short number, const unsigned long long stringLength);
    template std::string toFixedLengthString<int>(const int number, const unsigned long long stringLength);
    template std::string toFixedLengthString<long>(const long number, const unsigned long long stringLength);
    template std::string toFixedLengthString<long long>(const long long number, const unsigned long long stringLength);
    // Unsigned
    template std::string toFixedLengthString<unsigned char>(const unsigned char number, const unsigned long long stringLength);
    template std::string toFixedLengthString<unsigned short>(const unsigned short number, const unsigned long long stringLength);
    template std::string toFixedLengthString<unsigned int>(const unsigned int number, const unsigned long long stringLength);
    template std::string toFixedLengthString<unsigned long>(const unsigned long number, const unsigned long long stringLength);
    template std::string toFixedLengthString<unsigned long long>(const unsigned long long number, const unsigned long long stringLength);
}
