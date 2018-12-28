#include <algorithm> // std::transform
#include <openpose/utilities/string.hpp>

namespace op
{
    unsigned long long getLastNumber(const std::string& string)
    {
        try
        {
            const auto stringNumber = string.substr(string.find_last_not_of("0123456789") + 1);
            return std::stoull(stringNumber);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0ull;
        }
    }

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
    template OP_API std::string toFixedLengthString<char>(const char number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<signed char>(const signed char number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<short>(const short number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<int>(const int number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<long>(const long number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<long long>(const long long number, const unsigned long long stringLength);
    // Unsigned
    template OP_API std::string toFixedLengthString<unsigned char>(const unsigned char number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<unsigned short>(const unsigned short number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<unsigned int>(const unsigned int number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<unsigned long>(const unsigned long number, const unsigned long long stringLength);
    template OP_API std::string toFixedLengthString<unsigned long long>(const unsigned long long number, const unsigned long long stringLength);

    std::vector<std::string> splitString(const std::string& stringToSplit, const std::string& delimiter)
    {
        try
        {
            std::vector<std::string> result;
            size_t pos = 0;
            auto stringToSplitAux = stringToSplit;
            while ((pos = stringToSplitAux.find(delimiter)) != std::string::npos)
            {
                result.emplace_back(stringToSplitAux.substr(0, pos));
                stringToSplitAux.erase(0, pos + delimiter.length());
            }
            result.emplace_back(stringToSplitAux);
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string toLower(const std::string& string)
    {
        try
        {
            auto result = string;
            std::transform(string.begin(), string.end(), result.begin(), tolower);
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string toUpper(const std::string& string)
    {
        try
        {
            auto result = string;
            std::transform(string.begin(), string.end(), result.begin(), toupper);
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }
}
