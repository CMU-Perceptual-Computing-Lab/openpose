#include <openpose/utilities/string.hpp>
#include <algorithm> // std::transform
#include <cctype> // std::tolower, std::toupper
#include <locale> // std::tolower, std::toupper

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
            std::string result = string;
            std::transform(string.begin(), string.end(), result.begin(),
                [](unsigned char c) { return (unsigned char)std::tolower(c); });
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
            std::string result = string;
            std::transform(string.begin(), string.end(), result.begin(),
                [](unsigned char c) { return (unsigned char)std::toupper(c); });
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string remove0sFromString(const std::string& string)
    {
        try
        {
            std::string stringNo0s;
            if (string[0] == '0')
            {
                // Find first not 0
                const std::size_t found = string.find_first_not_of("0");
                if (found == std::string::npos)
                    error("This should not happen.", __LINE__, __FUNCTION__, __FILE__);
                // Make sure that 0 is not the only digit
                if (string.size() > found && std::isdigit(string[found]))
                    stringNo0s = string.substr(found);
                else
                    stringNo0s = string.substr(found-1);
            }
            else
                stringNo0s = string;
            return stringNo0s;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::string getFirstNumberOnString(const std::string& string)
    {
        try
        {
            const std::size_t found = string.find_first_not_of("0123456789");
            if (found == std::string::npos)
                error("This should not happen.", __LINE__, __FUNCTION__, __FILE__);
            return string.substr(0, found);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }
}
