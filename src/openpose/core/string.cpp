#include <openpose/core/string.hpp>
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
    struct String::ImplString
    {
        std::string mString;

        ImplString()
        {
        }

        ImplString(const char* charPtr) :
            mString{charPtr}
        {
        }
    };

    String::String() :
        spImpl{std::make_shared<ImplString>()}
    {
    }

    String::String(const char* charPtr) :
        spImpl{std::make_shared<ImplString>(charPtr)}
    {
    }

    String::String(const std::string& string) :
        spImpl{std::make_shared<ImplString>(string.c_str())}
    {
    }

    const std::string& String::getStdString() const
    {
        try
        {
            return spImpl->mString;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mString;
        }
    }

    bool String::empty() const
    {
        try
        {
            return spImpl->mString.empty();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return true;
        }
    }
}
