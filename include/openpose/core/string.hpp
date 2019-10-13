#ifndef OPENPOSE_CORE_STRING_HPP
#define OPENPOSE_CORE_STRING_HPP

#include <memory> // std::shared_ptr
#include <string>
#include <openpose/core/macros.hpp>

namespace op
{
    /**
     * String: Basic container for std::string to avoid std::string in the WrapperStructXXX classes. Otherwise,
     * cryptic runtime DLL errors could occur when exporting OpenPose to other projects using different STL DLLs.
     */
    class OP_API String
    {
    public:
        String();

        /**
         * It will force a copy of the char* of std::string to avoid DLL runtime errors. Example usages:
         * std::string stdString = "This is a std::string"; 
         * String string(stdString.c_str()); 
         */
        String(const char* charPtr);

        /**
         * It will force a copy of string
         */
        explicit String(const std::string& string);

        const std::string& getStdString() const;

        bool empty() const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplString;
        std::shared_ptr<ImplString> spImpl;
    };
}

#endif // OPENPOSE_CORE_STRING_HPP
