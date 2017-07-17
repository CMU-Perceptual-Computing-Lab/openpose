#ifndef OPENPOSE_CORE_MACROS_HPP
#define OPENPOSE_CORE_MACROS_HPP

#ifndef _WIN32
    #define OP_API
#elif defined OP_EXPORTS
    #define OP_API __declspec(dllexport)
#else
    #define OP_API __declspec(dllimport)
#endif

//Disable some Windows Warnings
#ifdef _WIN32
    #pragma warning ( disable : 4251 ) // XXX needs to have dll-interface to be used by clients of class YYY
    #pragma warning( disable: 4275 ) // non dll-interface structXXX used as base
#endif

#define DATUM_BASE_NO_PTR std::vector<Datum>
#define DATUM_BASE std::shared_ptr<DATUM_BASE_NO_PTR>
#define DEFINE_TEMPLATE_DATUM(templateName) template class OP_API templateName<DATUM_BASE>
#define COMPILE_TEMPLATE_DATUM(templateName) extern DEFINE_TEMPLATE_DATUM(templateName)

#define UNUSED(unusedVariable) (void)(unusedVariable)

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)

#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)

#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType OP_API className<char>; \
    template classType OP_API className<signed char>; \
    template classType OP_API className<short>; \
    template classType OP_API className<int>; \
    template classType OP_API className<long>; \
    template classType OP_API className<long long>; \
    template classType OP_API className<unsigned char>; \
    template classType OP_API className<unsigned short>; \
    template classType OP_API className<unsigned int>; \
    template classType OP_API className<unsigned long>; \
    template classType OP_API className<unsigned long long>; \
    template classType OP_API className<float>; \
    template classType OP_API className<double>; \
    template classType OP_API className<long double>

// Includes at the end, since this macros class does not need them, but the files that call this
// file. However, keeping the files at the beginning might create a circular include linking problem.
#include <memory> // std::shared_ptr
#include <vector>
#include <openpose/core/datum.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/rectangle.hpp>

#endif // OPENPOSE_CORE_MACROS_HPP
