#ifndef OPENPOSE_UTILITIES_MACROS_HPP
#define OPENPOSE_UTILITIES_MACROS_HPP

#include <memory> // std::shared_ptr
#include <vector>
#include <openpose/core/datum.hpp>

#define DATUM_BASE_NO_PTR std::vector<Datum>
#define DATUM_BASE std::shared_ptr<DATUM_BASE_NO_PTR>
#define DEFINE_TEMPLATE_DATUM(templateName) template class templateName<DATUM_BASE>
#define COMPILE_TEMPLATE_DATUM(templateName) extern DEFINE_TEMPLATE_DATUM(templateName)

#define UNUSED(unusedVariable) (void)(unusedVariable)

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)

#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)

#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType className<char>; \
    template classType className<signed char>; \
    template classType className<short>; \
    template classType className<int>; \
    template classType className<long>; \
    template classType className<long long>; \
    template classType className<unsigned char>; \
    template classType className<unsigned short>; \
    template classType className<unsigned int>; \
    template classType className<unsigned long>; \
    template classType className<unsigned long long>; \
    template classType className<float>; \
    template classType className<double>; \
    template classType className<long double>

#endif // OPENPOSE_UTILITIES_MACROS_HPP
