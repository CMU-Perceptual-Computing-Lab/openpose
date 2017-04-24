#ifndef OPENPOSE__UTILITIES__MACROS_HPP
#define OPENPOSE__UTILITIES__MACROS_HPP

#include <memory> // std::shared_ptr
#include <vector>
#include "../core/datum.hpp"

#define DATUM_BASE_NO_PTR std::vector<Datum>
#define DATUM_BASE std::shared_ptr<DATUM_BASE_NO_PTR>
#define DEFINE_TEMPLATE_DATUM(templateName) template class templateName<DATUM_BASE>
#define COMPILE_TEMPLATE_DATUM(templateName) extern DEFINE_TEMPLATE_DATUM(templateName)

#define UNUSED(unusedVariable) (void)(unusedVariable)

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

#define COMPILE_TEMPLATE_BASIC_TYPES(className) \
    template class className<char>; \
    template class className<signed char>; \
    template class className<short>; \
    template class className<int>; \
    template class className<long>; \
    template class className<long long>; \
    template class className<unsigned char>; \
    template class className<unsigned short>; \
    template class className<unsigned int>; \
    template class className<unsigned long>; \
    template class className<unsigned long long>; \
    template class className<float>; \
    template class className<double>; \
    template class className<long double>

#endif // OPENPOSE__UTILITIES__MACROS_HPP
