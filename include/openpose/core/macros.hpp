#ifndef OPENPOSE_CORE_MACROS_HPP
#define OPENPOSE_CORE_MACROS_HPP

#include <chrono> // std::chrono:: functionaligy, e.g., std::chrono::milliseconds
#include <memory> // std::shared_ptr
#include <ostream>
#include <string>
#include <thread> // std::this_thread
#include <vector>

// OpenPose name and version
const std::string OPEN_POSE_NAME_STRING = "OpenPose";
const std::string OPEN_POSE_VERSION_STRING = "1.5.1";
const std::string OPEN_POSE_NAME_AND_VERSION = OPEN_POSE_NAME_STRING + " " + OPEN_POSE_VERSION_STRING;
// #define COMMERCIAL_LICENSE

#ifndef _WIN32
    #define OP_API
#elif defined OP_EXPORTS
    #define OP_API __declspec(dllexport)
#else
    #define OP_API __declspec(dllimport)
#endif

// Disable some Windows Warnings
#ifdef _WIN32
    #pragma warning(disable: 4251) // 'XXX': class 'YYY' needs to have dll-interface to be used by clients of class 'ZZZ'
#endif

#define UNUSED(unusedVariable) (void)(unusedVariable)

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

// Instantiate a class with all the basic types
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

// Instantiate a class with float and double specifications
#define COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(className) COMPILE_TEMPLATE_FLOATING_TYPES(className, class)
#define COMPILE_TEMPLATE_FLOATING_TYPES_STRUCT(className) COMPILE_TEMPLATE_FLOATING_TYPES(className, struct)
#define COMPILE_TEMPLATE_FLOATING_TYPES(className, classType) \
  char gInstantiationGuard##className; \
  template classType OP_API className<float>; \
  template classType OP_API className<double>

// Instantiate a class with float and double specifications
#define COMPILE_TEMPLATE_FLOATING_INT_TYPES_CLASS(className) COMPILE_TEMPLATE_FLOATING_INT_TYPES(className, class)
#define COMPILE_TEMPLATE_FLOATING_INT_TYPES_STRUCT(className) COMPILE_TEMPLATE_FLOATING_INT_TYPES(className, struct)
#define COMPILE_TEMPLATE_FLOATING_INT_TYPES(className, classType) \
  char gInstantiationGuard##className; \
  template classType OP_API className<int>; \
  template classType OP_API className<unsigned int>; \
  template classType OP_API className<float>; \
  template classType OP_API className<double>

/**
 * cout operator overload calling toString() function
 * @return std::ostream containing output from toString()
 */
#define OVERLOAD_C_OUT(className) \
    template<typename T> std::ostream &operator<<(std::ostream& ostream, const op::className<T>& obj) \
    { \
        ostream << obj.toString(); \
        return ostream; \
    }

// PIMPL does not work if function arguments need the 3rd-party class. Alternative:
// stackoverflow.com/questions/13978775/how-to-avoid-include-dependency-to-external-library?answertab=active#tab-top
struct dim3;

#endif // OPENPOSE_CORE_MACROS_HPP
