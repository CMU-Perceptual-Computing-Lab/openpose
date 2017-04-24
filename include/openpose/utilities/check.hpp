#ifndef OPENPOSE__UTILITIES__CHECK_HPP
#define OPENPOSE__UTILITIES__CHECK_HPP

#include <string>
#include "errorAndLog.hpp"

namespace op
{
    // CHECK, CHECK_EQ, CHECK_NE, CHECK_LE, CHECK_LT, CHECK_GE, and CHECK_GT
    template<typename T>
    inline void check(const bool condition, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (!condition)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA != conditionB)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkNE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA == conditionB)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkLE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA > conditionB)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkLT(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA >= conditionB)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkGE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA < conditionB)
            error(message, line, function, file);
    }

    template<typename T, typename T1, typename T2>
    inline void checkGT(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        if (conditionA <= conditionB)
            error(message, line, function, file);
    }
}

#endif // OPENPOSE__UTILITIES__CHECK_HPP
