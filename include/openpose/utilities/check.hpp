#ifndef OPENPOSE_UTILITIES_CHECK_HPP
#define OPENPOSE_UTILITIES_CHECK_HPP

#include <openpose/core/common.hpp>

namespace op
{
    // CHECK, CHECK_EQ, CHECK_NE, CHECK_LE, CHECK_LT, CHECK_GE, and CHECK_GT
    template<typename T>
    void check(const bool condition, const T& message = "", const int line = -1, const std::string& function = "",
               const std::string& file = "")
    {
        if (!condition)
            error("Check failed: " + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                const std::string& function = "", const std::string& file = "")
    {
        if (conditionA != conditionB)
            error("CheckE failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkNE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                 const std::string& function = "", const std::string& file = "")
    {
        if (conditionA == conditionB)
            error("CheckNE failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkLE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                 const std::string& function = "", const std::string& file = "")
    {
        if (conditionA > conditionB)
            error("CheckLE failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkLT(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                 const std::string& function = "", const std::string& file = "")
    {
        if (conditionA >= conditionB)
            error("CheckLT failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkGE(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                 const std::string& function = "", const std::string& file = "")
    {
        if (conditionA < conditionB)
            error("CheckGE failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }

    template<typename T, typename T1, typename T2>
    void checkGT(const T1& conditionA, const T2& conditionB, const T& message = "", const int line = -1,
                 const std::string& function = "", const std::string& file = "")
    {
        if (conditionA <= conditionB)
            error("CheckGT failed (" + tToString(conditionA) + " vs. " + tToString(conditionB) + "): "
                  + tToString(message), line, function, file);
    }
}

#endif // OPENPOSE_UTILITIES_CHECK_HPP
