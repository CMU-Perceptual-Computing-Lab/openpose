#ifndef OPENPOSE_UTILITIES_MATH_HPP
#define OPENPOSE_UTILITIES_MATH_HPP

namespace op
{
    // Use op::round/max/min for basic types (int, char, long, float, double, etc). Never with classes!
    // `std::` alternatives uses 'const T&' instead of 'const T' as argument.
    // E.g., std::round is really slow (~300 ms vs ~10 ms when I individually apply it to each element of a whole
    // image array

    // VERY IMPORTANT: These fast functions does NOT work for negative integer numbers.
    // E.g., positiveIntRound(-180.f) = -179.

    // Round functions
    // Signed
    template<typename T>
    inline char positiveCharRound(const T a)
    {
        return char(a+0.5f);
    }

    template<typename T>
    inline signed char positiveSCharRound(const T a)
    {
        return (signed char)(a+0.5f);
    }

    template<typename T>
    inline int positiveIntRound(const T a)
    {
        return int(a+0.5f);
    }

    template<typename T>
    inline long positiveLongRound(const T a)
    {
        return long(a+0.5f);
    }

    template<typename T>
    inline long long positiveLongLongRound(const T a)
    {
        return (long long)(a+0.5f);
    }

    // Unsigned
    template<typename T>
    inline unsigned char uCharRound(const T a)
    {
        return (unsigned char)(a+0.5f);
    }

    template<typename T>
    inline unsigned int uIntRound(const T a)
    {
        return (unsigned int)(a+0.5f);
    }

    template<typename T>
    inline unsigned long ulongRound(const T a)
    {
        return (unsigned long)(a+0.5f);
    }

    template<typename T>
    inline unsigned long long uLongLongRound(const T a)
    {
        return (unsigned long long)(a+0.5f);
    }

    // Max/min functions
    template<typename T>
    inline T fastMax(const T a, const T b)
    {
        return (a > b ? a : b);
    }

    template<typename T>
    inline T fastMin(const T a, const T b)
    {
        return (a < b ? a : b);
    }

    template<class T>
    inline T fastTruncate(T value, T min = 0, T max = 1)
    {
        return fastMin(max, fastMax(min, value));
    }
}

#endif // OPENPOSE_UTILITIES_MATH_HPP
