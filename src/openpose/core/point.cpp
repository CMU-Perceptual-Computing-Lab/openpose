#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/core/point.hpp>

namespace op
{
    template<typename T>
    Point<T>::Point(const T x_, const T y_) :
        x{x_},
        y{y_}
    {
    }

    template<typename T>
    Point<T>::Point(const Point<T>& point)
    {
        try
        {
            x = point.x;
            y = point.y;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator=(const Point<T>& point)
    {
        try
        {
            x = point.x;
            y = point.y;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>::Point(Point<T>&& point)
    {
        try
        {
            x = point.x;
            y = point.y;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator=(Point<T>&& point)
    {
        try
        {
            x = point.x;
            y = point.y;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator+=(const Point<T>& point)
    {
        try
        {
            x += point.x;
            y += point.y;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator+=(const T value)
    {
        try
        {
            x += value;
            y += value;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator-=(const Point<T>& point)
    {
        try
        {
            x -= point.x;
            y -= point.y;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator-=(const T value)
    {
        try
        {
            x -= value;
            y -= value;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator*=(const T value)
    {
        try
        {
            x *= value;
            y *= value;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    Point<T>& Point<T>::operator/=(const T value)
    {
        try
        {
            x /= value;
            y /= value;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    COMPILE_TEMPLATE_BASIC_TYPES(Point);
}
