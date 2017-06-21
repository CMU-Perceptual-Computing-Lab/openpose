#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/core/rectangle.hpp>

namespace op
{
    template<typename T>
    Rectangle<T>::Rectangle(const T x_, const T y_, const T width_, const T height_) :
        x{x_},
        y{y_},
        width{width_},
        height{height_}
    {
    }

    template<typename T>
    Rectangle<T>::Rectangle(const Rectangle<T>& rectangle)
    {
        try
        {
            x = rectangle.x;
            y = rectangle.y;
            width = rectangle.width;
            height = rectangle.height;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Rectangle<T>& Rectangle<T>::operator=(const Rectangle<T>& rectangle)
    {
        try
        {
            x = rectangle.x;
            y = rectangle.y;
            width = rectangle.width;
            height = rectangle.height;
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
    Rectangle<T>::Rectangle(Rectangle<T>&& rectangle)
    {
        try
        {
            x = rectangle.x;
            y = rectangle.y;
            width = rectangle.width;
            height = rectangle.height;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Rectangle<T>& Rectangle<T>::operator=(Rectangle<T>&& rectangle)
    {
        try
        {
            x = rectangle.x;
            y = rectangle.y;
            width = rectangle.width;
            height = rectangle.height;
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
    Point<T> Rectangle<T>::center() const
    {
        try
        {
            return Point<T>{T(x + width / 2), T(y + height / 2)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<T>{};
        }
    }

    template<typename T>
    Point<T> Rectangle<T>::bottomRight() const
    {
        try
        {
            return Point<T>{T(x + width), T(y + height)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<T>{};
        }
    }

    template<typename T>
    std::string Rectangle<T>::toString() const
    {
        try
        {
            return '[' + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(width) + ", " + std::to_string(height) + ']';
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    template<typename T>
    Rectangle<T>& Rectangle<T>::operator*=(const T value)
    {
        try
        {
            x *= value;
            y *= value;
            width *= value;
            height *= value;
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
    Rectangle<T> Rectangle<T>::operator*(const T value) const
    {
        try
        {
            return Rectangle<T>{T(x * value), T(y * value), T(width * value), T(height * value)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<T>{};
        }
    }

    template<typename T>
    Rectangle<T>& Rectangle<T>::operator/=(const T value)
    {
        try
        {
            x /= value;
            y /= value;
            width /= value;
            height /= value;
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
    Rectangle<T> Rectangle<T>::operator/(const T value) const
    {
        try
        {
            return Rectangle<T>{T(x / value), T(y / value), T(width / value), T(height / value)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<T>{};
        }
    }

    COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(Rectangle);
}
