#include <openpose/utilities/errorAndLog.hpp>
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
    void Rectangle<T>::recenter(const T newWidth, const T newHeight)
    {
        try
        {
            const auto centerPoint = center();
            x = centerPoint.x - T(newWidth / 2.f);
            y = centerPoint.y - T(newHeight / 2.f);
            width = newWidth;
            height = newHeight;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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



    // Static methods
    template<typename T>
    Rectangle<T> recenter(const Rectangle<T>& rectangle, const T newWidth, const T newHeight)
    {
        try
        {
            Rectangle<T> result;
            const auto centerPoint = rectangle.center();
            result.x = centerPoint.x - T(newWidth / 2.f);
            result.y = centerPoint.y - T(newHeight / 2.f);
            result.width = newWidth;
            result.height = newHeight;
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<T>{};
        }
    }

    template Rectangle<char> recenter(const Rectangle<char>& rectangle, const char newWidth, const char newHeight);
    template Rectangle<signed char> recenter(const Rectangle<signed char>& rectangle, const signed char newWidth, const signed char newHeight);
    template Rectangle<short> recenter(const Rectangle<short>& rectangle, const short newWidth, const short newHeight);
    template Rectangle<int> recenter(const Rectangle<int>& rectangle, const int newWidth, const int newHeight);
    template Rectangle<long> recenter(const Rectangle<long>& rectangle, const long newWidth, const long newHeight);
    template Rectangle<long long> recenter(const Rectangle<long long>& rectangle, const long long newWidth, const long long newHeight);
    template Rectangle<unsigned char> recenter(const Rectangle<unsigned char>& rectangle, const unsigned char newWidth, const unsigned char newHeight);
    template Rectangle<unsigned short> recenter(const Rectangle<unsigned short>& rectangle, const unsigned short newWidth, const unsigned short newHeight);
    template Rectangle<unsigned int> recenter(const Rectangle<unsigned int>& rectangle, const unsigned int newWidth, const unsigned int newHeight);
    template Rectangle<unsigned long> recenter(const Rectangle<unsigned long>& rectangle, const unsigned long newWidth, const unsigned long newHeight);
    template Rectangle<unsigned long long> recenter(const Rectangle<unsigned long long>& rectangle, const unsigned long long newWidth, const unsigned long long newHeight);
    template Rectangle<float> recenter(const Rectangle<float>& rectangle, const float newWidth, const float newHeight);
    template Rectangle<double> recenter(const Rectangle<double>& rectangle, const double newWidth, const double newHeight);
    template Rectangle<long double> recenter(const Rectangle<long double>& rectangle, const long double newWidth, const long double newHeight);
}
