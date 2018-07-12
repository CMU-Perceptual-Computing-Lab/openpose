#ifndef OPENPOSE_CORE_POINT_HPP
#define OPENPOSE_CORE_POINT_HPP

#include <string>
#include <openpose/core/macros.hpp>

namespace op
{
    template<typename T>
    struct Point
    {
        T x;
        T y;

        Point(const T x = 0, const T y = 0);

        /**
         * Copy constructor.
         * It performs `fast copy`: For performance purpose, copying a Point<T> or Point<T> or cv::Mat just copies the
         * reference, it still shares the same internal data.
         * Modifying the copied element will modify the original one.
         * Use clone() for a slower but real copy, similarly to cv::Mat and Point<T>.
         * @param point Point to be copied.
         */
        Point<T>(const Point<T>& point);

        /**
         * Copy assignment.
         * Similar to Point<T>(const Point<T>& point).
         * @param point Point to be copied.
         * @return The resulting Point.
         */
        Point<T>& operator=(const Point<T>& point);

        /**
         * Move constructor.
         * It destroys the original Point to be moved.
         * @param point Point to be moved.
         */
        Point<T>(Point<T>&& point);

        /**
         * Move assignment.
         * Similar to Point<T>(Point<T>&& point).
         * @param point Point to be moved.
         * @return The resulting Point.
         */
        Point<T>& operator=(Point<T>&& point);

        inline T area() const
        {
            return x * y;
        }

        /**
         * It returns a string with the whole Point<T> data. Useful for debugging.
         * The format is: `[x, y]`
         * @return A string with the Point<T> values in the above format.
         */
        std::string toString() const;





        // ------------------------------ Comparison operators ------------------------------ //
        /**
         * Less comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator<(const Point<T>& point) const
        {
            return area() < point.area();
        }

        /**
         * Greater comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator>(const Point<T>& point) const
        {
            return area() > point.area();
        }

        /**
         * Less or equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator<=(const Point<T>& point) const
        {
            return area() <= point.area();
        }

        /**
         * Greater or equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator>=(const Point<T>& point) const
        {
            return area() >= point.area();
        }

        /**
         * Equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator==(const Point<T>& point) const
        {
            return area() == point.area();
        }

        /**
         * Not equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator!=(const Point<T>& point) const
        {
            return area() != point.area();
        }





        // ------------------------------ Basic Operators ------------------------------ //
        Point<T>& operator+=(const Point<T>& point);

        Point<T> operator+(const Point<T>& point) const;

        Point<T>& operator+=(const T value);

        Point<T> operator+(const T value) const;

        Point<T>& operator-=(const Point<T>& point);

        Point<T> operator-(const Point<T>& point) const;

        Point<T>& operator-=(const T value);

        Point<T> operator-(const T value) const;

        Point<T>& operator*=(const T value);

        Point<T> operator*(const T value) const;

        Point<T>& operator/=(const T value);

        Point<T> operator/(const T value) const;
    };

    // Static methods
    OVERLOAD_C_OUT(Point)
}

#endif // OPENPOSE_CORE_POINT_HPP
