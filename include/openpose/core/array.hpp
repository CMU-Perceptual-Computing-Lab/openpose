#ifndef OPENPOSE_CORE_ARRAY_HPP
#define OPENPOSE_CORE_ARRAY_HPP

#include <memory> // std::shared_ptr
#include <vector>
#include <openpose/core/macros.hpp>
#include <openpose/core/matrix.hpp>
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
    /**
     * Array<T>: The OpenPose Basic Raw Data Container
     * This template class implements a multidimensional data array. It is our basic data container, analogous to
     * Mat in OpenCV, Tensor in Torch/TensorFlow or Blob in Caffe.
     * It wraps a Matrix and a std::shared_ptr, both of them pointing to the same raw data. I.e. they both share the
     * same memory, so we can read and modify this data in both formats with no performance impact.
     * Hence, it keeps high performance while adding high-level functions.
     */
    template<typename T>
    class Array
    {
    public:
        // ------------------------------ Constructors and Data Allocator Functions ------------------------------ //
        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const int size).
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         */
        explicit Array(const int size);

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const std::vector<int>& size = {}).
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to
         * `new T[3*5*2]`.
         */
        explicit Array(const std::vector<int>& sizes = {});

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const int size, const T value).
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         * @param value Initial value for each component of the Array.
         */
        Array(const int size, const T value);

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const std::vector<int>& size, const T value).
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to:
         * `new T[3*5*2]`.
         * @param value Initial value for each component of the Array.
         */
        Array(const std::vector<int>& sizes, const T value);

        /**
         * Array constructor.
         * Equivalent to default constructor, but it does not allocate memory, but rather use dataPtr.
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         * @param dataPtr Pointer to the memory to be used by the Array.
         */
        Array(const int size, T* const dataPtr);

        /**
         * Array constructor.
         * Equivalent to default constructor, but it does not allocate memory, but rather use dataPtr.
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to:
         * `new T[3*5*2]`.
         * @param dataPtr Pointer to the memory to be used by the Array.
         */
        Array(const std::vector<int>& sizes, T* const dataPtr);

        /**
         * Array constructor.
         * @param array Array<T> with the original data array to slice.
         * @param index indicates the index of the array to extract.
         * @param noCopy indicates whether to perform a copy. Copy will never go to undefined behavior, however, if
         * noCopy == true, then:
         *     1. It is faster, as no data copy is involved, but...
         *     2. If the Array array goes out of scope, then the resulting Array will provoke an undefined behavior.
         *     3. If the returned Array is modified, the information in the Array array will also be.
         * @return Array<T> with the same dimension than array expect the first dimension being 1. E.g., if array
         * is {p,k,m}, the resulting Array<T> is {1,k,m}.
         */
        Array(const Array<T>& array, const int index, const bool noCopy = false);

        /**
         * Array constructor. It manually copies the Array<T2> into the new Array<T>
         * @param array Array<T2> with a format T2 different to the current Array type T.
         */
        template<typename T2>
        Array(const Array<T2>& array) :
            Array{array.getSize()}
        {
            try
            {
                // Copy
                for (auto i = 0u ; i < array.getVolume() ; i++)
                    pData[i] = T(array[i]);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        /**
         * Copy constructor.
         * It performs `fast copy`: For performance purpose, copying a Array<T> or Datum or cv::Mat just copies the
         * reference, it still shares the same internal data.
         * Modifying the copied element will modify the original one.
         * Use clone() for a slower but real copy, similarly to cv::Mat and Array<T>.
         * @param array Array to be copied.
         */
        Array<T>(const Array<T>& array);

        /**
         * Copy assignment.
         * Similar to Array<T>(const Array<T>& array).
         * @param array Array to be copied.
         * @return The resulting Array.
         */
        Array<T>& operator=(const Array<T>& array);

        /**
         * Move constructor.
         * It destroys the original Array to be moved.
         * @param array Array to be moved.
         */
        Array<T>(Array<T>&& array);

        /**
         * Move assignment.
         * Similar to Array<T>(Array<T>&& array).
         * @param array Array to be moved.
         * @return The resulting Array.
         */
        Array<T>& operator=(Array<T>&& array);

        /**
         * Clone function.
         * Similar to cv::Mat::clone and Datum::clone.
         * It performs a real but slow copy of the data, i.e., even if the copied element is modified, the original
         * one is not.
         * @return The resulting Array.
         */
        Array<T> clone() const;

        /**
         * Data allocation function.
         * It allocates the required space for the memory (it does not initialize that memory).
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         */
        void reset(const int size);

        /**
         * Data allocation function.
         * Similar to reset(const int size), but it allocates a multi-dimensional array of dimensions each of the
         * values of the argument.
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to
         * `new T[3*5*2]`.
         */
        void reset(const std::vector<int>& sizes = {});

        /**
         * Data allocation function.
         * Similar to reset(const int size), but initializing the data to the value specified by the second argument.
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         * @param value Initial value for each component of the Array.
         */
        void reset(const int size, const T value);

        /**
         * Data allocation function.
         * Similar to reset(const std::vector<int>& size), but initializing the data to the value specified by the
         * second argument.
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to
         * `new T[3*5*2]`.
         * @param value Initial value for each component of the Array.
         */
        void reset(const std::vector<int>& sizes, const T value);

        /**
         * Data allocation function.
         * Equivalent to default constructor, but it does not allocate memory, but rather use dataPtr.
         * @param size Integer with the number of T element to be allocated. E.g., size = 5 is internally similar to
         * `new T[5]`.
         * @param dataPtr Pointer to the memory to be used by the Array.
         */
        void reset(const int size, T* const dataPtr);

        /**
         * Data allocation function.
         * Equivalent to default constructor, but it does not allocate memory, but rather use dataPtr.
         * @param sizes Vector with the size of each dimension. E.g., size = {3, 5, 2} is internally similar to:
         * `new T[3*5*2]`.
         * @param dataPtr Pointer to the memory to be used by the Array.
         */
        void reset(const std::vector<int>& sizes, T* const dataPtr);

        /**
         * Data allocation function.
         * It internally allocates memory and copies the data of the argument to the Array allocated memory.
         * @param cvMat Matrix to be copied.
         */
        void setFrom(const Matrix& cvMat);

        /**
         * Data allocation function.
         * It internally assigns all the allocated memory to the value indicated by the argument.
         * @param value Value for each component of the Array.
         */
        void setTo(const T value);



        // ------------------------------ Data Information Functions ------------------------------ //
        /**
         * Check whether memory has been allocated.
         * @return True if no memory has been allocated, false otherwise.
         */
        inline bool empty() const
        {
            return (mVolume == 0);
        }

        /**
         * Return a vector with the size of each dimension allocated.
         * @return A std::vector<int> with the size of each dimension. If no memory has been allocated, it will return
         * an empty std::vector.
         */
        inline std::vector<int> getSize() const
        {
            return mSize;
        }

        /**
         * Return a vector with the size of the desired dimension.
         * @param index Dimension to check its size.
         * @return Size of the desired dimension. It will return 0 if the requested dimension is higher than the number
         * of dimensions.
         */
        int getSize(const int index) const;

        /**
         * Return a string with the size of each dimension allocated.
         * @return A std::stringwith the size of each dimension. If no memory has been allocated, it will return an
         * empty string.
         */
        std::string printSize() const;

        /**
         * Return the total number of dimensions, equivalent to getSize().size().
         * @return The number of dimensions. If no memory is allocated, it returns 0.
         */
        inline size_t getNumberDimensions() const
        {
            return mSize.size();
        }

        /**
         * Return the total number of elements allocated, equivalent to multiply all the components from getSize().
         * E.g., for a Array<T> of size = {2,5,3}, the volume or total number of elements is: 2x5x3 = 30.
         * @return The total volume of the allocated data. If no memory is allocated, it returns 0.
         */
        inline size_t getVolume() const
        {
            return mVolume;
        }

        /**
         * Similar to getVolume(), but in this case it just returns the volume between the desired dimensions.
         * E.g., for a Array<T> of size = {2,5,3}, the volume or total number of elements for getVolume(1,2) is
         * 5x3 = 15.
         * @param indexA Dimension where to start.
         * @param indexB Dimension where to stop. If indexB == -1, then it will take up to the last dimension.
         * @return The total volume of the allocated data between the desired dimensions. If the index are out of
         * bounds, it throws an error.
         */
        size_t getVolume(const int indexA, const int indexB = -1) const;

        /**
         * Return the stride or step size of the array.
         * E.g., given and Array<T> of size 5x3, getStride() would return the following vector:
         * {5x3sizeof(T), 3sizeof(T), sizeof(T)}.
         */
        std::vector<int> getStride() const;

        /**
         * Return the stride or step size of the array at the index-th dimension.
         * E.g., given and Array<T> of size 5x3, getStride(2) would return sizeof(T).
         */
        int getStride(const int index) const;



        // ------------------------------ Data Access Functions And Operators ------------------------------ //
        /**
         * Return a raw pointer to the data. Similar to: std::shared_ptr::get().
         * Note: if you modify the pointer data, you will directly modify it in the Array<T> instance too.
         * If you know you do not want to modify the data, then use getConstPtr() instead.
         * @return A raw pointer to the data.
         */
        inline T* getPtr()
        {
            return pData; // spData.get()
        }

        /**
         * Similar to getPtr(), but it forbids the data to be edited.
         * @return A raw const pointer to the data.
         */
        inline const T* getConstPtr() const
        {
            return pData; // spData.get()
        }

        /**
         * Similar to getConstPtr(), but it allows the data to be edited.
         * This function is only implemented for Pybind11 usage.
         * @return A raw pointer to the data.
         */
        inline T* getPseudoConstPtr() const
        {
            return pData; // spData.get()
        }

        /**
         * Return a Matrix wrapper to the data. It forbids the data to be modified.
         * OpenCV only admits unsigned char, signed char, int, float & double. If the T class is not supported by
         * OpenCV, it will throw an error.
         * Note: Array<T> does not return an editable Matrix because some OpenCV functions reallocate memory and it
         * would not longer point to the Array<T> instance.
         * If you want to perform some OpenCV operation on the Array data, you can use:
         *     editedCvMat = array.getConstCvMat().clone();
         *     // modify data
         *     array.setFrom(editedCvMat)
         * @return A const Matrix pointing to the data.
         */
        const Matrix& getConstCvMat() const;

        /**
         * Analogous to getConstCvMat, but in this case it returns a editable Matrix.
         * Very important: Only allowed functions which do not provoke data reallocation.
         * E.g., resizing functions will not work and they would provoke an undefined behavior and/or execution
         * crashes.
         * @return A Matrix pointing to the data.
         */
        Matrix& getCvMat();

        /**
         * [] operator
         * Similar to the [] operator for raw pointer data.
         * If debug mode is enabled, then it will check that the desired index is in the data range, and it will throw
         * an exception otherwise (similar to the at operator).
         * @param index The desired memory location.
         * @return A editable reference to the data on the desired index location.
         */
        inline T& operator[](const int index)
        {
            #ifdef NDEBUG
                return pData[index]; // spData.get()[index]
            #else
                return at(index);
            #endif
        }

        /**
         * [] operator
         * Same functionality as operator[](const int index), but it forbids modifying the value. Otherwise, const
         * functions would not be able to call the [] operator.
         * @param index The desired memory location.
         * @return A non-editable reference to the data on the desired index location.
         */
        inline const T& operator[](const int index) const
        {
            #ifdef NDEBUG
                return pData[index]; // spData.get()[index]
            #else
                return at(index);
            #endif
        }

        /**
         * [] operator
         * Same functionality as operator[](const int index), but it lets the user introduce the multi-dimensional
         * index.
         * E.g., given a (10 x 10 x 10) array, array[11] is equivalent to array[{1,1,0}]
         * @param indexes Vector with the desired memory location.
         * @return A editable reference to the data on the desired index location.
         */
        inline T& operator[](const std::vector<int>& indexes)
        {
            return operator[](getIndex(indexes));
        }

        /**
         * [] operator
         * Same functionality as operator[](const std::vector<int>& indexes), but it forbids modifying the value.
         * Otherwise, const functions would not be able to call the [] operator.
         * @param indexes Vector with the desired memory location.
         * @return A non-editable reference to the data on the desired index location.
         */
        inline const T& operator[](const std::vector<int>& indexes) const
        {
            return operator[](getIndex(indexes));
        }

        /**
         * at() function
         * Same functionality as operator[](const int index), but it always check whether the indexes are within the
         * data bounds. Otherwise, it will throw an error.
         * @param index The desired memory location.
         * @return A editable reference to the data on the desired index location.
         */
        inline T& at(const int index)
        {
            return commonAt(index);
        }

        /**
         * at() function
         * Same functionality as operator[](const int index) const, but it always check whether the indexes are within
         * the data bounds. Otherwise, it will throw an error.
         * @param index The desired memory location.
         * @return A non-editable reference to the data on the desired index location.
         */
        inline const T& at(const int index) const
        {
            return commonAt(index);
        }

        /**
         * at() function
         * Same functionality as operator[](const std::vector<int>& indexes), but it always check whether the indexes
         * are within the data bounds. Otherwise, it will throw an error.
         * @param indexes Vector with the desired memory location.
         * @return A editable reference to the data on the desired index location.
         */
        inline T& at(const std::vector<int>& indexes)
        {
            return at(getIndexAndCheck(indexes));
        }

        /**
         * at() function
         * Same functionality as operator[](const std::vector<int>& indexes) const, but it always check whether the
         * indexes are within the data bounds. Otherwise, it will throw an error.
         * @param indexes Vector with the desired memory location.
         * @return A non-editable reference to the data on the desired index location.
         */
        inline const T& at(const std::vector<int>& indexes) const
        {
            return at(getIndexAndCheck(indexes));
        }

        /**
         * It returns a string with the whole array data. Useful for debugging.
         * The format is: values separated by a space, and a enter for each dimension. E.g.,
         * For the Array{2, 2, 3}, it will print:
         * Array<T>::toString():
         * x1 x2 x3
         * x4 x5 x6
         *
         * x7 x8 x9
         * x10 x11 x12
         * @return A string with the array values in the above format.
         */
        const std::string toString() const;

    private:
        std::vector<int> mSize;
        size_t mVolume;
        std::shared_ptr<T> spData;
        T* pData; // pData is a wrapper of spData. Used for Pybind11 binding.
        std::pair<bool, Matrix> mCvMatData;

        /**
         * Auxiliary function that both operator[](const std::vector<int>& indexes) and
         * operator[](const std::vector<int>& indexes) const use.
         * It turn the multi-dimensions indexes into the 1-dimension equivalent index.
         * @param indexes Vector with the desired memory location.
         * @return The equivalent 1-D index.
         */
        int getIndex(const std::vector<int>& indexes) const;

        /**
         * Similar to getIndex(const std::vector<int>& indexes) const, but used for at(const std::vector<int>& indexes)
         * and at(const std::vector<int>& indexes) const.
         * It also checks whether the index is within the allocated memory.
         * @param indexes Vector with the desired memory location.
         * @return The equivalent 1-D index.
         */
        int getIndexAndCheck(const std::vector<int>& indexes) const;

        /**
         * Auxiliary function that both at(const int index) and at(const int index) const use.
         * @param index The desired memory location.
         * @return A non-editable reference to the data on the desired index location.
         */
        T& commonAt(const int index) const;

        void resetAuxiliary(const std::vector<int>& sizes, T* const dataPtr = nullptr);
    };

    // Static methods
    OVERLOAD_C_OUT(Array)
}

#endif // OPENPOSE_CORE_ARRAY_HPP
