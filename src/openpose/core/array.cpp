#include <typeinfo> // typeid
#include <numeric> // std::accumulate
#include <openpose/core/macros.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/array.hpp>

namespace op
{
    template<typename T>
    Array<T>::Array(const int size)
    {
        try
        {
            reset(size);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const std::vector<int>& sizes)
    {
        try
        {
            reset(sizes);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const int size, const T value)
    {
        try
        {
            reset(size, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const std::vector<int>& sizes, const T value)
    {
        try
        {
            reset(sizes, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const Array<T>& array) :
        mSize{array.mSize},
        mVolume{array.mVolume},
        spData{array.spData},
        mCvMatData{array.mCvMatData}
    {
    }

    template<typename T>
    Array<T>& Array<T>::operator=(const Array<T>& array)
    {
        try
        {
            mSize = array.mSize;
            mVolume = array.mVolume;
            spData = array.spData;
            mCvMatData = array.mCvMatData;
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
    Array<T>::Array(Array<T>&& array) :
        mSize{array.mSize},
        mVolume{array.mVolume}
    {
        try
        {
            std::swap(spData, array.spData);
            std::swap(mCvMatData, array.mCvMatData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>& Array<T>::operator=(Array<T>&& array)
    {
        try
        {
            mSize = array.mSize;
            mVolume = array.mVolume;
            std::swap(spData, array.spData);
            std::swap(mCvMatData, array.mCvMatData);
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
    Array<T> Array<T>::clone() const
    {
        try
        {
            // Constructor
            Array<T> array{mSize};
            // Clone data
            std::copy(spData.get(), spData.get() + mVolume, array.spData.get());
            // Return
            return std::move(array);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<T>{};
        }
    }

    template<typename T>
    void Array<T>::reset(const int size)
    {
        try
        {
            if (size > 0)
                reset(std::vector<int>{size});
            else
                reset(std::vector<int>{});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::reset(const std::vector<int>& sizes)
    {
        try
        {
            if (!sizes.empty())
            {
                // New size & volume
                mSize = sizes;
                mVolume = {std::accumulate(sizes.begin(), sizes.end(), 1ul, std::multiplies<size_t>())};
                // Prepare shared_ptr
                spData.reset(new T[mVolume]);
                setCvMatFromSharedPtr();
            }
            else
            {
                mSize = {};
                mVolume = 0;
                spData.reset();
                // cv::Mat available but empty
                mCvMatData = std::make_pair(true, cv::Mat());
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::reset(const int sizes, const T value)
    {
        try
        {
            reset(sizes);
            setTo(value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::reset(const std::vector<int>& sizes, const T value)
    {
        try
        {
            reset(sizes);
            setTo(value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::setFrom(const cv::Mat& cvMat)
    {
        try
        {
            if (!cvMat.empty())
            {
                // New size
                std::vector<int> newSize(cvMat.dims,0);
                for (auto i = 0 ; i < newSize.size() ; i++)
                    newSize[i] = cvMat.size[i];
                // Reset data & volume
                reset(newSize);
                // Integrity checks
                if (!mCvMatData.first || mCvMatData.second.type() != cvMat.type())
                    error("Array<T>: T type and cvMat type are different.", __LINE__, __FUNCTION__, __FILE__);
                // Fill data
                cvMat.copyTo(mCvMatData.second);
            }
            else
                reset();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::setTo(const T value)
    {
        try
        {
            if (mVolume > 0)
            {
                // OpenCV is efficient on copying (AVX, SSE, etc.)
                if (mCvMatData.first)
                    mCvMatData.second.setTo((double)value);
                else
                    for (auto i = 0 ; i < mVolume ; i++)
                        operator[](i) = value;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    int Array<T>::getSize(const int index) const
    {
        try
        {
            // Matlab style:
                // If empty -> return 0
                // If index >= # dimensions -> return 1
            if (index < mSize.size() && 0 <= index)
                return mSize[index];
            // Long version:
            // else if (mSize.empty())
            //     return 0;
            // else // if mSize.size() <= index 
            //     return 1;
            // Equivalent to:
            else
                return (!mSize.empty());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename T>
    size_t Array<T>::getVolume(const int indexA, const int indexB) const
    {
        try
        {
            if (indexA < indexB)
            {
                if (0 <= indexA && indexB < mSize.size()) // 0 <= indexA < indexB < mSize.size()
                    return std::accumulate(mSize.begin()+indexA, mSize.begin()+indexB+1, 1ul, std::multiplies<size_t>());
                else
                {
                    error("Indexes out of dimension.", __LINE__, __FUNCTION__, __FILE__);
                    return 0;
                }
            }
            else if (indexA == indexB)
                return mSize.at(indexA);
            else // if (indexA > indexB)
            {
                error("indexA > indexB.", __LINE__, __FUNCTION__, __FILE__);
                return 0;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename T>
    const cv::Mat& Array<T>::getConstCvMat() const
    {
        try
        {
            if (!mCvMatData.first)
                error("Array<T>: cv::Mat functions only valid for T types defined by OpenCV: unsigned char, signed char, int, float & double", __LINE__, __FUNCTION__, __FILE__);
            return mCvMatData.second;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCvMatData.second;
        }
    }

    template<typename T>
    cv::Mat& Array<T>::getCvMat()
    {
        try
        {
            if (!mCvMatData.first)
                error("Array<T>: cv::Mat functions only valid for T types defined by OpenCV: unsigned char, signed char, int, float & double", __LINE__, __FUNCTION__, __FILE__);
            return mCvMatData.second;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCvMatData.second;
        }
    }

    template<typename T>
    const std::string Array<T>::toString() const
    {
        try
        {
            // Initial value
            std::string string{"Array<T>::toString():\n"};
            // Add each element
            for (auto i = 0 ; i < mVolume ; i++)
            {
                // Adding element sepearted by an space
                string += std::to_string(spData[i]) + " ";
                // Introduce an enter for each dimension change
                // If comented, all values will be printed in the same line
                auto multiplier = 1;
                for (auto dimension = (int)(mSize.size() - 1u) ; dimension > 0 && ((i/multiplier) % getSize(dimension) == getSize(dimension)-1) ; dimension--)
                {
                    string += "\n";
                    multiplier *= getSize(dimension);
                }
            }
            // Return string
            return string;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    template<typename T>
    int Array<T>::getIndex(const std::vector<int>& indexes) const
    {
        try
        {
            auto index = 0;
            auto accumulated = 1;
            for (auto i = (int)indexes.size() - 1 ; i >= 0  ; i--)
            {
                index += accumulated * indexes[i];
                accumulated *= mSize[i];
            }
            return index;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename T>
    int Array<T>::getIndexAndCheck(const std::vector<int>& indexes) const
    {
        try
        {
            if (indexes.size() != mSize.size())
                error("Requested indexes size is different than Array size.", __LINE__, __FUNCTION__, __FILE__);
            return getIndex(indexes);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename T>
    T& Array<T>::commonAt(const int index) const
    {
        try
        {
            if (0 <= index && index < mVolume)
                return spData[index];
            else
            {
                error("Index out of bounds: 0 <= index && index < mVolume", __LINE__, __FUNCTION__, __FILE__);
                return spData[0];
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spData[0];
        }
    }

    template<typename T>
    void Array<T>::setCvMatFromSharedPtr()
    {
        try
        {
            // Prepare cv::Mat
            mCvMatData.first = true;
            auto cvFormat = CV_32FC1;
            if (typeid(T) == typeid(float))
                cvFormat = CV_32FC1;
            else if (typeid(T) == typeid(double))
                cvFormat = CV_64FC1;
            else if (typeid(T) == typeid(unsigned char))
                cvFormat = CV_8UC1;
            else if (typeid(T) == typeid(signed char))
                cvFormat = CV_8SC1;
            else if (typeid(T) == typeid(int))
                cvFormat = CV_32SC1;
            else
                mCvMatData.first = false;

            if (mCvMatData.first)
                mCvMatData.second = cv::Mat((int)mSize.size(), mSize.data(), cvFormat, spData.get());
            else
                mCvMatData.second = cv::Mat();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_BASIC_TYPES_CLASS(Array);
}
