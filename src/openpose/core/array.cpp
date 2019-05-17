#include <typeinfo> // typeid
#include <numeric> // std::accumulate
#include <openpose/utilities/avx.hpp>
#include <openpose/core/array.hpp>

// Note: std::shared_ptr not (fully) supported for array pointers:
// http://stackoverflow.com/questions/8947579/
// Solutions:
// 1) Using boost::shared_ptr from <boost/shared_ptr.hpp>: Very easy but requires Boost.
// 2) Using std::unique_ptr from <memory>: Same behaviour than 1, but only `unique`.
// 3) Using std::shared_ptr from <memory>: Harder to use, but benefits of 1 & 2. Solutions to its problems:
//     a) Accessing elements:
//        https://stackoverflow.com/questions/30780262/accessing-array-of-shared-ptr
//     b) Default delete:
//        https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used

namespace op
{
    /**
     * Private auxiliar function that sets the cv::Mat wrapper and makes it point to the same data than
     * std::shared_ptr points to.
     */
    template<typename T>
    void setCvMatFromPtr(std::pair<bool, cv::Mat>& cvMatData, T* const dataPtr, const std::vector<int>& sizes)
    {
        try
        {
            cvMatData.first = true;
            cvMatData.second = cv::Mat();
            // BGR image
            if (sizes.size() == 3 && sizes[2] == 3)
            {
                // Prepare cv::Mat
                auto cvFormat = CV_32FC3;
                if (typeid(T) == typeid(float))
                    cvFormat = CV_32FC3;
                else if (typeid(T) == typeid(double))
                    cvFormat = CV_64FC3;
                else if (typeid(T) == typeid(unsigned char))
                    cvFormat = CV_8UC3;
                else if (typeid(T) == typeid(signed char))
                    cvFormat = CV_8SC3;
                else if (typeid(T) == typeid(int))
                    cvFormat = CV_32SC3;
                else
                    cvMatData.first = false;

                if (cvMatData.first)
                    cvMatData.second = cv::Mat(sizes[0], sizes[1], cvFormat, dataPtr);
            }
            // Any other type
            else
            {
                // Prepare cv::Mat
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
                    cvMatData.first = false;

                if (cvMatData.first)
                    cvMatData.second = cv::Mat((int)sizes.size(), sizes.data(), cvFormat, dataPtr);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

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
    Array<T>::Array(const int size, T* const dataPtr)
    {
        try
        {
            reset(size, dataPtr);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const std::vector<int>& sizes, T* const dataPtr)
    {
        try
        {
            reset(sizes, dataPtr);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    Array<T>::Array(const Array<T>& array, const int index, const bool noCopy)
    {
        try
        {
            // Sanity check
            if (array.getSize(0) <= index)
                error("Index out of range.", __LINE__, __FUNCTION__, __FILE__);
            // Define new size
            auto sizes = array.getSize();
            sizes[0] = 1;
            // Move --> Temporary Array<T> as long as `array` is in scope
            if (noCopy)
                resetAuxiliary(sizes, array.getPseudoConstPtr() + index*array.getVolume(1));
            // Copy --> Slower but it will always stay in scope
            else
            {
                // Allocate memory
                reset(sizes);
                // Copy desired index
                const auto arrayArea = array.getVolume(1);
                const auto keypointsIndex = index*arrayArea;
                std::copy(&array[keypointsIndex], &array[keypointsIndex]+arrayArea, pData);
            }
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
        pData{array.pData},
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
            pData = array.pData;
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
            std::swap(pData, array.pData);
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
            std::swap(pData, array.pData);
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
            // Equivalent: std::copy(spData.get(), spData.get() + mVolume, array.spData.get());
            std::copy(pData, pData + mVolume, array.pData);
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
            resetAuxiliary(sizes);
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
    void Array<T>::reset(const int size, T* const dataPtr)
    {
        try
        {
            if (size > 0)
                resetAuxiliary(std::vector<int>{size}, dataPtr);
            else
                error("Size cannot be less than 1.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void Array<T>::reset(const std::vector<int>& sizes, T* const dataPtr)
    {
        try
        {
            if (!sizes.empty())
                resetAuxiliary(sizes, dataPtr);
            else
                error("Size cannot be empty or less than 1.", __LINE__, __FUNCTION__, __FILE__);
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
                for (auto i = 0u ; i < newSize.size() ; i++)
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
                    for (auto i = 0u ; i < mVolume ; i++)
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
            if ((unsigned int)index < mSize.size() && 0 <= index)
                return mSize[index];
            // Long version:
            // else if (mSize.empty())
            //     return 0;
            // else // if mSize.size() <= (unsigned int)index
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
    std::string Array<T>::printSize() const
    {
        try
        {
            auto counter = 0u;
            std::string sizeString = "[ ";
            for (const auto& i : mSize)
            {
                sizeString += std::to_string(i);
                if (++counter < mSize.size())
                    sizeString += " x ";
            }
            sizeString += " ]";
            return sizeString;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    template<typename T>
    size_t Array<T>::getVolume(const int indexA, const int indexB) const
    {
        try
        {
            const auto indexBFinal = (indexB != -1 ? indexB : (int)mSize.size()-1);
            if (indexA < indexBFinal)
            {
                // 0 <= indexA < indexBFinal < mSize.size()
                if (0 <= indexA && (unsigned int)indexBFinal < mSize.size())
                    return std::accumulate(
                        mSize.begin()+indexA, mSize.begin()+indexBFinal+1, 1ull, std::multiplies<size_t>());
                else
                {
                    error("Indexes out of dimension.", __LINE__, __FUNCTION__, __FILE__);
                    return 0;
                }
            }
            else if (indexA == indexBFinal)
                return mSize.at(indexA);
            else // if (indexA > indexBFinal)
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
    std::vector<int> Array<T>::getStride() const
    {
        try
        {
            std::vector<int> strides(mSize.size());
            if (!strides.empty())
            {
                strides.back() = sizeof(T);
                for (auto i = (int)strides.size()-2 ; i > -1 ; i--)
                    strides[i] = strides[i+1] * mSize[i+1];
            }
            return strides;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template<typename T>
    int Array<T>::getStride(const int index) const
    {
        try
        {
            return getStride()[index];
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    const cv::Mat& Array<T>::getConstCvMat() const
    {
        try
        {
            if (!mCvMatData.first)
                error("Array<T>: cv::Mat functions only valid for T types defined by OpenCV: unsigned char,"
                      " signed char, int, float & double", __LINE__, __FUNCTION__, __FILE__);
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
                error("Array<T>: cv::Mat functions only valid for T types defined by OpenCV: unsigned char,"
                      " signed char, int, float & double", __LINE__, __FUNCTION__, __FILE__);
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
            for (auto i = 0u ; i < mVolume ; i++)
            {
                // Adding element separated by a space
                string += std::to_string(pData[i]) + " ";
                // Introduce an enter for each dimension change
                // If comented, all values will be printed in the same line
                auto multiplier = 1;
                for (auto dimension = (int)(mSize.size() - 1u) ; dimension > 0
                      && (int(i/multiplier) % getSize(dimension) == getSize(dimension)-1) ; dimension--)
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
            if (0 <= index && (size_t)index < mVolume)
                return pData[index]; // spData.get()[index]
            else
            {
                error("Index out of bounds: 0 <= index && index < mVolume", __LINE__, __FUNCTION__, __FILE__);
                return pData[0]; // spData.get()[0]
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return pData[0]; // spData.get()[0]
        }
    }

    template<typename T>
    void Array<T>::resetAuxiliary(const std::vector<int>& sizes, T* const dataPtr)
    {
        try
        {
            if (!sizes.empty())
            {
                // New size & volume
                mSize = sizes;
                mVolume = {std::accumulate(sizes.begin(), sizes.end(), std::size_t(1), std::multiplies<size_t>())};
                // Prepare shared_ptr
                if (dataPtr == nullptr)
                {
                    #ifdef WITH_AVX
                        spData = aligned_shared_ptr<T>(mVolume);
                    #else
                        spData.reset(new T[mVolume], std::default_delete<T[]>());
                    #endif
                    pData = spData.get();
                    // Sanity check
                    if (pData == nullptr)
                        error("Shared pointer could not be allocated for Array data storage.",
                              __LINE__, __FUNCTION__, __FILE__);
                }
                else
                {
                    spData.reset();
                    pData = dataPtr;
                }
                setCvMatFromPtr(mCvMatData, pData, mSize); // spData.get()
            }
            else
            {
                mSize = {};
                mVolume = 0ul;
                spData.reset();
                pData = nullptr;
                // cv::Mat available but empty
                mCvMatData = std::make_pair(true, cv::Mat());
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_BASIC_TYPES_CLASS(Array);
}
