#ifndef OPENPOSE_UTILITIES_CAFFEUTIL_HPP
#define OPENPOSE_UTILITIES_CAFFEUTIL_HPP

#include <utility> // std::pair
#include <openpose/core/common.hpp>

namespace op
{

    template <typename T>
    inline std::string getCaffeBlobShapeAsString(caffe::Blob<T>& blob)
    {
        try
        {
            u_int8_t counter = 0;
            std::string sizeString = "[ ";
            for(const auto i : blob.shape()){
                sizeString += std::to_string(i);
                if(++counter < blob.shape().size()) sizeString += " x ";
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

}

#endif // OPENPOSE_UTILITIES_CAFFEUTIL_HPP
