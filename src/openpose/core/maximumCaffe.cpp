#ifdef USE_CAFFE
#include <openpose/core/maximumBase.hpp>
#include <openpose/core/maximumCaffe.hpp>

namespace op
{
    template <typename T>
    MaximumCaffe<T>::MaximumCaffe()
    {
    }

    template <typename T>
    void MaximumCaffe<T>::LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            if (top.size() != 1)
                error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
            if (bottom.size() != 1)
                error("bottom.size() != 1", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void MaximumCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            auto bottomBlob = bottom.at(0);
            auto topBlob = top.at(0);

            // Bottom shape
            std::vector<int> bottomShape = bottomBlob->shape();

            // Top shape
            std::vector<int> topShape{bottomShape};
            topShape[1] = 1; // Unnecessary
            topShape[2] = bottomShape[1]-1; // Number parts + bck - 1
            topShape[3] = 3;  // X, Y, score
            topBlob->Reshape(topShape);

            // Array sizes
            mTopSize = std::array<int, 4>{topBlob->shape(0), topBlob->shape(1), topBlob->shape(2), topBlob->shape(3)};
            mBottomSize = std::array<int, 4>{bottomBlob->shape(0), bottomBlob->shape(1), bottomBlob->shape(2), bottomBlob->shape(3)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void MaximumCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            maximumGpu(top.at(0)->mutable_cpu_data(), bottom.at(0)->cpu_data(), mTopSize, mBottomSize);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void MaximumCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            maximumGpu(top.at(0)->mutable_gpu_data(), bottom.at(0)->gpu_data(), mTopSize, mBottomSize);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void MaximumCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            NOT_IMPLEMENTED;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void MaximumCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            NOT_IMPLEMENTED;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    INSTANTIATE_CLASS(MaximumCaffe);
}

#endif
