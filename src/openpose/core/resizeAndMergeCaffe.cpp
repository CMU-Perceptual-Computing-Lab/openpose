#ifdef USE_CAFFE
#include <openpose/core/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>

namespace op
{
    template <typename T>
    ResizeAndMergeCaffe<T>::ResizeAndMergeCaffe() :
        mScaleRatios{1}
    {
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            if (top.size() != 1)
                error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
            if (bottom.size() != 1)
                error("bottom.size() != 2", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top,
                                         const float factor, const bool mergeFirstDimension)
    {
        try
        {
            auto bottomBlob = bottom.at(0);
            auto topBlob = top.at(0);

            // Top shape
            auto topShape = bottomBlob->shape();
            topShape[0] = (mergeFirstDimension ? 1 : bottomBlob->shape(0));
            topShape[2] = intRound(topShape[2] * factor);
            topShape[3] = intRound(topShape[3] * factor);
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
    void ResizeAndMergeCaffe<T>::setScaleRatios(const std::vector<T>& scaleRatios)
    {
        try
        {
            mScaleRatios = {scaleRatios};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            resizeAndMergeCpu(top.at(0)->mutable_cpu_data(), bottom.at(0)->cpu_data(), mTopSize, mBottomSize, mScaleRatios);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            resizeAndMergeGpu(top.at(0)->mutable_gpu_data(), bottom.at(0)->gpu_data(), mTopSize, mBottomSize, mScaleRatios);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
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
    void ResizeAndMergeCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
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

    INSTANTIATE_CLASS(ResizeAndMergeCaffe);
}

#endif
