#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/core/resizeAndMergeBase.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>

namespace op
{
    template <typename T>
    ResizeAndMergeCaffe<T>::ResizeAndMergeCaffe() :
        mScaleRatios{T(1)}
    {
        try
        {
            #ifndef USE_CAFFE
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom,
                                            const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                if (top.size() != 1)
                    error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.size() != 1)
                    error("bottom.size() != 2", __LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom,
                                         const std::vector<caffe::Blob<T>*>& top,
                                         const float netFactor,
                                         const float scaleFactor,
                                         const bool mergeFirstDimension)
    {
        try
        {
            #ifdef USE_CAFFE
                // Data
                const auto* bottomBlob = bottom.at(0);
                auto* topBlob = top.at(0);
                // Set top shape
                auto topShape = bottomBlob->shape();
                topShape[0] = (mergeFirstDimension ? 1 : bottomBlob->shape(0));
                // -1 and later +1 to take into account that we are using 0-based index
                // E.g. 100x100 image --> 200x200 --> 0-99 to 0-199 --> scale = 199/99 (not 2!)
                // E.g. 101x101 image --> 201x201 --> scale = 2
                // Test: pixel 0 --> 0, pixel 99 (ex 1) --> 199, pixel 100 (ex 2) --> 200
                topShape[2] = intRound((topShape[2]*netFactor - 1.f) * scaleFactor + 1);
                topShape[3] = intRound((topShape[3]*netFactor - 1.f) * scaleFactor + 1);
                topBlob->Reshape(topShape);
                // Array sizes
                mTopSize = std::array<int, 4>{topBlob->shape(0), topBlob->shape(1), topBlob->shape(2),
                                              topBlob->shape(3)};
                mBottomSize = std::array<int, 4>{bottomBlob->shape(0), bottomBlob->shape(1),
                                                 bottomBlob->shape(2), bottomBlob->shape(3)};
            #else
                UNUSED(bottom);
                UNUSED(top);
                UNUSED(factor);
                UNUSED(mergeFirstDimension);
            #endif
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
    void ResizeAndMergeCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                resizeAndMergeCpu(top.at(0)->mutable_cpu_data(), bottom.at(0)->cpu_data(), mTopSize, mBottomSize,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                resizeAndMergeGpu(top.at(0)->mutable_gpu_data(), bottom.at(0)->gpu_data(), mTopSize, mBottomSize,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top,
                                              const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top,
                                              const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
    {
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(ResizeAndMergeCaffe);
}
