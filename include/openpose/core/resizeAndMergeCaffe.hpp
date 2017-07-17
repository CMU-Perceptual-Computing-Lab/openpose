#ifdef USE_CAFFE
#ifndef OPENPOSE_CORE_RESIZE_AND_MERGE_CAFFE_HPP
#define OPENPOSE_CORE_RESIZE_AND_MERGE_CAFFE_HPP

#include <caffe/blob.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep the
    // compatibility with any generic Caffe version,
    // we keep this 'layer' inside our library rather than in the Caffe code.
    template <typename T>
    class OP_API ResizeAndMergeCaffe
    {
    public:
        explicit ResizeAndMergeCaffe();

        virtual void LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top,
                             const float factor, const bool mergeFirstDimension = true);

        virtual inline const char* type() const { return "ResizeAndMerge"; }

        void setScaleRatios(const std::vector<T>& scaleRatios);

        virtual void Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

    private:
        std::vector<T> mScaleRatios;
        std::array<int, 4> mBottomSize;
        std::array<int, 4> mTopSize;

        DELETE_COPY(ResizeAndMergeCaffe);
    };
}

#endif // OPENPOSE_CORE_RESIZE_AND_MERGE_CAFFE_HPP
#endif
