#ifndef OPENPOSE_NET_MAXIMUM_CAFFE_HPP
#define OPENPOSE_NET_MAXIMUM_CAFFE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep
    // the compatibility with any generic Caffe version, we keep this 'layer' inside our library rather than in the
    // Caffe code.
    template <typename T>
    class OP_API MaximumCaffe
    {
    public:
        explicit MaximumCaffe();

        virtual void LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual inline const char* type() const { return "Maximum"; }

        virtual void Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

    private:
        std::array<int, 4> mBottomSize;
        std::array<int, 4> mTopSize;
    };
}

#endif // OPENPOSE_NET_MAXIMUM_CAFFE_HPP
