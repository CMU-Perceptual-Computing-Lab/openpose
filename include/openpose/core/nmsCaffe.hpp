#ifdef USE_CAFFE
#ifndef OPENPOSE_CORE_NMS_CAFFE_HPP
#define OPENPOSE_CORE_NMS_CAFFE_HPP

#include <array>
#include "caffe/blob.hpp"

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep the compatibility with any generic Caffe version,
    // we keep this 'layer' inside our library rather than in the Caffe code.
    template <typename T>
    class NmsCaffe
    {
    public:
        explicit NmsCaffe();

        virtual void LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top, const int maxPeaks, const int numberParts);

        virtual inline const char* type() const { return "Nms"; }

        void setThreshold(const T threshold);

        virtual void Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<T>*>& bottom);

    private:
        T mThreshold;
        caffe::Blob<int> mKernelBlob;
        std::array<int, 4> mBottomSize;
        std::array<int, 4> mTopSize;
    };
}

#endif // OPENPOSE_CORE_NMS_CAFFE_HPP
#endif
