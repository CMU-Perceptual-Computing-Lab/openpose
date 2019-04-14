#ifndef OPENPOSE_NET_NMS_CAFFE_HPP
#define OPENPOSE_NET_NMS_CAFFE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep
    // the compatibility with any generic Caffe version, we keep this 'layer' inside our library rather than in the
    // Caffe code.
    template <typename T>
    class NmsCaffe
    {
    public:
        explicit NmsCaffe();

        virtual ~NmsCaffe();

        virtual void LayerSetUp(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top);

        virtual void Reshape(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top,
                             const int maxPeaks, const int outputChannels = -1, const int gpuID = 0);

        virtual inline const char* type() const { return "Nms"; }

        void setThreshold(const T threshold);

        // Empirically gives better results (copied from Matlab original code)
        void setOffset(const Point<T>& offset);

        virtual void Forward(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top);

        virtual void Forward_cpu(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top);

        virtual void Forward_gpu(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top);

        virtual void Forward_ocl(const std::vector<ArrayCpuGpu<T>*>& bottom, const std::vector<ArrayCpuGpu<T>*>& top);

        virtual void Backward_cpu(const std::vector<ArrayCpuGpu<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<ArrayCpuGpu<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<ArrayCpuGpu<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<ArrayCpuGpu<T>*>& bottom);

    private:
        T mThreshold;
        Point<T> mOffset;
        int mGpuID;

        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplNmsCaffe;
        std::unique_ptr<ImplNmsCaffe> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NmsCaffe);
    };
}

#endif // OPENPOSE_NET_NMS_CAFFE_HPP
