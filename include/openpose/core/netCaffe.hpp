#ifndef OPENPOSE_CORE_NET_CAFFE_HPP
#define OPENPOSE_CORE_NET_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/net.hpp>

namespace op
{
    class OP_API NetCaffe : public Net
    {
    public:
        NetCaffe(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto,
                 const std::string& caffeTrainedModel, const int gpuId = 0,
                 const std::string& lastBlobName = "net_output");

        virtual ~NetCaffe();

        void initializationOnThread();

        // Alternative a) getInputDataCpuPtr or getInputDataGpuPtr + forwardPass
        float* getInputDataCpuPtr() const;

        float* getInputDataGpuPtr() const;

        // Alternative b)
        void forwardPass(const float* const inputNetData = nullptr) const;

        boost::shared_ptr<caffe::Blob<float>> getOutputBlob() const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplNetCaffe;
        std::unique_ptr<ImplNetCaffe> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NetCaffe);
    };
}

#endif // OPENPOSE_CORE_NET_CAFFE_HPP
