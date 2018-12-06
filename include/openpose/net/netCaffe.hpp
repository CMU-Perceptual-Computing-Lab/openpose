#ifndef OPENPOSE_NET_NET_CAFFE_HPP
#define OPENPOSE_NET_NET_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/net.hpp>

namespace op
{
    class OP_API NetCaffe : public Net
    {
    public:
        NetCaffe(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId = 0,
                 const bool enableGoogleLogging = true, const std::string& lastBlobName = "net_output");

        virtual ~NetCaffe();

        void initializationOnThread();

        void forwardPass(const Array<float>& inputNetData) const;

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

#endif // OPENPOSE_NET_NET_CAFFE_HPP
