#ifndef OPENPOSE_NET_NET_OPEN_CV_HPP
#define OPENPOSE_NET_NET_OPEN_CV_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/net.hpp>

namespace op
{
    class OP_API NetOpenCv : public Net
    {
    public:
        NetOpenCv(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId = 0);

        virtual ~NetOpenCv();

        void initializationOnThread();

        void forwardPass(const Array<float>& inputNetData) const;

        boost::shared_ptr<caffe::Blob<float>> getOutputBlob() const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplNetOpenCv;
        std::unique_ptr<ImplNetOpenCv> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NetOpenCv);
    };
}

#endif // OPENPOSE_NET_NET_OPEN_CV_HPP
