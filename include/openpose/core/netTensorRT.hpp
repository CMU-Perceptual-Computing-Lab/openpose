#ifndef OPENPOSE_CORE_NET_TENSORRT_HPP
#define OPENPOSE_CORE_NET_TENSORRT_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/net.hpp>


#include "NvInfer.h"

namespace op
{
    class OP_API NetTensorRT : public Net
    {
    public:
        NetTensorRT(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId = 0,
                 const std::string& lastBlobName = "net_output");

        virtual ~NetTensorRT();

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
        struct ImplNetTensorRT;
        std::unique_ptr<ImplNetTensorRT> upImpl;
        
        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NetTensorRT);
    };
}

#endif // OPENPOSE_CORE_NET_TENSORRT_HPP
