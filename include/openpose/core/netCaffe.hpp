#ifdef USE_CAFFE
#ifndef OPENPOSE_CORE_NET_CAFFE_HPP
#define OPENPOSE_CORE_NET_CAFFE_HPP

#include <caffe/net.hpp>
#include <openpose/core/common.hpp>
#include <openpose/core/net.hpp>

namespace op
{
    class OP_API NetCaffe : public Net
    {
    public:
        NetCaffe(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId = 0,
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
        // Init with constructor
        const int mGpuId;
        const std::array<int, 4> mNetInputSize4D;
        const unsigned long mNetInputMemory;
        const std::string mCaffeProto;
        const std::string mCaffeTrainedModel;
        const std::string mLastBlobName;
        // Init with thread
        std::unique_ptr<caffe::Net<float>> upCaffeNet;
        boost::shared_ptr<caffe::Blob<float>> spOutputBlob;

        DELETE_COPY(NetCaffe);
    };
}

#endif // OPENPOSE_CORE_NET_CAFFE_HPP
#endif
