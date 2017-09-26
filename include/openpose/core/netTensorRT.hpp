#ifdef USE_CAFFE
#ifndef OPENPOSE_CORE_NET_TENSORRT_HPP
#define OPENPOSE_CORE_NET_TENSORRT_HPP

#include <caffe/net.hpp>
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
        // Init with constructor
        const int mGpuId;
        const std::array<int, 4> mNetInputSize4D;
        std::array<int, 4> mNetOutputSize4D;
        const unsigned long mNetInputMemory;
        const std::string mCaffeProto;
        const std::string mCaffeTrainedModel;
        const std::string mLastBlobName;
        // Init with thread
      
        boost::shared_ptr<caffe::Blob<float>> spInputBlob;
        boost::shared_ptr<caffe::Blob<float>> spOutputBlob;
      
        // TensorRT stuff
        nvinfer1::ICudaEngine* cudaEngine;
        nvinfer1::IExecutionContext* cudaContext;
        ICudaEngine* caffeToGIEModel();
        ICudaEngine* createEngine();

        DELETE_COPY(NetTensorRT);
    };
}

#endif // OPENPOSE_CORE_NET_TENSORRT_HPP
#endif
