#ifdef USE_CAFFE
#include <numeric> // std::accumulate
#include <openpose/utilities/cuda.hpp>
#include <openpose/core/netCaffe.hpp>

namespace op
{
    NetCaffe::NetCaffe(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId, const std::string& lastBlobName) :
        mGpuId{gpuId},
        // mNetInputSize4D{netInputSize4D}, // This line crashes on some devices with old G++
        mNetInputSize4D{netInputSize4D[0], netInputSize4D[1], netInputSize4D[2], netInputSize4D[3]},
        mNetInputMemory{std::accumulate(mNetInputSize4D.begin(), mNetInputSize4D.end(), 1, std::multiplies<int>()) * sizeof(float)},
        mCaffeProto{caffeProto},
        mCaffeTrainedModel{caffeTrainedModel},
        mLastBlobName{lastBlobName}
    {
    }

    NetCaffe::~NetCaffe()
    {
    }

    void NetCaffe::initializationOnThread()
    {
        try
        {
            // Initialize net
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
            caffe::Caffe::SetDevice(mGpuId);
            upCaffeNet.reset(new caffe::Net<float>{mCaffeProto, caffe::TEST});
            upCaffeNet->CopyTrainedLayersFrom(mCaffeTrainedModel);
            upCaffeNet->blobs()[0]->Reshape({mNetInputSize4D[0], mNetInputSize4D[1], mNetInputSize4D[2], mNetInputSize4D[3]});
            upCaffeNet->Reshape();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            // Set spOutputBlob
            spOutputBlob = upCaffeNet->blob_by_name(mLastBlobName);
            if (spOutputBlob == nullptr)
                error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: " + mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    float* NetCaffe::getInputDataCpuPtr() const
    {
        try
        {
            return upCaffeNet->blobs().at(0)->mutable_cpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    float* NetCaffe::getInputDataGpuPtr() const
    {
        try
        {
            return upCaffeNet->blobs().at(0)->mutable_gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    void NetCaffe::forwardPass(const float* const inputData) const
    {
        try
        {
            // Copy frame data to GPU memory
            if (inputData != nullptr)
            {
                auto* gpuImagePtr = upCaffeNet->blobs().at(0)->mutable_gpu_data();
                cudaMemcpy(gpuImagePtr, inputData, mNetInputMemory, cudaMemcpyHostToDevice);
            }
            // Perform deep network forward pass
            upCaffeNet->ForwardFrom(0);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    boost::shared_ptr<caffe::Blob<float>> NetCaffe::getOutputBlob() const
    {
        try
        {
            return spOutputBlob;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}

#endif
