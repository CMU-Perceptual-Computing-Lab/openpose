#include <numeric> // std::accumulate
#ifdef USE_CAFFE
    #include <caffe/net.hpp>
#endif
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/core/netCaffe.hpp>

namespace op
{
    struct NetCaffe::ImplNetCaffe
    {
        #ifdef USE_CAFFE
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

            ImplNetCaffe(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto,
                         const std::string& caffeTrainedModel, const int gpuId, const std::string& lastBlobName) :
                mGpuId{gpuId},
                // mNetInputSize4D{netInputSize4D}, // This line crashes on some devices with old G++
                mNetInputSize4D{netInputSize4D[0], netInputSize4D[1], netInputSize4D[2], netInputSize4D[3]},
                mNetInputMemory{sizeof(float) * std::accumulate(mNetInputSize4D.begin(), mNetInputSize4D.end(), 1,
                                                                std::multiplies<int>())},
                mCaffeProto{caffeProto},
                mCaffeTrainedModel{caffeTrainedModel},
                mLastBlobName{lastBlobName}
            {
                const std::string message{".\nPossible causes:\n\t1. Not downloading the OpenPose trained models."
                                          "\n\t2. Not running OpenPose from the same directory where the `model`"
                                          " folder is located.\n\t3. Using paths with spaces."};
                if (!existFile(mCaffeProto))
                    error("Prototxt file not found: " + mCaffeProto + message, __LINE__, __FUNCTION__, __FILE__);
                if (!existFile(mCaffeTrainedModel))
                    error("Caffe trained model file not found: " + mCaffeTrainedModel + message,
                          __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
    };

    NetCaffe::NetCaffe(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto,
                       const std::string& caffeTrainedModel, const int gpuId, const std::string& lastBlobName)
        #ifdef USE_CAFFE
            : upImpl{new ImplNetCaffe{netInputSize4D, caffeProto, caffeTrainedModel, gpuId, lastBlobName}}
        #endif
    {
        try
        {
            #ifndef USE_CAFFE
                UNUSED(netInputSize4D);
                UNUSED(caffeProto);
                UNUSED(caffeTrainedModel);
                UNUSED(gpuId);
                UNUSED(lastBlobName);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NetCaffe::~NetCaffe()
    {
    }

    void NetCaffe::initializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                // Initialize net
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
                caffe::Caffe::SetDevice(upImpl->mGpuId);
                upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
                upImpl->upCaffeNet->blobs()[0]->Reshape({upImpl->mNetInputSize4D[0], upImpl->mNetInputSize4D[1],
                                                         upImpl->mNetInputSize4D[2], upImpl->mNetInputSize4D[3]});
                upImpl->upCaffeNet->Reshape();
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                // Set spOutputBlob
                upImpl->spOutputBlob = upImpl->upCaffeNet->blob_by_name(upImpl->mLastBlobName);
                if (upImpl->spOutputBlob == nullptr)
                    error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: "
                          + upImpl->mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
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
            #ifdef USE_CAFFE
                return upImpl->upCaffeNet->blobs().at(0)->mutable_cpu_data();
            #else
                return nullptr;
            #endif
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
            #ifdef USE_CAFFE
                return upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
            #else
                return nullptr;
            #endif
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
            #ifdef USE_CAFFE
                // Copy frame data to GPU memory
                if (inputData != nullptr)
                {
                    #ifdef USE_CUDA
                        auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                        cudaMemcpy(gpuImagePtr, inputData, upImpl->mNetInputMemory, cudaMemcpyHostToDevice);
                    #else
                        auto* cpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_cpu_data();
                        std::copy(inputData,
                                  inputData + upImpl->mNetInputMemory/sizeof(float),
                                  cpuImagePtr);
                    #endif
                }
                // Perform deep network forward pass
                upImpl->upCaffeNet->ForwardFrom(0);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(inputData);
            #endif
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
            #ifdef USE_CAFFE
                return upImpl->spOutputBlob;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
