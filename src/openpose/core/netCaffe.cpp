#include <numeric> // std::accumulate
#ifdef USE_CAFFE
    #include <atomic>
    #include <mutex>
    #include <caffe/net.hpp>
    #include <glog/logging.h> // google::InitGoogleLogging
#endif
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/core/netCaffe.hpp>

namespace op
{
    std::mutex sMutexNetCaffe;
    std::atomic<bool> sGoogleLoggingInitialized{false};

    struct NetCaffe::ImplNetCaffe
    {
        #ifdef USE_CAFFE
            // Init with constructor
            const int mGpuId;
            const std::string mCaffeProto;
            const std::string mCaffeTrainedModel;
            const std::string mLastBlobName;
            std::vector<int> mNetInputSize4D;
            // Init with thread
            std::unique_ptr<caffe::Net<float>> upCaffeNet;
            boost::shared_ptr<caffe::Blob<float>> spOutputBlob;

            ImplNetCaffe(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                         const bool enableGoogleLogging, const std::string& lastBlobName) :
                mGpuId{gpuId},
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
                // Double if condition in order to speed up the program if it is called several times
                if (enableGoogleLogging && !sGoogleLoggingInitialized)
                {
                    std::lock_guard<std::mutex> lock{sMutexNetCaffe};
                    if (enableGoogleLogging && !sGoogleLoggingInitialized)
                    {
                        google::InitGoogleLogging("OpenPose");
                        sGoogleLoggingInitialized = true;
                    }
                }
            }
        #endif
    };

    #ifdef USE_CAFFE
        inline void reshapeNetCaffe(caffe::Net<float>* caffeNet, const std::vector<int>& dimensions)
        {
            try
            {
                caffeNet->blobs()[0]->Reshape(dimensions);
                caffeNet->Reshape();
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    NetCaffe::NetCaffe(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                       const bool enableGoogleLogging, const std::string& lastBlobName)
        #ifdef USE_CAFFE
            : upImpl{new ImplNetCaffe{caffeProto, caffeTrainedModel, gpuId, enableGoogleLogging,
                                      lastBlobName}}
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
                #ifdef USE_CUDA
                    caffe::Caffe::set_mode(caffe::Caffe::GPU);
                    caffe::Caffe::SetDevice(upImpl->mGpuId);
                #else
                    caffe::Caffe::set_mode(caffe::Caffe::CPU);
                #endif
                upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
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

    void NetCaffe::forwardPass(const Array<float>& inputData) const
    {
        try
        {
            #ifdef USE_CAFFE
                // Security checks
                if (inputData.empty())
                    error("The Array inputData cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
                    error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width].",
                          __LINE__, __FUNCTION__, __FILE__);
                // Reshape Caffe net if required
                if (!vectorsAreEqual(upImpl->mNetInputSize4D, inputData.getSize()))
                {
                    upImpl->mNetInputSize4D = inputData.getSize();
                    reshapeNetCaffe(upImpl->upCaffeNet.get(), inputData.getSize());
                }
                // Copy frame data to GPU memory
                #ifdef USE_CUDA
                    auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                    cudaMemcpy(gpuImagePtr, inputData.getConstPtr(), inputData.getVolume() * sizeof(float),
                               cudaMemcpyHostToDevice);
                #else
                    auto* cpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_cpu_data();
                    std::copy(inputData.getConstPtr(), inputData.getConstPtr() + inputData.getVolume(), cpuImagePtr);
                #endif
                // Perform deep network forward pass
                upImpl->upCaffeNet->ForwardFrom(0);
                // Cuda checks
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
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
