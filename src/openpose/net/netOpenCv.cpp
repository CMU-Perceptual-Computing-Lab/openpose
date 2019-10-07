// TODO: After completely adding the OpenCV DNN module, add this flag to CMake as alternative to USE_CAFFE
// #define USE_OPEN_CV_DNN

#include <openpose/net/netOpenCv.hpp>
// Note: OpenCV only uses CPU or OpenCL (for Intel GPUs). Used CUDA for following blobs (Resize + NMS)
#ifdef USE_CAFFE
    #include <caffe/net.hpp>
#endif
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef USE_OPEN_CV_DNN
    #if defined(USE_CAFFE) && defined(USE_CUDA) && defined(OPEN_CV_IS_4_OR_HIGHER)
        #include <opencv2/opencv.hpp>
        #include <openpose/gpu/cuda.hpp>
    #else
        #error In order to enable OpenCV DNN module in OpenPose, the CMake flags of Caffe and CUDA must be \
               enabled, and OpenCV version must be at least 4.0.0.
    #endif
#endif
#include <numeric> // std::accumulate
#include <openpose/utilities/fileSystem.hpp>

namespace op
{
    struct NetOpenCv::ImplNetOpenCv
    {
        #ifdef USE_OPEN_CV_DNN
            // Init with constructor
            const int mGpuId;
            const std::string mCaffeProto;
            const std::string mCaffeTrainedModel;
            // OpenCV DNN
            cv::dnn::Net mNet;
            cv::Mat mNetOutputBlob;
            boost::shared_ptr<caffe::Blob<float>> spOutputBlob;

            ImplNetOpenCv(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId) :
                mGpuId{gpuId},
                mCaffeProto{caffeProto},
                mCaffeTrainedModel{caffeTrainedModel},
                mNet{cv::dnn::readNetFromCaffe(caffeProto, caffeTrainedModel)},
                spOutputBlob{new caffe::Blob<float>(1,1,1,1)}
            {
                const std::string message{".\nPossible causes:\n\t1. Not downloading the OpenPose trained models."
                                          "\n\t2. Not running OpenPose from the same directory where the `model`"
                                          " folder is located.\n\t3. Using paths with spaces."};
                if (!existFile(mCaffeProto))
                    error("Prototxt file not found: " + mCaffeProto + message, __LINE__, __FUNCTION__, __FILE__);
                if (!existFile(mCaffeTrainedModel))
                    error("Caffe trained model file not found: " + mCaffeTrainedModel + message,
                          __LINE__, __FUNCTION__, __FILE__);

                // Set GPU
                mNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // 1.7 sec at -1x160
                // mNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL); // 1.2 sec at -1x160
                // mNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
                // mNet.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
                // mNet.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
                // // Set backen
                // mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                // mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
                // mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
                // mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                // mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
            }
        #endif
    };

    #ifdef USE_OPEN_CV_DNN
        inline void reshapeNetOpenCv(caffe::Net<float>* caffeNet, const std::vector<int>& dimensions)
        {
            try
            {
                caffeNet->blobs()[0]->Reshape(dimensions);
                caffeNet->Reshape();
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    NetOpenCv::NetOpenCv(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId)
        #ifdef USE_OPEN_CV_DNN
            : upImpl{new ImplNetOpenCv{caffeProto, caffeTrainedModel, gpuId}}
        #endif
    {
        try
        {
            #ifndef USE_OPEN_CV_DNN
                UNUSED(caffeProto);
                UNUSED(caffeTrainedModel);
                UNUSED(gpuId);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NetOpenCv::~NetOpenCv()
    {
    }

    void NetOpenCv::initializationOnThread()
    {
    }

    void NetOpenCv::forwardPass(const Array<float>& inputData) const
    {
        try
        {
            #ifdef USE_OPEN_CV_DNN
                upImpl->mNet.setInput(inputData.getConstCvMat());
                upImpl->mNetOutputBlob = upImpl->mNet.forward(); // 99% of the runtime here
                std::vector<int> outputSize(upImpl->mNetOutputBlob.dims,0);
                for (auto i = 0u ; i < outputSize.size() ; i++)
                    outputSize[i] = upImpl->mNetOutputBlob.size[i];
                upImpl->spOutputBlob->Reshape(outputSize);
                auto* gpuImagePtr = upImpl->spOutputBlob->mutable_gpu_data();
                cudaMemcpy(gpuImagePtr, (float*)upImpl->mNetOutputBlob.data,
                           upImpl->spOutputBlob->count() * sizeof(float),
                           cudaMemcpyHostToDevice);
            #else
                UNUSED(inputData);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::shared_ptr<ArrayCpuGpu<float>> NetOpenCv::getOutputBlobArray() const
    {
        try
        {
            #ifdef USE_OPEN_CV_DNN
                return std::make_shared<ArrayCpuGpu<float>>(upImpl->spOutputBlob.get());
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
