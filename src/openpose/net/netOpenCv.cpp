// Note: OpenCV only uses CPU or OpenCL (for Intel GPUs). Used CUDA for following blobs (Resize + NMS)
#include <openpose/core/macros.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef USE_CAFFE
    #include <caffe/net.hpp>
#endif
#if defined(USE_CAFFE) && defined(USE_CUDA) && defined(OPEN_CV_IS_4_OR_HIGHER)
    #define OPEN_CV_DNN_AVAILABLE
    #include <opencv2/opencv.hpp>
    #include <openpose/gpu/cuda.hpp>
#endif
#include <numeric> // std::accumulate
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/net/netOpenCv.hpp>

namespace op
{
    struct NetOpenCv::ImplNetOpenCv
    {
        #ifdef OPEN_CV_DNN_AVAILABLE
            // Init with constructor
            const int mGpuId;
            const std::string mCaffeProto;
            const std::string mCaffeTrainedModel;
            // OpenCV DNN
            cv::dnn::Net mNet;
            cv::Mat mNetOutputBlob;
            // std::shared_ptr<caffe::Blob<float>> spOutputBlob;
            boost::shared_ptr<caffe::Blob<float>> spOutputBlob;

            ImplNetOpenCv(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId) :
                mGpuId{gpuId},
                mCaffeProto{caffeProto},
                mCaffeTrainedModel{caffeTrainedModel},
                mNet{cv::dnn::readNetFromCaffe(caffeProto, caffeTrainedModel)},
                // spOutputBlob{std::make_shared<caffe::Blob<float>>(1,1,1,1)}
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

    #ifdef OPEN_CV_DNN_AVAILABLE
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
        #ifdef OPEN_CV_DNN_AVAILABLE
            : upImpl{new ImplNetOpenCv{caffeProto, caffeTrainedModel, gpuId}}
        #endif
    {
        try
        {
            #ifndef OPEN_CV_DNN_AVAILABLE
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
            #ifdef OPEN_CV_DNN_AVAILABLE
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

    boost::shared_ptr<caffe::Blob<float>> NetOpenCv::getOutputBlob() const
    {
        try
        {
            #ifdef OPEN_CV_DNN_AVAILABLE
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
