#include <numeric> // std::accumulate
#ifdef USE_TENSORRT
    #include <atomic>
    #include <mutex>
    #include <caffe/net.hpp>
    #include <glog/logging.h> // google::InitGoogleLogging
#endif
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/core/netTensorRT.hpp>

//#include <assert.h>
//#include <fstream>
//#include <sstream>
//#include <iostream>
//#include <cmath>
//#include <sys/stat.h>
//#include <cmath>
//#include <time.h>
//#include <cuda_runtime_api.h>
//#include <algorithm>
//#include <chrono>
//#include <string.h>
//#include <map>
//#include <random>
#include <boost/make_shared.hpp>


#ifdef USE_TENSORRT
    #include "NvInfer.h"
    #include "NvCaffeParser.h"

    using namespace nvinfer1;
    using namespace nvcaffeparser1;

    std::vector<std::string> gInputs;
    std::map<std::string, DimsCHW> gInputDimensions;
#endif // USE_TENSORRT

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // if suppress info-level message:  if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
} gLogger;

namespace op
{
    std::mutex sMutexNetTensorRT;
    std::atomic<bool> sGoogleLoggingInitializedTensorRT{false}; // Already defined in netCaffe
    
    struct NetTensorRT::ImplNetTensorRT
    {
        #ifdef USE_TENSORRT
            // Init with constructor
            const int mGpuId;
            const std::string mCaffeProto;
            const std::string mCaffeTrainedModel;
            const std::string mLastBlobName;
            std::vector<int> mNetInputSize4D;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> spInputBlob;
            boost::shared_ptr<caffe::Blob<float>> spOutputBlob;
        
            // Init with constructor
            //const std::array<int, 4> mNetInputSize4D;
            std::vector<int> mNetOutputSize4D;
            // Init with thread
        
            // TensorRT stuff
            nvinfer1::ICudaEngine* cudaEngine;
            nvinfer1::IExecutionContext* cudaContext;
            //nvinfer1::ICudaEngine* caffeToGIEModel();
            //nvinfer1::ICudaEngine* createEngine();
            cudaStream_t stream;
            cudaEvent_t start, end;
    
            ImplNetTensorRT(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                         const bool enableGoogleLogging, const std::string& lastBlobName) :
                mGpuId{gpuId},
                mCaffeProto{caffeProto + std::string("_368x656")}, // TODO, no size, how to proceed ?
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
                        if (enableGoogleLogging && !sGoogleLoggingInitializedTensorRT)
                        {
                            std::lock_guard<std::mutex> lock{sMutexNetTensorRT};
                            if (enableGoogleLogging && !sGoogleLoggingInitializedTensorRT)
                            {
                                google::InitGoogleLogging("OpenPose");
                                sGoogleLoggingInitializedTensorRT = true;
                            }
                        }
            }
        #endif
    };
    
    
#ifdef USE_TENSORRT
    ICudaEngine* NetTensorRT::caffeToGIEModel()
    {
        // create the builder
        IBuilder* builder = createInferBuilder(gLogger);
        
        // parse the caffe model to populate the network, then set the outputs
        INetworkDefinition* network = builder->createNetwork();
        ICaffeParser* parser = createCaffeParser();
        const IBlobNameToTensor* blobNameToTensor = parser->parse(upImpl->mCaffeProto.c_str(),
                                                                  upImpl->mCaffeTrainedModel.c_str(),
                                                                  *network,
                                                                  DataType::kFLOAT);
        
        if (!blobNameToTensor)
            return nullptr;
        
        
        for (int i = 0, n = network->getNbInputs(); i < n; i++)
        {
            DimsCHW dims = static_cast<DimsCHW&&>(network->getInput(i)->getDimensions());
            gInputs.push_back(network->getInput(i)->getName());
            gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
            std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
            if( i > 0)
                std::cerr << "Multiple output unsupported for now!";
        }
        
        // Specify which tensor is output (multiple unsupported)
        if (blobNameToTensor->find(upImpl->mLastBlobName.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << upImpl->mLastBlobName.c_str() << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(upImpl->mLastBlobName.c_str()));
        
        
        for (int i = 0, n = network->getNbOutputs(); i < n; i++)
        {
            DimsCHW dims = static_cast<DimsCHW&&>(network->getOutput(i)->getDimensions());
            std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
        }
        
        // Build the engine
        builder->setMaxBatchSize(1);
        // 16 megabytes, default in giexec. No idea what's best for Jetson though,
        // maybe check dusty_nv's code on github
        builder->setMaxWorkspaceSize(32<<20);
        builder->setHalf2Mode(false);
        
        ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (engine == nullptr)
            std::cout << "could not build engine" << std::endl;
        
        parser->destroy();
        network->destroy();
        builder->destroy();
        shutdownProtobufLibrary();
        
        return engine;
    }

    ICudaEngine* NetTensorRT::createEngine()
    {
        ICudaEngine *engine;
        
        std::string serializedEnginePath = upImpl->mCaffeProto + ".bin";
        
        std::cout << "Serialized engine path: " << serializedEnginePath.c_str() << std::endl;
        if (existFile(serializedEnginePath))
        {
            std::cout << "Found serialized TensorRT engine, deserializing..." << std::endl;
            char *gieModelStream{nullptr};
            size_t size{0};
            std::ifstream file(serializedEnginePath, std::ios::binary);
            if (file.good())
            {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                gieModelStream = new char[size];
                assert(gieModelStream);
                file.read(gieModelStream, size);
                file.close();
            }
            
            IRuntime* infer = createInferRuntime(gLogger);
            engine = infer->deserializeCudaEngine(gieModelStream, size, nullptr);
            if (gieModelStream) delete [] gieModelStream;
            
            return engine;
        }
        else
        {
            engine = caffeToGIEModel();
            if (!engine)
            {
                std::cerr << "Engine could not be created" << std::endl;
                return nullptr;
            }
            else // serialize engine
            {
                std::ofstream p(serializedEnginePath);
                if (!p)
                {
                    std::cerr << "could not serialize engine" << std::endl;
                }
                IHostMemory *ptr = engine->serialize();
                assert(ptr);
                p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
                ptr->destroy();
            }
        }
        return engine;
    }
    
    inline void reshapeNetTensorRT(boost::shared_ptr<caffe::Blob<float>> inputBlob, const std::vector<int>& dimensions)
    {
        try
        {
            inputBlob->Reshape(dimensions);
            //caffeNet->Reshape(); TODO find TensorRT equivalent
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
#endif
    
    NetTensorRT::NetTensorRT(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                       const bool enableGoogleLogging, const std::string& lastBlobName)
#ifdef USE_TENSORRT
    : upImpl{new ImplNetTensorRT{caffeProto, caffeTrainedModel, gpuId, enableGoogleLogging,
        lastBlobName}}
#endif
    {
        try
        {
            #ifdef USE_TENSORRT
                std::cout << "Caffe file: " << upImpl->mCaffeProto.c_str() << std::endl;
                CUDA_CHECK(cudaStreamCreate(&upImpl->stream));
                CUDA_CHECK(cudaEventCreate(&upImpl->start));
                CUDA_CHECK(cudaEventCreate(&upImpl->end));
            #else
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
    
    NetTensorRT::~NetTensorRT()
    {
        cudaStreamDestroy(upImpl->stream);
        cudaEventDestroy(upImpl->start);
        cudaEventDestroy(upImpl->end);
        
        if (upImpl->cudaEngine)
            upImpl->cudaEngine->destroy();
    }
    
    void NetTensorRT::initializationOnThread()
    {
        std::cout << "InitializationOnThread : start" << std::endl;
        try
        {
            #ifdef USE_TENSORRT
                std::cout << "InitializationOnThread : setting device" << std::endl;
                // Initialize net
                cudaSetDevice(upImpl->mGpuId);
            
                std::cout << "InitializationOnThread : creating engine" << std::endl;
            
                upImpl->cudaEngine = createEngine();
                if (!upImpl->cudaEngine)
                {
                    std::cerr << "cudaEngine could not be created" << std::endl;
                    return;
                }
            
                std::cout << "InitializationOnThread Pass : creating execution context" << std::endl;
            
                upImpl->cudaContext = upImpl->cudaEngine->createExecutionContext();
                if (!upImpl->cudaContext)
                {
                    std::cerr << "cudaContext could not be created" << std::endl;
                    return;
                }
            
                DimsCHW outputDims = static_cast<DimsCHW&&>(upImpl->cudaEngine->getBindingDimensions(upImpl->cudaEngine->getNbBindings() - 1));
                upImpl->mNetOutputSize4D = { 1, outputDims.c(), outputDims.h(), outputDims.w() };
            
            
                std::cout << "NetInputSize4D: " << upImpl->mNetInputSize4D[0] << " " << upImpl->mNetInputSize4D[1] << " " << upImpl->mNetInputSize4D[2] << " " << upImpl->mNetInputSize4D[3] << std::endl;
            
                upImpl->spInputBlob = boost::make_shared<caffe::Blob<float>>(upImpl->mNetInputSize4D[0], upImpl->mNetInputSize4D[1], upImpl->mNetInputSize4D[2], upImpl->mNetInputSize4D[3]);
                upImpl->spOutputBlob = boost::make_shared<caffe::Blob<float>>(upImpl->mNetOutputSize4D[0], upImpl->mNetOutputSize4D[1], upImpl->mNetOutputSize4D[2], upImpl->mNetOutputSize4D[3]);
            
                std::cout << "InitializationOnThread : done" << std::endl;
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
    
    void NetTensorRT::forwardPass(const Array<float>& inputData) const
    {
        try
        {
            #ifdef USE_TENSORRT
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
                reshapeNetTensorRT(upImpl->spInputBlob, inputData.getSize());
            }
            
            // Copy frame data to GPU memory
            auto* gpuImagePtr = upImpl->spInputBlob->mutable_gpu_data();
            CUDA_CHECK(cudaMemcpy(gpuImagePtr, inputData.getConstPtr(), inputData.getVolume() * sizeof(float), cudaMemcpyHostToDevice));
            
            // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
            // of these, but in this case we know that there is exactly one input and one output.
            std::vector<void*> buffers(2);
            buffers[0] = upImpl->spInputBlob->mutable_gpu_data();
            buffers[1] = upImpl->spOutputBlob->mutable_gpu_data();
            
            // Perform deep network forward pass
            upImpl->cudaContext->enqueue(1, &buffers[0], upImpl->stream, nullptr);
            
            // Cuda checks
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
    
    boost::shared_ptr<caffe::Blob<float>> NetTensorRT::getOutputBlob() const
    {
        try
        {
            #ifdef USE_TENSORRT
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
    
