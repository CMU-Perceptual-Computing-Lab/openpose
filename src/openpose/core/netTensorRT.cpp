#ifdef USE_CAFFE
#include <numeric> // std::accumulate
#include <openpose/utilities/cuda.hpp>
#include <openpose/core/netTensorRT.hpp>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <map>
#include <random>
#include <boost/make_shared.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;


std::vector<std::string> gInputs;
std::map<std::string, DimsCHW> gInputDimensions;


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
  NetTensorRT::NetTensorRT(const std::array<int, 4>& netInputSize4D, const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId, const std::string& lastBlobName) :
  mGpuId{gpuId},
  // mNetInputSize4D{netInputSize4D}, // This line crashes on some devices with old G++
  mNetInputSize4D{netInputSize4D[0], netInputSize4D[1], netInputSize4D[2], netInputSize4D[3]},
  mNetInputMemory{std::accumulate(mNetInputSize4D.begin(), mNetInputSize4D.end(), 1, std::multiplies<int>()) * sizeof(float)},
  mCaffeProto{caffeProto + "_" + std::to_string(mNetInputSize4D[2]) + "x" + std::to_string(mNetInputSize4D[3])},
  mCaffeTrainedModel{caffeTrainedModel},
  mLastBlobName{lastBlobName}
  {
    std::cout << "Caffe file: " << mCaffeProto.c_str() << std::endl;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
  }
  
  NetTensorRT::~NetTensorRT()
  {
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    if (cudaEngine)
      cudaEngine->destroy();
  }
  
  
  ICudaEngine* NetTensorRT::caffeToGIEModel()
  {
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(mCaffeProto.c_str(),
                                                              mCaffeTrainedModel.c_str(),
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
    if (blobNameToTensor->find(mLastBlobName.c_str()) == nullptr)
    {
      std::cout << "could not find output blob " << mLastBlobName.c_str() << std::endl;
      return nullptr;
    }
    network->markOutput(*blobNameToTensor->find(mLastBlobName.c_str()));
    
    
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
  
  inline bool file_exists(const std::string& file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
  }
  
  ICudaEngine* NetTensorRT::createEngine()
  {
    ICudaEngine *engine;
    
    std::string serializedEnginePath = mCaffeProto + ".bin";

    std::cout << "Serialized engine path: " << serializedEnginePath.c_str() << std::endl;
    if (file_exists(serializedEnginePath))
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
  
  void NetTensorRT::initializationOnThread()
  {
    
    std::cout << "InitializationOnThread : start" << std::endl;
    
    try
    {
      
      std::cout << "InitializationOnThread : setting device" << std::endl;
      // Initialize net
      cudaSetDevice(mGpuId);
      
      std::cout << "InitializationOnThread : creating engine" << std::endl;
      
      cudaEngine = createEngine();
      if (!cudaEngine)
      {
        std::cerr << "cudaEngine could not be created" << std::endl;
        return;
      }
      
      std::cout << "InitializationOnThread Pass : creating execution context" << std::endl;
      
      cudaContext = cudaEngine->createExecutionContext();
      if (!cudaContext)
      {
        std::cerr << "cudaContext could not be created" << std::endl;
        return;
      }

      DimsCHW outputDims = static_cast<DimsCHW&&>(cudaEngine->getBindingDimensions(cudaEngine->getNbBindings() - 1));      
      mNetOutputSize4D = { 1, outputDims.c(), outputDims.h(), outputDims.w() };

      
      std::cout << "NetInputSize4D: " << mNetInputSize4D[0] << " " << mNetInputSize4D[1] << " " << mNetInputSize4D[2] << " " << mNetInputSize4D[3] << std::endl;

      spInputBlob = boost::make_shared<caffe::Blob<float>>(mNetInputSize4D[0], mNetInputSize4D[1], mNetInputSize4D[2], mNetInputSize4D[3]);
      spOutputBlob = boost::make_shared<caffe::Blob<float>>(mNetOutputSize4D[0], mNetOutputSize4D[1], mNetOutputSize4D[2], mNetOutputSize4D[3]);
      
      std::cout << "InitializationOnThread : done" << std::endl;
      cudaCheck(__LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
  }
  
  float* NetTensorRT::getInputDataCpuPtr() const
  {
    try
    {
      return spInputBlob->mutable_cpu_data();
    }
    catch (const std::exception& e)
    {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
      return nullptr;
    }
  }
  
  float* NetTensorRT::getInputDataGpuPtr() const
  {
    try
    {
      return spInputBlob->mutable_gpu_data();
    }
    catch (const std::exception& e)
    {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
      return nullptr;
    }
  }
  
  void NetTensorRT::forwardPass(const float* const inputData) const
  {
    
    std::cout << "Forward Pass : start" << std::endl;
    try
    {
      const int batchSize = 1;
      // Copy frame data to GPU memory
      if (inputData != nullptr)
      {
        auto* gpuImagePtr = spInputBlob->mutable_gpu_data();
        CUDA_CHECK(cudaMemcpy(gpuImagePtr, inputData, mNetInputMemory, cudaMemcpyHostToDevice));
        
        // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        
        std::cout << "Forward Pass : creating CUDA memory" << std::endl;
        
        std::vector<void*> buffers(2);
        buffers[0] = spInputBlob->mutable_gpu_data();
        buffers[1] = spOutputBlob->mutable_gpu_data();
        
        size_t eltCount = mNetOutputSize4D[0]*mNetOutputSize4D[1]*mNetOutputSize4D[2]*mNetOutputSize4D[3]*batchSize, memSize = eltCount * sizeof(float);
          
        float* localMem = new float[eltCount];
        for (size_t i = 0; i < eltCount; i++)
          localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;
          
        void* deviceMem;
        CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
          std::cerr << "Out of memory" << std::endl;
          exit(1);
        }
        CUDA_CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));
          
        
        buffers[1] = deviceMem;
        delete[] localMem;
        
        std::cout << "Forward Pass : memory created" << std::endl;
        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
      
        std::cout << "Forward Pass : executing inference" << std::endl;
      
        cudaContext->enqueue(batchSize, &buffers[0], stream, nullptr);
      
        spOutputBlob->set_gpu_data((float*)deviceMem);
      
        std::cout << "Forward Pass : inference done !" << std::endl;
        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
      }
    }
    catch (const std::exception& e)
    {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
  }
  
  boost::shared_ptr<caffe::Blob<float>> NetTensorRT::getOutputBlob() const
  {
    std::cout << "Getting output blob." << std::endl;
    try
    {
      return spOutputBlob;
    }
    catch (const std::exception& e)
    {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
      return nullptr;
    }
    
    std::cout << "Got something..." << std::endl;
  }
}

#endif
