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
  mCaffeProto{caffeProto},
  mCaffeTrainedModel{caffeTrainedModel},
  mLastBlobName{lastBlobName}
  {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
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
  
  
  NetTensorRT::ICudaEngine* caffeToGIEModel()
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
        std::err << "Multiple output unsupported for now!" << std:endl;
    }
    
    // Specify which tensor is output (multiple unsupported)
    if (blobNameToTensor->find(mLastBlobName.c_str()) == nullptr)
    {
      std::cout << "could not find output blob " << s << std::endl;
      return nullptr;
    }
    network->markOutput(*blobNameToTensor->find(s.c_str()));
    
    
    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
      DimsCHW dims = static_cast<DimsCHW&&>(network->getOutput(i)->getDimensions());
      std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
      mNetOutputSize4D = { 1, dims.c(), dims.h(), dims.w() };
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
  
  
  NetTensorRT::ICudaEngine* createEngine()
  {
    ICudaEngine *engine;
    
    engine = caffeToGIEModel(caffeProto, caffeTrainedModel);
    if (!engine)
    {
      std::cerr << "Engine could not be created" << std::endl;
      return nullptr;
    }
    
    /* TODO Serialize and load engines for given net size as optim quite long
     if (!gParams.engine.empty())
     {
     std::ofstream p(gParams.engine);
     if (!p)
     {
     std::cerr << "could not open plan output file" << std::endl;
     return nullptr;
     }
     IHostMemory *ptr = engine->serialize();
     assert(ptr);
     p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
     ptr->destroy();
     }*/
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
      
      cudaEngine = createEngine(mCaffeProto, mCaffeTrainedModel);
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
      
      std::cout << "InitializationOnThread : done" << std::endl;
      
      
      std::cout << "NetInputSize4D: " << mNetInputSize4D[0] << " " << mNetInputSize4D[1] << " " << mNetInputSize4D[2] << " " << mNetInputSize4D[3] << std::endl;

      spInputBlob = boost::make_shared<caffe::Blob<float>>(mNetInputSize4D[0], mNetInputSize4D[1], mNetInputSize4D[2], mNetInputSize4D[3]);
      spOutputBlob = boost::make_shared<caffe::Blob<float>>(mNetOutputSize4D[0], mNetOutputSize4D[1], mNetOutputSize4D[2], mNetOutputSize4D[3]);
      
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
        
        //createMemory(*cudaEngine, buffers, gInputs[i]);
        
        const int batchSize = 1;
        size_t eltCount = 1*57*46*82*batchSize, memSize = eltCount * sizeof(float);
          
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
      }
      std::cout << "Forward Pass : executing inference" << std::endl;
      
      context->execute(batchSize, &buffers[0]);
      
      spOutputBlob->set_gpu_data((float*)deviceMem);
      
      std::cout << "Forward Pass : inference done !" << std::endl;
      cudaCheck(__LINE__, __FUNCTION__, __FILE__);
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
