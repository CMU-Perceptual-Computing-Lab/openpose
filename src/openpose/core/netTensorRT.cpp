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

#define CUDA_TENSORRT_CHECK(status)								      	\
{														                \
if (status != 0)									          \
{												                   	\
std::cout << "Cuda failure: " << status;		\
abort();									                	\
}										                   			\
}

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

void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
        size_t bindingIndex = engine.getBindingIndex(name.c_str());
        printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
        assert(bindingIndex < buffers.size());
        DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
        size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*1, memSize = eltCount * sizeof(float);

        float* localMem = new float[eltCount];
        for (size_t i = 0; i < eltCount; i++)
                localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
                std::cerr << "Out of memory" << std::endl;
                exit(1);
        }
        CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

        delete[] localMem;
        buffers[bindingIndex] = deviceMem;
}


ICudaEngine* caffeToGIEModel(const std::string& caffeProto, const std::string& caffeTrainedModel)
{
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);
  
  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* parser = createCaffeParser();
  const IBlobNameToTensor* blobNameToTensor = parser->parse(caffeProto.c_str(),
                                                            caffeTrainedModel.c_str(),
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
  }
  
  // specify which tensors are outputs
  
  
  // TODO, if it works switch to something more generic, add as parameter etc
  std::string s("net_output");
  if (blobNameToTensor->find(s.c_str()) == nullptr)
  {
    std::cout << "could not find output blob " << s << std::endl;
    return nullptr;
  }
  network->markOutput(*blobNameToTensor->find(s.c_str()));
  
  
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


static ICudaEngine* createEngine(const std::string& caffeProto, const std::string& caffeTrainedModel)
{
  ICudaEngine *engine;
  
  engine = caffeToGIEModel(caffeProto, caffeTrainedModel);
  if (!engine)
  {
    std::cerr << "Engine could not be created" << std::endl;
    return nullptr;
  }
  
  /* TODO seems unneeded, remove if so.
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
  }
  
  NetTensorRT::~NetTensorRT()
  {
    if (cudaEngine)
      cudaEngine->destroy();
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
        std::cerr << "Engine could not be created" << std::endl;
        return;
      }
      
      std::cout << "InitializationOnThread : done" << std::endl;
      
      
      
      spInputBlob = boost::make_shared<caffe::Blob<float>>(1, 3, 368, 656);
      spOutputBlob = boost::make_shared<caffe::Blob<float>>(1, 57, 46, 82);
      
      // For tensor RT is done in caffeToGIE
      /*
      //caffe::TensorRT::SetDevice(mGpuId);
      upTensorRTNet.reset(new caffe::Net<float>{mTensorRTProto, caffe::TEST});
      upTensorRTNet->CopyTrainedLayersFrom(mTensorRTTrainedModel);
      upTensorRTNet->blobs()[0]->Reshape({mNetInputSize4D[0], mNetInputSize4D[1], mNetInputSize4D[2], mNetInputSize4D[3]});
      upTensorRTNet->Reshape();
      cudaCheck(__LINE__, __FUNCTION__, __FILE__);*/
      // Set spOutputBlob
      /*
      if (spOutputBlob == nullptr)
        error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: " + mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
      cudaCheck(__LINE__, __FUNCTION__, __FILE__);*/
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
        
      
        
        // OLD
        auto* gpuImagePtr = spInputBlob->mutable_gpu_data();
        CUDA_TENSORRT_CHECK(cudaMemcpy(gpuImagePtr, inputData, mNetInputMemory, cudaMemcpyHostToDevice));
        
        // Tensor RT version
        
        // TODO maybe move this to init and keep only the execute part
        
        std::cout << "Forward Pass : creating execution context" << std::endl;
        
        IExecutionContext *context = cudaEngine->createExecutionContext();
        // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        
        std::cout << "Forward Pass : creating CUDA memory" << std::endl;
        
        
        /*
          const int batchSize = 1;
          size_t bindingIndex = engine.getBindingIndex(name.c_str());
        std::cout"name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
          assert(bindingIndex < buffers.size());
          DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
          size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*batchSize, memSize = eltCount * sizeof(float);
          
          float* localMem = new float[eltCount];
          for (size_t i = 0; i < eltCount; i++)
            localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;
          
          void* deviceMem;
          CUDA_TENSORRT_CHECK(cudaMalloc(&deviceMem, memSize));
          if (deviceMem == nullptr)
          {
            std::cerr << "Out of memory" << std::endl;
            exit(1);
          }
          CUDA_TENSORRT_CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));
          
          delete[] localMem;
          buffers[bindingIndex] = deviceMem;
          */
        
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
        CUDA_TENSORRT_CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
          std::cerr << "Out of memory" << std::endl;
          exit(1);
        }
        CUDA_TENSORRT_CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));
          
        
        buffers[1] = deviceMem;
        //spOutputBlob->set_gpu_data((float*)deviceMem);

        
        //createMemory(*cudaEngine, buffers, std::string("net_output"));
        
        
        std::cout << "Forward Pass : memory created" << std::endl;
        
        cudaStream_t stream;
        CUDA_TENSORRT_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t start, end;
        CUDA_TENSORRT_CHECK(cudaEventCreate(&start));
        CUDA_TENSORRT_CHECK(cudaEventCreate(&end));
        
        
        std::cout << "Forward Pass : executing inference" << std::endl;
        
        //spInputBlob->Update();
        context->execute(batchSize, &buffers[0]);
        //spOutputBlob->Update();
        spOutputBlob->set_gpu_data((float*)deviceMem);
        //CUDA_TENSORRT_CHECK(cudaMemcpy(localMem, buffers[1], memSize, cudaMemcpyDeviceToHost)); 
        //spOutputBlob->set_cpu_data((float*)localMem);

        delete[] localMem;
        std::cout << "Forward Pass : inference done !" << std::endl;
        
        
        
        cudaStreamDestroy(stream);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        
      }
      // Old Perform deep network forward pass
      //upTensorRTNet->ForwardFrom(0);
      //cudaCheck(__LINE__, __FUNCTION__, __FILE__);
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
