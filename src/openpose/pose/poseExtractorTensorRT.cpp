#ifdef USE_CAFFE
#include <openpose/core/netCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/pose/poseExtractorTensorRT.hpp>

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

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)								      	\
{														                \
if (status != 0)									          \
{												                   	\
std::cout << "Cuda failure: " << status;		\
abort();									                	\
}										                   			\
}

struct Params
{
  std::string deployFile, modelFile, engine, calibrationCache;
  std::vector<std::string> outputs;
  int device{ 0 }, batchSize{ 1 }, workspaceSize{ 16 }, iterations{ 10 }, avgRuns{ 10 };
  bool half2{ false }, int8{ false }, verbose{ false }, hostTime{ false };
} gParams;

static inline int volume(DimsCHW dims)
{
  return dims.c()*dims.h()*dims.w();
}

std::vector<std::string> gInputs;
std::map<std::string, DimsCHW> gInputDimensions;

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    // suppress info-level messages
    if (severity != Severity::kINFO || gParams.verbose)
      std::cout << msg << std::endl;
  }
} gLogger;


ICudaEngine* caffeToGIEModel()
{
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);
  
  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* parser = createCaffeParser();
  const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                            gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
                                                            *network,
                                                            gParams.half2 ? DataType::kHALF:DataType::kFLOAT);
  
  
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
  
  for (auto& s : gParams.outputs)
  {
    if (blobNameToTensor->find(s.c_str()) == nullptr)
    {
      std::cout << "could not find output blob " << s << std::endl;
      return nullptr;
    }
    network->markOutput(*blobNameToTensor->find(s.c_str()));
  }
  
  for (int i = 0, n = network->getNbOutputs(); i < n; i++)
  {
    DimsCHW dims = static_cast<DimsCHW&&>(network->getOutput(i)->getDimensions());
    std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
  }
  
  // Build the engine
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(gParams.workspaceSize<<20);
  builder->setHalf2Mode(true);
  
  ICudaEngine* engine = builder->buildCudaEngine(*network);
  if (engine == nullptr)
    std::cout << "could not build engine" << std::endl;
  
  parser->destroy();
  network->destroy();
  builder->destroy();
  shutdownProtobufLibrary();
  return engine;
}

void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
  size_t bindingIndex = engine.getBindingIndex(name.c_str());
  printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
  assert(bindingIndex < buffers.size());
  DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
  size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);
  
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

void doInference(ICudaEngine& engine)
{
  IExecutionContext *context = engine.createExecutionContext();
  // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
  // of these, but in this case we know that there is exactly one input and one output.
  
  std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
  for (size_t i = 0; i < gInputs.size(); i++)
    createMemory(engine, buffers, gInputs[i]);
  
  for (size_t i = 0; i < gParams.outputs.size(); i++)
    createMemory(engine, buffers, gParams.outputs[i]);
  
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  
  for (int j = 0; j < gParams.iterations; j++)
  {
    float total = 0, ms;
    for (int i = 0; i < gParams.avgRuns; i++)
    {
      if (gParams.hostTime)
      {
        auto t_start = std::chrono::high_resolution_clock::now();
        context->execute(gParams.batchSize, &buffers[0]);
        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
      }
      else
      {
        cudaEventRecord(start, stream);
        context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
      }
      total += ms;
    }
    total /= gParams.avgRuns;
    std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms." << std::endl;
  }
  
  
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}


static ICudaEngine* createEngine()
{
  // TODO replace all gParams with corresponding parameters
  ICudaEngine *engine;
  
  if (!gParams.deployFile.empty()) {
    engine = caffeToGIEModel();
    if (!engine)
    {
      std::cerr << "Engine could not be created" << std::endl;
      return nullptr;
    }
    
    
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
    }
    return engine;
  }
  
  // load directly from serialized engine file if deploy not specified
  if (!gParams.engine.empty()) {
    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(gParams.engine, std::ios::binary);
    if (file.good()) {
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
    
    // assume input to be "data" for deserialized engine
    gInputs.push_back("data");
    return engine;
  }
  
  // complain about empty deploy file
  std::cerr << "Deploy file not specified" << std::endl;
  return nullptr;
}



namespace op
{
    PoseExtractorTensorRT::PoseExtractorTensorRT(const Point<int>& netInputSize, const Point<int>& netOutputSize, const Point<int>& outputSize, const int scaleNumber,
                                           const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale) :
        PoseExtractor{netOutputSize, outputSize, poseModel, heatMapTypes, heatMapScale},
        mResizeScale{mNetOutputSize.x / (float)netInputSize.x},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{scaleNumber, 3, (int)netInputSize.y, (int)netInputSize.x},
                                         modelFolder + POSE_PROTOTXT[(int)poseModel], modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId)},
        spResizeAndMergeTensorRT{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsTensorRT{std::make_shared<NmsCaffe<float>>()},
        spBodyPartConnectorTensorRT{std::make_shared<BodyPartConnectorCaffe<float>>()},
        mNetInputSize(netInputSize),
        mNetOutputSize(netOutputSize),
        mOutputSize(outputSize),
        mScaleNumber(scaleNumber),
        mPoseModel(poseModel),
        mModelFolder(modelFolder),
        mGpuId(gpuId),
        mHeatMapTypes(heatMapTypes),
        mHeatMapScale(heatMapScale)
    {
        try
        {
            const auto resizeScale = mNetOutputSize.x / (float)netInputSize.x;
            const auto resizeScaleCheck = resizeScale / (mNetOutputSize.y/(float)netInputSize.y);
            if (1+1e-6 < resizeScaleCheck || resizeScaleCheck < 1-1e-6)
                error("Net input and output size must be proportional. resizeScaleCheck = " + std::to_string(resizeScaleCheck), __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorTensorRT::~PoseExtractorTensorRT()
    {
      if(cudaEngine)
        engine->destroy();
        
    }

    void PoseExtractorTensorRT::netInitializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
          
            cudaEngine = createEngine();
            if (!cudaEngine)
            {
              std::cerr << "Engine could not be created" << std::endl;
              return -1;
            }

            // TensorRT net
            spNet->initializationOnThread();
            spTensorRTNetOutputBlob = ((NetCaffe*)spNet.get())->getOutputBlob();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // HeatMaps extractor blob and layer
            spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spResizeAndMergeTensorRT->Reshape({spTensorRTNetOutputBlob.get()}, {spHeatMapsBlob.get()}, mResizeScale * POSE_CCN_DECREASE_FACTOR[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spNmsTensorRT->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, POSE_MAX_PEAKS[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spBodyPartConnectorTensorRT->setPoseModel(mPoseModel);
            spBodyPartConnectorTensorRT->Reshape({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()});
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorTensorRT::forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize, const std::vector<float>& scaleRatios)
    {
        try
        {
            // Security checks
            if (inputNetData.empty())
                error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);

            // 1. TensorRT deep network
            //spNet->forwardPass(inputNetData.getConstPtr());                                                     // ~79.3836ms
          
            doInference(inputNetData.getConstPtr());
          
            // Replace spNet->forward pass, but how to propagate to next
            // Replace spTensorRTNetOututBlob.get() ?

            // 2. Resize heat maps + merge different scales
            spResizeAndMergeTensorRT->setScaleRatios(scaleRatios);
            #ifndef CPU_ONLY
                spResizeAndMergeTensorRT->Forward_gpu({spTensorRTNetOutputBlob.get()}, {spHeatMapsBlob.get()});       // ~5ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                error("ResizeAndMergeTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif

            // 3. Get peaks by Non-Maximum Suppression
            spNmsTensorRT->setThreshold((float)get(PoseProperty::NMSThreshold));
            #ifndef CPU_ONLY
                spNmsTensorRT->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});                           // ~2ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                error("NmsTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif

            // Get scale net to output
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x), intRound(scaleProducerToNetInput*inputDataSize.y)};
            mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, mOutputSize)};

            // 4. Connecting body parts
            spBodyPartConnectorTensorRT->setScaleNetToOutput(mScaleNetToOutput);
            spBodyPartConnectorTensorRT->setInterMinAboveThreshold((int)get(PoseProperty::ConnectInterMinAboveThreshold));
            spBodyPartConnectorTensorRT->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
            spBodyPartConnectorTensorRT->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
            spBodyPartConnectorTensorRT->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));

            // GPU version not implemented yet
            spBodyPartConnectorTensorRT->Forward_cpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints);
            // spBodyPartConnectorTensorRT->Forward_gpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()}, mPoseKeypoints);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorTensorRT::getHeatMapCpuConstPtr() const
    {
        try
        {
            checkThread();
            return spHeatMapsBlob->cpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorTensorRT::getHeatMapGpuConstPtr() const
    {
        try
        {
            checkThread();
            return spHeatMapsBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorTensorRT::getPoseGpuConstPtr() const
    {
        try
        {
            error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            checkThread();
            return spPoseBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}

#endif



