#ifdef USE_CAFFE
#include <caffe/blob.hpp>
#endif
#include <openpose/core/netTensorRT.hpp>
#include <openpose/core/nmsCaffe.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/pose/poseExtractorTensorRT.hpp>

typedef std::vector<std::pair<std::string, std::chrono::high_resolution_clock::time_point>> OpTimings;

static OpTimings timings;

static void timeNow(const std::string& label){
    const auto now = std::chrono::high_resolution_clock::now();
    const auto timing = std::make_pair(label, now);
    timings.push_back(timing);
}

static std::string timeDiffToString(const std::chrono::high_resolution_clock::time_point& t1,
                                const std::chrono::high_resolution_clock::time_point& t2 ) {
    return std::to_string((double)std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t2).count() * 1e3) + " ms";
}


nameupImpl->space op
{

    struct PoseExtractorTensorRT::ImplPoseExtractorTensorRT
    {
        #ifdef USE_TENSORRT // implies USE_CAFFE for now
            const float upImpl->mResizeScale;
            std::shared_ptr<Net> upImpl->spNet;
            std::shared_ptr<ResizeAndMergeCaffe<float>> upImpl->spResizeAndMergeTensorRT;
            std::shared_ptr<NmsCaffe<float>> upImpl->spNmsTensorRT;
            std::shared_ptr<BodyPartConnectorCaffe<float>> upImpl->spBodyPartConnectorTensorRT;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> upImpl->spTensorRTNetOutputBlob;
            std::shared_ptr<caffe::Blob<float>> upImpl->spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> upImpl->spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> upImpl->spPoseBlob;


            ImplPoseExtractorTensorRT(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                      const Point<int>& outputSize, const int scaleNumber,
                                      const PoseModel poseModel, const int gpuId,
                                      const std::string& modelFolder, const bool enableGoogleLogging) :
        mResizeScale{mNetOutputSize.x / (float)netInputSize.x},
        spNet{std::make_shared<NetTensorRT>(std::array<int,4>{scaleNumber, 3, 
                                            (int)netInputSize.y, (int)netInputSize.x},
                                            modelFolder + POSE_PROTOTXT[(int)poseModel],
                                            modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId)},
        spResizeAndMergeTensorRT{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsTensorRT{std::make_shared<NmsCaffe<float>>()},
        spBodyPartConnectorTensorRT{std::make_shared<BodyPartConnectorCaffe<float>>()}
        {
        }
        #endif
    }
 
    PoseExtractorTensorRT::PoseExtractorTensorRT(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                                 const Point<int>& outputSize, const int scaleNumber,
                                                 const PoseModel poseModel, const std::string& modelFolder, 
                                                 const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                                 const ScaleMode heatMapScale, const bool enableGoogleLogging) :
        PoseExtractor{netOutputSize, outputSize, poseModel, heatMapTypes, heatMapScale},
        #ifdef USE_TENSORRT
        , upImpl{new ImplPoseExtractorTensorRT{netInputSize, netOutputSize, scaleNumber, poseModel,
                                               gpuId, modelFolder, enableGoogleLogging}}
        #endif
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
    }

    void PoseExtractorTensorRT::netInitializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
          

            // TensorRT net
            upImpl->spNet->initializationOnThread();
            upImpl->spTensorRTNetOutputBlob = ((NetTensorRT*)upImpl->spNet.get())->getOutputBlob();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // HeatMaps extractor blob and layer
            upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            upImpl->spResizeAndMergeTensorRT->Reshape({upImpl->spTensorRTNetOutputBlob.get()}, {upImpl->spHeatMapsBlob.get()}, upImpl->mResizeScale * POSE_CCN_DECREASE_FACTOR[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            upImpl->spNmsTensorRT->Reshape({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}, POSE_MAX_PEAKS[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            upImpl->spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            upImpl->spBodyPartConnectorTensorRT->setPoseModel(mPoseModel);
            upImpl->spBodyPartConnectorTensorRT->Reshape({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, {upImpl->spPoseBlob.get()});
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
            timeNow("Start");
            // 1. TensorRT deep network
            upImpl->spNet->forwardPass(inputNetData.getConstPtr());
            timeNow("TensorRT forward");
            // 2. Resize heat maps + merge different scales
            upImpl->spResizeAndMergeTensorRT->setScaleRatios(scaleRatios);
            timeNow("SpResizeAndMergeTensorRT");
            #ifndef CPU_ONLY
                upImpl->spResizeAndMergeTensorRT->Forward_gpu({upImpl->spTensorRTNetOutputBlob.get()}, {upImpl->spHeatMapsBlob.get()});       // ~5ms
                timeNow("RaM forward_gpu");
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                timeNow("CudaCheck");
            #else
                error("ResizeAndMergeTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            timeNow("Resize heat Maps");
            // 3. Get peaks by Non-Maximum Suppression
            upImpl->spNmsTensorRT->setThreshold((float)get(PoseProperty::NMSThreshold));
            #ifndef CPU_ONLY
                upImpl->spNmsTensorRT->Forward_gpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});                           // ~2ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                error("NmsTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            timeNow("Peaks by nms");
            // Get scale net to output
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x), intRound(scaleProducerToNetInput*inputDataSize.y)};
            mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, mOutputSize)};
            timeNow("Scale net to output");
            // 4. Connecting body parts
            upImpl->spBodyPartConnectorTensorRT->setScaleNetToOutput(mScaleNetToOutput);
            upImpl->spBodyPartConnectorTensorRT->setInterMinAboveThreshold((int)get(PoseProperty::ConnectInterMinAboveThreshold));
            upImpl->spBodyPartConnectorTensorRT->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
            upImpl->spBodyPartConnectorTensorRT->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
            upImpl->spBodyPartConnectorTensorRT->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
            // GPU version not implemented yet
            upImpl->spBodyPartConnectorTensorRT->Forward_cpu({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, mPoseKeypoints);
            // upImpl->spBodyPartConnectorTensorRT->Forward_gpu({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, {upImpl->spPoseBlob.get()}, mPoseKeypoints);
            timeNow("Connect Body Parts");
             
            const auto totalTimeSec = timeDiffToString(timings.back().second, timings.front().second);
            const auto message = "Pose estimation successfully finished. Total time: " + totalTimeSec + " seconds.";
            op::log(message, op::Priority::High);

            for(OpTimings::iterator timing = timings.begin()+1; timing != timings.end(); ++timing) {
              const auto log_time = (*timing).first + " - " + timeDiffToString((*timing).second, (*(timing-1)).second);
              op::log(log_time, op::Priority::High);
            }
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
            return upImpl->spHeatMapsBlob->cpu_data();
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
            return upImpl->spHeatMapsBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }


    std::vector<int> PoseExtractorTensorRT::getHeatMapSize() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->shape();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }


    const float* PoseExtractorTensorRT::getPoseGpuConstPtr() const
    {
        try
        {
            error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            checkThread();
            return upImpl->spPoseBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}




