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
#include <openpose/utilities/standard.hpp>
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


namespace op
{

    struct PoseExtractorTensorRT::ImplPoseExtractorTensorRT
    {
        #ifdef USE_TENSORRT // implies USE_TENSORRT for now
            const PoseModel mPoseModel;
            const int mGpuId;
            const std::string mModelFolder;
            const bool mEnableGoogleLogging;
            // General parameters
            std::vector<std::shared_ptr<NetTensorRT>> spTensorRTNets;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            std::vector<std::vector<int>> mNetInput4DSizes;
            std::vector<double> mScaleInputToNetInputs;
            // Init with thread
            std::vector<boost::shared_ptr<caffe::Blob<float>>> spTensorRTNetOutputBlobs;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> spPoseBlob;

            ImplPoseExtractorTensorRT(const PoseModel poseModel, const int gpuId,
                                      const std::string& modelFolder, const bool enableGoogleLogging) :
                mPoseModel{poseModel},
                mGpuId{gpuId},
                mModelFolder{modelFolder},
                mEnableGoogleLogging{enableGoogleLogging},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
                                                             std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
        {
            try
            {
                // Prepare spTensorRTNetOutputBlobss
                std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
                for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
                    caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
                return caffeNetOutputBlobs;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }
    
        inline void reshapePoseExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
                                              std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
                                              std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
                                              std::shared_ptr<caffe::Blob<float>>& poseBlob,
                                              const float scaleInputToNetInput,
                                              const PoseModel poseModel)
        {
            try
            {
                // HeatMaps extractor blob and layer
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);
                resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {heatMapsBlob.get()},
                                             POSE_CCN_DECREASE_FACTOR[(int)poseModel], 1.f/scaleInputToNetInput);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, POSE_MAX_PEAKS[(int)poseModel]);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, {poseBlob.get()});
                // Cuda check
    #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
    #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    
        void addTensorRTNetOnThread(std::vector<std::shared_ptr<NetTensorRT>>& netTensorRT,
                                 std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                 const PoseModel poseModel, const int gpuId,
                                 const std::string& modelFolder, const bool enableGoogleLogging)
        {
            try
            {
                // Add Caffe Net
                netTensorRT.emplace_back(
                                      std::make_shared<NetTensorRT>(modelFolder + POSE_PROTOTXT[(int)poseModel],
                                                                 modelFolder + POSE_TRAINED_MODEL[(int)poseModel],
                                                                 gpuId, enableGoogleLogging)
                                      );
                // Initializing them on the thread
                netTensorRT.back()->initializationOnThread();
                caffeNetOutputBlob.emplace_back(netTensorRT.back()->getOutputBlob());
                // Security checks
                if (netTensorRT.size() != caffeNetOutputBlob.size())
                    error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
                // Cuda check
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

 
    PoseExtractorTensorRT::PoseExtractorTensorRT(const PoseModel poseModel, const std::string& modelFolder,
                                                 const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                                 const ScaleMode heatMapScale, const bool enableGoogleLogging) :
        PoseExtractor{poseModel, heatMapTypes, heatMapScale}
        #ifdef USE_TENSORRT
        , upImpl{new ImplPoseExtractorTensorRT{poseModel, gpuId, modelFolder, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifdef USE_TENSORRT
                // Layers parameters
                upImpl->spBodyPartConnectorCaffe->setPoseModel(mPoseModel);
            #else
            UNUSED(poseModel);
            UNUSED(modelFolder);
            UNUSED(gpuId);
            UNUSED(heatMapTypes);
            UNUSED(heatMapScale);
            error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
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
            #ifdef USE_TENSORRT
            
                // Logging
                log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Initialize Caffe net
                addTensorRTNetOnThread(upImpl->spTensorRTNets, upImpl->spTensorRTNetOutputBlobs, upImpl->mPoseModel,
                                    upImpl->mGpuId, upImpl->mModelFolder, upImpl->mEnableGoogleLogging);
            
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            
                // Initialize blobs
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};

                cudaCheck(__LINE__, __FUNCTION__, __FILE__);

                // Logging
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorTensorRT::forwardPass(const std::vector<Array<float>>& inputNetData,
                                            const Point<int>& inputDataSize,
                                            const std::vector<double>& scaleInputToNetInputs)
    {
        try
        {
            #ifdef USE_TENSORRT
                // Security checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                for (const auto& inputNetDataI : inputNetData)
                    if (inputNetDataI.empty())
                        error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                if (inputNetData.size() != scaleInputToNetInputs.size())
                    error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                          __LINE__, __FUNCTION__, __FILE__);
            
                timeNow("Start");
            
                // Resize std::vectors if required
                const auto numberScales = inputNetData.size();
                upImpl->mNetInput4DSizes.resize(numberScales);
                while (upImpl->spTensorRTNets.size() < numberScales)
                    addTensorRTNetOnThread(upImpl->spTensorRTNets, upImpl->spTensorRTNetOutputBlobs, upImpl->mPoseModel,
                                        upImpl->mGpuId, upImpl->mModelFolder, false);
            
                // Process each image
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    // 1. TensorRT deep network
                    upImpl->spTensorRTNets.at(i)->forwardPass(inputNetData[i]);
                    
                    // Reshape blobs if required
                    // Note: In order to resize to input size to have same results as Matlab, uncomment the commented
                    // lines
                    if (!vectorsAreEqual(upImpl->mNetInput4DSizes.at(i), inputNetData[i].getSize()))
                        // || !vectorsAreEqual(upImpl->mScaleInputToNetInputs, scaleInputToNetInputs))
                    {
                        upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                        mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
                            upImpl->mNetInput4DSizes[0][2]};
                        // upImpl->mScaleInputToNetInputs = scaleInputToNetInputs;
                        reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                                  upImpl->spBodyPartConnectorCaffe, upImpl->spTensorRTNetOutputBlobs,
                                                  upImpl->spHeatMapsBlob, upImpl->spPeaksBlob, upImpl->spPoseBlob,
                                                  1.f, mPoseModel);
                        // scaleInputToNetInputs[i], mPoseModel);
                    }
                }
            
                timeNow("TensorRT forwards");
            
                // 2. Resize heat maps + merge different scales
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spTensorRTNetOutputBlobs);
                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                #ifdef USE_CUDA // Implied by tensorrt
                upImpl->spResizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs,                             // ~5ms
                                                           {upImpl->spHeatMapsBlob.get()});
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #else // Never reached, suppress ?
                upImpl->spResizeAndMergeCaffe->Forward_cpu({upImpl->spCaffeNetOutputBlob.get()},
                                                           {upImpl->spHeatMapsBlob.get()});
                #endif
            
                timeNow("Resize heat Maps");
            
                // 3. Get peaks by Non-Maximum Suppression
                upImpl->spNmsCaffe->setThreshold((float)get(PoseProperty::NMSThreshold));
                #ifdef USE_CUDA
                upImpl->spNmsCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});// ~2ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #else
                error("NmsCaffe CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                #endif
            
                timeNow("Peaks by nms");
            
                // Get scale net to output (i.e. image input)
                // Note: In order to resize to input size, (un)comment the following lines
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                    intRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
                // mScaleNetToOutput = 1.f;
            
                timeNow("Scale net to output");
            
                // 4. Connecting body parts
                // Get scale net to output (i.e. image input)
                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                                                                            (float)get(PoseProperty::ConnectInterMinAboveThreshold)
                                                                            );
                upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
            
                // GPU version not implemented yet
                // #ifdef USE_CUDA
                //     upImpl->spBodyPartConnectorCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get(),
                //                                                    upImpl->spPeaksBlob.get()},
                //                                                   {upImpl->spPoseBlob.get()}, mPoseKeypoints);
                // #else
                upImpl->spBodyPartConnectorCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get(),
                    upImpl->spPeaksBlob.get()},
                                                              mPoseKeypoints, mPoseScores);
                // #endif
            
                timeNow("Connect Body Parts");
                 
                const auto totalTimeSec = timeDiffToString(timings.back().second, timings.front().second);
                const auto message = "Pose estimation successfully finished. Total time: " + totalTimeSec + " seconds.";
                op::log(message, op::Priority::High);

                for(OpTimings::iterator timing = timings.begin()+1; timing != timings.end(); ++timing) {
                  const auto log_time = (*timing).first + " - " + timeDiffToString((*timing).second, (*(timing-1)).second);
                  op::log(log_time, op::Priority::High);
                }
            #else
                UNUSED(inputNetData);
                UNUSED(inputDataSize);
                UNUSED(scaleInputToNetInputs);
            #endif
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
            #ifdef USE_TENSORRT
                checkThread();
                return upImpl->spHeatMapsBlob->cpu_data();
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

    const float* PoseExtractorTensorRT::getHeatMapGpuConstPtr() const
    {
        try
        {
            #ifdef USE_TENSORRT
                checkThread();
                return upImpl->spHeatMapsBlob->gpu_data();
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

    std::vector<int> PoseExtractorTensorRT::getHeatMapSize() const
    {
        try
        {
            #ifdef USE_TENSORRT
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
            #ifdef USE_TENSORRT
                error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                checkThread();
                return upImpl->spPoseBlob->gpu_data();
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
