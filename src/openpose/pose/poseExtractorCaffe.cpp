#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/core/netCaffe.hpp>
#include <openpose/core/nmsCaffe.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>

namespace op
{
    struct PoseExtractorCaffe::ImplPoseExtractorCaffe
    {
        #ifdef USE_CAFFE
            const float mResizeScale;
            std::shared_ptr<NetCaffe> spNetCaffe;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> spPoseBlob;

            ImplPoseExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                   const int scaleNumber, const PoseModel poseModel, const int gpuId,
                                   const std::string& modelFolder) :
                mResizeScale{netOutputSize.x / (float)netInputSize.x},
                spNetCaffe{std::make_shared<NetCaffe>(std::array<int,4>{scaleNumber, 3, (int)netInputSize.y,
                                                      (int)netInputSize.x},
                                                      modelFolder + POSE_PROTOTXT[(int)poseModel],
                                                      modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId)},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
            {
            }
        #endif
    };

    PoseExtractorCaffe::PoseExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                                           const Point<int>& outputSize, const int scaleNumber,
                                           const PoseModel poseModel, const std::string& modelFolder,
                                           const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale) :
        PoseExtractor{netOutputSize, outputSize, poseModel, heatMapTypes, heatMapScale}
        #ifdef USE_CAFFE
        , upImpl{new ImplPoseExtractorCaffe{netInputSize, netOutputSize, scaleNumber, poseModel,
                                            gpuId, modelFolder}}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
                const auto resizeScale = mNetOutputSize.x / (float)netInputSize.x;
                const auto resizeScaleCheck = resizeScale / (mNetOutputSize.y/(float)netInputSize.y);
                if (1+1e-6 < resizeScaleCheck || resizeScaleCheck < 1-1e-6)
                    error("Net input and output size must be proportional. resizeScaleCheck = "
                          + std::to_string(resizeScaleCheck), __LINE__, __FUNCTION__, __FILE__);
                // Layers parameters
                upImpl->spBodyPartConnectorCaffe->setPoseModel(mPoseModel);
            #else
                UNUSED(netInputSize);
                UNUSED(netOutputSize);
                UNUSED(outputSize);
                UNUSED(scaleNumber);
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

    PoseExtractorCaffe::~PoseExtractorCaffe()
    {
    }

    void PoseExtractorCaffe::netInitializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                // Logging
                log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Caffe net
                upImpl->spNetCaffe->initializationOnThread();
                upImpl->spCaffeNetOutputBlob = upImpl->spNetCaffe->getOutputBlob();
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                // HeatMaps extractor blob and layer
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spResizeAndMergeCaffe->Reshape({upImpl->spCaffeNetOutputBlob.get()}, {upImpl->spHeatMapsBlob.get()},
                                                        upImpl->mResizeScale * POSE_CCN_DECREASE_FACTOR[(int)mPoseModel]);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                // Pose extractor blob and layer
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spNmsCaffe->Reshape({upImpl->spHeatMapsBlob.get()},
                                            {upImpl->spPeaksBlob.get()}, POSE_MAX_PEAKS[(int)mPoseModel]);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                // Pose extractor blob and layer
                upImpl->spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spBodyPartConnectorCaffe->Reshape({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()},
                                                          {upImpl->spPoseBlob.get()});
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

    void PoseExtractorCaffe::forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize,
                                         const std::vector<double>& scaleRatios)
    {
        try
        {
            #ifdef USE_CAFFE
                // Security checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);

                // 1. Caffe deep network
                upImpl->spNetCaffe->forwardPass(inputNetData.getConstPtr());                                    // ~80ms

                // 2. Resize heat maps + merge different scales
                const std::vector<float> floatScaleRatios(scaleRatios.begin(), scaleRatios.end());
                upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                #ifdef USE_CUDA
                    upImpl->spResizeAndMergeCaffe->Forward_gpu({upImpl->spCaffeNetOutputBlob.get()},            // ~5ms
                                                               {upImpl->spHeatMapsBlob.get()});
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #else
                    error("ResizeAndMergeCaffe CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                #endif

                // 3. Get peaks by Non-Maximum Suppression
                upImpl->spNmsCaffe->setThreshold((float)get(PoseProperty::NMSThreshold));
                #ifdef USE_CUDA
                    upImpl->spNmsCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});// ~2ms
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #else
                    error("NmsCaffe CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                #endif

                // Get scale net to output
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                         intRound(scaleProducerToNetInput*inputDataSize.y)};
                if (mOutputSize.x > 0 && mOutputSize.y > 0)
                    mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, mOutputSize)};
                else
                    mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};

                // 4. Connecting body parts
                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (int)get(PoseProperty::ConnectInterMinAboveThreshold)
                );
                upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));

                // GPU version not implemented yet
                upImpl->spBodyPartConnectorCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()},
                                                               mPoseKeypoints);
                // upImpl->spBodyPartConnectorCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()},
                //                                               {upImpl->spPoseBlob.get()}, mPoseKeypoints);
            #else
                UNUSED(inputNetData);
                UNUSED(inputDataSize);
                UNUSED(scaleRatios);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorCaffe::getHeatMapCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
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

    const float* PoseExtractorCaffe::getHeatMapGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
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

    std::vector<int> PoseExtractorCaffe::getHeatMapSize() const
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

    const float* PoseExtractorCaffe::getPoseGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
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
