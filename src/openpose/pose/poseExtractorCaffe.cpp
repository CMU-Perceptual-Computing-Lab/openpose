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
            std::vector<int> mNetInputSize4D;
            std::shared_ptr<NetCaffe> spNetCaffe;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            // Init with thread
            boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> spPoseBlob;

            ImplPoseExtractorCaffe(const PoseModel poseModel, const int gpuId,
                                   const std::string& modelFolder, const bool enableGoogleLogging) :
                mNetInputSize4D{0,0,0,0},
                spNetCaffe{std::make_shared<NetCaffe>(modelFolder + POSE_PROTOTXT[(int)poseModel],
                                                      modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId,
                                                      enableGoogleLogging)},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        inline void reshapePoseExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
                                              std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
                                              boost::shared_ptr<caffe::Blob<float>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
                                              std::shared_ptr<caffe::Blob<float>>& poseBlob,
                                              const float scaleInputToNetInput,
                                              const PoseModel poseModel)
        {
            try
            {
                // HeatMaps extractor blob and layer
                UNUSED(scaleInputToNetInput);
                resizeAndMergeCaffe->Reshape({caffeNetOutputBlob.get()}, {heatMapsBlob.get()},
                                             POSE_CCN_DECREASE_FACTOR[(int)poseModel]);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, POSE_MAX_PEAKS[(int)poseModel]);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, {poseBlob.get()});
                // Cuda check
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        inline bool requiredReshapePoseExtractorCaffe(const std::vector<int>& dimensionsA,
                                                      const std::vector<int>& dimensionsB)
        {
            try
            {
                return (dimensionsA[0] != dimensionsB[0] || dimensionsA[1] != dimensionsB[1]
                        || dimensionsA[2] != dimensionsB[2] || dimensionsA[3] != dimensionsB[3]);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }
    #endif

    PoseExtractorCaffe::PoseExtractorCaffe(const PoseModel poseModel, const std::string& modelFolder,
                                           const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale, const bool enableGoogleLogging) :
        PoseExtractor{poseModel, heatMapTypes, heatMapScale}
        #ifdef USE_CAFFE
        , upImpl{new ImplPoseExtractorCaffe{poseModel, gpuId, modelFolder, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
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
                // Initialize Caffe net
                upImpl->spNetCaffe->initializationOnThread();
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                // Initialize blobs
                upImpl->spCaffeNetOutputBlob = upImpl->spNetCaffe->getOutputBlob();
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

    void PoseExtractorCaffe::forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize,
                                         const std::vector<double>& scaleInputToNetInputs)
    {
        try
        {
            #ifdef USE_CAFFE
                // Security checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);

                // 1. Caffe deep network
                upImpl->spNetCaffe->forwardPass(inputNetData);                                                  // ~80ms

                // Reshape blobs if required
                if (requiredReshapePoseExtractorCaffe(upImpl->mNetInputSize4D, inputNetData.getSize()))
                {
                    upImpl->mNetInputSize4D = inputNetData.getSize();
                    mNetOutputSize = Point<int>{upImpl->mNetInputSize4D[3], upImpl->mNetInputSize4D[2]};
                    reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                              upImpl->spBodyPartConnectorCaffe, upImpl->spCaffeNetOutputBlob,
                                              upImpl->spHeatMapsBlob, upImpl->spPeaksBlob, upImpl->spPoseBlob,
                                              scaleInputToNetInputs[0], mPoseModel);
                }

                // 2. Resize heat maps + merge different scales
                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
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

                // Get scale net to output (i.e. image input)
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                         intRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};

                // 4. Connecting body parts
                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                // upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(1);
                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (int)get(PoseProperty::ConnectInterMinAboveThreshold)
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
                                                                  mPoseKeypoints);
                // #endif
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
