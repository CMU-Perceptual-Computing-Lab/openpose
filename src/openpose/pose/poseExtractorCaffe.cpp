#ifdef USE_CAFFE
#include "openpose/core/netCaffe.hpp"
#include "openpose/pose/poseParameters.hpp"
#include "openpose/utilities/check.hpp"
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/openCv.hpp"
#include "openpose/pose/poseExtractorCaffe.hpp"

namespace op
{
    PoseExtractorCaffe::PoseExtractorCaffe(const cv::Size& netInputSize, const cv::Size& netOutputSize, const cv::Size& outputSize, const int scaleNumber,
                                           const float scaleGap, const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScaleMode) :
        PoseExtractor{netOutputSize, outputSize, poseModel, heatMapTypes, heatMapScaleMode},
        spNet{std::make_shared<NetCaffe>(std::array<int,4>{scaleNumber, 3, (int)netInputSize.height, (int)netInputSize.width},
                                         modelFolder + POSE_PROTOTXT[(int)poseModel], modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId)},
        spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
        spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
    {
        try
        {
            checkE(netOutputSize, netInputSize, "Net input and output size must be equal.", __LINE__, __FUNCTION__, __FILE__);
            spResizeAndMergeCaffe->setScaleGap(scaleGap);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorCaffe::netInitializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Caffe net
            spNet->initializationOnThread();
            spCaffeNetOutputBlob = ((NetCaffe*)spNet.get())->getOutputBlob();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // HeatMaps extractor blob and layer
            spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spResizeAndMergeCaffe->Reshape({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()}, POSE_CCN_DECREASE_FACTOR[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, POSE_MAX_PEAKS[(int)mPoseModel], POSE_NUMBER_BODY_PARTS[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spBodyPartConnectorCaffe->setPoseModel(mPoseModel);
            spBodyPartConnectorCaffe->Reshape({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()});
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorCaffe::forwardPass(const Array<float>& inputNetData, const cv::Size& inputDataSize)
    {
        try
        {
            // Security checks
            if (inputNetData.empty())
                error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);

            // 1. Caffe deep network
            spNet->forwardPass(inputNetData.getConstPtr());                                                     // ~79.3836ms

            // 2. Resize heat maps + merge different scales
            #ifndef CPU_ONLY
                spResizeAndMergeCaffe->Forward_gpu({spCaffeNetOutputBlob.get()}, {spHeatMapsBlob.get()});       // ~5ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                error("ResizeAndMergeCaffe CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif

            // 3. Get peaks by Non-Maximum Suppression
            spNmsCaffe->setThreshold(get(PoseProperty::NMSThreshold));
            #ifndef CPU_ONLY
                spNmsCaffe->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});                           // ~2ms
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #else
                error("NmsCaffe CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            #endif

            // Get scale net to output
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const cv::Size netSize{intRound(scaleProducerToNetInput*inputDataSize.width), intRound(scaleProducerToNetInput*inputDataSize.height)};
            mScaleNetToOutput = {resizeGetScaleFactor(netSize, mOutputSize)};

            // 4. Connecting body parts
            spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
            spBodyPartConnectorCaffe->setInterMinAboveThreshold((int)get(PoseProperty::ConnectInterMinAboveThreshold));
            spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
            spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
            spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));

            // GPU version not implemented yet
            spBodyPartConnectorCaffe->Forward_cpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeyPoints);
            // spBodyPartConnectorCaffe->Forward_gpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()}, mPoseKeyPoints);
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
            checkThread();
            return spHeatMapsBlob->cpu_data();
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
            checkThread();
            return spHeatMapsBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getPoseGpuConstPtr() const
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
