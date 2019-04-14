#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/bodyPartConnectorCaffe.hpp>
#include <openpose/net/maximumCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/netOpenCv.hpp>
#include <openpose/net/nmsCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractorNet.hpp>

namespace op
{
    class OP_API PoseExtractorCaffe : public PoseExtractorNet
    {
    public:
        PoseExtractorCaffe(
            const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
            const std::vector<HeatMapType>& heatMapTypes = {},
            const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne,
            const bool addPartCandidates = false, const bool maximizePositives = false,
            const std::string& protoTxtPath = "", const std::string& caffeModelPath = "",
            const float upsamplingRatio = 0.f, const bool enableNet = true,
            const bool enableGoogleLogging = true);

        virtual ~PoseExtractorCaffe();

        virtual void netInitializationOnThread();

        /**
         * @param poseNetOutput If it is not empty, OpenPose will not run its internal body pose estimation network
         * and will instead use this data as the substitute of its network. The size of this element must match the
         * size of the output of its internal network, or it will lead to core dumped (segmentation) errors. You can
         * modify the pose estimation flags to match the dimension of both elements (e.g., `--net_resolution`,
         * `--scale_number`, etc.).
         */
        virtual void forwardPass(
            const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
            const std::vector<double>& scaleInputToNetInputs = {1.f},
            const Array<float>& poseNetOutput = Array<float>{});

        const float* getCandidatesCpuConstPtr() const;

        const float* getCandidatesGpuConstPtr() const;

        const float* getHeatMapCpuConstPtr() const;

        const float* getHeatMapGpuConstPtr() const;

        std::vector<int> getHeatMapSize() const;

        const float* getPoseGpuConstPtr() const;

    private:
        // Used when increasing spNets
        const PoseModel mPoseModel;
        const int mGpuId;
        const std::string mModelFolder;
        const std::string mProtoTxtPath;
        const std::string mCaffeModelPath;
        const float mUpsamplingRatio;
        const bool mEnableNet;
        const bool mEnableGoogleLogging;
        // General parameters
        std::vector<std::shared_ptr<Net>> spNets;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
        std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
        std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
        std::vector<std::vector<int>> mNetInput4DSizes;
        // Init with thread
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> spCaffeNetOutputBlobs;
        std::shared_ptr<ArrayCpuGpu<float>> spHeatMapsBlob;
        std::shared_ptr<ArrayCpuGpu<float>> spPeaksBlob;
        std::shared_ptr<ArrayCpuGpu<float>> spMaximumPeaksBlob;

        DELETE_COPY(PoseExtractorCaffe);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
