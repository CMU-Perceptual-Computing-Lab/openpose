#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP

#include <openpose/core/common.hpp>
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

        void netInitializationOnThread();

        /**
         * @param poseNetOutput If it is not empty, OpenPose will not run its internal body pose estimation network
         * and will instead use this data as the substitute of its network. The size of this element must match the
         * size of the output of its internal network, or it will lead to core dumped (segmentation) errors. You can
         * modify the pose estimation flags to match the dimension of both elements (e.g., `--net_resolution`,
         * `--scale_number`, etc.).
         */
        void forwardPass(
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
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplPoseExtractorCaffe;
        std::unique_ptr<ImplPoseExtractorCaffe> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(PoseExtractorCaffe);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
