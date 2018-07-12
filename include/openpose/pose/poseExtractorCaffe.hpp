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
        PoseExtractorCaffe(const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
                           const std::vector<HeatMapType>& heatMapTypes = {},
                           const ScaleMode heatMapScale = ScaleMode::ZeroToOne,
                           const bool addPartCandidates = false,
                           const bool enableGoogleLogging = true);

        virtual ~PoseExtractorCaffe();

        void netInitializationOnThread();

        void forwardPass(const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
                         const std::vector<double>& scaleInputToNetInputs = {1.f});

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
