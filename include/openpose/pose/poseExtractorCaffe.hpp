#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractor.hpp>

namespace op
{
    class OP_API PoseExtractorCaffe : public PoseExtractor
    {
    public:
        PoseExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize,
                           const Point<int>& outputSize, const int scaleNumber, const PoseModel poseModel,
                           const std::string& modelFolder, const int gpuId,
                           const std::vector<HeatMapType>& heatMapTypes = {},
                           const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        virtual ~PoseExtractorCaffe();

        void netInitializationOnThread();

        void forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize,
                         const std::vector<double>& scaleRatios = {1.f});

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
