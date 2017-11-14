#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractor.hpp>

namespace op
{
    class OP_API PoseExtractorTensorRT : public PoseExtractor
    {
    public:
        PoseExtractorTensorRT(const std::string& modelFolder, const int gpuId,
                              const std::vector<HeatMapType>& heatMapTypes = {},
                              const ScaleMode heatMapScale = ScaleMode::ZeroToOne,
                              const bool enableGoogleLogging = true);

        virtual ~PoseExtractorTensorRT();

        void netInitializationOnThread();

        void forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize,
                         const std::vector<float>& scaleRatios = {1.f});

         
        const float* getHeatMapCpuConstPtr() const;

        const float* getHeatMapGpuConstPtr() const;

        std::vector<int> getHeatMapSize() const;

        const float* getPoseGpuConstPtr() const;

  private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplPoseExtractorTensorRT;
        std::unique_ptr<ImplPoseExtractorTensorRT> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(PoseExtractorTensorRT);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP
