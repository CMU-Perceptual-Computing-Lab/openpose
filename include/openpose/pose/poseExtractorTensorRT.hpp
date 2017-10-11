#ifdef USE_TENSORRT
#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP

#include <caffe/blob.hpp>
#include <openpose/core/common.hpp>
#include <openpose/core/net.hpp>
#include <openpose/core/nmsCaffe.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractor.hpp>

namespace op
{
    class OP_API PoseExtractorTensorRT : public PoseExtractor
    {
    public:
        PoseExtractorTensorRT(const Point<int>& netInputSize, const Point<int>& netOutputSize, const Point<int>& outputSize, const int scaleNumber,
                           const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes = {},
                           const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        virtual ~PoseExtractorTensorRT();

        void netInitializationOnThread();

        void forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize, const std::vector<float>& scaleRatios = {1.f});

        const float* getHeatMapCpuConstPtr() const;

        const float* getHeatMapGpuConstPtr() const;

        const float* getPoseGpuConstPtr() const;

    private:
        const float mResizeScale;
        std::shared_ptr<Net> spNet;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeTensorRT;
        std::shared_ptr<NmsCaffe<float>> spNmsTensorRT;
        std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorTensorRT;
        // Init with thread
        boost::shared_ptr<caffe::Blob<float>> spTensorRTNetOutputBlob;
        std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
        std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
        std::shared_ptr<caffe::Blob<float>> spPoseBlob;

        DELETE_COPY(PoseExtractorTensorRT);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_TENSORRT_HPP
#endif // USE_TENSORRT
