#ifdef USE_CAFFE
#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP

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
    class OP_API PoseExtractorCaffe : public PoseExtractor
    {
    public:
        PoseExtractorCaffe(const Point<int>& netInputSize, const Point<int>& netOutputSize, const Point<int>& outputSize, const int scaleNumber,
                           const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes = {},
                           const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        virtual ~PoseExtractorCaffe();

        void netInitializationOnThread();

        void forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize, const std::vector<float>& scaleRatios = {1.f});

        const float* getHeatMapCpuConstPtr() const;

        const float* getHeatMapGpuConstPtr() const;

        const float* getPoseGpuConstPtr() const;

    private:
        const float mResizeScale;
        std::shared_ptr<Net> spNet;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
        std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
        // Init with thread
        boost::shared_ptr<caffe::Blob<float>> spCaffeNetOutputBlob;
        std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
        std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
        std::shared_ptr<caffe::Blob<float>> spPoseBlob;

        DELETE_COPY(PoseExtractorCaffe);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_HPP
#endif
