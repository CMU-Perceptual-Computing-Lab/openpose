#ifdef USE_CAFFE
#ifndef OPENPOSE__POSE__POSE_EXTRACTOR_CAFFE_HPP
#define OPENPOSE__POSE__POSE_EXTRACTOR_CAFFE_HPP

#include <memory> // std::shared_ptr
#include <opencv2/core/core.hpp>
#include <caffe/blob.hpp>
#include "../core/array.hpp"
#include "../core/net.hpp"
#include "../core/nmsCaffe.hpp"
#include "../core/resizeAndMergeCaffe.hpp"
#include "../utilities/macros.hpp"
#include "bodyPartConnectorCaffe.hpp"
#include "enumClasses.hpp"
#include "poseExtractor.hpp"

namespace op
{
    class PoseExtractorCaffe : public PoseExtractor
    {
    public:
        PoseExtractorCaffe(const cv::Size& netInputSize, const cv::Size& netOutputSize, const cv::Size& outputSize, const int scaleNumber,
                           const float scaleGap, const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes = {},
                           const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne);

        void netInitializationOnThread();

        void forwardPass(const Array<float>& inputNetData, const cv::Size& inputDataSize);

        const float* getHeatMapCpuConstPtr() const;

        const float* getHeatMapGpuConstPtr() const;

        const float* getPoseGpuConstPtr() const;

    private:
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

#endif // OPENPOSE__POSE__POSE_EXTRACTOR_CAFFE_HPP
#endif
