#include "openpose/wrapper/wrapperStructPose.hpp"

namespace op
{
    WrapperStructPose::WrapperStructPose(const cv::Size& netInputSize_, const cv::Size& outputSize_, const ScaleMode poseScaleMode_, const int gpuNumber_,
                                         const int gpuNumberStart_, const int scalesNumber_, const float scaleGap_, const bool renderOutput_,
                                         const PoseModel poseModel_, const bool blendOriginalFrame_, const float alphaPose_, const float alphaHeatMap_,
                                         const int defaultPartToRender_, const std::string& modelFolder_, const std::vector<HeatMapType>& heatMapTypes_,
                                         const ScaleMode heatMapScaleMode_) :
        netInputSize{netInputSize_},
        outputSize{outputSize_},
        poseScaleMode{poseScaleMode_},
        gpuNumber{gpuNumber_},
        gpuNumberStart{gpuNumberStart_},
        scalesNumber{scalesNumber_},
        scaleGap{scaleGap_},
        renderOutput{renderOutput_},
        poseModel{poseModel_},
        blendOriginalFrame{blendOriginalFrame_},
        alphaPose{alphaPose_},
        alphaHeatMap{alphaHeatMap_},
        defaultPartToRender{defaultPartToRender_},
        modelFolder{modelFolder_},
        heatMapTypes{heatMapTypes_},
        heatMapScaleMode{heatMapScaleMode_}
    {
    }
}
