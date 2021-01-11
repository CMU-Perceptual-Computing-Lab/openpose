#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    WrapperStructPose::WrapperStructPose(
        const PoseMode poseMode_, const Point<int>& netInputSize_, const double netInputSizeDynamicBehavior_,
        const Point<int>& outputSize_, const ScaleMode keypointScaleMode_, const int gpuNumber_,
        const int gpuNumberStart_, const int scalesNumber_, const float scaleGap_, const RenderMode renderMode_,
        const PoseModel poseModel_, const bool blendOriginalFrame_, const float alphaKeypoint_,
        const float alphaHeatMap_, const int defaultPartToRender_, const String& modelFolder_,
        const std::vector<HeatMapType>& heatMapTypes_, const ScaleMode heatMapScaleMode_,
        const bool addPartCandidates_, const float renderThreshold_, const int numberPeopleMax_,
        const bool maximizePositives_, const double fpsMax_, const String& protoTxtPath_,
        const String& caffeModelPath_, const float upsamplingRatio_, const bool enableGoogleLogging_) :
        poseMode{poseMode_},
        netInputSize{netInputSize_},
        netInputSizeDynamicBehavior{netInputSizeDynamicBehavior_},
        outputSize{outputSize_},
        keypointScaleMode{keypointScaleMode_},
        gpuNumber{gpuNumber_},
        gpuNumberStart{gpuNumberStart_},
        scalesNumber{scalesNumber_},
        scaleGap{scaleGap_},
        renderMode{renderMode_},
        poseModel{poseModel_},
        blendOriginalFrame{blendOriginalFrame_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_},
        defaultPartToRender{defaultPartToRender_},
        modelFolder{modelFolder_},
        heatMapTypes{heatMapTypes_},
        heatMapScaleMode{heatMapScaleMode_},
        addPartCandidates{addPartCandidates_},
        renderThreshold{renderThreshold_},
        numberPeopleMax{numberPeopleMax_},
        maximizePositives{maximizePositives_},
        fpsMax{fpsMax_},
        protoTxtPath{protoTxtPath_},
        caffeModelPath{caffeModelPath_},
        upsamplingRatio{upsamplingRatio_},
        enableGoogleLogging{enableGoogleLogging_}
    {
    }
}
