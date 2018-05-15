#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    WrapperStructPose::WrapperStructPose(const bool enable_, const Point<int>& netInputSize_,
                                         const Point<int>& outputSize_, const ScaleMode keypointScale_,
                                         const int gpuNumber_, const int gpuNumberStart_,
                                         const int scalesNumber_, const float scaleGap_,
                                         const RenderMode renderMode_, const PoseModel poseModel_,
                                         const bool blendOriginalFrame_, const float alphaKeypoint_,
                                         const float alphaHeatMap_, const int defaultPartToRender_,
                                         const std::string& modelFolder_,
                                         const std::vector<HeatMapType>& heatMapTypes_,
                                         const ScaleMode heatMapScale_, const bool addPartCandidates_,
                                         const float renderThreshold_, const int numberPeopleMax_,
                                         const bool enableGoogleLogging_, const bool reconstruct3d_,
                                         const int minViews3d_, const bool identification_,
                                         const int tracking_) :
        enable{enable_},
        netInputSize{netInputSize_},
        outputSize{outputSize_},
        keypointScale{keypointScale_},
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
        heatMapScale{heatMapScale_},
        addPartCandidates{addPartCandidates_},
        renderThreshold{renderThreshold_},
        numberPeopleMax{numberPeopleMax_},
        enableGoogleLogging{enableGoogleLogging_},
        reconstruct3d{reconstruct3d_},
        minViews3d{minViews3d_},
        identification{identification_},
        tracking{tracking_}
    {
    }
}
