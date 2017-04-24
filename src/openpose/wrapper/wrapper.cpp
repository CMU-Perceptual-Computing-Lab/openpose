#include "openpose/wrapper/wrapper.hpp"

namespace op
{
    WrapperPoseStruct::WrapperPoseStruct(const cv::Size& netInputSize_, const cv::Size& outputSize_, const ScaleMode scaleMode_, const int gpuNumber_,
                                         const int gpuNumberStart_, const int scalesNumber_, const float scaleGap_, const bool renderOutput_, const PoseModel poseModel_,
                                         const bool blendOriginalFrame_, const float alphaPose_, const float alphaHeatMap_, const int defaultPartToRender_,
                                         const std::string& modelFolder_, const std::vector<HeatMapType>& heatMapTypes_, const ScaleMode heatMapScaleMode_) :
        netInputSize{netInputSize_},
        outputSize{outputSize_},
        scaleMode{scaleMode_},
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

    namespace experimental
    {
        WrapperHandsStruct::WrapperHandsStruct(const bool extractAndRenderHands_) :
            extractAndRenderHands{extractAndRenderHands_}
        {
        }
    }

    WrapperInputStruct::WrapperInputStruct(const std::shared_ptr<Producer> producerSharedPtr_, const unsigned long long frameFirst_, const unsigned long long frameLast_,
                                           const bool realTimeProcessing_, const bool frameFlip_, const int frameRotate_, const bool framesRepeat_) :
        producerSharedPtr{producerSharedPtr_},
        frameFirst{frameFirst_},
        frameLast{frameLast_},
        realTimeProcessing{realTimeProcessing_},
        frameFlip{frameFlip_},
        frameRotate{frameRotate_},
        framesRepeat{framesRepeat_}
    {
    }

    WrapperOutputStruct::WrapperOutputStruct(const bool displayGui_, const bool guiVerbose_, const bool fullScreen_, const std::string& writePose_,
                                             const DataFormat dataFormat_, const std::string& writePoseJson_, const std::string& writeCocoJson_,
                                             const std::string& writeImages_, const std::string& writeImagesFormat_, const std::string& writeVideo_,
                                             const std::string& writeHeatMaps_, const std::string& writeHeatMapsFormat_) :
        displayGui{displayGui_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_},
        writePose{writePose_},
        dataFormat{dataFormat_},
        writePoseJson{writePoseJson_},
        writeCocoJson{writeCocoJson_},
        writeImages{writeImages_},
        writeImagesFormat{writeImagesFormat_},
        writeVideo{writeVideo_},
        writeHeatMaps{writeHeatMaps_},
        writeHeatMapsFormat{writeHeatMapsFormat_}
    {
    }
}
