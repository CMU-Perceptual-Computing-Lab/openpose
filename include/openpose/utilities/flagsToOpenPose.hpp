#ifndef OPENPOSE_UTILITIES_G_FLAGS_TO_OPEN_POSE_HPP
#define OPENPOSE_UTILITIES_G_FLAGS_TO_OPEN_POSE_HPP

#include <memory> // std::shared_ptr
#include <string>
#include <vector>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    PoseModel flagsToPoseModel(const std::string& poseModeString);

    ScaleMode flagsToScaleMode(const int keypointScale);

    // Determine type of frame source
    ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex);

    std::shared_ptr<Producer> flagsToProducer(const std::string& imageDirectory, const std::string& videoPath,
                                              const int webcamIndex, const std::string& webcamResolution, const double webcamFps);

    std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg, const bool heatMapsAddPAFs);

    DetectionMode flagsToDetectionMode(const int handDetectionModeFlag, const std::shared_ptr<Producer>& producer = nullptr);

    RenderMode flagsToRenderMode(const int renderFlag, const int renderPoseFlag = -2);

    Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample = "1280x720");
}

#endif // OPENPOSE_UTILITIES_G_FLAGS_TO_OPEN_POSE_HPP
