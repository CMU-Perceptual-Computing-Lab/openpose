#ifndef OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
#define OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP

#include <memory> // std::shared_ptr
#include <string>
#include <vector>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/point.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/producer/producer.hpp>
#include <openpose/core/macros.hpp>

namespace op
{
    OP_API PoseModel flagsToPoseModel(const std::string& poseModeString);

    OP_API ScaleMode flagsToScaleMode(const int keypointScale);

    // Determine type of frame source
    OP_API ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex);

    OP_API std::shared_ptr<Producer> flagsToProducer(const std::string& imageDirectory, const std::string& videoPath,
                                                     const int webcamIndex, const std::string& webcamResolution = "1280x720",
                                                     const double webcamFps = 30.);

    OP_API std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts = false, const bool heatMapsAddBkg = false,
                                                    const bool heatMapsAddPAFs = false);

    OP_API RenderMode flagsToRenderMode(const int renderFlag, const int renderPoseFlag = -2);

    OP_API Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample = "1280x720");
}

#endif // OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
