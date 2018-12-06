#ifndef OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
#define OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/producer/enumClasses.hpp>

namespace op
{
    OP_API PoseModel flagsToPoseModel(const std::string& poseModeString);

    OP_API ScaleMode flagsToScaleMode(const int keypointScale);

    OP_API ScaleMode flagsToHeatMapScaleMode(const int heatMapScale);

    // Determine type of frame source
    OP_API ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath,
                                            const std::string& ipCameraPath, const int webcamIndex,
                                            const bool flirCamera);

    OP_API std::pair<ProducerType, std::string> flagsToProducer(
        const std::string& imageDirectory, const std::string& videoPath, const std::string& ipCameraPath = "",
        const int webcamIndex = -1, const bool flirCamera = false, const int flirCameraIndex = -1);

    OP_API std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts = false,
                                                    const bool heatMapsAddBkg = false,
                                                    const bool heatMapsAddPAFs = false);

    OP_API RenderMode flagsToRenderMode(const int renderFlag, const bool gpuBuggy = false,
                                        const int renderPoseFlag = -2);

    OP_API DisplayMode flagsToDisplayMode(const int display, const bool enabled3d);

    OP_API Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample = "1280x720");
}

#endif // OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
