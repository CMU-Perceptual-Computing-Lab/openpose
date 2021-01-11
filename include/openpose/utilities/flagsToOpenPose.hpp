#ifndef OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
#define OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/producer/enumClasses.hpp>
#include <openpose/wrapper/enumClasses.hpp>

namespace op
{
    OP_API PoseMode flagsToPoseMode(const int poseModeInt);

    OP_API PoseModel flagsToPoseModel(const String& poseModeString);

    OP_API ScaleMode flagsToScaleMode(const int keypointScaleMode);

    OP_API ScaleMode flagsToHeatMapScaleMode(const int heatMapScaleMode);

    OP_API Detector flagsToDetector(const int detector);

    // Determine type of frame source
    OP_API ProducerType flagsToProducerType(
        const String& imageDirectory, const String& videoPath, const String& ipCameraPath,
        const int webcamIndex, const bool flirCamera);

    OP_API std::pair<ProducerType, String> flagsToProducer(
        const String& imageDirectory, const String& videoPath, const String& ipCameraPath = String(""),
        const int webcamIndex = -1, const bool flirCamera = false, const int flirCameraIndex = -1);

    OP_API std::vector<HeatMapType> flagsToHeatMaps(
        const bool heatMapsAddParts = false, const bool heatMapsAddBkg = false,
        const bool heatMapsAddPAFs = false);

    OP_API RenderMode flagsToRenderMode(
        const int renderFlag, const bool gpuBuggy = false, const int renderPoseFlag = -2);

    OP_API DisplayMode flagsToDisplayMode(const int display, const bool enabled3d);

    /**
     * E.g., const Point<int> netInputSize = flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
     * E.g., const Point<int> resolution = flagsToPoint(resolutionString, "1280x720");
     */
    OP_API Point<int> flagsToPoint(const String& pointString, const String& pointExample);
}

#endif // OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
