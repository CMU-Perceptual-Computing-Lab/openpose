#include <openpose/wrapper/wrapperStructOutput.hpp>

namespace op
{
    WrapperStructOutput::WrapperStructOutput(
        const double verbose_, const std::string& writeKeypoint_, const DataFormat writeKeypointFormat_,
        const std::string& writeJson_, const std::string& writeCocoJson_, const std::string& writeCocoFootJson_,
        const int writeCocoJsonVariant_, const std::string& writeImages_, const std::string& writeImagesFormat_,
        const std::string& writeVideo_, const double writeVideoFps_, const std::string& writeHeatMaps_,
        const std::string& writeHeatMapsFormat_, const std::string& writeVideo3D_,
        const std::string& writeVideoAdam_, const std::string& writeBvh_,
        const std::string& udpHost_, const std::string& udpPort_) :
        verbose{verbose_},
        writeKeypoint{writeKeypoint_},
        writeKeypointFormat{writeKeypointFormat_},
        writeJson{writeJson_},
        writeCocoJson{writeCocoJson_},
        writeCocoFootJson{writeCocoFootJson_},
        writeCocoJsonVariant{writeCocoJsonVariant_},
        writeImages{writeImages_},
        writeImagesFormat{writeImagesFormat_},
        writeVideo{writeVideo_},
        writeHeatMaps{writeHeatMaps_},
        writeHeatMapsFormat{writeHeatMapsFormat_},
        writeVideoFps{writeVideoFps_},
        writeVideo3D{writeVideo3D_},
        writeVideoAdam{writeVideoAdam_},
        writeBvh{writeBvh_},
        udpHost{udpHost_},
        udpPort{udpPort_}
    {
    }
}
