#include <openpose/wrapper/wrapperStructOutput.hpp>

namespace op
{
    WrapperStructOutput::WrapperStructOutput(const DisplayMode displayMode_, const bool guiVerbose_,
                                             const bool fullScreen_, const std::string& writeKeypoint_,
                                             const DataFormat writeKeypointFormat_, const std::string& writeJson_,
                                             const std::string& writeCocoJson_, const std::string& writeImages_,
                                             const std::string& writeImagesFormat_, const std::string& writeVideo_,
                                             const double writeVideoFps_, const std::string& writeHeatMaps_,
                                             const std::string& writeHeatMapsFormat_) :
        displayMode{displayMode_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_},
        writeKeypoint{writeKeypoint_},
        writeKeypointFormat{writeKeypointFormat_},
        writeJson{writeJson_},
        writeCocoJson{writeCocoJson_},
        writeImages{writeImages_},
        writeImagesFormat{writeImagesFormat_},
        writeVideo{writeVideo_},
        writeHeatMaps{writeHeatMaps_},
        writeHeatMapsFormat{writeHeatMapsFormat_},
        writeVideoFps{writeVideoFps_}
    {
    }
}
