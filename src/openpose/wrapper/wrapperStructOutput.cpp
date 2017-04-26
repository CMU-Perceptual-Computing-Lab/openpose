#include "openpose/wrapper/wrapperStructOutput.hpp"

namespace op
{
    WrapperStructOutput::WrapperStructOutput(const bool displayGui_, const bool guiVerbose_, const bool fullScreen_, const std::string& writePose_,
                                             const DataFormat writePoseDataFormat_, const std::string& writePoseJson_, const std::string& writeCocoJson_,
                                             const std::string& writeImages_, const std::string& writeImagesFormat_, const std::string& writeVideo_,
                                             const std::string& writeHeatMaps_, const std::string& writeHeatMapsFormat_) :
        displayGui{displayGui_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_},
        writePose{writePose_},
        writePoseDataFormat{writePoseDataFormat_},
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
