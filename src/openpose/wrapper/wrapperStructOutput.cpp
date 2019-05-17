#include <openpose/wrapper/wrapperStructOutput.hpp>

namespace op
{
    WrapperStructOutput::WrapperStructOutput(
        const double verbose_, const std::string& writeKeypoint_, const DataFormat writeKeypointFormat_,
        const std::string& writeJson_, const std::string& writeCocoJson_, const int writeCocoJsonVariants_,
        const int writeCocoJsonVariant_, const std::string& writeImages_, const std::string& writeImagesFormat_,
        const std::string& writeVideo_, const double writeVideoFps_, const bool writeVideoWithAudio_,
        const std::string& writeHeatMaps_, const std::string& writeHeatMapsFormat_, const std::string& writeVideo3D_,
        const std::string& writeVideoAdam_, const std::string& writeBvh_, const std::string& udpHost_,
        const std::string& udpPort_) :
        verbose{verbose_},
        writeKeypoint{writeKeypoint_},
        writeKeypointFormat{writeKeypointFormat_},
        writeJson{writeJson_},
        writeCocoJson{writeCocoJson_},
        writeCocoJsonVariants{writeCocoJsonVariants_},
        writeCocoJsonVariant{writeCocoJsonVariant_},
        writeImages{writeImages_},
        writeImagesFormat{writeImagesFormat_},
        writeVideo{writeVideo_},
        writeVideoFps{writeVideoFps_},
        writeVideoWithAudio{writeVideoWithAudio_},
        writeHeatMaps{writeHeatMaps_},
        writeHeatMapsFormat{writeHeatMapsFormat_},
        writeVideo3D{writeVideo3D_},
        writeVideoAdam{writeVideoAdam_},
        writeBvh{writeBvh_},
        udpHost{udpHost_},
        udpPort{udpPort_}
    {
        try
        {
            if (!writeBvh.empty())
                error("BVH writing is experimental and not available yet (flag `--write_bvh`). Please, disable this"
                      " flag and do not open a GitHub issue asking for it.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
