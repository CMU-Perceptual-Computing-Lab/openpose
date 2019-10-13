#include <openpose/wrapper/wrapperStructOutput.hpp>

namespace op
{
    WrapperStructOutput::WrapperStructOutput(
        const double verbose_, const String& writeKeypoint_, const DataFormat writeKeypointFormat_,
        const String& writeJson_, const String& writeCocoJson_, const int writeCocoJsonVariants_,
        const int writeCocoJsonVariant_, const String& writeImages_, const String& writeImagesFormat_,
        const String& writeVideo_, const double writeVideoFps_, const bool writeVideoWithAudio_,
        const String& writeHeatMaps_, const String& writeHeatMapsFormat_, const String& writeVideo3D_,
        const String& writeVideoAdam_, const String& writeBvh_, const String& udpHost_,
        const String& udpPort_) :
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
