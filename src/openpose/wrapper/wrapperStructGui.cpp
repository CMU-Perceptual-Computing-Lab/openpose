#include <openpose/wrapper/wrapperStructGui.hpp>

namespace op
{
    WrapperStructGui::WrapperStructGui(
        const DisplayMode displayMode_, const bool guiVerbose_, const bool fullScreen_) :
        displayMode{displayMode_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_}
    {
    }
}
