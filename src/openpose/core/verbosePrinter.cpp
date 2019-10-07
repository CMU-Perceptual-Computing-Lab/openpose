#include <openpose/core/verbosePrinter.hpp>
#include <openpose/utilities/fastMath.hpp>

namespace op
{
    VerbosePrinter::VerbosePrinter(const double verbose, const unsigned long long numberFrames) :
    mNumberFrames{numberFrames},
    mNumberFramesString{"/" + std::to_string(numberFrames) + "..."},
    mVerbose{verbose}
    {
        try
        {
            if (mVerbose > 0. && mVerbose < 1. && mNumberFrames <= 0.)
                error("Number of total frames could not be retrieved from the frames producer. Disable"
                      " `--verbose` or use a frames producer with known number of frames.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    VerbosePrinter::~VerbosePrinter()
    {
    }

    void VerbosePrinter::printVerbose(const unsigned long long frameNumber) const
    {
        try
        {
            // If verbose enabled
            if (mVerbose > 0.)
            {
                bool plotResults = false;
                // If first or last frame
                if (frameNumber == 0 || frameNumber >= mNumberFrames-1)
                    plotResults = true;
                // mVerbose = (0,1) --> Percentage --> Every mVerbose*numberFrames frames
                else if (mVerbose < 1.)
                    plotResults = ((frameNumber+1) % uLongLongRound(mVerbose*mNumberFrames) == 0);
                // mVerbose = integer >= 1 --> Every mVerbose frames
                else
                    plotResults = ((frameNumber+1) % uLongLongRound(mVerbose) == 0);
                // Plot results
                if (plotResults)
                    opLog("Processing frame " + std::to_string(frameNumber+1) + mNumberFramesString);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
