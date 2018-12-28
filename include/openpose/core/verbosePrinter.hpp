#ifndef OPENPOSE_CORE_VERBOSE_PRINTER_HPP
#define OPENPOSE_CORE_VERBOSE_PRINTER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API VerbosePrinter
    {
    public:
        VerbosePrinter(const double verbose, const unsigned long long numberFrames);

        virtual ~VerbosePrinter();

        void printVerbose(const unsigned long long frameNumber) const;

    private:
        const unsigned long long mNumberFrames;
        const std::string mNumberFramesString;
        const double mVerbose;
    };
}

#endif // OPENPOSE_CORE_VERBOSE_PRINTER_HPP
