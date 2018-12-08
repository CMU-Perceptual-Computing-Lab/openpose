#ifndef OPENPOSE_FILESTREAM_UDP_SENDER_HPP
#define OPENPOSE_FILESTREAM_UDP_SENDER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API UdpSender
    {
    public:
        UdpSender(const std::string& udpHost, const std::string& udpPort);

        virtual ~UdpSender();

        void sendJointAngles(const double* const adamPosePtr, const int adamPoseRows,
                             const double* const adamTranslationPtr,
                             const double* const adamFaceCoeffsExpPtr, const int faceCoeffRows);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplUdpSender;
        std::shared_ptr<ImplUdpSender> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(UdpSender);
    };
}

#endif // OPENPOSE_FILESTREAM_UDP_SENDER_HPP
