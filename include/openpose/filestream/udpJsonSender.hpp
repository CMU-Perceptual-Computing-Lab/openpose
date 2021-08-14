#ifndef OPENPOSE_FILESTREAM_UDP_JSON_SENDER_HPP
#define OPENPOSE_FILESTREAM_UDP_JSON_SENDER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API UdpJsonSender
    {
    public:
        UdpJsonSender(const std::string& udpHost, const std::string& udpPort);

        virtual ~UdpJsonSender();

        void sendJson(const std::vector<std::pair<Array<float>, std::string>>& keypointVector);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplJsonUdpSender;
        std::shared_ptr<ImplJsonUdpSender> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(UdpJsonSender);
    };
}

#endif // OPENPOSE_FILESTREAM_UDP_JSON_SENDER_HPP
