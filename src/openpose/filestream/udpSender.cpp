#ifdef USE_ASIO
    #include <iostream>
    #include <asio.hpp>
#endif
#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/udpSender.hpp>

namespace op
{
    #ifdef USE_ASIO
        class UdpClient
        {
        public:
            UdpClient(const std::string& host, const std::string& port) :
                mIoService{},
                mUdpSocket{io_service, asio::ip::udp::endpoint(asio::ip::udp::v4(), 0)}
            {
                try
                {
                    asio::ip::udp::resolver resolver{mIoService};
                    asio::ip::udp::resolver::query query{asio::ip::udp::v4(), host, port};
                    asio::ip::udp::resolver::iterator iter = resolver.resolve(query);
                    mUdpEndpoint = *iter;
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            ~UdpClient()
            {
                try
                {
                    mUdpSocket.close();
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void send(const std::string& msg)
            {
                try
                {
                    mUdpSocket.send_to(asio::buffer(msg, msg.size()), mUdpEndpoint);
                    // std::cout << "sent data: " << msg << std::endl;
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

        private:
            asio::io_service& mIoService;
            asio::ip::udp::socket mUdpSocket;
            asio::ip::udp::endpoint mUdpEndpoint;
        };
    #endif

    struct UdpSender::ImplUdpSender
    {
        #ifdef USE_ASIO
            // Used when increasing spCaffeNets
            const UdpClient mUdpClient;

            ImplUdpSender(const std::string& udpHost, const std::string& udpPort) :
                mUdpClient(udpHost, udpPort)
            {
            }
        #endif
    };

    UdpSender::UdpSender(const std::string& udpHost, const std::string& udpPort)
        #ifdef USE_ASIO
            : spImpl{new ImplUdpSender{udpHost, udpPort}}
        #endif
    {
        try
        {
            // error("UdpSender (`--udp_host` and `--udp_port` flags) buggy and not working yet, but we are"
            //       "working on it! Coming soon!", __LINE__, __FUNCTION__, __FILE__);
            #ifndef USE_ASIO
                UNUSED(udpHost);
                UNUSED(udpPort);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void UdpSender::sendJointAngles(const double* const adamPosePtr, const int adamPoseRows,
                                    const double* const adamTranslationPtr,
                                    const double* const adamFaceCoeffsExpPtr, const int faceCoeffRows)
    {
        #ifdef USE_ASIO
            try
            {
                const Eigen::Map<Vector3d> adamTranslation(adamTranslationPtr);
                const Eigen::Map<Eigen::Matrix<double, adamPoseRows, 3, Eigen::RowMajor>> adamPose(adamPosePtr, adamPoseRows);
                const Eigen::Map<VectorXd> adamFaceCoeffsExpPtr(adamFaceCoeffsExpPtr, faceCoeffRows);

                const std::string prefix = "AnimData:";
                const std::string totalPositionString = "\"totalPosition\":"
                    + vectorToJson(adamTranslation(0), adamTranslation(1), adamTranslation(2));
                std::string jointAnglesString = "\"jointAngles\":[";
                for (int i = 0; i < adamPoseRows; i++)
                {
                    jointAnglesString += vectorToJson(adamPose(i, 0), adamPose(i, 1), adamPose(i, 2));
                    if (i != adamPoseRows - 1)
                    {
                        jointAnglesString += ",";
                    }
                }
                jointAnglesString += "]";

                std::string facialParamsString = "\"facialParams\":[";
                for (int i = 0; i < faceCoeffRows; i++)
                {
                    facialParamsString += std::to_string(adamFaceCoeffsExp(i));
                    if (i != faceCoeffRows - 1)
                    {
                        facialParamsString += ",";
                    }
                }
                facialParamsString += "]";

                // facialParamsString + std::to_string(mouth_open) + "," + std::to_string(leye_open) + "," + std::to_string(reye_open) + "]";

                // std::string rootHeightString = "\"rootHeight\":" + std::to_string(dist_root_foot);

                const std::string data = prefix + "{" + facialParamsString
                                       + "," + totalPositionString
                                       + "," + jointAnglesString + "}";

                spImpl->mUdpClient.send(data);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            UNUSED(adamPosePtr);
            UNUSED(adamPoseRows);
            UNUSED(adamTranslationPtr);
            UNUSED(adamFaceCoeffsExpPtr);
            UNUSED(faceCoeffRows);
        #endif
    }
}
