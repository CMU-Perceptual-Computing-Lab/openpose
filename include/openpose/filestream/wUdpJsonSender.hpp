#ifndef OPENPOSE_FILESTREAM_W_UDP_JSON_SENDER_HPP
#define OPENPOSE_FILESTREAM_W_UDP_JSON_SENDER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/udpJsonSender.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WUdpJsonSender : public WorkerConsumer<TDatums>
    {
    public:
        explicit WUdpJsonSender(const std::shared_ptr<UdpJsonSender>& udpJsonSender);

        virtual ~WUdpJsonSender();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<UdpJsonSender> spUdpSender;

        DELETE_COPY(WUdpJsonSender);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WUdpJsonSender<TDatums>::WUdpJsonSender(const std::shared_ptr<UdpJsonSender>& udpSender) :
        spUdpSender{udpSender}
    {
    }

    template<typename TDatums>
    WUdpJsonSender<TDatums>::~WUdpJsonSender()
    {
    }

    template<typename TDatums>
    void WUdpJsonSender<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WUdpJsonSender<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);

                for (auto i = 0u; i < tDatums->size(); i++)
                {
                    const auto& tDatumPtr = (*tDatums)[i];

                    // Pose IDs from long long to float
                    Array<float> poseIds{ tDatumPtr->poseIds };

                    const std::vector<std::pair<Array<float>, std::string>> keypointVector{
                        // Pose IDs
                        std::make_pair(poseIds, "person_id"),
                        // 2D
                        std::make_pair(tDatumPtr->poseKeypoints, "pose_keypoints_2d"),
                        std::make_pair(tDatumPtr->faceKeypoints, "face_keypoints_2d"),
                        std::make_pair(tDatumPtr->handKeypoints[0], "hand_left_keypoints_2d"),
                        std::make_pair(tDatumPtr->handKeypoints[1], "hand_right_keypoints_2d"),
                        // 3D
                        std::make_pair(tDatumPtr->poseKeypoints3D, "pose_keypoints_3d"),
                        std::make_pair(tDatumPtr->faceKeypoints3D, "face_keypoints_3d"),
                        std::make_pair(tDatumPtr->handKeypoints3D[0], "hand_left_keypoints_3d"),
                        std::make_pair(tDatumPtr->handKeypoints3D[1], "hand_right_keypoints_3d")
                    };

                    spUdpSender->sendJson(keypointVector);
                }
        
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WUdpJsonSender);
}

#endif // OPENPOSE_FILESTREAM_W_UDP_SENDER_HPP
