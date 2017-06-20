#ifndef OPENPOSE_FILESTREAM_W_HAND_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_HAND_JSON_SAVER_HPP

#include <memory> // std::shared_ptr
#include <string>
#include <openpose/thread/workerConsumer.hpp>
#include "keypointJsonSaver.hpp"

namespace op
{
    template<typename TDatums>
    class WHandJsonSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WHandJsonSaver(const std::shared_ptr<KeypointJsonSaver>& keypointJsonSaver);

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<KeypointJsonSaver> spKeypointJsonSaver;

        DELETE_COPY(WHandJsonSaver);
    };
}





// Implementation
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/macros.hpp>
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/utilities/profiler.hpp>
namespace op
{
    template<typename TDatums>
    WHandJsonSaver<TDatums>::WHandJsonSaver(const std::shared_ptr<KeypointJsonSaver>& keypointJsonSaver) :
        spKeypointJsonSaver{keypointJsonSaver}
    {
    }

    template<typename TDatums>
    void WHandJsonSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandJsonSaver<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // T* to T
                auto& tDatumsNoPtr = *tDatums;
                // Record people hand data in json format
                const auto fileName = (!tDatumsNoPtr[0].name.empty() ? tDatumsNoPtr[0].name : std::to_string(tDatumsNoPtr[0].id));
                std::vector<Array<float>> keypointVector(tDatumsNoPtr.size());
                // Left hand
                for (auto i = 0; i < tDatumsNoPtr.size(); i++)
                    keypointVector[i] = tDatumsNoPtr[i].handKeypoints[0];
                spKeypointJsonSaver->save(keypointVector, fileName, "hand_left_keypoints");
                // Right hand
                for (auto i = 0; i < tDatumsNoPtr.size(); i++)
                    keypointVector[i] = tDatumsNoPtr[i].handKeypoints[1];
                spKeypointJsonSaver->save(keypointVector, fileName, "hand_right_keypoints");
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, Profiler::DEFAULT_X);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WHandJsonSaver);
}

#endif // OPENPOSE_FILESTREAM_W_HAND_JSON_SAVER_HPP
