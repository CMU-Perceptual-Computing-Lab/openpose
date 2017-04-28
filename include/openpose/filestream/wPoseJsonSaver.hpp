#ifndef OPENPOSE__FILESTREAM__W_POSE_JSON_SAVER_HPP
#define OPENPOSE__FILESTREAM__W_POSE_JSON_SAVER_HPP

#include <memory> // std::shared_ptr
#include <string>
#include "../thread/workerConsumer.hpp"
#include "poseJsonSaver.hpp"

namespace op
{
    template<typename TDatums>
    class WPoseJsonSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WPoseJsonSaver(const std::shared_ptr<PoseJsonSaver>& poseJsonSaver);

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<PoseJsonSaver> spPoseJsonSaver;

        DELETE_COPY(WPoseJsonSaver);
    };
}





// Implementation
#include "../utilities/errorAndLog.hpp"
#include "../utilities/macros.hpp"
#include "../utilities/pointerContainer.hpp"
#include "../utilities/profiler.hpp"
namespace op
{
    template<typename TDatums>
    WPoseJsonSaver<TDatums>::WPoseJsonSaver(const std::shared_ptr<PoseJsonSaver>& poseJsonSaver) :
        spPoseJsonSaver{poseJsonSaver}
    {
    }

    template<typename TDatums>
    void WPoseJsonSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WPoseJsonSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record people pose data in json format
                std::vector<Array<float>> poseKeyPointsVector(tDatumsNoPtr.size());
                for (auto i = 0; i < tDatumsNoPtr.size(); i++)
                    poseKeyPointsVector[i] = tDatumsNoPtr[i].poseKeyPoints;
                const auto fileName = (!tDatumsNoPtr[0].name.empty() ? tDatumsNoPtr[0].name : std::to_string(tDatumsNoPtr[0].id));
                spPoseJsonSaver->savePoseKeyPoints(poseKeyPointsVector, fileName);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 1000);
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

    COMPILE_TEMPLATE_DATUM(WPoseJsonSaver);
}

#endif // OPENPOSE__FILESTREAM__W_POSE_JSON_SAVER_HPP
