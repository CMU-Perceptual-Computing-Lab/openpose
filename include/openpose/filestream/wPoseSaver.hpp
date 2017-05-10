#ifndef OPENPOSE__FILESTREAM__W_POSE_SAVER_HPP
#define OPENPOSE__FILESTREAM__W_POSE_SAVER_HPP

#include <memory> // std::shared_ptr
#include <string>
#include "../thread/workerConsumer.hpp"
#include "enumClasses.hpp"
#include "poseSaver.hpp"

namespace op
{
    template<typename TDatums>
    class WPoseSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WPoseSaver(const std::shared_ptr<PoseSaver>& poseSaver);

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<PoseSaver> spPoseSaver;

        DELETE_COPY(WPoseSaver);
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
    WPoseSaver<TDatums>::WPoseSaver(const std::shared_ptr<PoseSaver>& poseSaver) :
        spPoseSaver{poseSaver}
    {
    }

    template<typename TDatums>
    void WPoseSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WPoseSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record people pose data
                std::vector<Array<float>> poseKeyPointsVector(tDatumsNoPtr.size());
                for (auto i = 0; i < tDatumsNoPtr.size(); i++)
                    poseKeyPointsVector[i] = tDatumsNoPtr[i].poseKeyPoints;
                const auto fileName = (!tDatumsNoPtr[0].name.empty() ? tDatumsNoPtr[0].name : std::to_string(tDatumsNoPtr[0].id));
                spPoseSaver->savePoseKeyPoints(poseKeyPointsVector, fileName);
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

    COMPILE_TEMPLATE_DATUM(WPoseSaver);
}

#endif // OPENPOSE__FILESTREAM__W_POSE_SAVER_HPP
