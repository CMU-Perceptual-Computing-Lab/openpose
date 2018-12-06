#ifndef OPENPOSE_CORE_W_KEEP_TOP_N_PEOPLE_HPP
#define OPENPOSE_CORE_W_KEEP_TOP_N_PEOPLE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/keepTopNPeople.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WKeepTopNPeople : public Worker<TDatums>
    {
    public:
        explicit WKeepTopNPeople(const std::shared_ptr<KeepTopNPeople>& keepTopNPeople);

        virtual ~WKeepTopNPeople();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<KeepTopNPeople> spKeepTopNPeople;
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WKeepTopNPeople<TDatums>::WKeepTopNPeople(const std::shared_ptr<KeepTopNPeople>& keepTopNPeople) :
        spKeepTopNPeople{keepTopNPeople}
    {
    }

    template<typename TDatums>
    WKeepTopNPeople<TDatums>::~WKeepTopNPeople()
    {
    }

    template<typename TDatums>
    void WKeepTopNPeople<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WKeepTopNPeople<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Rescale pose data
                for (auto& tDatum : *tDatums)
                {
                    tDatum.poseKeypoints = spKeepTopNPeople->keepTopPeople(tDatum.poseKeypoints, tDatum.poseScores);
                    tDatum.faceKeypoints = spKeepTopNPeople->keepTopPeople(tDatum.faceKeypoints, tDatum.poseScores);
                    tDatum.handKeypoints[0] = spKeepTopNPeople->keepTopPeople(tDatum.handKeypoints[0],
                                                                              tDatum.poseScores);
                    tDatum.handKeypoints[1] = spKeepTopNPeople->keepTopPeople(tDatum.handKeypoints[1],
                                                                              tDatum.poseScores);
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WKeepTopNPeople);
}

#endif // OPENPOSE_CORE_W_KEEP_TOP_N_PEOPLE_HPP
