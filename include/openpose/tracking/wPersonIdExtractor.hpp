#ifndef OPENPOSE_TRACKING_W_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_W_PERSON_ID_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/worker.hpp>
#include <openpose/tracking/personIdExtractor.hpp>

namespace op
{
    template<typename TDatums>
    class WPersonIdExtractor : public Worker<TDatums>
    {
    public:
        explicit WPersonIdExtractor(const std::shared_ptr<PersonIdExtractor>& personIdExtractor);

        virtual ~WPersonIdExtractor();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PersonIdExtractor> spPersonIdExtractor;

        DELETE_COPY(WPersonIdExtractor);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPersonIdExtractor<TDatums>::WPersonIdExtractor(const std::shared_ptr<PersonIdExtractor>& personIdExtractor) :
        spPersonIdExtractor{personIdExtractor}
    {
    }

    template<typename TDatums>
    WPersonIdExtractor<TDatums>::~WPersonIdExtractor()
    {
    }

    template<typename TDatums>
    void WPersonIdExtractor<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WPersonIdExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Render people pose
                for (auto& tDatum : *tDatums)
                    tDatum.poseIds = spPersonIdExtractor->extractIds(tDatum.poseKeypoints, tDatum.cvInputData);
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

    COMPILE_TEMPLATE_DATUM(WPersonIdExtractor);
}

#endif // OPENPOSE_TRACKING_W_PERSON_ID_EXTRACTOR_HPP
