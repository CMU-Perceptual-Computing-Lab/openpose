#ifndef OPENPOSE_HAND_W_HAND_EXTRACTOR_NET_HPP
#define OPENPOSE_HAND_W_HAND_EXTRACTOR_NET_HPP

#include <openpose/core/common.hpp>
#include <openpose/hand/handRenderer.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WHandExtractorNet : public Worker<TDatums>
    {
    public:
        explicit WHandExtractorNet(const std::shared_ptr<HandExtractorNet>& handExtractorNet);

        virtual ~WHandExtractorNet();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<HandExtractorNet> spHandExtractorNet;

        DELETE_COPY(WHandExtractorNet);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandExtractorNet<TDatums>::WHandExtractorNet(const std::shared_ptr<HandExtractorNet>& handExtractorNet) :
        spHandExtractorNet{handExtractorNet}
    {
    }

    template<typename TDatums>
    WHandExtractorNet<TDatums>::~WHandExtractorNet()
    {
    }

    template<typename TDatums>
    void WHandExtractorNet<TDatums>::initializationOnThread()
    {
        spHandExtractorNet->initializationOnThread();
    }

    template<typename TDatums>
    void WHandExtractorNet<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Extract people hands
                for (auto& tDatumPtr : *tDatums)
                {
                    spHandExtractorNet->forwardPass(tDatumPtr->handRectangles, tDatumPtr->cvInputData);
                    for (auto hand = 0 ; hand < 2 ; hand++)
                    {
                        tDatumPtr->handHeatMaps[hand] = spHandExtractorNet->getHeatMaps()[hand].clone();
                        tDatumPtr->handKeypoints[hand] = spHandExtractorNet->getHandKeypoints()[hand].clone();
                    }
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
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WHandExtractorNet);
}

#endif // OPENPOSE_HAND_W_HAND_EXTRACTOR_NET_HPP
