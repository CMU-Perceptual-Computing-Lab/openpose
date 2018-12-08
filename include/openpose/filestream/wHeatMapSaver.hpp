#ifndef OPENPOSE_FILESTREAM_W_HEAT_MAP_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_HEAT_MAP_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/heatMapSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WHeatMapSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WHeatMapSaver(const std::shared_ptr<HeatMapSaver>& heatMapSaver);

        virtual ~WHeatMapSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<HeatMapSaver> spHeatMapSaver;

        DELETE_COPY(WHeatMapSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHeatMapSaver<TDatums>::WHeatMapSaver(const std::shared_ptr<HeatMapSaver>& heatMapSaver) :
        spHeatMapSaver{heatMapSaver}
    {
    }

    template<typename TDatums>
    WHeatMapSaver<TDatums>::~WHeatMapSaver()
    {
    }

    template<typename TDatums>
    void WHeatMapSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHeatMapSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record pose heatmap image(s) on disk
                std::vector<Array<float>> poseHeatMaps(tDatumsNoPtr.size());
                for (auto i = 0u; i < tDatumsNoPtr.size(); i++)
                    poseHeatMaps[i] = tDatumsNoPtr[i].poseHeatMaps;
                const auto fileName = (!tDatumsNoPtr[0].name.empty()
                                       ? tDatumsNoPtr[0].name : std::to_string(tDatumsNoPtr[0].id)) + "_pose_heatmaps";
                spHeatMapSaver->saveHeatMaps(poseHeatMaps, fileName);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey,
                                                          __LINE__, __FUNCTION__, __FILE__);
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

    COMPILE_TEMPLATE_DATUM(WHeatMapSaver);
}

#endif // OPENPOSE_FILESTREAM_W_HEAT_MAP_SAVER_HPP
