#ifndef OPENPOSE_FILESTREAM_W_HAND_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_HAND_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/filestream/keypointSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WHandSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WHandSaver(const std::shared_ptr<KeypointSaver>& keypointSaver);

        virtual ~WHandSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<KeypointSaver> spKeypointSaver;

        DELETE_COPY(WHandSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WHandSaver<TDatums>::WHandSaver(const std::shared_ptr<KeypointSaver>& keypointSaver) :
        spKeypointSaver{keypointSaver}
    {
    }

    template<typename TDatums>
    WHandSaver<TDatums>::~WHandSaver()
    {
    }

    template<typename TDatums>
    void WHandSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WHandSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record people hand keypoint data
                const auto fileName = (!tDatumsNoPtr[0].name.empty() ? tDatumsNoPtr[0].name : std::to_string(tDatumsNoPtr[0].id));
                std::vector<Array<float>> keypointVector(tDatumsNoPtr.size());
                // Left hand
                for (auto i = 0u; i < tDatumsNoPtr.size(); i++)
                    keypointVector[i] = tDatumsNoPtr[i].handKeypoints[0];
                spKeypointSaver->saveKeypoints(keypointVector, fileName, "hand_left");
                // Right hand
                for (auto i = 0u; i < tDatumsNoPtr.size(); i++)
                    keypointVector[i] = tDatumsNoPtr[i].handKeypoints[1];
                spKeypointSaver->saveKeypoints(keypointVector, fileName, "hand_right");
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
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WHandSaver);
}

#endif // OPENPOSE_FILESTREAM_W_HAND_SAVER_HPP
