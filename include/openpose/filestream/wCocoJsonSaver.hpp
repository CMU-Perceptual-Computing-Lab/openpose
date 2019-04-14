#ifndef OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/cocoJsonSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WCocoJsonSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WCocoJsonSaver(const std::shared_ptr<CocoJsonSaver>& cocoJsonSaver);

        virtual ~WCocoJsonSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<CocoJsonSaver> spCocoJsonSaver;

        DELETE_COPY(WCocoJsonSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WCocoJsonSaver<TDatums>::WCocoJsonSaver(const std::shared_ptr<CocoJsonSaver>& cocoJsonSaver) :
        spCocoJsonSaver{cocoJsonSaver}
    {
    }

    template<typename TDatums>
    WCocoJsonSaver<TDatums>::~WCocoJsonSaver()
    {
    }

    template<typename TDatums>
    void WCocoJsonSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WCocoJsonSaver<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Check tDatums->size() == 1
                if (tDatums->size() > 1)
                    error("Function only ready for tDatums->size() == 1", __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // T* to T
                const auto& tDatumPtr = tDatums->at(0);
                // Record json in COCO format
                spCocoJsonSaver->record(
                    tDatumPtr->poseKeypoints, tDatumPtr->poseScores, tDatumPtr->name, tDatumPtr->frameNumber);
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

    COMPILE_TEMPLATE_DATUM(WCocoJsonSaver);
}

#endif // OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP
