#ifndef OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP

#include <memory> // std::shared_ptr
#include <openpose/filestream/cocoJsonSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WCocoJsonSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WCocoJsonSaver(const std::shared_ptr<CocoJsonSaver>& cocoJsonSaver);

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<CocoJsonSaver> spCocoJsonSaver;

        DELETE_COPY(WCocoJsonSaver);
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
    WCocoJsonSaver<TDatums>::WCocoJsonSaver(const std::shared_ptr<CocoJsonSaver>& cocoJsonSaver) :
        spCocoJsonSaver{cocoJsonSaver}
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
                const auto& tDatum = tDatums->at(0);
                // Record json in COCO format
                const std::string stringToRemove = "COCO_val2014_";
                const auto stringToRemoveEnd = tDatum.name.find(stringToRemove) + stringToRemove.size();
                const auto imageId = std::stoull(tDatum.name.substr(stringToRemoveEnd, tDatum.name.size() - stringToRemoveEnd));
                // Record json in COCO format if file within desired range of images
                spCocoJsonSaver->record(tDatum.poseKeypoints, imageId);
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

    COMPILE_TEMPLATE_DATUM(WCocoJsonSaver);
}

#endif // OPENPOSE_FILESTREAM_W_COCO_JSON_SAVER_HPP
