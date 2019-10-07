#ifndef OPENPOSE_FILESTREAM_W_VIDEO_SAVER_3D_HPP
#define OPENPOSE_FILESTREAM_W_VIDEO_SAVER_3D_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/videoSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WVideoSaver3D : public WorkerConsumer<TDatums>
    {
    public:
        explicit WVideoSaver3D(const std::shared_ptr<VideoSaver>& videoSaver);

        virtual ~WVideoSaver3D();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<VideoSaver> spVideoSaver;

        DELETE_COPY(WVideoSaver3D);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WVideoSaver3D<TDatums>::WVideoSaver3D(const std::shared_ptr<VideoSaver>& videoSaver) :
        spVideoSaver{videoSaver}
    {
    }

    template<typename TDatums>
    WVideoSaver3D<TDatums>::~WVideoSaver3D()
    {
    }

    template<typename TDatums>
    void WVideoSaver3D<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WVideoSaver3D<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // T* to T
                auto& tDatumsNoPtr = *tDatums;
                // Record video(s)
                if (!tDatumsNoPtr.empty())
                    spVideoSaver->write(tDatumsNoPtr[0]->cvOutputData3D);
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
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WVideoSaver3D);
}

#endif // OPENPOSE_FILESTREAM_W_VIDEO_SAVER_3D_HPP
