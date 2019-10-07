#ifndef OPENPOSE_FILESTREAM_W_IMAGE_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_IMAGE_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/imageSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WImageSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WImageSaver(const std::shared_ptr<ImageSaver>& imageSaver);

        virtual ~WImageSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<ImageSaver> spImageSaver;

        DELETE_COPY(WImageSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WImageSaver<TDatums>::WImageSaver(const std::shared_ptr<ImageSaver>& imageSaver) :
        spImageSaver{imageSaver}
    {
    }

    template<typename TDatums>
    WImageSaver<TDatums>::~WImageSaver()
    {
    }

    template<typename TDatums>
    void WImageSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WImageSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record image(s) on disk
                std::vector<Matrix> opOutputDatas(tDatumsNoPtr.size());
                for (auto i = 0u; i < tDatumsNoPtr.size(); i++)
                    opOutputDatas[i] = tDatumsNoPtr[i]->cvOutputData;
                const auto fileName = (!tDatumsNoPtr[0]->name.empty()
                    ? tDatumsNoPtr[0]->name : std::to_string(tDatumsNoPtr[0]->id));
                spImageSaver->saveImages(opOutputDatas, fileName);
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

    COMPILE_TEMPLATE_DATUM(WImageSaver);
}

#endif // OPENPOSE_FILESTREAM_W_IMAGE_SAVER_HPP
