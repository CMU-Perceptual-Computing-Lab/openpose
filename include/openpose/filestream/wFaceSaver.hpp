#ifndef OPENPOSE_FILESTREAM_W_FACE_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_FACE_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/filestream/keypointSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WFaceSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WFaceSaver(const std::shared_ptr<KeypointSaver>& keypointSaver);

        virtual ~WFaceSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<KeypointSaver> spKeypointSaver;

        DELETE_COPY(WFaceSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WFaceSaver<TDatums>::WFaceSaver(const std::shared_ptr<KeypointSaver>& keypointSaver) :
        spKeypointSaver{keypointSaver}
    {
    }

    template<typename TDatums>
    WFaceSaver<TDatums>::~WFaceSaver()
    {
    }

    template<typename TDatums>
    void WFaceSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WFaceSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                // Record people face keypoint data
                std::vector<Array<float>> keypointVector(tDatumsNoPtr.size());
                for (auto i = 0u; i < tDatumsNoPtr.size(); i++)
                    keypointVector[i] = tDatumsNoPtr[i]->faceKeypoints;
                const auto fileName = (!tDatumsNoPtr[0]->name.empty()
                    ? tDatumsNoPtr[0]->name : std::to_string(tDatumsNoPtr[0]->id));
                spKeypointSaver->saveKeypoints(keypointVector, fileName, "face");
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

    COMPILE_TEMPLATE_DATUM(WFaceSaver);
}

#endif // OPENPOSE_FILESTREAM_W_FACE_SAVER_HPP
