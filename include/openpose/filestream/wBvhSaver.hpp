#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_FILESTREAM_W_BVH_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_BVH_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/bvhSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WBvhSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WBvhSaver(const std::shared_ptr<BvhSaver>& bvhSaver);

        virtual ~WBvhSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        std::shared_ptr<BvhSaver> spBvhSaver;

        DELETE_COPY(WBvhSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WBvhSaver<TDatums>::WBvhSaver(const std::shared_ptr<BvhSaver>& bvhSaver) :
        spBvhSaver{bvhSaver}
    {
    }

    template<typename TDatums>
    WBvhSaver<TDatums>::~WBvhSaver()
    {
    }

    template<typename TDatums>
    void WBvhSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WBvhSaver<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Record BVH file
                const auto& tDatumPtr = (*tDatums)[0];
                if (!tDatumPtr->poseKeypoints3D.empty())
                    spBvhSaver->updateBvh(tDatumPtr->adamPose, tDatumPtr->adamTranslation, tDatumPtr->j0Vec);
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

    COMPILE_TEMPLATE_DATUM(WBvhSaver);
}

#endif // OPENPOSE_FILESTREAM_W_BVH_SAVER_HPP
#endif
