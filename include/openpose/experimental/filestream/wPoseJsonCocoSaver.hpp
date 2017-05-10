// THE CLASSES INSIDE THE experimental NAMESPACE ARE STILL BEING DEVELOPED, WE HIGHLY RECOMMEND NOT TO USE THEM
// The results are not the same as the one obtained with Matlab, so there is a bug somewhere in this class or in PoseJsonCocoSaver.

#ifndef OPENPOSE__EXPERIMENTAL__FILESTREAM__W_POSE_JSON_COCO_SAVER_HPP
#define OPENPOSE__EXPERIMENTAL__FILESTREAM__W_POSE_JSON_COCO_SAVER_HPP

#include <memory> // std::shared_ptr
#include "../../filestream/poseJsonCocoSaver.hpp"
#include "../../thread/workerConsumer.hpp"

namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        class WPoseJsonCocoSaver : public WorkerConsumer<TDatums>
        {
        public:
            explicit WPoseJsonCocoSaver(const std::shared_ptr<PoseJsonCocoSaver>& poseJsonCocoSaver);

            void initializationOnThread();

            void workConsumer(const TDatums& tDatums);

        private:
            std::shared_ptr<PoseJsonCocoSaver> spPoseJsonCocoSaver;

            DELETE_COPY(WPoseJsonCocoSaver);
        };
    }
}





// Implementation
#include "../../utilities/errorAndLog.hpp"
#include "../../utilities/macros.hpp"
#include "../../utilities/pointerContainer.hpp"
#include "../../utilities/profiler.hpp"
namespace op
{
    namespace experimental
    {
        template<typename TDatums>
        WPoseJsonCocoSaver<TDatums>::WPoseJsonCocoSaver(const std::shared_ptr<PoseJsonCocoSaver>& poseJsonCocoSaver) :
            spPoseJsonCocoSaver{poseJsonCocoSaver}
        {
        }

        template<typename TDatums>
        void WPoseJsonCocoSaver<TDatums>::initializationOnThread()
        {
        }

        template<typename TDatums>
        void WPoseJsonCocoSaver<TDatums>::workConsumer(const TDatums& tDatums)
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
                    const auto& tDatum = (*tDatums)[0];
                    // Record json in COCO format
                    const std::string stringToRemove = "COCO_val2014_";
                    const auto stringToRemoveEnd = tDatum.name.find(stringToRemove) + stringToRemove.size();
                    const auto imageId = std::stoull(tDatum.name.substr(stringToRemoveEnd, tDatum.name.size() - stringToRemoveEnd));
                    // if (imageId <= 20671)   // 1471 images ~ 1.5 min at 15fps
                    if (imageId <= 50006)   // 3559 images ~ 4 min at 15fps // move this to producer as --frame_last 50006
                    {
                        // Record json in COCO format if file within desired range of images
                        spPoseJsonCocoSaver->record(tDatum.poseKeyPoints, imageId);
                        // Profiling speed
                        Profiler::timerEnd(profilerKey);
                        Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, Profiler::DEFAULT_X);
                    }
                    else
                    {
                        // Debugging log
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        // Stop worker
                        this->stop();
                    }
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

        COMPILE_TEMPLATE_DATUM(WPoseJsonCocoSaver);
    }
}

#endif // OPENPOSE__EXPERIMENTAL__FILESTREAM__W_POSE_JSON_COCO_SAVER_HPP
