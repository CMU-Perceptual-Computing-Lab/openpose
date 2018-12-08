#ifndef OPENPOSE_FILESTREAM_W_PEOPLE_JSON_SAVER_HPP
#define OPENPOSE_FILESTREAM_W_PEOPLE_JSON_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/peopleJsonSaver.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WPeopleJsonSaver : public WorkerConsumer<TDatums>
    {
    public:
        explicit WPeopleJsonSaver(const std::shared_ptr<PeopleJsonSaver>& peopleJsonSaver);

        virtual ~WPeopleJsonSaver();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<PeopleJsonSaver> spPeopleJsonSaver;

        DELETE_COPY(WPeopleJsonSaver);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPeopleJsonSaver<TDatums>::WPeopleJsonSaver(const std::shared_ptr<PeopleJsonSaver>& peopleJsonSaver) :
        spPeopleJsonSaver{peopleJsonSaver}
    {
    }

    template<typename TDatums>
    WPeopleJsonSaver<TDatums>::~WPeopleJsonSaver()
    {
    }

    template<typename TDatums>
    void WPeopleJsonSaver<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WPeopleJsonSaver<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Save body/face/hand keypoints to JSON file
                const auto& tDatumFirst = (*tDatums)[0];
                const auto baseFileName = (!tDatumFirst.name.empty() ? tDatumFirst.name
                                            : std::to_string(tDatumFirst.id)) + "_keypoints";
                const bool humanReadable = false;
                for (auto i = 0u ; i < tDatums->size() ; i++)
                {
                    const auto& tDatum = (*tDatums)[i];
                    // const auto fileName = baseFileName;
                    const auto fileName = baseFileName + (i != 0 ? "_" + std::to_string(i) : "");

                    const std::vector<std::pair<Array<float>, std::string>> keypointVector{
                        // 2D
                        std::make_pair(tDatum.poseKeypoints, "pose_keypoints_2d"),
                        std::make_pair(tDatum.faceKeypoints, "face_keypoints_2d"),
                        std::make_pair(tDatum.handKeypoints[0], "hand_left_keypoints_2d"),
                        std::make_pair(tDatum.handKeypoints[1], "hand_right_keypoints_2d"),
                        // 3D
                        std::make_pair(tDatum.poseKeypoints3D, "pose_keypoints_3d"),
                        std::make_pair(tDatum.faceKeypoints3D, "face_keypoints_3d"),
                        std::make_pair(tDatum.handKeypoints3D[0], "hand_left_keypoints_3d"),
                        std::make_pair(tDatum.handKeypoints3D[1], "hand_right_keypoints_3d")
                    };
                    // Save keypoints
                    spPeopleJsonSaver->save(keypointVector, tDatum.poseCandidates, fileName, humanReadable);
                }
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

    COMPILE_TEMPLATE_DATUM(WPeopleJsonSaver);
}

#endif // OPENPOSE_FILESTREAM_W_PEOPLE_JSON_SAVER_HPP
