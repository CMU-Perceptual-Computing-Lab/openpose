#ifndef OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP
#define OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/filestream/jsonOfstream.hpp>

namespace op
{
    /**
     *  The CocoJsonSaver class creates a COCO validation json file with details about the processed images. It
     * inherits from Recorder.
     */
    class OP_API CocoJsonSaver
    {
    public:
        /**
         * This constructor of CocoJsonSaver extends the Recorder::Recorder(const std::string & filePathToSave)
         * constructor.
         * @param filePathToSave const std::string parameter with the final file path where the generated json file
         * will be saved.
         */
        explicit CocoJsonSaver(
            const std::string& filePathToSave, const PoseModel poseModel, const bool humanReadable = true,
            const int cocoJsonVariants = 1, const CocoJsonFormat cocoJsonFormat = CocoJsonFormat::Body,
            const int cocoJsonVariant = 0);

        virtual ~CocoJsonSaver();

        void record(
            const Array<float>& poseKeypoints, const Array<float>& poseScores, const std::string& imageName,
            const unsigned long long frameNumber);

    private:
        const PoseModel mPoseModel;
        const int mCocoJsonVariant;
        std::vector<std::tuple<JsonOfstream, CocoJsonFormat, bool>> mJsonOfstreams;

        DELETE_COPY(CocoJsonSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP
