#ifndef OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP
#define OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP

#include <string>
#include <openpose/core/array.hpp>
#include <openpose/utilities/macros.hpp>
#include "jsonOfstream.hpp"

namespace op
{
    /**
     *  The CocoJsonSaver class creates a COCO validation json file with details about the processed images. It inherits from Recorder.
     */
    class CocoJsonSaver
    {
    public:
        /**
         * This constructor of CocoJsonSaver extends the Recorder::Recorder(const std::string & filePathToSave) constructor.
         * @param filePathToSave const std::string parameter with the final file path where the generated json file will be saved.
         */
        explicit CocoJsonSaver(const std::string& filePathToSave, const bool humanReadable = true);

        ~CocoJsonSaver();

        void record(const Array<float>& poseKeypoints, const unsigned long long imageId);

    private:
        JsonOfstream mJsonOfstream;
        bool mFirstElementAdded;

        DELETE_COPY(CocoJsonSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_POSE_JSON_COCO_SAVER_HPP
