#ifndef OPENPOSE__FILESTREAM__POSE_JSON_COCO_SAVER_HPP
#define OPENPOSE__FILESTREAM__POSE_JSON_COCO_SAVER_HPP

#include <string>
#include "../core/array.hpp"
#include "../utilities/macros.hpp"
#include "jsonOfstream.hpp"

namespace op
{
    /**
     *  The PoseJsonCocoSaver class creates a COCO validation json file with details about the processed images. It inherits from Recorder.
     */
    class PoseJsonCocoSaver
    {
    public:
        /**
         * This constructor of PoseJsonCocoSaver extends the Recorder::Recorder(const std::string & filePathToSave) constructor.
         * @param filePathToSave const std::string parameter with the final file path where the generated json file will be saved.
         */
        explicit PoseJsonCocoSaver(const std::string& filePathToSave, const bool humanReadable = true);

        ~PoseJsonCocoSaver();

        void record(const Array<float>& poseKeyPoints, const int imageId);

    private:
        JsonOfstream mJsonOfstream;
        bool mFirstElementAdded;

        DELETE_COPY(PoseJsonCocoSaver);
    };
}

#endif // OPENPOSE__FILESTREAM__POSE_JSON_COCO_SAVER_HPP
