#ifndef OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP
#define OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API VideoSaver
    {
    public:
        VideoSaver(
            const std::string& videoSaverPath, const int cvFourcc, const double fps,
            const std::string& addAudioFromThisVideo = "");

        virtual ~VideoSaver();

        bool isOpened();

        void write(const Matrix& matToSave);

        void write(const std::vector<Matrix>& matsToSave);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplVideoSaver;
        std::unique_ptr<ImplVideoSaver> upImpl;

        DELETE_COPY(VideoSaver);
    };
}

#endif // OPENPOSE_FILESTREAM_VIDEO_SAVER_HPP
