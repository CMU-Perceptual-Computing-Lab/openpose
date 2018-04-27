#ifndef OPENPOSE_PRODUCER_VIDEO_READER_HPP
#define OPENPOSE_PRODUCER_VIDEO_READER_HPP

#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/core/common.hpp>
#include <openpose/producer/videoCaptureReader.hpp>

namespace op
{
    /**
     * VideoReader is a wrapper of the cv::VideoCapture class for video. It allows controlling a video (e.g. extracting
     * frames, setting resolution & fps, etc).
     */
    class OP_API VideoReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of VideoReader. It opens the video as a wrapper of cv::VideoCapture. It includes a flag to
         * indicate whether the video should be repeated once it is completely read.
         * @param videoPath const std::string parameter with the full video path location.
         * @param imageDirectoryStereo const int parameter with the number of images per iteration (>1 would represent
         * stereo processing).
         * @param cameraParameterPath const std::string parameter with the folder path containing the camera
         * parameters (only required if imageDirectorystereo > 1).
         */
        explicit VideoReader(const std::string& videoPath, const unsigned int imageDirectoryStereo = 1,
                             const std::string& cameraParameterPath = "");

        std::vector<cv::Mat> getCameraMatrices();

        std::vector<cv::Mat> getCameraExtrinsics();

        std::vector<cv::Mat> getCameraIntrinsics();

        std::string getNextFrameName();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        const unsigned int mImageDirectoryStereo;
        const std::string mPathName;
        CameraParameterReader mCameraParameterReader;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        DELETE_COPY(VideoReader);
    };
}

#endif // OPENPOSE_PRODUCER_VIDEO_READER_HPP
