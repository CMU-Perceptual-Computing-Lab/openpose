#ifndef OPENPOSE__PRODUCER__WEBCAM_READER_HPP
#define OPENPOSE__PRODUCER__WEBCAM_READER_HPP

#include <atomic>
#include <thread>
#include "videoCaptureReader.hpp"

namespace op
{
    /**
     *  WebcamReader is a wrapper of the cv::VideoCapture class for webcam. It allows controlling a video (extracting
     * frames, setting resolution & fps, seeking to a particular frame, etc).
     */
    class WebcamReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of WebcamReader. It opens the webcam as a wrapper of cv::VideoCapture. It includes an argument
         * to indicate the desired resolution.
         * @param webcamIndex const int indicating the camera source (see the OpenCV documentation about
         * cv::VideoCapture for more details), in the range [0, 9].
         * @param webcamResolution const cv::Size parameter which specifies the desired camera resolution.
         */
        explicit WebcamReader(const int webcamIndex = 0, const cv::Size webcamResolution = cv::Size{});

        ~WebcamReader();

        std::string getFrameName();

        double get(const int capProperty);

    private:
        long long mFrameNameCounter;
        cv::Mat mBuffer;
        std::mutex mBufferMutex;
        std::atomic<bool> mCloseThread;
        std::thread mThread;

        cv::Mat getRawFrame();

        void bufferingThread();

        DELETE_COPY(WebcamReader);
    };
}

#endif // OPENPOSE__PRODUCER__WEBCAM_READER_HPP
