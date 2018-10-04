#ifndef OPENPOSE_PRODUCER_WEBCAM_READER_HPP
#define OPENPOSE_PRODUCER_WEBCAM_READER_HPP

#include <atomic>
#include <mutex>
#include <openpose/core/common.hpp>
#include <openpose/producer/videoCaptureReader.hpp>

namespace op
{
    /**
     *  WebcamReader is a wrapper of the cv::VideoCapture class for webcam. It allows controlling a video (extracting
     * frames, setting resolution & fps, seeking to a particular frame, etc).
     */
    class OP_API WebcamReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of WebcamReader. It opens the webcam as a wrapper of cv::VideoCapture. It includes an argument
         * to indicate the desired resolution.
         * @param webcamIndex const int indicating the camera source (see the OpenCV documentation about
         * cv::VideoCapture for more details), in the range [0, 9].
         * @param webcamResolution const Point<int> parameter which specifies the desired camera resolution.
         * @param fps Double parameter which specifies the desired camera frame rate.
         */
        explicit WebcamReader(const int webcamIndex = 0, const Point<int>& webcamResolution = Point<int>{},
                              const double fps = 30., const bool throwExceptionIfNoOpened = true);

        virtual ~WebcamReader();

        std::vector<cv::Mat> getCameraMatrices();

        std::vector<cv::Mat> getCameraExtrinsics();

        std::vector<cv::Mat> getCameraIntrinsics();

        std::string getNextFrameName();

        bool isOpened() const;

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        const int mIndex;
        double mFps;
        long long mFrameNameCounter;
        bool mThreadOpened;
        cv::Mat mBuffer;
        std::mutex mBufferMutex;
        std::atomic<bool> mCloseThread;
        std::thread mThread;
        // Detect camera unplugged
        double mLastNorm;
        std::atomic<int> mDisconnectedCounter;
        Point<int> mResolution;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        void bufferingThread();

        bool reset();

        DELETE_COPY(WebcamReader);
    };
}

#endif // OPENPOSE_PRODUCER_WEBCAM_READER_HPP
