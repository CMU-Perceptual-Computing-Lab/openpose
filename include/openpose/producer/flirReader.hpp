#ifndef OPENPOSE_PRODUCER_FLIR_READER_HPP
#define OPENPOSE_PRODUCER_FLIR_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>
#include <openpose/producer/spinnakerWrapper.hpp>

namespace op
{
    /**
     * FlirReader is an abstract class to extract frames from a FLIR stereo-camera system. Its interface imitates the
     * cv::VideoCapture class, so it can be used quite similarly to the cv::VideoCapture class. Thus,
     * it is quite similar to VideoReader and WebcamReader.
     */
    class OP_API FlirReader : public Producer
    {
    public:
        /**
         * Constructor of FlirReader. It opens all the available FLIR cameras
         */
        explicit FlirReader(const std::string& cameraParametersPath, const Point<int>& cameraResolution,
                            const bool undistortImage = true, const int cameraIndex = -1);

        virtual ~FlirReader();

        std::vector<Matrix> getCameraMatrices();

        std::vector<Matrix> getCameraExtrinsics();

        std::vector<Matrix> getCameraIntrinsics();

        std::string getNextFrameName();

        bool isOpened() const;

        void release();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        SpinnakerWrapper mSpinnakerWrapper;
        Point<int> mResolution;
        unsigned long long mFrameNameCounter;

        Matrix getRawFrame();

        std::vector<Matrix> getRawFrames();

        DELETE_COPY(FlirReader);
    };
}

#endif // OPENPOSE_PRODUCER_FLIR_READER_HPP
