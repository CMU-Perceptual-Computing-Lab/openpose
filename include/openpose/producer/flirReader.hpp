#ifndef OPENPOSE_PRODUCER_FLIR_READER_HPP
#define OPENPOSE_PRODUCER_FLIR_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    /**
     * FlirReader is an abstract class to extract frames from a image directory. Its interface imitates the
     * cv::VideoCapture class, so it can be used quite similarly to the cv::VideoCapture class. Thus,
     * it is quite similar to VideoReader and WebcamReader.
     */
    class OP_API FlirReader : public Producer
    {
    public:
        /**
         * Constructor of FlirReader. It sets the image directory path from which the images will be loaded
         * and generates a std::vector<std::string> with the list of images on that directory.
         * @param imageDirectoryPath const std::string parameter with the folder path containing the images.
         */
        explicit FlirReader();

        ~FlirReader();

        std::vector<cv::Mat> getCameraMatrices();

        std::string getFrameName();

        inline bool isOpened() const
        {
            return true;
        }

        void release();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplFlirReader;
        std::shared_ptr<ImplFlirReader> upImpl;

        Point<int> mResolution;
        long long mFrameNameCounter;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        DELETE_COPY(FlirReader);
    };
}

#endif // OPENPOSE_PRODUCER_FLIR_READER_HPP
