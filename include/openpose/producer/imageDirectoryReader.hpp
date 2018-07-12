#ifndef OPENPOSE_PRODUCER_IMAGE_DIRECTORY_READER_HPP
#define OPENPOSE_PRODUCER_IMAGE_DIRECTORY_READER_HPP

#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    /**
     * ImageDirectoryReader is an abstract class to extract frames from a image directory. Its interface imitates the
     * cv::VideoCapture class, so it can be used quite similarly to the cv::VideoCapture class. Thus,
     * it is quite similar to VideoReader and WebcamReader.
     */
    class OP_API ImageDirectoryReader : public Producer
    {
    public:
        /**
         * Constructor of ImageDirectoryReader. It sets the image directory path from which the images will be loaded
         * and generates a std::vector<std::string> with the list of images on that directory.
         * @param imageDirectoryPath const std::string parameter with the folder path containing the images.
         * @param imageDirectoryStereo const int parameter with the number of images per iteration (>1 would represent
         * stereo processing).
         * @param cameraParameterPath const std::string parameter with the folder path containing the camera
         * parameters (only required if imageDirectorystereo > 1).
         */
        explicit ImageDirectoryReader(const std::string& imageDirectoryPath, const unsigned int imageDirectoryStereo = 1,
                                      const std::string& cameraParameterPath = "");

        std::vector<cv::Mat> getCameraMatrices();

        std::vector<cv::Mat> getCameraExtrinsics();

        std::vector<cv::Mat> getCameraIntrinsics();

        std::string getNextFrameName();

        inline bool isOpened() const
        {
            return (mFrameNameCounter >= 0);
        }

        inline void release()
        {
            mFrameNameCounter = {-1ll};
        }

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        const std::string mImageDirectoryPath;
        const unsigned int mImageDirectoryStereo;
        const std::vector<std::string> mFilePaths;
        CameraParameterReader mCameraParameterReader;
        Point<int> mResolution;
        long long mFrameNameCounter;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        DELETE_COPY(ImageDirectoryReader);
    };
}

#endif // OPENPOSE_PRODUCER_IMAGE_DIRECTORY_READER_HPP
