#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/producer/imageDirectoryReader.hpp>

namespace op
{
    std::vector<std::string> getImagePathsOnDirectory(const std::string& imageDirectoryPath)
    {
        try
        {
            // Get files on directory with the desired extensions
            const std::vector<std::string> extensions{
                // Completely supported by OpenCV
                "bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras",
                // Most of them supported by OpenCV
                "jpg", "jpeg", "png"};
            const auto imagePaths = getFilesOnDirectory(imageDirectoryPath, extensions);

            // Check #files > 0
            if (imagePaths.empty())
                error("No images were found on " + imageDirectoryPath, __LINE__, __FUNCTION__, __FILE__);

            return imagePaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    ImageDirectoryReader::ImageDirectoryReader(const std::string& imageDirectoryPath,
                                               const unsigned int imageDirectoryStereo,
                                               const std::string& cameraParameterPath) :
        Producer{ProducerType::ImageDirectory},
        mImageDirectoryPath{imageDirectoryPath},
        mImageDirectoryStereo{imageDirectoryStereo},
        mFilePaths{getImagePathsOnDirectory(imageDirectoryPath)},
        mFrameNameCounter{0ll}
    {
        try
        {
            // If stereo setting --> load camera parameters
            if (imageDirectoryStereo > 1)
            {
                // Read camera parameters from SN
                auto serialNumbers = getFilesOnDirectory(cameraParameterPath, ".xml");
                // Sanity check
                if (serialNumbers.size() != mImageDirectoryStereo && mImageDirectoryStereo > 1)
                    error("Found different number of camera parameter files than the number indicated by"
                          " `--3d_views` ("
                          + std::to_string(serialNumbers.size()) + " vs. "
                          + std::to_string(mImageDirectoryStereo) + "). Make them equal or add"
                          + " `--3d_views 1`",
                          __LINE__, __FUNCTION__, __FILE__);
                // Get serial numbers
                for (auto& serialNumber : serialNumbers)
                    serialNumber = getFileNameNoExtension(serialNumber);
                // Get camera paremeters
                mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    ImageDirectoryReader::~ImageDirectoryReader()
    {
    }

    std::vector<cv::Mat> ImageDirectoryReader::getCameraMatrices()
    {
        try
        {
            return mCameraParameterReader.getCameraMatrices();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<cv::Mat> ImageDirectoryReader::getCameraExtrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraExtrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<cv::Mat> ImageDirectoryReader::getCameraIntrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraIntrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string ImageDirectoryReader::getNextFrameName()
    {
        try
        {
            return getFileNameNoExtension(mFilePaths.at(mFrameNameCounter));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    cv::Mat ImageDirectoryReader::getRawFrame()
    {
        try
        {
            // Read frame
            auto frame = loadImage(mFilePaths.at(mFrameNameCounter++).c_str(), CV_LOAD_IMAGE_COLOR);
            // Skip frames if frame step > 1
            const auto frameStep = Producer::get(ProducerProperty::FrameStep);
            if (frameStep > 1)
                set(CV_CAP_PROP_POS_FRAMES, mFrameNameCounter + frameStep-1);
            // Check frame integrity. This function also checks width/height changes. However, if it is performed
            // after setWidth/setHeight this is performed over the new resolution (so they always match).
            checkFrameIntegrity(frame);
            // Update size, since images might have different size between each one of them
            mResolution = Point<int>{frame.cols, frame.rows};
            return frame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    std::vector<cv::Mat> ImageDirectoryReader::getRawFrames()
    {
        try
        {
            std::vector<cv::Mat> rawFrames;
            for (auto i = 0u ; i < mImageDirectoryStereo ; i++)
                rawFrames.emplace_back(getRawFrame());
            return rawFrames;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    double ImageDirectoryReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
            {
                if (Producer::get(ProducerProperty::Rotation) == 0.
                    || Producer::get(ProducerProperty::Rotation) == 180.)
                    return mResolution.x;
                else
                    return mResolution.y;
            }
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
            {
                if (Producer::get(ProducerProperty::Rotation) == 0.
                    || Producer::get(ProducerProperty::Rotation) == 180.)
                    return mResolution.y;
                else
                    return mResolution.x;
            }
            else if (capProperty == CV_CAP_PROP_POS_FRAMES)
                return (double)mFrameNameCounter;
            else if (capProperty == CV_CAP_PROP_FRAME_COUNT)
                return (double)mFilePaths.size();
            else if (capProperty == CV_CAP_PROP_FPS)
                return -1.;
            else
            {
                log("Unknown property", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                return -1.;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void ImageDirectoryReader::set(const int capProperty, const double value)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                mResolution.x = {(int)value};
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
                mResolution.y = {(int)value};
            else if (capProperty == CV_CAP_PROP_POS_FRAMES)
                mFrameNameCounter = fastTruncate((long long)value, 0ll, (long long)mFilePaths.size()-1);
            else if (capProperty == CV_CAP_PROP_FRAME_COUNT || capProperty == CV_CAP_PROP_FPS)
                log("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else
                log("Unknown property", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
