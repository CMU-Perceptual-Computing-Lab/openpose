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
            const std::vector<std::string> extensions{"bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras",   // Completely supported by OpenCV
                                                      "jpg", "jpeg", "png"};                            // Most of them supported by OpenCV
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

    ImageDirectoryReader::ImageDirectoryReader(const std::string& imageDirectoryPath) :
        Producer{ProducerType::ImageDirectory},
        mImageDirectoryPath{imageDirectoryPath},
        mFilePaths{getImagePathsOnDirectory(imageDirectoryPath)},
        mFrameNameCounter{0}
    {
    }

    std::string ImageDirectoryReader::getFrameName()
    {
        return getFileNameNoExtension(mFilePaths.at(mFrameNameCounter));
    }

    cv::Mat ImageDirectoryReader::getRawFrame()
    {
        try
        {
            auto frame = loadImage(mFilePaths.at(mFrameNameCounter++).c_str(), CV_LOAD_IMAGE_COLOR);
            // Check frame integrity. This function also checks width/height changes. However, if it is performed after setWidth/setHeight this is performed over the new resolution (so they always match).
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

    double ImageDirectoryReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
            {
                if (get(ProducerProperty::Rotation) == 0. || get(ProducerProperty::Rotation) == 180.)
                    return mResolution.x;
                else
                    return mResolution.y;
            }
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
            {
                if (get(ProducerProperty::Rotation) == 0. || get(ProducerProperty::Rotation) == 180.)
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
                mFrameNameCounter = fastTruncate((long long)value, 0ll, (long long)mImageDirectoryPath.size()-1);
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
