#include <openpose/producer/imageDirectoryReader.hpp>
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    std::vector<std::string> getImagePathsOnDirectory(const std::string& imageDirectoryPath)
    {
        try
        {
            // Get files on directory with the desired extensions
            const auto imagePaths = getFilesOnDirectory(imageDirectoryPath, Extensions::Images);
            // Check #files > 0
            if (imagePaths.empty())
                error("No images were found on " + imageDirectoryPath, __LINE__, __FUNCTION__, __FILE__);
            // Return result
            return imagePaths;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    ImageDirectoryReader::ImageDirectoryReader(const std::string& imageDirectoryPath,
                                               const std::string& cameraParameterPath,
                                               const bool undistortImage,
                                               const int numberViews) :
        Producer{ProducerType::ImageDirectory, cameraParameterPath, undistortImage, numberViews},
        mImageDirectoryPath{imageDirectoryPath},
        mFilePaths{getImagePathsOnDirectory(imageDirectoryPath)},
        mFrameNameCounter{0ll}
    {
    }

    ImageDirectoryReader::~ImageDirectoryReader()
    {
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

    Matrix ImageDirectoryReader::getRawFrame()
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
            mResolution = Point<int>{frame.cols(), frame.rows()};
            // Return final frame
            return frame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    std::vector<Matrix> ImageDirectoryReader::getRawFrames()
    {
        try
        {
            std::vector<Matrix> rawFrames;
            for (auto i = 0 ; i < positiveIntRound(Producer::get(ProducerProperty::NumberViews)) ; i++)
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
                opLog("Unknown property", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
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
                opLog("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else
                opLog("Unknown property", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
