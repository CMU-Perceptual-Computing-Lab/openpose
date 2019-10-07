#include <openpose/producer/flirReader.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    FlirReader::FlirReader(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                           const bool undistortImage, const int cameraIndex) :
        Producer{ProducerType::FlirCamera, cameraParameterPath, undistortImage, -1},
        mSpinnakerWrapper{cameraParameterPath, cameraResolution, undistortImage, cameraIndex},
        mFrameNameCounter{0ull}
    {
        try
        {
            // Get resolution
            const auto resolution = mSpinnakerWrapper.getResolution();
            // Set resolution
            set(CV_CAP_PROP_FRAME_WIDTH, resolution.x);
            set(CV_CAP_PROP_FRAME_HEIGHT, resolution.y);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    FlirReader::~FlirReader()
    {
        try
        {
            release();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Matrix> FlirReader::getCameraMatrices()
    {
        try
        {
            return mSpinnakerWrapper.getCameraMatrices();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> FlirReader::getCameraExtrinsics()
    {
        try
        {
            return mSpinnakerWrapper.getCameraExtrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> FlirReader::getCameraIntrinsics()
    {
        try
        {
            return mSpinnakerWrapper.getCameraIntrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string FlirReader::getNextFrameName()
    {
        try
        {
            const auto stringLength = 12u;
            return toFixedLengthString(mFrameNameCounter, stringLength);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    bool FlirReader::isOpened() const
    {
        try
        {
            return mSpinnakerWrapper.isOpened();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void FlirReader::release()
    {
        try
        {
            mSpinnakerWrapper.release();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix FlirReader::getRawFrame()
    {
        try
        {
            return mSpinnakerWrapper.getRawFrames().at(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    std::vector<Matrix> FlirReader::getRawFrames()
    {
        try
        {
            mFrameNameCounter++; // Simple counter: 0,1,2,3,...
            return mSpinnakerWrapper.getRawFrames();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    double FlirReader::get(const int capProperty)
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
                return -1.;
            else if (capProperty == CV_CAP_PROP_FPS)
                return -1.;
            else
            {
                opLog("Unknown property.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                return -1.;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void FlirReader::set(const int capProperty, const double value)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                mResolution.x = {(int)value};
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
                mResolution.y = {(int)value};
            else if (capProperty == CV_CAP_PROP_POS_FRAMES)
                opLog("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else if (capProperty == CV_CAP_PROP_FRAME_COUNT || capProperty == CV_CAP_PROP_FPS)
                opLog("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else
                opLog("Unknown property.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
