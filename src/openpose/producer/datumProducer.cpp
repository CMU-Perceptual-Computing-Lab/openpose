#include <openpose/producer/datumProducer.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    void datumProducerConstructor(
        const std::shared_ptr<Producer>& producerSharedPtr,
        const unsigned long long frameFirst, const unsigned long long frameStep, const unsigned long long frameLast)
    {
        try
        {
            // Sanity check
            if (frameLast < frameFirst)
                error("The desired initial frame must be lower than the last one (flags `--frame_first` vs."
                      " `--frame_last`). Current: " + std::to_string(frameFirst) + " vs. " + std::to_string(frameLast)
                      + ".", __LINE__, __FUNCTION__, __FILE__);
            if (frameLast != std::numeric_limits<unsigned long long>::max()
                && frameLast > producerSharedPtr->get(getCvCapPropFrameCount())-1)
                error("The desired last frame must be lower than the length of the video or the number of images."
                      " Current: " + std::to_string(frameLast) + " vs. "
                      + std::to_string(positiveIntRound(producerSharedPtr->get(getCvCapPropFrameCount()))-1) + ".",
                      __LINE__, __FUNCTION__, __FILE__);
            // Set frame first and step
            if (producerSharedPtr->getType() != ProducerType::FlirCamera
                && producerSharedPtr->getType() != ProducerType::IPCamera
                && producerSharedPtr->getType() != ProducerType::Webcam)
            {
                // Frame first
                producerSharedPtr->set(CV_CAP_PROP_POS_FRAMES, (double)frameFirst);
                // Frame step
                producerSharedPtr->set(ProducerProperty::FrameStep, (double)frameStep);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void datumProducerConstructorTooManyConsecutiveEmptyFrames(
        unsigned int& numberConsecutiveEmptyFrames, const bool emptyFrame)
    {
        try
        {
            numberConsecutiveEmptyFrames = (emptyFrame ? numberConsecutiveEmptyFrames+1 : 0);
            const auto threshold = 500u;
            if (numberConsecutiveEmptyFrames >= threshold)
                error("Detected too many (" + std::to_string(numberConsecutiveEmptyFrames)
                    + ") empty frames in a row.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool datumProducerConstructorRunningAndGetDatumIsDatumProducerRunning(
        const std::shared_ptr<Producer>& producerSharedPtr, const unsigned long long numberFramesToProcess,
        const unsigned long long globalCounter)
    {
        try
        {
            // Check last desired frame has not been reached
            if (numberFramesToProcess != std::numeric_limits<unsigned long long>::max()
                && globalCounter > numberFramesToProcess)
            {
                producerSharedPtr->release();
            }
            // If producer released -> it sends an empty Mat + a datumProducerRunning signal
            return producerSharedPtr->isOpened();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void datumProducerConstructorRunningAndGetDatumApplyPlayerControls(
        const std::shared_ptr<Producer>& producerSharedPtr,
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr)
    {
        try
        {
            // Fast forward/backward - Seek to specific frame index desired
            if (videoSeekSharedPtr != nullptr)
            {
                // Fake pause vs. normal mode
                const auto increment = videoSeekSharedPtr->second - (videoSeekSharedPtr->first ? 1 : 0);
                // Normal mode
                if (increment != 0)
                    producerSharedPtr->set(
                        CV_CAP_PROP_POS_FRAMES, producerSharedPtr->get(CV_CAP_PROP_POS_FRAMES) + increment);
                // It must be always reset or bug in fake pause
                videoSeekSharedPtr->second = 0;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    unsigned long long datumProducerConstructorRunningAndGetNextFrameNumber(
        const std::shared_ptr<Producer>& producerSharedPtr)
    {
        try
        {
            // Get next frame number
            return (unsigned long long)producerSharedPtr->get(CV_CAP_PROP_POS_FRAMES);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0ull;
        }
    }

    void datumProducerConstructorRunningAndGetDatumFrameIntegrity(Matrix& inputDataMatrix)
    {
        try
        {
            // Image integrity
            if (inputDataMatrix.channels() != 3)
            {
                const std::string commonMessage{"Input images must be 3-channel BGR."};
                // Grey to RGB if required
                if (inputDataMatrix.channels() == 1)
                {
                    opLog(commonMessage + " Converting grey image into BGR.", Priority::High);
                    cv::Mat inputData = OP_OP2CVMAT(inputDataMatrix);
                    cv::cvtColor(inputData, inputData, CV_GRAY2BGR);
                    // Diferent memory size --> new cv::Mat raw ptr memory --> new Matrix
                    inputDataMatrix = OP_CV2OPMAT(inputData);
                }
                else
                    error(commonMessage, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
