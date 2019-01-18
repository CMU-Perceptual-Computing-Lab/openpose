#ifndef OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
#define OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP

#include <atomic>
#include <limits> // std::numeric_limits
#include <tuple>
#include <openpose/core/common.hpp>
#include <openpose/core/datum.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    template<typename TDatum>
    class DatumProducer
    {
    public:
        explicit DatumProducer(
            const std::shared_ptr<Producer>& producerSharedPtr,
            const unsigned long long frameFirst = 0, const unsigned long long frameStep = 1,
            const unsigned long long frameLast = std::numeric_limits<unsigned long long>::max(),
            const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr);

        virtual ~DatumProducer();

        std::pair<bool, std::shared_ptr<std::vector<std::shared_ptr<TDatum>>>> checkIfRunningAndGetDatum();

    private:
        const unsigned long long mNumberFramesToProcess;
        std::shared_ptr<Producer> spProducer;
        unsigned long long mGlobalCounter;
        unsigned long long mFrameStep;
        unsigned int mNumberConsecutiveEmptyFrames;
        std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;

        void checkIfTooManyConsecutiveEmptyFrames(
            unsigned int& numberConsecutiveEmptyFrames, const bool emptyFrame) const;

        DELETE_COPY(DatumProducer);
    };
}





// Implementation
#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor
#include <openpose/producer/datumProducer.hpp>
namespace op
{
    template<typename TDatum>
    DatumProducer<TDatum>::DatumProducer(
        const std::shared_ptr<Producer>& producerSharedPtr,
        const unsigned long long frameFirst, const unsigned long long frameStep,
        const unsigned long long frameLast,
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr) :
        mNumberFramesToProcess{(frameLast != std::numeric_limits<unsigned long long>::max()
                                ? frameLast - frameFirst : frameLast)},
        spProducer{producerSharedPtr},
        mGlobalCounter{0ll},
        mFrameStep{frameStep},
        mNumberConsecutiveEmptyFrames{0u},
        spVideoSeek{videoSeekSharedPtr}
    {
        try
        {
            // Sanity check
            if (frameLast < frameFirst)
                error("The desired initial frame must be lower than the last one (flags `--frame_first` vs."
                      " `--frame_last`). Current: " + std::to_string(frameFirst) + " vs. " + std::to_string(frameLast)
                      + ".", __LINE__, __FUNCTION__, __FILE__);
            if (frameLast != std::numeric_limits<unsigned long long>::max()
                && frameLast > spProducer->get(CV_CAP_PROP_FRAME_COUNT)-1)
                error("The desired last frame must be lower than the length of the video or the number of images."
                      " Current: " + std::to_string(frameLast) + " vs. "
                      + std::to_string(positiveIntRound(spProducer->get(CV_CAP_PROP_FRAME_COUNT))-1) + ".",
                      __LINE__, __FUNCTION__, __FILE__);
            // Set frame first and step
            if (spProducer->getType() != ProducerType::FlirCamera && spProducer->getType() != ProducerType::IPCamera
                && spProducer->getType() != ProducerType::Webcam)
            {
                // Frame first
                spProducer->set(CV_CAP_PROP_POS_FRAMES, (double)frameFirst);
                // Frame step
                spProducer->set(ProducerProperty::FrameStep, (double)frameStep);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum>
    DatumProducer<TDatum>::~DatumProducer()
    {
    }

    template<typename TDatum>
    std::pair<bool, std::shared_ptr<std::vector<std::shared_ptr<TDatum>>>> DatumProducer<TDatum>::checkIfRunningAndGetDatum()
    {
        try
        {
            auto datums = std::make_shared<std::vector<std::shared_ptr<TDatum>>>();
            // Check last desired frame has not been reached
            if (mNumberFramesToProcess != std::numeric_limits<unsigned long long>::max()
                && mGlobalCounter > mNumberFramesToProcess)
            {
                spProducer->release();
            }
            // If producer released -> it sends an empty cv::Mat + a datumProducerRunning signal
            const bool datumProducerRunning = spProducer->isOpened();
            // If device is open
            if (datumProducerRunning)
            {
                // Fast forward/backward - Seek to specific frame index desired
                if (spVideoSeek != nullptr)
                {
                    // Fake pause vs. normal mode
                    const auto increment = spVideoSeek->second - (spVideoSeek->first ? 1 : 0);
                    // Normal mode
                    if (increment != 0)
                        spProducer->set(CV_CAP_PROP_POS_FRAMES, spProducer->get(CV_CAP_PROP_POS_FRAMES) + increment);
                    // It must be always reset or bug in fake pause
                    spVideoSeek->second = 0;
                }
                auto nextFrameName = spProducer->getNextFrameName();
                const auto nextFrameNumber = (unsigned long long)spProducer->get(CV_CAP_PROP_POS_FRAMES);
                const auto cvMats = spProducer->getFrames();
                const auto cameraMatrices = spProducer->getCameraMatrices();
                auto cameraExtrinsics = spProducer->getCameraExtrinsics();
                auto cameraIntrinsics = spProducer->getCameraIntrinsics();
                // Check frames are not empty
                checkIfTooManyConsecutiveEmptyFrames(mNumberConsecutiveEmptyFrames, cvMats.empty() || cvMats[0].empty());
                if (!cvMats.empty())
                {
                    datums->resize(cvMats.size());
                    // Datum cannot be assigned before resize()
                    auto& datumPtr = (*datums)[0];
                    datumPtr = std::make_shared<TDatum>();
                    // Filling first element
                    std::swap(datumPtr->name, nextFrameName);
                    datumPtr->frameNumber = nextFrameNumber;
                    datumPtr->cvInputData = cvMats[0];
                    if (!cameraMatrices.empty())
                    {
                        datumPtr->cameraMatrix = cameraMatrices[0];
                        datumPtr->cameraExtrinsics = cameraExtrinsics[0];
                        datumPtr->cameraIntrinsics = cameraIntrinsics[0];
                    }
                    // Image integrity
                    if (datumPtr->cvInputData.channels() != 3)
                    {
                        const std::string commonMessage{"Input images must be 3-channel BGR."};
                        // Grey to RGB if required
                        if (datumPtr->cvInputData.channels() == 1)
                        {
                            log(commonMessage + " Converting grey image into BGR.", Priority::High);
                            cv::cvtColor(datumPtr->cvInputData, datumPtr->cvInputData, CV_GRAY2BGR);
                        }
                        else
                            error(commonMessage, __LINE__, __FUNCTION__, __FILE__);
                    }
                    datumPtr->cvOutputData = datumPtr->cvInputData;
                    // Resize if it's stereo-system
                    if (datums->size() > 1)
                    {
                        // Stereo-system: Assign all cv::Mat
                        for (auto i = 1u ; i < datums->size() ; i++)
                        {
                            auto& datumIPtr = (*datums)[i];
                            datumIPtr = std::make_shared<TDatum>();
                            datumIPtr->name = datumPtr->name;
                            datumIPtr->frameNumber = datumPtr->frameNumber;
                            datumIPtr->cvInputData = cvMats[i];
                            datumIPtr->cvOutputData = datumIPtr->cvInputData;
                            if (cameraMatrices.size() > i)
                            {
                                datumIPtr->cameraMatrix = cameraMatrices[i];
                                datumIPtr->cameraExtrinsics = cameraExtrinsics[i];
                                datumIPtr->cameraIntrinsics = cameraIntrinsics[i];
                            }
                        }
                    }
                    // Check producer is running
                    if (!datumProducerRunning || (*datums)[0]->cvInputData.empty())
                        datums = nullptr;
                    // Increase counter if successful image
                    if (datums != nullptr)
                        mGlobalCounter += mFrameStep;
                }
            }
            // Return result
            return std::make_pair(datumProducerRunning, datums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::make_shared<std::vector<std::shared_ptr<TDatum>>>());
        }
    }

    template<typename TDatum>
    void DatumProducer<TDatum>::checkIfTooManyConsecutiveEmptyFrames(
        unsigned int& numberConsecutiveEmptyFrames, const bool emptyFrame) const
    {
        numberConsecutiveEmptyFrames = (emptyFrame ? numberConsecutiveEmptyFrames+1 : 0);
        const auto threshold = 500u;
        if (numberConsecutiveEmptyFrames >= threshold)
            error("Detected too many (" + std::to_string(numberConsecutiveEmptyFrames) + ") empty frames in a row.",
                  __LINE__, __FUNCTION__, __FILE__);
    }

    extern template class DatumProducer<BASE_DATUM>;
}


#endif // OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
