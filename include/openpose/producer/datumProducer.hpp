#ifndef OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
#define OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP

#include <atomic>
#include <limits> // std::numeric_limits
#include <tuple>
#include <openpose/core/common.hpp>
#include <openpose/core/datum.hpp>
#include <openpose/producer/producer.hpp>

namespace op
{
    template<typename TDatumsNoPtr>
    class DatumProducer
    {
    public:
        explicit DatumProducer(const std::shared_ptr<Producer>& producerSharedPtr,
                               const unsigned long long frameFirst = 0,
                               const unsigned long long frameLast = std::numeric_limits<unsigned long long>::max(),
                               const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr);

        std::pair<bool, std::shared_ptr<TDatumsNoPtr>> checkIfRunningAndGetDatum();

    private:
        const unsigned long long mNumberFramesToProcess;
        std::shared_ptr<Producer> spProducer;
        unsigned long long mGlobalCounter;
        unsigned int mNumberConsecutiveEmptyFrames;
        std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;

        void checkIfTooManyConsecutiveEmptyFrames(unsigned int& numberConsecutiveEmptyFrames,
                                                  const bool emptyFrame) const;

        DELETE_COPY(DatumProducer);
    };
}





// Implementation
#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor
#include <openpose/producer/datumProducer.hpp>
namespace op
{
    template<typename TDatumsNoPtr>
    DatumProducer<TDatumsNoPtr>::DatumProducer(const std::shared_ptr<Producer>& producerSharedPtr,
                                               const unsigned long long frameFirst, const unsigned long long frameLast,
                                               const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr) :
        mNumberFramesToProcess{(frameLast != std::numeric_limits<unsigned long long>::max()
                                ? frameLast - frameFirst : frameLast)},
        spProducer{producerSharedPtr},
        mGlobalCounter{0ll},
        mNumberConsecutiveEmptyFrames{0u},
        spVideoSeek{videoSeekSharedPtr}
    {
        try
        {
            if (spProducer->getType() != ProducerType::Webcam)
                spProducer->set(CV_CAP_PROP_POS_FRAMES, (double)frameFirst);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatumsNoPtr>
    std::pair<bool, std::shared_ptr<TDatumsNoPtr>> DatumProducer<TDatumsNoPtr>::checkIfRunningAndGetDatum()
    {
        try
        {
            // Check last desired frame has not been reached
            if (mNumberFramesToProcess != std::numeric_limits<unsigned long long>::max()
                && mGlobalCounter > mNumberFramesToProcess)
            {
                spProducer->release();
            }
            // If producer released -> it sends an empty cv::Mat + a datumProducerRunning signal
            const bool datumProducerRunning = spProducer->isOpened();
            auto datums = std::make_shared<TDatumsNoPtr>(1);
            auto& datum = (*datums)[0];
            // Check producer device is open
            if (datumProducerRunning)
            {
                // Fast forward/backward - Seek to specific frame index desired
                if (spVideoSeek != nullptr)
                {
                    // Fake pause vs. normal mode
                    const auto increment = spVideoSeek->second  + (spVideoSeek->first ? 1 : 0);
                    // Normal mode
                    if (increment != 0)
                    {
                        spProducer->set(CV_CAP_PROP_POS_FRAMES, spProducer->get(CV_CAP_PROP_POS_FRAMES) + increment);
                        spVideoSeek->second = 0;
                    }
                }
                // Get cv::Mat
                datum.name = spProducer->getFrameName();
                datum.cvInputData = spProducer->getFrame();
                if (datum.cvInputData.channels() != 3)
                {
                    const std::string commonMessage{"Input images must be 3-channel BGR."};
                    if (datum.cvInputData.channels() == 1)
                    {
                        log(commonMessage + " Converting grey image into BGR.", Priority::High);
                        cv::cvtColor(datum.cvInputData, datum.cvInputData, CV_GRAY2BGR);
                    }
                    else
                        error(commonMessage, __LINE__, __FUNCTION__, __FILE__);
                }
                datum.cvOutputData = datum.cvInputData;
                // Check frames are not empty
                checkIfTooManyConsecutiveEmptyFrames(mNumberConsecutiveEmptyFrames, datum.cvInputData.empty());
            }
            // Check producer is running
            if (!datumProducerRunning || datum.cvInputData.empty())
                datums = nullptr;
            // Increase counter if successful image
            if (datums != nullptr)
                mGlobalCounter++;
            // Return result
            return std::make_pair(datumProducerRunning, datums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(false, std::make_shared<TDatumsNoPtr>());
        }
    }

    template<typename TDatumsNoPtr>
    void DatumProducer<TDatumsNoPtr>::checkIfTooManyConsecutiveEmptyFrames(unsigned int& numberConsecutiveEmptyFrames,
                                                                           const bool emptyFrame) const
    {
        numberConsecutiveEmptyFrames = (emptyFrame ? numberConsecutiveEmptyFrames+1 : 0);
        const auto threshold = 500u;
        if (numberConsecutiveEmptyFrames >= threshold)
            error("Detected too many (" + std::to_string(numberConsecutiveEmptyFrames) + ") empty frames in a row.",
                  __LINE__, __FUNCTION__, __FILE__);
    }

    extern template class DatumProducer<DATUM_BASE_NO_PTR>;
}


#endif // OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
