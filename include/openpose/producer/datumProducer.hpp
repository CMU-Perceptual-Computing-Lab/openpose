#ifndef OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
#define OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP

#include <atomic>
#include <limits> // std::numeric_limits
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
#include <openpose/producer/datumProducer.hpp>
#include <openpose/utilities/openCv.hpp>
namespace op
{
    // Auxiliary functions for DatumProducer in order to 1) Reduce compiling time and 2) Remove OpenCV deps.
    OP_API void datumProducerConstructor(
        const std::shared_ptr<Producer>& producerSharedPtr, const unsigned long long frameFirst,
        const unsigned long long frameStep, const unsigned long long frameLast);
    OP_API void datumProducerConstructorTooManyConsecutiveEmptyFrames(
            unsigned int& numberConsecutiveEmptyFrames, const bool emptyFrame);
    OP_API bool datumProducerConstructorRunningAndGetDatumIsDatumProducerRunning(
        const std::shared_ptr<Producer>& producerSharedPtr, const unsigned long long numberFramesToProcess,
        const unsigned long long globalCounter);
    OP_API void datumProducerConstructorRunningAndGetDatumApplyPlayerControls(
        const std::shared_ptr<Producer>& producerSharedPtr,
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr);
    OP_API unsigned long long datumProducerConstructorRunningAndGetNextFrameNumber(
        const std::shared_ptr<Producer>& producerSharedPtr);
    OP_API void datumProducerConstructorRunningAndGetDatumFrameIntegrity(Matrix& matrix);

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
            datumProducerConstructor(producerSharedPtr, frameFirst, frameStep, frameLast);
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
            // If producer released -> it sends an empty Matrix + a datumProducerRunning signal
            const bool datumProducerRunning = datumProducerConstructorRunningAndGetDatumIsDatumProducerRunning(
                spProducer, mNumberFramesToProcess, mGlobalCounter);
            // If device is open
            auto datums = std::make_shared<std::vector<std::shared_ptr<TDatum>>>();
            if (datumProducerRunning)
            {
                // Fast forward/backward - Seek to specific frame index desired
                datumProducerConstructorRunningAndGetDatumApplyPlayerControls(spProducer, spVideoSeek);
                // Get Matrix vector
                std::string nextFrameName = spProducer->getNextFrameName();
                const unsigned long long nextFrameNumber = datumProducerConstructorRunningAndGetNextFrameNumber(
                    spProducer);
                const std::vector<Matrix> matrices = spProducer->getFrames();
                // Check frames are not empty
                checkIfTooManyConsecutiveEmptyFrames(
                    mNumberConsecutiveEmptyFrames, matrices.empty() || matrices[0].empty());
                if (!matrices.empty())
                {
                    // Get camera parameters
                    const std::vector<Matrix> cameraMatrices = spProducer->getCameraMatrices();
                    const std::vector<Matrix> cameraExtrinsics = spProducer->getCameraExtrinsics();
                    const std::vector<Matrix> cameraIntrinsics = spProducer->getCameraIntrinsics();
                    // Resize datum
                    datums->resize(matrices.size());
                    // Datum cannot be assigned before resize()
                    auto& datumPtr = (*datums)[0];
                    datumPtr = std::make_shared<TDatum>();
                    // Filling first element
                    std::swap(datumPtr->name, nextFrameName);
                    datumPtr->frameNumber = nextFrameNumber;
                    datumPtr->cvInputData = matrices[0];
                    datumProducerConstructorRunningAndGetDatumFrameIntegrity(datumPtr->cvInputData);
                    if (!cameraMatrices.empty())
                    {
                        datumPtr->cameraMatrix = cameraMatrices[0];
                        datumPtr->cameraExtrinsics = cameraExtrinsics[0];
                        datumPtr->cameraIntrinsics = cameraIntrinsics[0];
                    }
                    // Initially, cvOutputData = cvInputData. No performance hit (both cv::Mat share raw memory)
                    datumPtr->cvOutputData = datumPtr->cvInputData;
                    // Resize if it's stereo-system
                    if (datums->size() > 1)
                    {
                        // Stereo-system: Assign all Matrices
                        for (auto i = 1u ; i < datums->size() ; i++)
                        {
                            auto& datumIPtr = (*datums)[i];
                            datumIPtr = std::make_shared<TDatum>();
                            datumIPtr->name = datumPtr->name;
                            datumIPtr->frameNumber = datumPtr->frameNumber;
                            datumIPtr->cvInputData = matrices[i];
                            datumProducerConstructorRunningAndGetDatumFrameIntegrity(datumPtr->cvInputData);
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
                    if ((*datums)[0]->cvInputData.empty())
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
        datumProducerConstructorTooManyConsecutiveEmptyFrames(
            numberConsecutiveEmptyFrames, emptyFrame);
    }

    extern template class DatumProducer<BASE_DATUM>;
}


#endif // OPENPOSE_PRODUCER_DATUM_PRODUCER_HPP
