#ifndef OPENPOSE_WRAPPER_WRAPPER_HAND_FROM_JSON_TEST_HPP
#define OPENPOSE_WRAPPER_WRAPPER_HAND_FROM_JSON_TEST_HPP

// Third-party dependencies
#include <opencv2/opencv.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

namespace op
{
    template<typename TDatum,
             typename TDatums = std::vector<std::shared_ptr<TDatum>>,
             typename TWorker = std::shared_ptr<Worker<std::shared_ptr<TDatums>>>,
             typename TQueue = Queue<std::shared_ptr<TDatums>>>
    class WrapperHandFromJsonTest
    {
    public:
        /**
         * Constructor.
         */
        explicit WrapperHandFromJsonTest();

        /**
         * Destructor.
         * It automatically frees resources.
         */
        ~WrapperHandFromJsonTest();

        void configure(const WrapperStructPose& wrapperStructPose,
                       const WrapperStructHand& wrapperStructHand,
                       const std::shared_ptr<Producer>& producerSharedPtr,
                       const std::string& handGroundTruth,
                       const std::string& writeJson,
                       const DisplayMode displayMode = DisplayMode::NoDisplay);

        /**
         * Function to start multi-threading.
         * Similar to start(), but exec() blocks the thread that calls the function (it saves 1 thread). Use exec() instead of
         * start() if the calling thread will otherwise be waiting for the WrapperHandFromJsonTest to end.
         */
        void exec();

    private:
        ThreadManager<std::shared_ptr<TDatums>> mThreadManager;
        // Workers
        TWorker wDatumProducer;
        TWorker spWIdGenerator;
        TWorker spWScaleAndSizeExtractor;
        TWorker spWCvMatToOpInput;
        TWorker spWCvMatToOpOutput;
        std::vector<std::vector<TWorker>> spWPoses;
        std::vector<TWorker> mPostProcessingWs;
        std::vector<TWorker> mOutputWs;
        TWorker spWGui;

        /**
         * Frees TWorker variables (private internal function).
         * For most cases, this class is non-necessary, since std::shared_ptr are automatically cleaned on destruction of each class.
         * However, it might be useful if the same WrapperHandFromJsonTest is gonna be started twice (not recommended on most cases).
         */
        void reset();

        /**
         * Set ThreadManager from TWorkers (private internal function).
         * After any configure() has been called, the TWorkers are initialized. This function resets the ThreadManager and adds them.
         * Common code for start() and exec().
         */
        void configureThreadManager();

        /**
         * TWorker concatenator (private internal function).
         * Auxiliary function that concatenate std::vectors of TWorker. Since TWorker is some kind of smart pointer (usually
         * std::shared_ptr), its copy still shares the same internal data. It will not work for TWorker classes that do not share
         * the data when moved.
         * @param workersA First std::shared_ptr<TDatums> element to be concatenated.
         * @param workersB Second std::shared_ptr<TDatums> element to be concatenated.
         * @return Concatenated std::vector<TWorker> of both workersA and workersB.
         */
        std::vector<TWorker> mergeWorkers(const std::vector<TWorker>& workersA, const std::vector<TWorker>& workersB);

        DELETE_COPY(WrapperHandFromJsonTest);
    };
}





// Implementation
#include <openpose/core/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/gpu/gpu.hpp>
#include <openpose/hand/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/producer/headers.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fileSystem.hpp>
namespace op
{
    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::WrapperHandFromJsonTest()
    {
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::~WrapperHandFromJsonTest()
    {
        try
        {
            mThreadManager.stop();
            reset();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    void WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::configure(
        const WrapperStructPose& wrapperStructPose,
        const WrapperStructHand& wrapperStructHand,
        const std::shared_ptr<Producer>& producerSharedPtr,
        const std::string& handGroundTruth,
        const std::string& writeJson,
        const DisplayMode displayMode)
    {
        try
        {
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Shortcut
            typedef std::shared_ptr<TDatums> TDatumsPtr;

            // Check no wrong/contradictory flags enabled
            if (wrapperStructPose.scaleGap <= 0.f && wrapperStructPose.scalesNumber > 1)
                error("The scale gap must be greater than 0 (it has no effect if the number of scales is 1).",
                      __LINE__, __FUNCTION__, __FILE__);
            const std::string additionalMessage = " You could also set mThreadManagerMode = mThreadManagerMode::Asynchronous(Out)"
                                                  " and/or add your own output worker class before calling this function.";
            const auto savingSomething = !writeJson.empty();
            const auto displayGui = (displayMode != DisplayMode::NoDisplay);
            if (!displayGui && !savingSomething)
            {
                const auto message = "No output is selected (`--display 0`) and no results are generated (no `write_X` flags enabled). Thus,"
                                     " no output would be generated." + additionalMessage;
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }

            // Get number GPUs
            auto gpuNumber = wrapperStructPose.gpuNumber;
            auto gpuNumberStart = wrapperStructPose.gpuNumberStart;
            // If number GPU < 0 --> set it to all the available GPUs
            if (gpuNumber < 0)
            {
                // Get total number GPUs
                gpuNumber = getGpuNumber();
                // Reset initial GPU to 0 (we want them all)
                gpuNumberStart = 0;
                // Logging message
                opLog("Auto-detecting GPUs... Detected " + std::to_string(gpuNumber) + " GPU(s), using them all.", Priority::High);
            }

            // Proper format
            const auto writeJsonCleaned = formatAsDirectory(writeJson);

            // Common parameters
            const auto finalOutputSize = wrapperStructPose.outputSize;
            const Point<int> producerSize{
                (int)producerSharedPtr->get(getCvCapPropFrameWidth()),
                (int)producerSharedPtr->get(getCvCapPropFrameHeight())};
            if (finalOutputSize.x == -1 || finalOutputSize.y == -1)
            {
                const auto message = "Output resolution cannot be (-1 x -1) unless producerSharedPtr is also set.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }

            // Producer
            const auto datumProducer = std::make_shared<DatumProducer<TDatum>>(producerSharedPtr);
            wDatumProducer = std::make_shared<WDatumProducer<TDatum>>(datumProducer);

            // Get input scales and sizes
            const auto scaleAndSizeExtractor = std::make_shared<ScaleAndSizeExtractor>(
                wrapperStructPose.netInputSize, (float)wrapperStructPose.netInputSizeDynamicBehavior, finalOutputSize,
                wrapperStructPose.scalesNumber, wrapperStructPose.scaleGap);
            spWScaleAndSizeExtractor = std::make_shared<WScaleAndSizeExtractor<TDatumsPtr>>(scaleAndSizeExtractor);

            // Input cvMat to OpenPose format
            const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(wrapperStructPose.poseModel);
            spWCvMatToOpInput = std::make_shared<WCvMatToOpInput<TDatumsPtr>>(cvMatToOpInput);
            if (displayGui)
            {
                const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>();
                spWCvMatToOpOutput = std::make_shared<WCvMatToOpOutput<TDatumsPtr>>(cvMatToOpOutput);
            }

            // Hand extractor(s)
            if (wrapperStructHand.enable)
            {
                spWPoses.resize(gpuNumber);
                const auto handDetector = std::make_shared<HandDetectorFromTxt>(handGroundTruth);
                for (auto gpuId = 0u; gpuId < spWPoses.size(); gpuId++)
                {
                    // Hand detector
                    // If tracking
                    if (wrapperStructHand.detector == Detector::BodyWithTracking)
                        error("Tracking not valid for hand detector from JSON files.", __LINE__, __FUNCTION__, __FILE__);
                    // If detection
                    else
                        spWPoses.at(gpuId) = {std::make_shared<WHandDetectorFromTxt<TDatumsPtr>>(handDetector)};
                    // Hand keypoint extractor
                    const auto netOutputSize = wrapperStructHand.netInputSize;
                    const auto handExtractor = std::make_shared<HandExtractorCaffe>(
                        wrapperStructHand.netInputSize, netOutputSize, wrapperStructPose.modelFolder.getStdString(),
                        gpuId + gpuNumberStart, wrapperStructHand.scalesNumber, wrapperStructHand.scaleRange
                    );
                    spWPoses.at(gpuId).emplace_back(std::make_shared<WHandExtractorNet<TDatumsPtr>>(handExtractor));
                }
            }

            // Hand renderer(s)
            std::vector<TWorker> cpuRenderers;
            if (displayGui)
            {
                // Construct hand renderer
                const auto handRenderer = std::make_shared<HandCpuRenderer>(wrapperStructHand.renderThreshold,
                                                                            wrapperStructHand.alphaKeypoint,
                                                                            wrapperStructHand.alphaHeatMap);
                // Add worker
                cpuRenderers.emplace_back(std::make_shared<WHandRenderer<TDatumsPtr>>(handRenderer));
            }

            // Itermediate workers (e.g., OpenPose format to cv::Mat, json & frames recorder, ...)
            mPostProcessingWs.clear();
            // Frame buffer and ordering
            if (spWPoses.size() > 1)
                mPostProcessingWs.emplace_back(std::make_shared<WQueueOrderer<TDatumsPtr>>());
            // Frames processor (OpenPose format -> cv::Mat format)
            if (displayGui)
            {
                mPostProcessingWs = mergeWorkers(mPostProcessingWs, cpuRenderers);
                const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>();
                mPostProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsPtr>>(opOutputToCvMat));
            }
            // Re-scale pose if desired
            if (wrapperStructPose.keypointScaleMode != ScaleMode::InputResolution)
                error("Only wrapperStructPose.keypointScaleMode == ScaleMode::InputResolution.",
                      __LINE__, __FUNCTION__, __FILE__);

            mOutputWs.clear();
            // Write people pose data on disk (json format)
            if (!writeJsonCleaned.empty())
            {
                const auto jsonSaver = std::make_shared<PeopleJsonSaver>(writeJsonCleaned);
                mOutputWs.emplace_back(std::make_shared<WPeopleJsonSaver<TDatumsPtr>>(jsonSaver));
            }
            // Minimal graphical user interface (GUI)
            spWGui = nullptr;
            if (displayGui)
            {
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(gpuNumber, displayGui);
                mOutputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsPtr>>(guiInfoAdder));
                const auto gui = std::make_shared<Gui>(
                    finalOutputSize, false, mThreadManager.getIsRunningSharedPtr()
                );
                spWGui = {std::make_shared<WGui<TDatumsPtr>>(gui)};
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    void WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::exec()
    {
        try
        {
            configureThreadManager();
            mThreadManager.exec();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    void WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::reset()
    {
        try
        {
            mThreadManager.reset();
            // Reset
            wDatumProducer = nullptr;
            spWScaleAndSizeExtractor = nullptr;
            spWCvMatToOpInput = nullptr;
            spWCvMatToOpOutput = nullptr;
            spWPoses.clear();
            mPostProcessingWs.clear();
            mOutputWs.clear();
            spWGui = nullptr;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    void WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::configureThreadManager()
    {
        try
        {
            // Sanity checks
            if (spWCvMatToOpInput == nullptr)
                error("Configure the WrapperHandFromJsonTest class before calling `start()`.",
                      __LINE__, __FUNCTION__, __FILE__);
            if (wDatumProducer == nullptr)
            {
                const auto message = "You need to use the OpenPose default producer.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (mOutputWs.empty() && spWGui == nullptr)
            {
                error("No output selected.", __LINE__, __FUNCTION__, __FILE__);
            }

            // Thread Manager:
            // Clean previous thread manager (avoid configure to crash the program if used more than once)
            mThreadManager.reset();
            auto threadId = 0ull;
            auto queueIn = 0ull;
            auto queueOut = 1ull;
            // If custom user Worker in same thread or producer on same thread
            spWIdGenerator = std::make_shared<WIdGenerator<std::shared_ptr<TDatums>>>();
            // OpenPose producer
            // Thread 0 or 1, queues 0 -> 1
            if (spWCvMatToOpOutput == nullptr)
                mThreadManager.add(threadId++, {wDatumProducer, spWIdGenerator, spWScaleAndSizeExtractor,
                                   spWCvMatToOpInput}, queueIn++, queueOut++);
            else
                mThreadManager.add(threadId++, {wDatumProducer, spWIdGenerator, spWScaleAndSizeExtractor,
                                   spWCvMatToOpInput, spWCvMatToOpOutput}, queueIn++, queueOut++);
            // Pose estimation & rendering
            // Thread 1 or 2...X, queues 1 -> 2, X = 2 + #GPUs
            if (!spWPoses.empty())
            {
                for (auto& wPose : spWPoses)
                    mThreadManager.add(threadId++, wPose, queueIn, queueOut);
                queueIn++;
                queueOut++;
            }
            // If custom user Worker in same thread or producer on same thread
            // Post processing workers + User post processing workers + Output workers
            // Thread 2 or 3, queues 2 -> 3
            mThreadManager.add(threadId++, mergeWorkers(mPostProcessingWs, mOutputWs), queueIn++, queueOut++);
            // OpenPose GUI
            // Thread Y+1, queues Q+1 -> Q+2
            if (spWGui != nullptr)
                mThreadManager.add(threadId++, spWGui, queueIn++, queueOut++);
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TWorker, typename TQueue>
    std::vector<TWorker> WrapperHandFromJsonTest<TDatum, TDatums, TWorker, TQueue>::mergeWorkers(const std::vector<TWorker>& workersA, const std::vector<TWorker>& workersB)
    {
        try
        {
            auto workersToReturn(workersA);
            for (auto& worker : workersB)
                workersToReturn.emplace_back(worker);
            return workersToReturn;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<TWorker>{};
        }
    }
}

#endif // OPENPOSE_WRAPPER_WRAPPER_HAND_FROM_JSON_TEST_HPP
