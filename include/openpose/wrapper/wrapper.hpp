#ifndef OPENPOSE_WRAPPER_WRAPPER_HPP
#define OPENPOSE_WRAPPER_WRAPPER_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/headers.hpp>
#include <openpose/wrapper/wrapperStructExtra.hpp>
#include <openpose/wrapper/wrapperStructFace.hpp>
#include <openpose/wrapper/wrapperStructHand.hpp>
#include <openpose/wrapper/wrapperStructInput.hpp>
#include <openpose/wrapper/wrapperStructOutput.hpp>
#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    /**
     * Wrapper: OpenPose all-in-one wrapper template class.
     * Wrapper allows the user to set up the input (video, webcam, custom input, etc.), pose, face and/or hands
     * estimation and rendering, and output (integrated small GUI, custom output, etc.).
     *
     * This function can be used in 2 ways:
     *     - Synchronous mode: call the full constructor with your desired input and output workers.
     *     - Asynchronous mode: call the empty constructor Wrapper() + use the emplace and pop functions to push the
     *       original frames and retrieve the processed ones.
     *     - Mix of them:
     *         - Synchronous input + asynchronous output: call the constructor Wrapper(ThreadManagerMode::Synchronous,
     *           workersInput, {}, true)
     *         - Asynchronous input + synchronous output: call the constructor
     *           Wrapper(ThreadManagerMode::Synchronous, nullptr, workersOutput, irrelevantBoolean, true)
     */
    template<typename TDatums,
             typename TDatumsSP = std::shared_ptr<TDatums>,
             typename TWorker = std::shared_ptr<Worker<TDatumsSP>>,
             typename TQueue = Queue<TDatumsSP>>
    class Wrapper
    {
    public:
        /**
         * Constructor.
         * @param threadManagerMode Thread syncronization mode. If set to ThreadManagerMode::Synchronous, everything
         * will run inside the Wrapper. If ThreadManagerMode::Synchronous(In/Out), then input (frames producer) and/or
         * output (GUI, writing results, etc.) will be controlled outside the Wrapper class by the user. See
         * ThreadManagerMode for a detailed explanation of when to use each one.
         */
        explicit Wrapper(const ThreadManagerMode threadManagerMode = ThreadManagerMode::Synchronous);

        /**
         * Destructor.
         * It automatically frees resources.
         */
        ~Wrapper();

        /**
         * Disable multi-threading.
         * Useful for debugging and logging, all the Workers will run in the same thread.
         * Note that workerOnNewThread (argument for setWorkerInput, setWorkerPostProcessing and setWorkerOutput) will
         * not make any effect.
         */
        void disableMultiThreading();

        /**
         * Add an user-defined extra Worker as frames generator.
         * @param worker TWorker to be added.
         * @param workerOnNewThread Whether to add this TWorker on a new thread (if it is computationally demanding) or
         * simply reuse existing threads (for light functions). Set to true if the performance time is unknown.
         */
        void setWorkerInput(const TWorker& worker, const bool workerOnNewThread = true);

        /**
         * Add an user-defined extra Worker as frames post-processor.
         * @param worker TWorker to be added.
         * @param workerOnNewThread Whether to add this TWorker on a new thread (if it is computationally demanding) or
         * simply reuse existing threads (for light functions). Set to true if the performance time is unknown.
         */
        void setWorkerPostProcessing(const TWorker& worker, const bool workerOnNewThread = true);

        /**
         * Add an user-defined extra Worker as frames consumer (custom display and/or saving).
         * @param worker TWorker to be added.
         * @param workerOnNewThread Whether to add this TWorker on a new thread (if it is computationally demanding) or
         * simply reuse existing threads (for light functions). Set to true if the performance time is unknown.
         */
        void setWorkerOutput(const TWorker& worker, const bool workerOnNewThread = true);

        // If output is not required, just use this function until the renderOutput argument. Keep the default values
        // for the other parameters in order not to display/save any output.
        void configure(const WrapperStructPose& wrapperStructPose,
                       // Producer: set producerSharedPtr=nullptr or use default WrapperStructInput{} to disable input
                       const WrapperStructInput& wrapperStructInput,
                       // Consumer (keep default values to disable any output)
                       const WrapperStructOutput& wrapperStructOutput = WrapperStructOutput{});

        // Similar to the previos configure, but it includes hand extraction and rendering
        void configure(const WrapperStructPose& wrapperStructPose,
                       // Hand (use the default WrapperStructHand{} to disable any hand detector)
                       const WrapperStructHand& wrapperStructHand,
                       // Producer: set producerSharedPtr=nullptr or use default WrapperStructInput{} to disable input
                       const WrapperStructInput& wrapperStructInput,
                       // Consumer (keep default values to disable any output)
                       const WrapperStructOutput& wrapperStructOutput = WrapperStructOutput{});

        // Similar to the previos configure, but it includes hand extraction and rendering
        void configure(const WrapperStructPose& wrapperStructPose,
                       // Face (use the default WrapperStructFace{} to disable any face detector)
                       const WrapperStructFace& wrapperStructFace,
                       // Producer: set producerSharedPtr=nullptr or use default WrapperStructInput{} to disable input
                       const WrapperStructInput& wrapperStructInput,
                       // Consumer (keep default values to disable any output)
                       const WrapperStructOutput& wrapperStructOutput = WrapperStructOutput{});

        // Similar to the previos configure, but it includes hand extraction and rendering
        void configure(const WrapperStructPose& wrapperStructPose = WrapperStructPose{},
                       // Face (use the default WrapperStructFace{} to disable any face detector)
                       const WrapperStructFace& wrapperStructFace = WrapperStructFace{},
                       // Hand (use the default WrapperStructHand{} to disable any hand detector)
                       const WrapperStructHand& wrapperStructHand = WrapperStructHand{},
                       // Hand (use the default WrapperStructExtra{} to disable any hand detector)
                       const WrapperStructExtra& wrapperStructExtra = WrapperStructExtra{},
                       // Producer: set producerSharedPtr=nullptr or use default WrapperStructInput{} to disable input
                       const WrapperStructInput& wrapperStructInput = WrapperStructInput{},
                       // Consumer (keep default values to disable any output)
                       const WrapperStructOutput& wrapperStructOutput = WrapperStructOutput{});

        /**
         * Function to start multi-threading.
         * Similar to start(), but exec() blocks the thread that calls the function (it saves 1 thread). Use exec()
         * instead of start() if the calling thread will otherwise be waiting for the Wrapper to end.
         */
        void exec();

        /**
         * Function to start multi-threading.
         * Similar to exec(), but start() does not block the thread that calls the function. It just opens new threads,
         * so it lets the user perform other tasks meanwhile on the calling thread.
         * VERY IMPORTANT NOTE: if the GUI is selected and OpenCV is compiled with Qt support, this option will not
         * work. Qt needs the main thread to plot visual results, so the final GUI (which uses OpenCV) would return an
         * exception similar to: `QMetaMethod::invoke: Unable to invoke methods with return values in queued
         * connections`. Use exec() in that case.
         */
        void start();

        /**
         * Function to stop multi-threading.
         * It can be called internally or externally.
         */
        void stop();

        /**
         * Whether the Wrapper is running.
         * It will return true after exec() or start() and before stop(), and false otherwise.
         * @return Boolean specifying whether the Wrapper is running.
         */
        bool isRunning() const;

        /**
         * Emplace (move) an element on the first (input) queue.
         * Only valid if ThreadManagerMode::Asynchronous or ThreadManagerMode::AsynchronousIn.
         * If the input queue is full or the Wrapper was stopped, it will return false and not emplace it.
         * @param tDatums TDatumsSP element to be emplaced.
         * @return Boolean specifying whether the tDatums could be emplaced.
         */
        bool tryEmplace(TDatumsSP& tDatums);

        /**
         * Emplace (move) an element on the first (input) queue.
         * Similar to tryEmplace.
         * However, if the input queue is full, it will wait until it can emplace it.
         * If the Wrapper class is stopped before adding the element, it will return false and not emplace it.
         * @param tDatums TDatumsSP element to be emplaced.
         * @return Boolean specifying whether the tDatums could be emplaced.
         */
        bool waitAndEmplace(TDatumsSP& tDatums);

        /**
         * Push (copy) an element on the first (input) queue.
         * Same as tryEmplace, but it copies the data instead of moving it.
         * @param tDatums TDatumsSP element to be pushed.
         * @return Boolean specifying whether the tDatums could be pushed.
         */
        bool tryPush(const TDatumsSP& tDatums);

        /**
         * Push (copy) an element on the first (input) queue.
         * Same as waitAndEmplace, but it copies the data instead of moving it.
         * @param tDatums TDatumsSP element to be pushed.
         * @return Boolean specifying whether the tDatums could be pushed.
         */
        bool waitAndPush(const TDatumsSP& tDatums);

        /**
         * Pop (retrieve) an element from the last (output) queue.
         * Only valid if ThreadManagerMode::Asynchronous or ThreadManagerMode::AsynchronousOut.
         * If the output queue is empty or the Wrapper was stopped, it will return false and not retrieve it.
         * @param tDatums TDatumsSP element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool tryPop(TDatumsSP& tDatums);

        /**
         * Pop (retrieve) an element from the last (output) queue.
         * Similar to tryPop.
         * However, if the output queue is empty, it will wait until it can pop an element.
         * If the Wrapper class is stopped before popping the element, it will return false and not retrieve it.
         * @param tDatums TDatumsSP element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool waitAndPop(TDatumsSP& tDatums);

    private:
        const ThreadManagerMode mThreadManagerMode;
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
        bool mConfigured;
        ThreadManager<TDatumsSP> mThreadManager;
        bool mUserInputWsOnNewThread;
        bool mUserPostProcessingWsOnNewThread;
        bool mUserOutputWsOnNewThread;
        unsigned long long mThreadId;
        bool mMultiThreadEnabled;
        // Workers
        std::vector<TWorker> mUserInputWs;
        TWorker wDatumProducer;
        TWorker spWScaleAndSizeExtractor;
        TWorker spWCvMatToOpInput;
        TWorker spWCvMatToOpOutput;
        std::vector<std::vector<TWorker>> spWPoseExtractors;
        std::vector<std::vector<TWorker>> spWPoseTriangulations;
        std::vector<std::vector<TWorker>> spWJointAngleEstimations;
        std::vector<TWorker> mPostProcessingWs;
        std::vector<TWorker> mUserPostProcessingWs;
        std::vector<TWorker> mOutputWs;
        TWorker spWGui;
        std::vector<TWorker> mUserOutputWs;

        /**
         * Frees TWorker variables (private internal function).
         * For most cases, this class is non-necessary, since std::shared_ptr are automatically cleaned on destruction
         * of each class.
         * However, it might be useful if the same Wrapper is gonna be started twice (not recommended on most cases).
         */
        void reset();

        /**
         * Set ThreadManager from TWorkers (private internal function).
         * After any configure() has been called, the TWorkers are initialized. This function resets the ThreadManager
         * and adds them.
         * Common code for start() and exec().
         */
        void configureThreadManager();

        /**
         * Thread ID increase (private internal function).
         * If multi-threading mode, it increases the thread ID.
         * If single-threading mode (for debugging), it does not modify it.
         * Note that mThreadId must be re-initialized to 0 before starting a new Wrapper configuration.
         * @return unsigned int with the next thread id value.
         */
        unsigned long long threadIdPP();

        DELETE_COPY(Wrapper);
    };
}





// Implementation
#include <openpose/3d/headers.hpp>
#include <openpose/core/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gpu/gpu.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/hand/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/producer/headers.hpp>
#include <openpose/tracking/headers.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/wrapper/wrapperAuxiliary.hpp>
namespace op
{
    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::Wrapper(const ThreadManagerMode threadManagerMode) :
        mThreadManagerMode{threadManagerMode},
        spVideoSeek{std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>()},
        mConfigured{false},
        mThreadManager{threadManagerMode},
        mMultiThreadEnabled{true}
    {
        try
        {
            // It cannot be directly included in the constructor (compiler error for copying std::atomic)
            spVideoSeek->first = false;
            spVideoSeek->second = 0;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::~Wrapper()
    {
        try
        {
            stop();
            reset();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::disableMultiThreading()
    {
        try
        {
            mMultiThreadEnabled = false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::setWorkerInput(const TWorker& worker,
                                                                      const bool workerOnNewThread)
    {
        try
        {
            mUserInputWs.clear();
            if (worker == nullptr)
                error("Your worker is a nullptr.", __LINE__, __FILE__, __FUNCTION__);
            mUserInputWs.emplace_back(worker);
            mUserInputWsOnNewThread = {workerOnNewThread};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::setWorkerPostProcessing(const TWorker& worker,
                                                                               const bool workerOnNewThread)
    {
        try
        {
            mUserPostProcessingWs.clear();
            if (worker == nullptr)
                error("Your worker is a nullptr.", __LINE__, __FILE__, __FUNCTION__);
            mUserPostProcessingWs.emplace_back(worker);
            mUserPostProcessingWsOnNewThread = {workerOnNewThread};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::setWorkerOutput(const TWorker& worker,
                                                                       const bool workerOnNewThread)
    {
        try
        {
            mUserOutputWs.clear();
            if (worker == nullptr)
                error("Your worker is a nullptr.", __LINE__, __FILE__, __FUNCTION__);
            mUserOutputWs.emplace_back(worker);
            mUserOutputWsOnNewThread = {workerOnNewThread};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                                 const WrapperStructInput& wrapperStructInput,
                                                                 const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, WrapperStructFace{}, WrapperStructHand{},
                      WrapperStructExtra{}, wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                                 const WrapperStructFace& wrapperStructFace,
                                                                 const WrapperStructInput& wrapperStructInput,
                                                                 const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, wrapperStructFace, WrapperStructHand{},
                      WrapperStructExtra{}, wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                                 const WrapperStructHand& wrapperStructHand,
                                                                 const WrapperStructInput& wrapperStructInput,
                                                                 const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, WrapperStructFace{}, wrapperStructHand,
                      WrapperStructExtra{}, wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPoseTemp,
                                                                 const WrapperStructFace& wrapperStructFace,
                                                                 const WrapperStructHand& wrapperStructHand,
                                                                 const WrapperStructExtra& wrapperStructExtra,
                                                                 const WrapperStructInput& wrapperStructInput,
                                                                 const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            auto wrapperStructPose = wrapperStructPoseTemp;
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Required parameters
            const auto renderOutput = wrapperStructPose.renderMode != RenderMode::None
                                        || wrapperStructFace.renderMode != RenderMode::None
                                        || wrapperStructHand.renderMode != RenderMode::None;
            const auto renderOutputGpu = wrapperStructPose.renderMode == RenderMode::Gpu
                                            || wrapperStructFace.renderMode == RenderMode::Gpu
                                            || wrapperStructHand.renderMode == RenderMode::Gpu;
            const auto renderFace = wrapperStructFace.enable && wrapperStructFace.renderMode != RenderMode::None;
            const auto renderHand = wrapperStructHand.enable && wrapperStructHand.renderMode != RenderMode::None;
            const auto renderHandGpu = wrapperStructHand.enable && wrapperStructHand.renderMode == RenderMode::Gpu;

            // Check no wrong/contradictory flags enabled
            const auto userOutputWsEmpty = mUserOutputWs.empty();
            wrapperConfigureSecurityChecks(wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructExtra,
                                           wrapperStructInput, wrapperStructOutput, renderOutput, userOutputWsEmpty,
                                           mThreadManagerMode);

            // Get number threads
            auto numberThreads = wrapperStructPose.gpuNumber;
            auto gpuNumberStart = wrapperStructPose.gpuNumberStart;
            // CPU --> 1 thread or no pose extraction
            if (getGpuMode() == GpuMode::NoGpu)
            {
                numberThreads = (wrapperStructPose.gpuNumber == 0 ? 0 : 1);
                gpuNumberStart = 0;
                // Disabling multi-thread makes the code 400 ms faster (2.3 sec vs. 2.7 in i7-6850K)
                // and fixes the bug that the screen was not properly displayed and only refreshed sometimes
                // Note: The screen bug could be also fixed by using waitKey(30) rather than waitKey(1)
                disableMultiThreading();
            }
            // GPU --> user picks (<= #GPUs)
            else
            {
                // Get total number GPUs
                const auto totalGpuNumber = getGpuNumber();
                // If number GPU < 0 --> set it to all the available GPUs
                if (numberThreads < 0)
                {
                    if (totalGpuNumber <= gpuNumberStart)
                        error("Number of initial GPU (`--number_gpu_start`) must be lower than the total number of"
                              " used GPUs (`--number_gpu`)", __LINE__, __FUNCTION__, __FILE__);
                    numberThreads = totalGpuNumber - gpuNumberStart;
                    // Reset initial GPU to 0 (we want them all)
                    // Logging message
                    log("Auto-detecting all available GPUs... Detected " + std::to_string(totalGpuNumber)
                        + " GPU(s), using " + std::to_string(numberThreads) + " of them starting at GPU "
                        + std::to_string(gpuNumberStart) + ".", Priority::High);
                }
                // Security check
                if (gpuNumberStart + numberThreads > totalGpuNumber)
                    error("Initial GPU selected (`--number_gpu_start`) + number GPUs to use (`--number_gpu`) must"
                          " be lower or equal than the total number of GPUs in your machine ("
                          + std::to_string(gpuNumberStart) + " + "
                          + std::to_string(numberThreads) + " vs. "
                          + std::to_string(totalGpuNumber) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
            }

            // Proper format
            const auto writeImagesCleaned = formatAsDirectory(wrapperStructOutput.writeImages);
            const auto writeKeypointCleaned = formatAsDirectory(wrapperStructOutput.writeKeypoint);
            const auto writeJsonCleaned = formatAsDirectory(wrapperStructOutput.writeJson);
            const auto writeHeatMapsCleaned = formatAsDirectory(wrapperStructOutput.writeHeatMaps);
            const auto modelFolder = formatAsDirectory(wrapperStructPose.modelFolder);

            // Common parameters
            auto finalOutputSize = wrapperStructPose.outputSize;
            Point<int> producerSize{-1,-1};
            const auto oPProducer = (wrapperStructInput.producerSharedPtr != nullptr);
            if (oPProducer)
            {
                // 1. Set producer properties
                const auto displayProducerFpsMode = (wrapperStructInput.realTimeProcessing
                                                      ? ProducerFpsMode::OriginalFps : ProducerFpsMode::RetrievalFps);
                wrapperStructInput.producerSharedPtr->setProducerFpsMode(displayProducerFpsMode);
                wrapperStructInput.producerSharedPtr->set(ProducerProperty::Flip, wrapperStructInput.frameFlip);
                wrapperStructInput.producerSharedPtr->set(ProducerProperty::Rotation, wrapperStructInput.frameRotate);
                wrapperStructInput.producerSharedPtr->set(ProducerProperty::AutoRepeat,
                                                          wrapperStructInput.framesRepeat);
                // 2. Set finalOutputSize
                producerSize = Point<int>{(int)wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FRAME_WIDTH),
                                          (int)wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FRAME_HEIGHT)};
                // Set finalOutputSize to input size if desired
                if (finalOutputSize.x == -1 || finalOutputSize.y == -1)
                    finalOutputSize = producerSize;
            }

            // Producer
            if (oPProducer)
            {
                const auto datumProducer = std::make_shared<DatumProducer<TDatums>>(
                    wrapperStructInput.producerSharedPtr, wrapperStructInput.frameFirst, wrapperStructInput.frameLast,
                    spVideoSeek
                );
                wDatumProducer = std::make_shared<WDatumProducer<TDatumsSP, TDatums>>(datumProducer);
            }
            else
                wDatumProducer = nullptr;

            std::vector<std::shared_ptr<PoseExtractorNet>> poseExtractorNets;
            std::vector<std::shared_ptr<FaceExtractorNet>> faceExtractorNets;
            std::vector<std::shared_ptr<HandExtractorNet>> handExtractorNets;
            std::vector<std::shared_ptr<PoseGpuRenderer>> poseGpuRenderers;
            std::shared_ptr<PoseCpuRenderer> poseCpuRenderer;
            if (numberThreads > 0)
            {
                // Get input scales and sizes
                const auto scaleAndSizeExtractor = std::make_shared<ScaleAndSizeExtractor>(
                    wrapperStructPose.netInputSize, finalOutputSize, wrapperStructPose.scalesNumber,
                    wrapperStructPose.scaleGap
                );
                spWScaleAndSizeExtractor = std::make_shared<WScaleAndSizeExtractor<TDatumsSP>>(scaleAndSizeExtractor);

                // Input cvMat to OpenPose input & output format
                const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(wrapperStructPose.poseModel);
                spWCvMatToOpInput = std::make_shared<WCvMatToOpInput<TDatumsSP>>(cvMatToOpInput);
                if (renderOutput)
                {
                    const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>();
                    spWCvMatToOpOutput = std::make_shared<WCvMatToOpOutput<TDatumsSP>>(cvMatToOpOutput);
                }

                // Pose estimators & renderers
                std::vector<TWorker> cpuRenderers;
                spWPoseExtractors.clear();
                spWPoseExtractors.resize(numberThreads);
                if (wrapperStructPose.enable)
                {
                    // Pose estimators
                    for (auto gpuId = 0; gpuId < numberThreads; gpuId++)
                        poseExtractorNets.emplace_back(std::make_shared<PoseExtractorCaffe>(
                            wrapperStructPose.poseModel, modelFolder, gpuId + gpuNumberStart,
                            wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                            wrapperStructPose.addPartCandidates, wrapperStructPose.enableGoogleLogging
                        ));

                    // Pose renderers
                    if (renderOutputGpu || wrapperStructPose.renderMode == RenderMode::Cpu)
                    {
                        // If wrapperStructPose.renderMode != RenderMode::Gpu but renderOutput, then we create an
                        // alpha = 0 pose renderer in order to keep the removing background option
                        const auto alphaKeypoint = (wrapperStructPose.renderMode != RenderMode::None
                                                    ? wrapperStructPose.alphaKeypoint : 0.f);
                        const auto alphaHeatMap = (wrapperStructPose.renderMode != RenderMode::None
                                                    ? wrapperStructPose.alphaHeatMap : 0.f);
                        // GPU rendering
                        if (renderOutputGpu)
                        {
                            for (const auto& poseExtractorNet : poseExtractorNets)
                            {
                                poseGpuRenderers.emplace_back(std::make_shared<PoseGpuRenderer>(
                                    wrapperStructPose.poseModel, poseExtractorNet, wrapperStructPose.renderThreshold,
                                    wrapperStructPose.blendOriginalFrame, alphaKeypoint,
                                    alphaHeatMap, wrapperStructPose.defaultPartToRender
                                ));
                            }
                        }
                        // CPU rendering
                        if (wrapperStructPose.renderMode == RenderMode::Cpu)
                        {
                            poseCpuRenderer = std::make_shared<PoseCpuRenderer>(
                                wrapperStructPose.poseModel, wrapperStructPose.renderThreshold,
                                wrapperStructPose.blendOriginalFrame, alphaKeypoint, alphaHeatMap,
                                wrapperStructPose.defaultPartToRender);
                            cpuRenderers.emplace_back(std::make_shared<WPoseRenderer<TDatumsSP>>(poseCpuRenderer));
                        }
                    }
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                    // Pose extractor(s)
                    spWPoseExtractors.resize(poseExtractorNets.size());
                    const auto personIdExtractor = (wrapperStructExtra.identification
                        ? std::make_shared<PersonIdExtractor>() : nullptr);
                    // Keep top N people
                    // Added right after PoseExtractorNet to avoid:
                    // 1) Rendering people that are later deleted (wrong visualization).
                    // 2) Processing faces and hands on people that will be deleted (speed up).
                    // 3) Running tracking before deleting the people.
                    // Add KeepTopNPeople for each PoseExtractorNet
                    const auto keepTopNPeople = (wrapperStructPose.numberPeopleMax > 0 ?
                        std::make_shared<KeepTopNPeople>(wrapperStructPose.numberPeopleMax)
                        : nullptr);
                    // Person tracker
                    auto personTrackers = std::make_shared<std::vector<std::shared_ptr<PersonTracker>>>();
                    if (wrapperStructExtra.tracking > -1)
                        personTrackers->emplace_back(
                            std::make_shared<PersonTracker>(wrapperStructExtra.tracking == 0));
                    for (auto i = 0u; i < spWPoseExtractors.size(); i++)
                    {
                        // OpenPose keypoint detector + keepTopNPeople
                        //    + ID extractor (experimental) + tracking (experimental)
                        const auto poseExtractor = std::make_shared<PoseExtractor>(
                            poseExtractorNets.at(i), keepTopNPeople, personIdExtractor, personTrackers,
                            wrapperStructPose.numberPeopleMax, wrapperStructExtra.tracking);
                        spWPoseExtractors.at(i) = {std::make_shared<WPoseExtractor<TDatumsSP>>(poseExtractor)};
                        // // Just OpenPose keypoint detector
                        // spWPoseExtractors.at(i) = {std::make_shared<WPoseExtractorNet<TDatumsSP>>(
                        //     poseExtractorNets.at(i))};
                    }

                    // // (Before tracking / id extractor)
                    // // Added right after PoseExtractorNet to avoid:
                    // // 1) Rendering people that are later deleted (wrong visualization).
                    // // 2) Processing faces and hands on people that will be deleted (speed up).
                    // if (wrapperStructPose.numberPeopleMax > 0)
                    // {
                    //     // Add KeepTopNPeople for each PoseExtractorNet
                    //     const auto keepTopNPeople = std::make_shared<KeepTopNPeople>(
                    //         wrapperStructPose.numberPeopleMax);
                    //     for (auto& wPose : spWPoseExtractors)
                    //         wPose.emplace_back(std::make_shared<WKeepTopNPeople<TDatumsSP>>(keepTopNPeople));
                    // }
                }


                // Face extractor(s)
                if (wrapperStructFace.enable)
                {
                    // Face detector
                    // OpenPose face detector
                    if (wrapperStructPose.enable)
                    {
                        const auto faceDetector = std::make_shared<FaceDetector>(wrapperStructPose.poseModel);
                        for (auto& wPose : spWPoseExtractors)
                            wPose.emplace_back(std::make_shared<WFaceDetector<TDatumsSP>>(faceDetector));
                    }
                    // OpenCV face detector
                    else
                    {
                        log("Body keypoint detection is disabled. Hence, using OpenCV face detector (much less"
                            " accurate but faster).", Priority::High);
                        for (auto& wPose : spWPoseExtractors)
                        {
                            // 1 FaceDetectorOpenCV per thread, OpenCV face detector is not thread-safe
                            const auto faceDetectorOpenCV = std::make_shared<FaceDetectorOpenCV>(modelFolder);
                            wPose.emplace_back(
                                std::make_shared<WFaceDetectorOpenCV<TDatumsSP>>(faceDetectorOpenCV)
                            );
                        }
                    }
                    // Face keypoint extractor
                    for (auto gpu = 0u; gpu < spWPoseExtractors.size(); gpu++)
                    {
                        // Face keypoint extractor
                        const auto netOutputSize = wrapperStructFace.netInputSize;
                        const auto faceExtractorNet = std::make_shared<FaceExtractorCaffe>(
                            wrapperStructFace.netInputSize, netOutputSize, modelFolder,
                            gpu + gpuNumberStart, wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                            wrapperStructPose.enableGoogleLogging
                        );
                        faceExtractorNets.emplace_back(faceExtractorNet);
                        spWPoseExtractors.at(gpu).emplace_back(
                            std::make_shared<WFaceExtractorNet<TDatumsSP>>(faceExtractorNet));
                    }
                }

                // Hand extractor(s)
                if (wrapperStructHand.enable)
                {
                    const auto handDetector = std::make_shared<HandDetector>(wrapperStructPose.poseModel);
                    for (auto gpu = 0u; gpu < spWPoseExtractors.size(); gpu++)
                    {
                        // Hand detector
                        // If tracking
                        if (wrapperStructHand.tracking)
                            spWPoseExtractors.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorTracking<TDatumsSP>>(handDetector)
                            );
                        // If detection
                        else
                            spWPoseExtractors.at(gpu).emplace_back(
                                std::make_shared<WHandDetector<TDatumsSP>>(handDetector));
                        // Hand keypoint extractor
                        const auto netOutputSize = wrapperStructHand.netInputSize;
                        const auto handExtractorNet = std::make_shared<HandExtractorCaffe>(
                            wrapperStructHand.netInputSize, netOutputSize, modelFolder,
                            gpu + gpuNumberStart, wrapperStructHand.scalesNumber, wrapperStructHand.scaleRange,
                            wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                            wrapperStructPose.enableGoogleLogging
                        );
                        handExtractorNets.emplace_back(handExtractorNet);
                        spWPoseExtractors.at(gpu).emplace_back(
                            std::make_shared<WHandExtractorNet<TDatumsSP>>(handExtractorNet)
                            );
                        // If tracking
                        if (wrapperStructHand.tracking)
                            spWPoseExtractors.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorUpdate<TDatumsSP>>(handDetector)
                            );
                    }
                }

                // Pose renderer(s)
                if (!poseGpuRenderers.empty())
                    for (auto i = 0u; i < spWPoseExtractors.size(); i++)
                        spWPoseExtractors.at(i).emplace_back(std::make_shared<WPoseRenderer<TDatumsSP>>(
                            poseGpuRenderers.at(i)
                        ));

                // Face renderer(s)
                if (renderFace)
                {
                    // CPU rendering
                    if (wrapperStructFace.renderMode == RenderMode::Cpu)
                    {
                        // Construct face renderer
                        const auto faceRenderer = std::make_shared<FaceCpuRenderer>(wrapperStructFace.renderThreshold,
                                                                                    wrapperStructFace.alphaKeypoint,
                                                                                    wrapperStructFace.alphaHeatMap);
                        // Add worker
                        cpuRenderers.emplace_back(std::make_shared<WFaceRenderer<TDatumsSP>>(faceRenderer));
                    }
                    // GPU rendering
                    else if (wrapperStructFace.renderMode == RenderMode::Gpu)
                    {
                        for (auto i = 0u; i < spWPoseExtractors.size(); i++)
                        {
                            // Construct face renderer
                            const auto faceRenderer = std::make_shared<FaceGpuRenderer>(
                                wrapperStructFace.renderThreshold, wrapperStructFace.alphaKeypoint,
                                wrapperStructFace.alphaHeatMap
                            );
                            // Performance boost -> share spGpuMemory for all renderers
                            if (!poseGpuRenderers.empty())
                            {
                                const bool isLastRenderer = !renderHandGpu;
                                const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(
                                    poseGpuRenderers.at(i)
                                );
                                faceRenderer->setSharedParametersAndIfLast(renderer->getSharedParameters(),
                                                                           isLastRenderer);
                            }
                            // Add worker
                            spWPoseExtractors.at(i).emplace_back(
                                std::make_shared<WFaceRenderer<TDatumsSP>>(faceRenderer));
                        }
                    }
                    else
                        error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
                }

                // Hand renderer(s)
                if (renderHand)
                {
                    // CPU rendering
                    if (wrapperStructHand.renderMode == RenderMode::Cpu)
                    {
                        // Construct hand renderer
                        const auto handRenderer = std::make_shared<HandCpuRenderer>(wrapperStructHand.renderThreshold,
                                                                                    wrapperStructHand.alphaKeypoint,
                                                                                    wrapperStructHand.alphaHeatMap);
                        // Add worker
                        cpuRenderers.emplace_back(std::make_shared<WHandRenderer<TDatumsSP>>(handRenderer));
                    }
                    // GPU rendering
                    else if (wrapperStructHand.renderMode == RenderMode::Gpu)
                    {
                        for (auto i = 0u; i < spWPoseExtractors.size(); i++)
                        {
                            // Construct hands renderer
                            const auto handRenderer = std::make_shared<HandGpuRenderer>(
                                wrapperStructHand.renderThreshold, wrapperStructHand.alphaKeypoint,
                                wrapperStructHand.alphaHeatMap
                            );
                            // Performance boost -> share spGpuMemory for all renderers
                            if (!poseGpuRenderers.empty())
                            {
                                const bool isLastRenderer = true;
                                const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(
                                    poseGpuRenderers.at(i)
                                    );
                                handRenderer->setSharedParametersAndIfLast(renderer->getSharedParameters(),
                                                                           isLastRenderer);
                            }
                            // Add worker
                            spWPoseExtractors.at(i).emplace_back(
                                std::make_shared<WHandRenderer<TDatumsSP>>(handRenderer));
                        }
                    }
                    else
                        error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
                }

                // 3-D reconstruction
                spWPoseTriangulations.clear();
                if (wrapperStructExtra.reconstruct3d)
                {
                    // For all (body/face/hands): PoseTriangulations ~30 msec, 8 GPUS ~30 msec for keypoint estimation
                    spWPoseTriangulations.resize(fastMax(1, int(spWPoseExtractors.size() / 4)));
                    for (auto i = 0u ; i < spWPoseTriangulations.size() ; i++)
                    {
                        const auto poseTriangulation = std::make_shared<PoseTriangulation>(
                            wrapperStructExtra.minViews3d);
                        spWPoseTriangulations.at(i) = {std::make_shared<WPoseTriangulation<TDatumsSP>>(
                            poseTriangulation)};
                    }
                }
                // Itermediate workers (e.g. OpenPose format to cv::Mat, json & frames recorder, ...)
                mPostProcessingWs.clear();
                // // Person ID identification (when no multi-thread and no dependency on tracking)
                // if (wrapperStructExtra.identification)
                // {
                //     const auto personIdExtractor = std::make_shared<PersonIdExtractor>();
                //     mPostProcessingWs.emplace_back(
                //         std::make_shared<WPersonIdExtractor<TDatumsSP>>(personIdExtractor)
                //     );
                // }
                // Frames processor (OpenPose format -> cv::Mat format)
                if (renderOutput)
                {
                    mPostProcessingWs = mergeVectors(mPostProcessingWs, cpuRenderers);
                    const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>();
                    mPostProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsSP>>(opOutputToCvMat));
                }
                // Re-scale pose if desired
                // If desired scale is not the current input
                if (wrapperStructPose.keypointScale != ScaleMode::InputResolution
                    // and desired scale is not output when size(input) = size(output)
                    && !(wrapperStructPose.keypointScale == ScaleMode::OutputResolution &&
                         (finalOutputSize == producerSize || finalOutputSize.x <= 0 || finalOutputSize.y <= 0))
                    // and desired scale is not net output when size(input) = size(net output)
                    && !(wrapperStructPose.keypointScale == ScaleMode::NetOutputResolution
                         && producerSize == wrapperStructPose.netInputSize))
                {
                    // Then we must rescale the keypoints
                    auto keypointScaler = std::make_shared<KeypointScaler>(wrapperStructPose.keypointScale);
                    mPostProcessingWs.emplace_back(std::make_shared<WKeypointScaler<TDatumsSP>>(keypointScaler));
                }
            }

            // IK/Adam
            const auto displayAdam = wrapperStructOutput.displayMode == DisplayMode::DisplayAdam
                                     || (wrapperStructOutput.displayMode == DisplayMode::DisplayAll
                                         && wrapperStructExtra.ikThreads > 0);
            spWJointAngleEstimations.clear();
#ifdef USE_3D_ADAM_MODEL
            if (wrapperStructExtra.ikThreads > 0)
            {
                spWJointAngleEstimations.resize(wrapperStructExtra.ikThreads);
                // Pose extractor(s)
                for (auto i = 0u; i < spWJointAngleEstimations.size(); i++)
                {
                    const auto jointAngleEstimation = std::make_shared<JointAngleEstimation>(displayAdam);
                    spWJointAngleEstimations.at(i) = {std::make_shared<WJointAngleEstimation<TDatumsSP>>(
                        jointAngleEstimation)};
                }
            }
#endif

            // Output workers
            mOutputWs.clear();
            // Send information (e.g., to Unity) though UDP client-server communication
#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.udpHost.empty() && !wrapperStructOutput.udpPort.empty())
            {
                const auto udpSender = std::make_shared<UdpSender>(wrapperStructOutput.udpHost,
                                                                   wrapperStructOutput.udpPort);
                mOutputWs.emplace_back(std::make_shared<WUdpSender<TDatumsSP>>(udpSender));
            }
#endif
            // Write people pose data on disk (json for OpenCV >= 3, xml, yml...)
            if (!writeKeypointCleaned.empty())
            {
                const auto keypointSaver = std::make_shared<KeypointSaver>(writeKeypointCleaned,
                                                                           wrapperStructOutput.writeKeypointFormat);
                mOutputWs.emplace_back(std::make_shared<WPoseSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructFace.enable)
                    mOutputWs.emplace_back(std::make_shared<WFaceSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructHand.enable)
                    mOutputWs.emplace_back(std::make_shared<WHandSaver<TDatumsSP>>(keypointSaver));
            }
            // Write OpenPose output data on disk in json format (body/hand/face keypoints, body part locations if
            // enabled, etc.)
            if (!writeJsonCleaned.empty())
            {
                const auto peopleJsonSaver = std::make_shared<PeopleJsonSaver>(writeJsonCleaned);
                mOutputWs.emplace_back(std::make_shared<WPeopleJsonSaver<TDatumsSP>>(peopleJsonSaver));
            }
            // Write people pose data on disk (COCO validation json format)
            if (!wrapperStructOutput.writeCocoJson.empty())
            {
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(wrapperStructOutput.writeCocoJson,
                                                                           humanFormat, CocoJsonFormat::Body);
                mOutputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsSP>>(cocoJsonSaver));
            }
            // Write people foot pose data on disk (COCO validation json format for foot data)
            if (!wrapperStructOutput.writeCocoFootJson.empty())
            {
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(wrapperStructOutput.writeCocoFootJson,
                                                                           humanFormat, CocoJsonFormat::Foot);
                mOutputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsSP>>(cocoJsonSaver));
            }
            // Write frames as desired image format on hard disk
            if (!writeImagesCleaned.empty())
            {
                const auto imageSaver = std::make_shared<ImageSaver>(writeImagesCleaned,
                                                                     wrapperStructOutput.writeImagesFormat);
                mOutputWs.emplace_back(std::make_shared<WImageSaver<TDatumsSP>>(imageSaver));
            }
            // Write frames as *.avi video on hard disk
            const auto producerFps = (wrapperStructInput.producerSharedPtr == nullptr ?
                                        0. : wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FPS));
            const auto originalVideoFps = (wrapperStructOutput.writeVideoFps > 0 ?
                                            wrapperStructOutput.writeVideoFps
                                            : producerFps);
            if (!wrapperStructOutput.writeVideo.empty())
            {
                if (!oPProducer)
                    error("Video file can only be recorded inside `wrapper/wrapper.hpp` if the producer"
                          " is one of the default ones (e.g. video, webcam, ...).",
                          __LINE__, __FUNCTION__, __FILE__);
                if (finalOutputSize.x <= 0 || finalOutputSize.y <= 0)
                    error("Video can only be recorded if outputSize is fixed (e.g. video, webcam, IP camera),"
                          "but not for a image directory.", __LINE__, __FUNCTION__, __FILE__);
                const auto videoSaver = std::make_shared<VideoSaver>(
                    wrapperStructOutput.writeVideo, CV_FOURCC('M','J','P','G'), originalVideoFps, finalOutputSize
                );
                mOutputWs.emplace_back(std::make_shared<WVideoSaver<TDatumsSP>>(videoSaver));
            }
            // Write joint angles as *.bvh file on hard disk
#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.writeBvh.empty())
            {
                const auto bvhSaver = std::make_shared<BvhSaver>(
                    wrapperStructOutput.writeBvh, JointAngleEstimation::getTotalModel(), originalVideoFps
                );
                mOutputWs.emplace_back(std::make_shared<WBvhSaver<TDatumsSP>>(bvhSaver));
            }
#endif
            // Write heat maps as desired image format on hard disk
            if (!writeHeatMapsCleaned.empty())
            {
                const auto heatMapSaver = std::make_shared<HeatMapSaver>(writeHeatMapsCleaned,
                                                                         wrapperStructOutput.writeHeatMapsFormat);
                mOutputWs.emplace_back(std::make_shared<WHeatMapSaver<TDatumsSP>>(heatMapSaver));
            }
            // Add frame information for GUI
            const bool guiEnabled = (wrapperStructOutput.displayMode != DisplayMode::NoDisplay);
            // If this WGuiInfoAdder instance is placed before the WImageSaver or WVideoSaver, then the resulting
            // recorded frames will look exactly as the final displayed image by the GUI
            if (wrapperStructOutput.guiVerbose && (guiEnabled || !mUserOutputWs.empty()
                                                   || mThreadManagerMode == ThreadManagerMode::Asynchronous
                                                   || mThreadManagerMode == ThreadManagerMode::AsynchronousOut))
            {
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(numberThreads, guiEnabled);
                mOutputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsSP>>(guiInfoAdder));
            }
            // Minimal graphical user interface (GUI)
            spWGui = nullptr;
            if (guiEnabled)
            {
                // PoseRenderers to Renderers
                std::vector<std::shared_ptr<Renderer>> renderers;
                if (wrapperStructPose.renderMode == RenderMode::Cpu)
                    renderers.emplace_back(std::static_pointer_cast<Renderer>(poseCpuRenderer));
                else
                    for (const auto& poseGpuRenderer : poseGpuRenderers)
                        renderers.emplace_back(std::static_pointer_cast<Renderer>(poseGpuRenderer));
                // Display
                // Adam (+3-D/2-D) display
                if (displayAdam)
                {
#ifdef USE_3D_ADAM_MODEL
                    // Gui
                    const auto gui = std::make_shared<GuiAdam>(
                        finalOutputSize, wrapperStructOutput.fullScreen, mThreadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers,
                        wrapperStructOutput.displayMode, JointAngleEstimation::getTotalModel(),
                        wrapperStructOutput.writeVideoAdam
                    );
                    // WGui
                    spWGui = {std::make_shared<WGuiAdam<TDatumsSP>>(gui)};
#endif
                }
                // 3-D (+2-D) display
                else if (wrapperStructOutput.displayMode == DisplayMode::Display3D
                    || wrapperStructOutput.displayMode == DisplayMode::DisplayAll)
                {
                    // Gui
                    const auto gui = std::make_shared<Gui3D>(
                        finalOutputSize, wrapperStructOutput.fullScreen, mThreadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers,
                        wrapperStructPose.poseModel, wrapperStructOutput.displayMode
                    );
                    // WGui
                    spWGui = {std::make_shared<WGui3D<TDatumsSP>>(gui)};
                }
                // 2-D display
                else if (wrapperStructOutput.displayMode == DisplayMode::Display2D)
                {
                    // Gui
                    const auto gui = std::make_shared<Gui>(
                        finalOutputSize, wrapperStructOutput.fullScreen, mThreadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers
                    );
                    // WGui
                    spWGui = {std::make_shared<WGui<TDatumsSP>>(gui)};
                }
                else
                    error("Unknown DisplayMode.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Set wrapper as configured
            mConfigured = true;
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::exec()
    {
        try
        {
            configureThreadManager();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            mThreadManager.exec();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::start()
    {
        try
        {
            configureThreadManager();
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            mThreadManager.start();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::stop()
    {
        try
        {
            mThreadManager.stop();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::isRunning() const
    {
        try
        {
            return mThreadManager.isRunning();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::tryEmplace(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserInputWs.empty())
                error("Emplace cannot be called if an input worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.tryEmplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::waitAndEmplace(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserInputWs.empty())
                error("Emplace cannot be called if an input worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.waitAndEmplace(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::tryPush(const TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserInputWs.empty())
                error("Push cannot be called if an input worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.tryPush(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::waitAndPush(const TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserInputWs.empty())
                error("Push cannot be called if an input worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.waitAndPush(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::tryPop(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserOutputWs.empty())
                error("Pop cannot be called if an output worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.tryPop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::waitAndPop(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserOutputWs.empty())
                error("Pop cannot be called if an output worker was already selected.",
                      __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.waitAndPop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::reset()
    {
        try
        {
            mConfigured = false;
            mThreadManager.reset();
            mThreadId = 0ull;
            // Reset
            mUserInputWs.clear();
            wDatumProducer = nullptr;
            spWScaleAndSizeExtractor = nullptr;
            spWCvMatToOpInput = nullptr;
            spWCvMatToOpOutput = nullptr;
            spWPoseExtractors.clear();
            spWPoseTriangulations.clear();
            spWJointAngleEstimations.clear();
            mPostProcessingWs.clear();
            mUserPostProcessingWs.clear();
            mOutputWs.clear();
            spWGui = nullptr;
            mUserOutputWs.clear();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::configureThreadManager()
    {
        try
        {
            // The less number of queues -> the less threads opened, and potentially the less lag

            // Security checks
            if (!mConfigured)
                error("Configure the Wrapper class before calling `start()`.", __LINE__, __FUNCTION__, __FILE__);
            if ((wDatumProducer == nullptr) == (mUserInputWs.empty())
                && mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
            {
                const auto message = "You need to have 1 and only 1 producer selected. You can introduce your own"
                                     " producer by using setWorkerInput() or use the OpenPose default producer by"
                                     " configuring it in the configure function) or use the"
                                     " ThreadManagerMode::Asynchronous(In) mode.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (mOutputWs.empty() && mUserOutputWs.empty() && spWGui == nullptr
                && mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousOut)
            {
                error("No output selected.", __LINE__, __FUNCTION__, __FILE__);
            }

            // Thread Manager
            // Clean previous thread manager (avoid configure to crash the program if used more than once)
            mThreadManager.reset();
            mThreadId = 0ull;
            auto queueIn = 0ull;
            auto queueOut = 1ull;
            // After producer
            // ID generator (before any multi-threading or any function that requires the ID)
            const auto wIdGenerator = std::make_shared<WIdGenerator<TDatumsSP>>();
            std::vector<TWorker> workersAux{wIdGenerator};
            // Scale & cv::Mat to OP format
            if (spWScaleAndSizeExtractor != nullptr)
                workersAux = mergeVectors(workersAux, {spWScaleAndSizeExtractor});
            if (spWCvMatToOpInput != nullptr)
                workersAux = mergeVectors(workersAux, {spWCvMatToOpInput});
            // cv::Mat to output format
            if (spWCvMatToOpOutput != nullptr)
                workersAux = mergeVectors(workersAux, {spWCvMatToOpOutput});

            // Producer
            // If custom user Worker and uses its own thread
            if (!mUserInputWs.empty() && mUserInputWsOnNewThread)
            {
                // Thread 0, queues 0 -> 1
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                mThreadManager.add(mThreadId, mUserInputWs, queueIn++, queueOut++);
                threadIdPP();
            }
            // If custom user Worker in same thread
            else if (!mUserInputWs.empty())
                workersAux = mergeVectors(mUserInputWs, workersAux);
            // If OpenPose producer (same thread)
            else if (wDatumProducer != nullptr)
                workersAux = mergeVectors({wDatumProducer}, workersAux);
            // Otherwise
            else if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                        && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("No input selected.", __LINE__, __FUNCTION__, __FILE__);
            // Thread 0 or 1, queues 0 -> 1
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            mThreadManager.add(mThreadId, workersAux, queueIn++, queueOut++);
            // Increase thread
            threadIdPP();

            // Pose estimation & rendering
            // Thread 1 or 2...X, queues 1 -> 2, X = 2 + #GPUs
            if (!spWPoseExtractors.empty())
            {
                if (mMultiThreadEnabled)
                {
                    for (auto& wPose : spWPoseExtractors)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wPose, queueIn, queueOut);
                        threadIdPP();
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames - Required own thread
                    if (spWPoseExtractors.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP();
                    }
                }
                else
                {
                    if (spWPoseExtractors.size() > 1)
                        log("Multi-threading disabled, only 1 thread running. All GPUs have been disabled but the"
                            " first one, which is defined by gpuNumberStart (e.g. in the OpenPose demo, it is set"
                            " with the `--num_gpu_start` flag).", Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId, spWPoseExtractors.at(0), queueIn++, queueOut++);
                }
            }
            // Assemble all frames from same time instant (3-D module)
            const auto wQueueAssembler = std::make_shared<WQueueAssembler<TDatumsSP, TDatums>>();
            // 3-D reconstruction
            if (!spWPoseTriangulations.empty())
            {
                // Assemble frames
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                mThreadManager.add(mThreadId, wQueueAssembler, queueIn++, queueOut++);
                threadIdPP();
                // 3-D reconstruction
                if (mMultiThreadEnabled)
                {
                    for (auto& wPoseTriangulations : spWPoseTriangulations)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wPoseTriangulations, queueIn, queueOut);
                        threadIdPP();
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (spWPoseTriangulations.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP();
                    }
                }
                else
                {
                    if (spWPoseTriangulations.size() > 1)
                        log("Multi-threading disabled, only 1 thread running for 3-D triangulation.",
                            Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId, spWPoseTriangulations.at(0), queueIn++, queueOut++);
                }
            }
            else
                mPostProcessingWs = mergeVectors({wQueueAssembler}, mPostProcessingWs);
            // Adam/IK step
            if (!spWJointAngleEstimations.empty())
            {
                if (mMultiThreadEnabled)
                {
                    for (auto& wJointAngleEstimator : spWJointAngleEstimations)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wJointAngleEstimator, queueIn, queueOut);
                        threadIdPP();
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (spWJointAngleEstimations.size() > 1)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        mThreadManager.add(mThreadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP();
                    }
                }
                else
                {
                    if (spWJointAngleEstimations.size() > 1)
                        log("Multi-threading disabled, only 1 thread running for joint angle estimation.",
                            Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId, spWJointAngleEstimations.at(0), queueIn++, queueOut++);
                }
            }
            // Post processing workers
            if (!mPostProcessingWs.empty())
            {
                // Combining mPostProcessingWs and mOutputWs
                mOutputWs = mergeVectors(mPostProcessingWs, mOutputWs);
                // // If I wanna split them
                // log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // mThreadManager.add(mThreadId, mPostProcessingWs, queueIn++, queueOut++);
                // threadIdPP();
            }
            // If custom user Worker and uses its own thread
            if (!mUserPostProcessingWs.empty())
            {
                // If custom user Worker in its own thread
                if (mUserPostProcessingWsOnNewThread)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId, mUserPostProcessingWs, queueIn++, queueOut++);
                    threadIdPP();
                }
                // If custom user Worker in same thread
                // Merge with mOutputWs
                else
                    mOutputWs = mergeVectors(mOutputWs, mUserPostProcessingWs);
            }
            // Output workers
            if (!mOutputWs.empty())
            {
                // Thread 4 or 5, queues 4 -> 5
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                mThreadManager.add(mThreadId, mOutputWs, queueIn++, queueOut++);
                threadIdPP();
            }
            // User output worker
            // Thread Y, queues Q -> Q+1
            if (!mUserOutputWs.empty())
            {
                if (mUserOutputWsOnNewThread)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId, mUserOutputWs, queueIn++, queueOut++);
                    threadIdPP();
                }
                else
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    mThreadManager.add(mThreadId-1, mUserOutputWs, queueIn++, queueOut++);
                }
            }
            // OpenPose GUI
            if (spWGui != nullptr)
            {
                // Thread Y+1, queues Q+1 -> Q+2
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                mThreadManager.add(mThreadId, spWGui, queueIn++, queueOut++);
                threadIdPP();
            }
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker, typename TQueue>
    unsigned long long Wrapper<TDatums, TDatumsSP, TWorker, TQueue>::threadIdPP()
    {
        try
        {
            if (mMultiThreadEnabled)
                mThreadId++;
            return mThreadId;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0ull;
        }
    }

    extern template class Wrapper<DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE_WRAPPER_WRAPPER_HPP
