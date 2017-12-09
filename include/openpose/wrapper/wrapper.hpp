#ifndef OPENPOSE_WRAPPER_WRAPPER_HPP
#define OPENPOSE_WRAPPER_WRAPPER_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/headers.hpp>
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
             typename TWorker = std::shared_ptr<Worker<std::shared_ptr<TDatums>>>,
             typename TQueue = Queue<std::shared_ptr<TDatums>>>
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
         * @param tDatums std::shared_ptr<TDatums> element to be emplaced.
         * @return Boolean specifying whether the tDatums could be emplaced.
         */
        bool tryEmplace(std::shared_ptr<TDatums>& tDatums);

        /**
         * Emplace (move) an element on the first (input) queue.
         * Similar to tryEmplace.
         * However, if the input queue is full, it will wait until it can emplace it.
         * If the Wrapper class is stopped before adding the element, it will return false and not emplace it.
         * @param tDatums std::shared_ptr<TDatums> element to be emplaced.
         * @return Boolean specifying whether the tDatums could be emplaced.
         */
        bool waitAndEmplace(std::shared_ptr<TDatums>& tDatums);

        /**
         * Push (copy) an element on the first (input) queue.
         * Same as tryEmplace, but it copies the data instead of moving it.
         * @param tDatums std::shared_ptr<TDatums> element to be pushed.
         * @return Boolean specifying whether the tDatums could be pushed.
         */
        bool tryPush(const std::shared_ptr<TDatums>& tDatums);

        /**
         * Push (copy) an element on the first (input) queue.
         * Same as waitAndEmplace, but it copies the data instead of moving it.
         * @param tDatums std::shared_ptr<TDatums> element to be pushed.
         * @return Boolean specifying whether the tDatums could be pushed.
         */
        bool waitAndPush(const std::shared_ptr<TDatums>& tDatums);

        /**
         * Pop (retrieve) an element from the last (output) queue.
         * Only valid if ThreadManagerMode::Asynchronous or ThreadManagerMode::AsynchronousOut.
         * If the output queue is empty or the Wrapper was stopped, it will return false and not retrieve it.
         * @param tDatums std::shared_ptr<TDatums> element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool tryPop(std::shared_ptr<TDatums>& tDatums);

        /**
         * Pop (retrieve) an element from the last (output) queue.
         * Similar to tryPop.
         * However, if the output queue is empty, it will wait until it can pop an element.
         * If the Wrapper class is stopped before popping the element, it will return false and not retrieve it.
         * @param tDatums std::shared_ptr<TDatums> element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool waitAndPop(std::shared_ptr<TDatums>& tDatums);

    private:
        const ThreadManagerMode mThreadManagerMode;
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
        ThreadManager<std::shared_ptr<TDatums>> mThreadManager;
        bool mUserInputWsOnNewThread;
        bool mUserPostProcessingWsOnNewThread;
        bool mUserOutputWsOnNewThread;
        unsigned long long mThreadId;
        bool mMultiThreadEnabled;
        // Workers
        std::vector<TWorker> mUserInputWs;
        TWorker wDatumProducer;
        TWorker spWIdGenerator;
        TWorker spWScaleAndSizeExtractor;
        TWorker spWCvMatToOpInput;
        TWorker spWCvMatToOpOutput;
        std::vector<std::vector<TWorker>> spWPoses;
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
#include <openpose/core/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/hand/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/producer/headers.hpp>
#include <openpose/experimental/tracking/headers.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    Wrapper<TDatums, TWorker, TQueue>::Wrapper(const ThreadManagerMode threadManagerMode) :
        mThreadManagerMode{threadManagerMode},
        spVideoSeek{std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>()},
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

    template<typename TDatums, typename TWorker, typename TQueue>
    Wrapper<TDatums, TWorker, TQueue>::~Wrapper()
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::disableMultiThreading()
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::setWorkerInput(const TWorker& worker, const bool workerOnNewThread)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::setWorkerPostProcessing(const TWorker& worker,
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::setWorkerOutput(const TWorker& worker, const bool workerOnNewThread)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                      const WrapperStructInput& wrapperStructInput,
                                                      const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, WrapperStructFace{}, WrapperStructHand{},
                      wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                      const WrapperStructFace& wrapperStructFace,
                                                      const WrapperStructInput& wrapperStructInput,
                                                      const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, wrapperStructFace, WrapperStructHand{},
                      wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                      const WrapperStructHand& wrapperStructHand,
                                                      const WrapperStructInput& wrapperStructInput,
                                                      const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            configure(wrapperStructPose, WrapperStructFace{}, wrapperStructHand,
                      wrapperStructInput, wrapperStructOutput);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperStructPose& wrapperStructPose,
                                                      const WrapperStructFace& wrapperStructFace,
                                                      const WrapperStructHand& wrapperStructHand,
                                                      const WrapperStructInput& wrapperStructInput,
                                                      const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Shortcut
            typedef std::shared_ptr<TDatums> TDatumsPtr;

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
            if (wrapperStructPose.alphaKeypoint < 0. || wrapperStructPose.alphaKeypoint > 1.
                || wrapperStructFace.alphaHeatMap < 0. || wrapperStructFace.alphaHeatMap > 1.
                || wrapperStructHand.alphaHeatMap < 0. || wrapperStructHand.alphaHeatMap > 1.)
                error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
            if (wrapperStructPose.scaleGap <= 0.f && wrapperStructPose.scalesNumber > 1)
                error("The scale gap must be greater than 0 (it has no effect if the number of scales is 1).",
                      __LINE__, __FUNCTION__, __FILE__);
            if (!renderOutput && (!wrapperStructOutput.writeImages.empty() || !wrapperStructOutput.writeVideo.empty()))
            {
                const auto message = "In order to save the rendered frames (`write_images` or `write_video`), you must"
                                     " set `render_output` to true.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (!wrapperStructOutput.writeHeatMaps.empty() && wrapperStructPose.heatMapTypes.empty())
            {
                const auto message = "In order to save the heatmaps (`write_heatmaps`), you need to pick which heat"
                                     " maps you want to save: `heatmaps_add_X` flags or fill the"
                                     " wrapperStructPose.heatMapTypes.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (!wrapperStructOutput.writeHeatMaps.empty()
                && (wrapperStructPose.heatMapScale != ScaleMode::UnsignedChar &&
                        wrapperStructOutput.writeHeatMapsFormat != "float"))
            {
                const auto message = "In order to save the heatmaps, you must either set"
                                     " wrapperStructPose.heatMapScale to ScaleMode::UnsignedChar (i.e. range [0, 255])"
                                     " or `write_heatmaps_format` to `float` to storage floating numbers in binary"
                                     " mode.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (mUserOutputWs.empty() && mThreadManagerMode != ThreadManagerMode::Asynchronous
                && mThreadManagerMode != ThreadManagerMode::AsynchronousOut)
            {
                const std::string additionalMessage{
                    " You could also set mThreadManagerMode = mThreadManagerMode::Asynchronous(Out) and/or add your"
                    " own output worker class before calling this function."
                };
                const auto savingSomething = (
                    !wrapperStructOutput.writeImages.empty() || !wrapperStructOutput.writeVideo.empty()
                        || !wrapperStructOutput.writeKeypoint.empty() || !wrapperStructOutput.writeKeypointJson.empty()
                        || !wrapperStructOutput.writeCocoJson.empty() || !wrapperStructOutput.writeHeatMaps.empty()
                );
                if (!wrapperStructOutput.displayGui && !savingSomething)
                {
                    const auto message = "No output is selected (`no_display`) and no results are generated (no"
                                         " `write_X` flags enabled). Thus, no output would be generated."
                                         + additionalMessage;
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                if (wrapperStructInput.framesRepeat && savingSomething)
                {
                    const auto message = "Frames repetition (`frames_repeat`) is enabled as well as some writing"
                                         " function (`write_X`). This program would never stop recording the same"
                                         " frames over and over. Please, disable repetition or remove writing.";
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                // Warnings
                if ((wrapperStructOutput.displayGui && wrapperStructOutput.guiVerbose) && !renderOutput)
                {
                    const auto message = "No render is enabled (e.g. `render_pose 0`), so you might also want to"
                                         " remove the display (set `no_display` or `no_gui_verbose`). If you simply"
                                         " want to use OpenPose to record video/images without keypoints, you only"
                                         " need to set `num_gpu 0`." + additionalMessage;
                    log(message, Priority::High);
                }
                if (wrapperStructInput.realTimeProcessing && savingSomething)
                {
                    const auto message = "Real time processing is enabled as well as some writing function. Thus, some"
                                         " frames might be skipped. Consider disabling real time processing if you"
                                         " intend to save any results.";
                    log(message, Priority::High);
                }
            }
            if (!wrapperStructOutput.writeVideo.empty() && wrapperStructInput.producerSharedPtr == nullptr)
                error("Writting video is only available if the OpenPose producer is used (i.e."
                      " wrapperStructInput.producerSharedPtr cannot be a nullptr).", __LINE__, __FUNCTION__, __FILE__);
            if (!wrapperStructPose.enable)
            {
                if (!wrapperStructFace.enable)
                    error("Body keypoint detection must be enabled.", __LINE__, __FUNCTION__, __FILE__);
                if (wrapperStructHand.enable)
                    error("Body keypoint detection must be enabled in order to run hand keypoint detection.",
                          __LINE__, __FUNCTION__, __FILE__);
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
                log("Auto-detecting GPUs... Detected " + std::to_string(gpuNumber) + " GPU(s), using them all.",
                    Priority::High);
            }

            // Proper format
            const auto writeImagesCleaned = formatAsDirectory(wrapperStructOutput.writeImages);
            const auto writeKeypointCleaned = formatAsDirectory(wrapperStructOutput.writeKeypoint);
            const auto writeKeypointJsonCleaned = formatAsDirectory(wrapperStructOutput.writeKeypointJson);
            const auto writeHeatMapsCleaned = formatAsDirectory(wrapperStructOutput.writeHeatMaps);
            const auto modelFolder = formatAsDirectory(wrapperStructPose.modelFolder);

            // Common parameters
            auto finalOutputSize = wrapperStructPose.outputSize;
            Point<int> producerSize{-1,-1};
            if (wrapperStructInput.producerSharedPtr != nullptr)
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
            if (wrapperStructInput.producerSharedPtr != nullptr)
            {
                const auto datumProducer = std::make_shared<DatumProducer<TDatums>>(
                    wrapperStructInput.producerSharedPtr, wrapperStructInput.frameFirst, wrapperStructInput.frameLast,
                    spVideoSeek
                );
                wDatumProducer = std::make_shared<WDatumProducer<TDatumsPtr, TDatums>>(datumProducer);
            }
            else
                wDatumProducer = nullptr;

            // Get input scales and sizes
            const auto scaleAndSizeExtractor = std::make_shared<ScaleAndSizeExtractor>(
                wrapperStructPose.netInputSize, finalOutputSize, wrapperStructPose.scalesNumber,
                wrapperStructPose.scaleGap
            );
            spWScaleAndSizeExtractor = std::make_shared<WScaleAndSizeExtractor<TDatumsPtr>>(scaleAndSizeExtractor);

            // Input cvMat to OpenPose input & output format
            const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>();
            spWCvMatToOpInput = std::make_shared<WCvMatToOpInput<TDatumsPtr>>(cvMatToOpInput);
            if (renderOutput)
            {
                const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>();
                spWCvMatToOpOutput = std::make_shared<WCvMatToOpOutput<TDatumsPtr>>(cvMatToOpOutput);
            }

            // Pose estimators & renderers
            std::vector<std::shared_ptr<PoseExtractor>> poseExtractors;
            std::vector<std::shared_ptr<PoseGpuRenderer>> poseGpuRenderers;
            std::shared_ptr<PoseCpuRenderer> poseCpuRenderer;
            std::vector<TWorker> cpuRenderers;
            spWPoses.clear();
            spWPoses.resize(gpuNumber);
            if (wrapperStructPose.enable)
            {
                // Pose estimators
                for (auto gpuId = 0; gpuId < gpuNumber; gpuId++)
                    poseExtractors.emplace_back(std::make_shared<PoseExtractorCaffe>(
                        wrapperStructPose.poseModel, modelFolder, gpuId + gpuNumberStart,
                        wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                        wrapperStructPose.enableGoogleLogging
                    ));

                // Pose renderers
                if (renderOutputGpu || wrapperStructPose.renderMode == RenderMode::Cpu)
                {
                    // If wrapperStructPose.renderMode != RenderMode::Gpu but renderOutput, then we create an alpha = 0
                    // pose renderer in order to keep the removing background option
                    const auto alphaKeypoint = (wrapperStructPose.renderMode != RenderMode::None
                                                ? wrapperStructPose.alphaKeypoint : 0.f);
                    const auto alphaHeatMap = (wrapperStructPose.renderMode != RenderMode::None
                                                ? wrapperStructPose.alphaHeatMap : 0.f);
                    // GPU rendering
                    if (renderOutputGpu)
                    {
                        for (const auto& poseExtractor : poseExtractors)
                        {
                            poseGpuRenderers.emplace_back(std::make_shared<PoseGpuRenderer>(
                                wrapperStructPose.poseModel, poseExtractor, wrapperStructPose.renderThreshold,
                                wrapperStructPose.blendOriginalFrame, alphaKeypoint,
                                alphaHeatMap, wrapperStructPose.defaultPartToRender
                            ));
                        }
                    }
                    // CPU rendering
                    if (wrapperStructPose.renderMode == RenderMode::Cpu)
                    {
                        poseCpuRenderer = std::make_shared<PoseCpuRenderer>(wrapperStructPose.poseModel,
                                                                            wrapperStructPose.renderThreshold,
                                                                            wrapperStructPose.blendOriginalFrame);
                        cpuRenderers.emplace_back(std::make_shared<WPoseRenderer<TDatumsPtr>>(poseCpuRenderer));
                    }
                }
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Pose extractor(s)
                spWPoses.resize(poseExtractors.size());
                for (auto i = 0u; i < spWPoses.size(); i++)
                    spWPoses.at(i) = {std::make_shared<WPoseExtractor<TDatumsPtr>>(poseExtractors.at(i))};
            }


            // Face extractor(s)
            if (wrapperStructFace.enable)
            {
                // Face detector
                // OpenPose face detector
                if (wrapperStructPose.enable)
                {
                    const auto faceDetector = std::make_shared<FaceDetector>(wrapperStructPose.poseModel);
                    for (auto gpu = 0u; gpu < spWPoses.size(); gpu++)
                        spWPoses.at(gpu).emplace_back(std::make_shared<WFaceDetector<TDatumsPtr>>(faceDetector));
                }
                // OpenCV face detector
                else
                {
                    log("Body keypoint detection is disabled. Hence, using OpenCV face detector (much less accurate"
                        " but faster).", Priority::High);
                    for (auto gpu = 0u; gpu < spWPoses.size(); gpu++)
                    {
                        // 1 FaceDetectorOpenCV per thread, OpenCV face detector is not thread-safe
                        const auto faceDetectorOpenCV = std::make_shared<FaceDetectorOpenCV>(modelFolder);
                        spWPoses.at(gpu).emplace_back(
                            std::make_shared<WFaceDetectorOpenCV<TDatumsPtr>>(faceDetectorOpenCV)
                        );
                    }
                }
                // Face keypoint extractor
                for (auto gpu = 0u; gpu < spWPoses.size(); gpu++)
                {
                    // Face keypoint extractor
                    const auto netOutputSize = wrapperStructFace.netInputSize;
                    const auto faceExtractor = std::make_shared<FaceExtractorCaffe>(
                        wrapperStructFace.netInputSize, netOutputSize, modelFolder,
                        gpu + gpuNumberStart, wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                        wrapperStructPose.enableGoogleLogging
                    );
                    spWPoses.at(gpu).emplace_back(std::make_shared<WFaceExtractor<TDatumsPtr>>(faceExtractor));
                }
            }

            // Hand extractor(s)
            if (wrapperStructHand.enable)
            {
                const auto handDetector = std::make_shared<HandDetector>(wrapperStructPose.poseModel);
                for (auto gpu = 0u; gpu < spWPoses.size(); gpu++)
                {
                    // Hand detector
                    // If tracking
                    if (wrapperStructHand.tracking)
                        spWPoses.at(gpu).emplace_back(
                            std::make_shared<WHandDetectorTracking<TDatumsPtr>>(handDetector)
                        );
                    // If detection
                    else
                        spWPoses.at(gpu).emplace_back(std::make_shared<WHandDetector<TDatumsPtr>>(handDetector));
                    // Hand keypoint extractor
                    const auto netOutputSize = wrapperStructHand.netInputSize;
                    const auto handExtractor = std::make_shared<HandExtractorCaffe>(
                        wrapperStructHand.netInputSize, netOutputSize, modelFolder,
                        gpu + gpuNumberStart, wrapperStructHand.scalesNumber, wrapperStructHand.scaleRange,
                        wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                        wrapperStructPose.enableGoogleLogging
                    );
                    spWPoses.at(gpu).emplace_back(std::make_shared<WHandExtractor<TDatumsPtr>>(handExtractor));
                    // If tracking
                    if (wrapperStructHand.tracking)
                        spWPoses.at(gpu).emplace_back(std::make_shared<WHandDetectorUpdate<TDatumsPtr>>(handDetector));
                }
            }

            // Pose renderer(s)
            if (!poseGpuRenderers.empty())
                for (auto i = 0u; i < spWPoses.size(); i++)
                    spWPoses.at(i).emplace_back(std::make_shared<WPoseRenderer<TDatumsPtr>>(poseGpuRenderers.at(i)));

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
                    cpuRenderers.emplace_back(std::make_shared<WFaceRenderer<TDatumsPtr>>(faceRenderer));
                }
                // GPU rendering
                else if (wrapperStructFace.renderMode == RenderMode::Gpu)
                {
                    for (auto i = 0u; i < spWPoses.size(); i++)
                    {
                        // Construct face renderer
                        const auto faceRenderer = std::make_shared<FaceGpuRenderer>(wrapperStructFace.renderThreshold,
                                                                                    wrapperStructFace.alphaKeypoint,
                                                                                    wrapperStructFace.alphaHeatMap);
                        // Performance boost -> share spGpuMemory for all renderers
                        if (!poseGpuRenderers.empty())
                        {
                            const bool isLastRenderer = !renderHandGpu;
                            const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(poseGpuRenderers.at(i));
                            faceRenderer->setSharedParametersAndIfLast(renderer->getSharedParameters(),
                                                                       isLastRenderer);
                        }
                        // Add worker
                        spWPoses.at(i).emplace_back(std::make_shared<WFaceRenderer<TDatumsPtr>>(faceRenderer));
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
                    cpuRenderers.emplace_back(std::make_shared<WHandRenderer<TDatumsPtr>>(handRenderer));
                }
                // GPU rendering
                else if (wrapperStructHand.renderMode == RenderMode::Gpu)
                {
                    for (auto i = 0u; i < spWPoses.size(); i++)
                    {
                        // Construct hands renderer
                        const auto handRenderer = std::make_shared<HandGpuRenderer>(wrapperStructHand.renderThreshold,
                                                                                    wrapperStructHand.alphaKeypoint,
                                                                                    wrapperStructHand.alphaHeatMap);
                        // Performance boost -> share spGpuMemory for all renderers
                        if (!poseGpuRenderers.empty())
                        {
                            const bool isLastRenderer = true;
                            const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(poseGpuRenderers.at(i));
                            handRenderer->setSharedParametersAndIfLast(renderer->getSharedParameters(),
                                                                       isLastRenderer);
                        }
                        // Add worker
                        spWPoses.at(i).emplace_back(std::make_shared<WHandRenderer<TDatumsPtr>>(handRenderer));
                    }
                }
                else
                    error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
            }

            // Itermediate workers (e.g. OpenPose format to cv::Mat, json & frames recorder, ...)
            mPostProcessingWs.clear();
            // Frame buffer and ordering
            if (spWPoses.size() > 1u)
                mPostProcessingWs.emplace_back(std::make_shared<WQueueOrderer<TDatumsPtr>>());
            // Person ID identification
            if (wrapperStructPose.identification)
            {
                const auto personIdExtractor = std::make_shared<PersonIdExtractor>();
                mPostProcessingWs.emplace_back(std::make_shared<WPersonIdExtractor<TDatumsPtr>>(personIdExtractor));
            }
            // Frames processor (OpenPose format -> cv::Mat format)
            if (renderOutput)
            {
                mPostProcessingWs = mergeVectors(mPostProcessingWs, cpuRenderers);
                const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>();
                mPostProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsPtr>>(opOutputToCvMat));
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
                mPostProcessingWs.emplace_back(std::make_shared<WKeypointScaler<TDatumsPtr>>(keypointScaler));
            }

            mOutputWs.clear();
            // Write people pose data on disk (json for OpenCV >= 3, xml, yml...)
            if (!writeKeypointCleaned.empty())
            {
                const auto keypointSaver = std::make_shared<KeypointSaver>(writeKeypointCleaned,
                                                                           wrapperStructOutput.writeKeypointFormat);
                mOutputWs.emplace_back(std::make_shared<WPoseSaver<TDatumsPtr>>(keypointSaver));
                if (wrapperStructFace.enable)
                    mOutputWs.emplace_back(std::make_shared<WFaceSaver<TDatumsPtr>>(keypointSaver));
                if (wrapperStructHand.enable)
                    mOutputWs.emplace_back(std::make_shared<WHandSaver<TDatumsPtr>>(keypointSaver));
            }
            // Write people pose data on disk (json format)
            if (!writeKeypointJsonCleaned.empty())
            {
                const auto keypointJsonSaver = std::make_shared<KeypointJsonSaver>(writeKeypointJsonCleaned);
                mOutputWs.emplace_back(std::make_shared<WKeypointJsonSaver<TDatumsPtr>>(keypointJsonSaver));
            }
            // Write people pose data on disk (COCO validation json format)
            if (!wrapperStructOutput.writeCocoJson.empty())
            {
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(wrapperStructOutput.writeCocoJson,
                                                                           humanFormat);
                mOutputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsPtr>>(cocoJsonSaver));
            }
            // Write frames as desired image format on hard disk
            if (!writeImagesCleaned.empty())
            {
                const auto imageSaver = std::make_shared<ImageSaver>(writeImagesCleaned,
                                                                     wrapperStructOutput.writeImagesFormat);
                mOutputWs.emplace_back(std::make_shared<WImageSaver<TDatumsPtr>>(imageSaver));
            }
            // Write frames as *.avi video on hard disk
            if (!wrapperStructOutput.writeVideo.empty() && wrapperStructInput.producerSharedPtr != nullptr)
            {
                if (finalOutputSize.x <= 0 || finalOutputSize.y <= 0)
                    error("Video can only be recorded if outputSize is fixed (e.g. video, webcam, IP camera),"
                          "but not for a image directory.", __LINE__, __FUNCTION__, __FILE__);
                const auto originalVideoFps = (wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FPS) > 0.
                                               ? wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FPS) : 30.);
                const auto videoSaver = std::make_shared<VideoSaver>(
                    wrapperStructOutput.writeVideo, CV_FOURCC('M','J','P','G'), originalVideoFps, finalOutputSize
                );
                mOutputWs.emplace_back(std::make_shared<WVideoSaver<TDatumsPtr>>(videoSaver));
            }
            // Write heat maps as desired image format on hard disk
            if (!writeHeatMapsCleaned.empty())
            {
                const auto heatMapSaver = std::make_shared<HeatMapSaver>(writeHeatMapsCleaned,
                                                                         wrapperStructOutput.writeHeatMapsFormat);
                mOutputWs.emplace_back(std::make_shared<WHeatMapSaver<TDatumsPtr>>(heatMapSaver));
            }
            // Add frame information for GUI
            // If this WGuiInfoAdder instance is placed before the WImageSaver or WVideoSaver, then the resulting
            // recorded frames will look exactly as the final displayed image by the GUI
            if (wrapperStructOutput.guiVerbose && (wrapperStructOutput.displayGui || !mUserOutputWs.empty()
                                                   || mThreadManagerMode == ThreadManagerMode::Asynchronous
                                                   || mThreadManagerMode == ThreadManagerMode::AsynchronousOut))
            {
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(gpuNumber, wrapperStructOutput.displayGui);
                mOutputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsPtr>>(guiInfoAdder));
            }
            // Minimal graphical user interface (GUI)
            spWGui = nullptr;
            if (wrapperStructOutput.displayGui)
            {
                // PoseRenderers to Renderers
                std::vector<std::shared_ptr<Renderer>> renderers;
                if (wrapperStructPose.renderMode == RenderMode::Cpu)
                    renderers.emplace_back(std::static_pointer_cast<Renderer>(poseCpuRenderer));
                else
                    for (const auto& poseGpuRenderer : poseGpuRenderers)
                        renderers.emplace_back(std::static_pointer_cast<Renderer>(poseGpuRenderer));
                // Gui
                const auto gui = std::make_shared<Gui>(
                    finalOutputSize, wrapperStructOutput.fullScreen, mThreadManager.getIsRunningSharedPtr(),
                    spVideoSeek, poseExtractors, renderers
                );
                // WGui
                spWGui = {std::make_shared<WGui<TDatumsPtr>>(gui)};
            }
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::exec()
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::start()
    {
        try
        {
            configureThreadManager();
            mThreadManager.start();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::stop()
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::isRunning() const
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::tryEmplace(std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::waitAndEmplace(std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::tryPush(const std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::waitAndPush(const std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::tryPop(std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    bool Wrapper<TDatums, TWorker, TQueue>::waitAndPop(std::shared_ptr<TDatums>& tDatums)
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::reset()
    {
        try
        {
            mThreadManager.reset();
            mThreadId = 0ull;
            // Reset
            mUserInputWs.clear();
            wDatumProducer = nullptr;
            spWScaleAndSizeExtractor = nullptr;
            spWCvMatToOpInput = nullptr;
            spWCvMatToOpOutput = nullptr;
            spWPoses.clear();
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

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configureThreadManager()
    {
        try
        {
            // The less number of queues -> the less lag

            // Security checks
            if (spWScaleAndSizeExtractor == nullptr || spWCvMatToOpInput == nullptr)
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

            // Thread Manager:
            // Clean previous thread manager (avoid configure to crash the program if used more than once)
            mThreadManager.reset();
            mThreadId = 0ull;
            auto queueIn = 0ull;
            auto queueOut = 1ull;
            // If custom user Worker and uses its own thread
            spWIdGenerator = std::make_shared<WIdGenerator<std::shared_ptr<TDatums>>>();
            if (!mUserInputWs.empty() && mUserInputWsOnNewThread)
            {
                // Thread 0, queues 0 -> 1
                mThreadManager.add(mThreadId, mUserInputWs, queueIn++, queueOut++);
                threadIdPP();
                // Thread 1, queues 1 -> 2
                if (spWCvMatToOpOutput == nullptr)
                    mThreadManager.add(mThreadId, {spWIdGenerator, spWScaleAndSizeExtractor, spWCvMatToOpInput},
                                       queueIn++, queueOut++);
                else
                    mThreadManager.add(mThreadId, {spWIdGenerator, spWScaleAndSizeExtractor, spWCvMatToOpInput,
                                       spWCvMatToOpOutput}, queueIn++, queueOut++);
            }
            // If custom user Worker in same thread or producer on same thread
            else
            {
                std::vector<TWorker> workersAux;
                // Custom user Worker
                if (!mUserInputWs.empty())
                    workersAux = mergeVectors(workersAux, mUserInputWs);
                // OpenPose producer
                else if (wDatumProducer != nullptr)
                    workersAux = mergeVectors(workersAux, {wDatumProducer});
                // Otherwise
                else if (mThreadManagerMode != ThreadManagerMode::Asynchronous
                            && mThreadManagerMode != ThreadManagerMode::AsynchronousIn)
                    error("No input selected.", __LINE__, __FUNCTION__, __FILE__);

                if (spWCvMatToOpOutput == nullptr)
                    workersAux = mergeVectors(workersAux, {spWIdGenerator, spWScaleAndSizeExtractor,
                                                           spWCvMatToOpInput});
                else
                    workersAux = mergeVectors(workersAux, {spWIdGenerator, spWScaleAndSizeExtractor,
                                                           spWCvMatToOpInput, spWCvMatToOpOutput});
                // Thread 0 or 1, queues 0 -> 1
                mThreadManager.add(mThreadId, workersAux, queueIn++, queueOut++);
            }
            threadIdPP();
            // Pose estimation & rendering
            // Thread 1 or 2...X, queues 1 -> 2, X = 2 + #GPUs
            if (!spWPoses.empty())
            {
                if (mMultiThreadEnabled)
                {
                    for (auto& wPose : spWPoses)
                    {
                        mThreadManager.add(mThreadId, wPose, queueIn, queueOut);
                        threadIdPP();
                    }
                }
                else
                {
                    if (spWPoses.size() > 1)
                        log("Multi-threading disabled, only 1 thread running. All GPUs have been disabled but the"
                            " first one, which is defined by gpuNumberStart (e.g. in the OpenPose demo, it is set"
                            " with the `num_gpu_start` flag).", Priority::High);
                    mThreadManager.add(mThreadId, spWPoses.at(0), queueIn, queueOut);
                }
                queueIn++;
                queueOut++;
            }
            // If custom user Worker and uses its own thread
            if (!mUserPostProcessingWs.empty() && mUserPostProcessingWsOnNewThread)
            {
                // Post processing workers
                if (!mPostProcessingWs.empty())
                {
                    // Thread 2 or 3, queues 2 -> 3
                    mThreadManager.add(mThreadId, mPostProcessingWs, queueIn++, queueOut++);
                    threadIdPP();
                }
                // User processing workers
                // Thread 3 or 4, queues 3 -> 4
                mThreadManager.add(mThreadId, mUserPostProcessingWs, queueIn++, queueOut++);
                threadIdPP();
                // Output workers
                if (!mOutputWs.empty())
                {
                    // Thread 4 or 5, queues 4 -> 5
                    mThreadManager.add(mThreadId, mOutputWs, queueIn++, queueOut++);
                    threadIdPP();
                }
            }
            // If custom user Worker in same thread or producer on same thread
            else
            {
                // Post processing workers + User post processing workers + Output workers
                auto workersAux = mergeVectors(mPostProcessingWs, mUserPostProcessingWs);
                workersAux = mergeVectors(workersAux, mOutputWs);
                if (!workersAux.empty())
                {
                    // Thread 2 or 3, queues 2 -> 3
                    mThreadManager.add(mThreadId, workersAux, queueIn++, queueOut++);
                    threadIdPP();
                }
            }
            // User output worker
            // Thread Y, queues Q -> Q+1
            if (!mUserOutputWs.empty())
            {
                if (mUserOutputWsOnNewThread)
                {
                    mThreadManager.add(mThreadId, mUserOutputWs, queueIn++, queueOut++);
                    threadIdPP();
                }
                else
                    mThreadManager.add(mThreadId-1, mUserOutputWs, queueIn++, queueOut++);
            }
            // OpenPose GUI
            if (spWGui != nullptr)
            {
                // Thread Y+1, queues Q+1 -> Q+2
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

    template<typename TDatums, typename TWorker, typename TQueue>
    unsigned long long Wrapper<TDatums, TWorker, TQueue>::threadIdPP()
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
