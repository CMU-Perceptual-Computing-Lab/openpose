#ifndef OPENPOSE_WRAPPER_WRAPPER_HPP
#define OPENPOSE_WRAPPER_WRAPPER_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/headers.hpp>
#include <openpose/wrapper/enumClasses.hpp>
#include <openpose/wrapper/wrapperStructExtra.hpp>
#include <openpose/wrapper/wrapperStructFace.hpp>
#include <openpose/wrapper/wrapperStructHand.hpp>
#include <openpose/wrapper/wrapperStructInput.hpp>
#include <openpose/wrapper/wrapperStructOutput.hpp>
#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    /**
     * WrapperT: OpenPose all-in-one wrapper template class. Simplified into Wrapper for WrapperT<std::vector<Datum>>
     * WrapperT allows the user to set up the input (video, webcam, custom input, etc.), pose, face and/or hands
     * estimation and rendering, and output (integrated small GUI, custom output, etc.).
     *
     * This function can be used in 2 ways:
     *     - Synchronous mode: call the full constructor with your desired input and output workers.
     *     - Asynchronous mode: call the empty constructor WrapperT() + use the emplace and pop functions to push the
     *       original frames and retrieve the processed ones.
     *     - Mix of them:
     *         - Synchronous input + asynchronous output: call the constructor WrapperT(ThreadManagerMode::Synchronous,
     *           workersInput, {}, true)
     *         - Asynchronous input + synchronous output: call the constructor
     *           WrapperT(ThreadManagerMode::Synchronous, nullptr, workersOutput, irrelevantBoolean, true)
     */
    template<typename TDatums = std::vector<Datum>,
             typename TDatumsSP = std::shared_ptr<TDatums>,
             typename TWorker = std::shared_ptr<Worker<TDatumsSP>>>
    class WrapperT
    {
    public:
        /**
         * Constructor.
         * @param threadManagerMode Thread syncronization mode. If set to ThreadManagerMode::Synchronous, everything
         * will run inside the WrapperT. If ThreadManagerMode::Synchronous(In/Out), then input (frames producer) and/or
         * output (GUI, writing results, etc.) will be controlled outside the WrapperT class by the user. See
         * ThreadManagerMode for a detailed explanation of when to use each one.
         */
        explicit WrapperT(const ThreadManagerMode threadManagerMode = ThreadManagerMode::Synchronous);

        /**
         * Destructor.
         * It automatically frees resources.
         */
        virtual ~WrapperT();

        /**
         * Disable multi-threading.
         * Useful for debugging and logging, all the Workers will run in the same thread.
         * Note that workerOnNewThread (argument for setWorker function) will not make any effect.
         */
        void disableMultiThreading();

        /**
         * Add an user-defined extra Worker for a desired task (input, output, ...).
         * @param workerType WorkerType to configure (e.g., Input, PostProcessing, Output).
         * @param worker TWorker to be added.
         * @param workerOnNewThread Whether to add this TWorker on a new thread (if it is computationally demanding) or
         * simply reuse existing threads (for light functions). Set to true if the performance time is unknown.
         */
        void setWorker(const WorkerType workerType, const TWorker& worker, const bool workerOnNewThread = true);

        /**
         * It configures the pose parameters. Do not call for default values.
         */
        void configure(const WrapperStructPose& wrapperStructPose);

        /**
         * Analogous to configure() but applied to only pose (WrapperStructFace)
         */
        void configure(const WrapperStructFace& wrapperStructFace);

        /**
         * Analogous to configure() but applied to only pose (WrapperStructHand)
         */
        void configure(const WrapperStructHand& wrapperStructHand);

        /**
         * Analogous to configure() but applied to only pose (WrapperStructExtra)
         */
        void configure(const WrapperStructExtra& wrapperStructExtra);

        /**
         * Analogous to configure() but applied to only pose (WrapperStructInput)
         */
        void configure(const WrapperStructInput& wrapperStructInput);

        /**
         * Analogous to configure() but applied to only pose (WrapperStructInput)
         */
        void configure(const WrapperStructOutput& wrapperStructOutput);

        /**
         * Function to start multi-threading.
         * Similar to start(), but exec() blocks the thread that calls the function (it saves 1 thread). Use exec()
         * instead of start() if the calling thread will otherwise be waiting for the WrapperT to end.
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
         * Whether the WrapperT is running.
         * It will return true after exec() or start() and before stop(), and false otherwise.
         * @return Boolean specifying whether the WrapperT is running.
         */
        bool isRunning() const;

        /**
         * Emplace (move) an element on the first (input) queue.
         * Only valid if ThreadManagerMode::Asynchronous or ThreadManagerMode::AsynchronousIn.
         * If the input queue is full or the WrapperT was stopped, it will return false and not emplace it.
         * @param tDatums TDatumsSP element to be emplaced.
         * @return Boolean specifying whether the tDatums could be emplaced.
         */
        bool tryEmplace(TDatumsSP& tDatums);

        /**
         * Emplace (move) an element on the first (input) queue.
         * Similar to tryEmplace.
         * However, if the input queue is full, it will wait until it can emplace it.
         * If the WrapperT class is stopped before adding the element, it will return false and not emplace it.
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
         * If the output queue is empty or the WrapperT was stopped, it will return false and not retrieve it.
         * @param tDatums TDatumsSP element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool tryPop(TDatumsSP& tDatums);

        /**
         * Pop (retrieve) an element from the last (output) queue.
         * Similar to tryPop.
         * However, if the output queue is empty, it will wait until it can pop an element.
         * If the WrapperT class is stopped before popping the element, it will return false and not retrieve it.
         * @param tDatums TDatumsSP element where the retrieved element will be placed.
         * @return Boolean specifying whether the tDatums could be retrieved.
         */
        bool waitAndPop(TDatumsSP& tDatums);

        /**
         * Runs both waitAndEmplace and waitAndPop
         */
        bool emplaceAndPop(TDatumsSP& tDatums);

        /**
         * Runs both waitAndEmplace and waitAndPop
         */
        TDatumsSP emplaceAndPop(const cv::Mat& cvMat);

    private:
        const ThreadManagerMode mThreadManagerMode;
        ThreadManager<TDatumsSP> mThreadManager;
        bool mMultiThreadEnabled;
        // Configuration
        WrapperStructPose mWrapperStructPose;
        WrapperStructFace mWrapperStructFace;
        WrapperStructHand mWrapperStructHand;
        WrapperStructExtra mWrapperStructExtra;
        WrapperStructInput mWrapperStructInput;
        WrapperStructOutput mWrapperStructOutput;
        // User configurable workers
        std::array<bool, int(WorkerType::Size)> mUserWsOnNewThread;
        std::array<std::vector<TWorker>, int(WorkerType::Size)> mUserWs;

        DELETE_COPY(WrapperT);
    };

    // Type
    typedef WrapperT<DATUM_BASE_NO_PTR> Wrapper;
}





// Implementation
#include <openpose/wrapper/wrapperAuxiliary.hpp>
namespace op
{
    template<typename TDatums, typename TDatumsSP, typename TWorker>
    WrapperT<TDatums, TDatumsSP, TWorker>::WrapperT(const ThreadManagerMode threadManagerMode) :
        mThreadManagerMode{threadManagerMode},
        mThreadManager{threadManagerMode},
        mMultiThreadEnabled{true}
    {
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    WrapperT<TDatums, TDatumsSP, TWorker>::~WrapperT()
    {
        try
        {
            stop();
            // Reset mThreadManager
            mThreadManager.reset();
            // Reset user workers
            for (auto& userW : mUserWs)
                userW.clear();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::disableMultiThreading()
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::setWorker(
        const WorkerType workerType, const TWorker& worker, const bool workerOnNewThread)
    {
        try
        {
            // Sanity check
            if (worker == nullptr)
                error("Your worker is a nullptr.", __LINE__, __FILE__, __FUNCTION__);
            // Add worker
            mUserWs[int(workerType)].clear();
            mUserWs[int(workerType)].emplace_back(worker);
            mUserWsOnNewThread[int(workerType)] = workerOnNewThread;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructPose& wrapperStructPose)
    {
        try
        {
            mWrapperStructPose = wrapperStructPose;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructFace& wrapperStructFace)
    {
        try
        {
            mWrapperStructFace = wrapperStructFace;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructHand& wrapperStructHand)
    {
        try
        {
            mWrapperStructHand = wrapperStructHand;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructExtra& wrapperStructExtra)
    {
        try
        {
            mWrapperStructExtra = wrapperStructExtra;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructInput& wrapperStructInput)
    {
        try
        {
            mWrapperStructInput = wrapperStructInput;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::configure(const WrapperStructOutput& wrapperStructOutput)
    {
        try
        {
            mWrapperStructOutput = wrapperStructOutput;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::exec()
    {
        try
        {
            configureThreadManager<TDatums, TDatumsSP, TWorker>(
                mThreadManager, mMultiThreadEnabled, mThreadManagerMode, mWrapperStructPose, mWrapperStructFace,
                mWrapperStructHand, mWrapperStructExtra, mWrapperStructInput, mWrapperStructOutput,
                mUserWs, mUserWsOnNewThread);
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            mThreadManager.exec();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::start()
    {
        try
        {
            configureThreadManager<TDatums, TDatumsSP, TWorker>(
                mThreadManager, mMultiThreadEnabled, mThreadManagerMode, mWrapperStructPose, mWrapperStructFace,
                mWrapperStructHand, mWrapperStructExtra, mWrapperStructInput, mWrapperStructOutput,
                mUserWs, mUserWsOnNewThread);
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            mThreadManager.start();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void WrapperT<TDatums, TDatumsSP, TWorker>::stop()
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::isRunning() const
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::tryEmplace(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Input)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::waitAndEmplace(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Input)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::tryPush(const TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Input)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::waitAndPush(const TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Input)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::tryPop(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Output)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::waitAndPop(TDatumsSP& tDatums)
    {
        try
        {
            if (!mUserWs[int(WorkerType::Output)].empty())
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

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    bool WrapperT<TDatums, TDatumsSP, TWorker>::emplaceAndPop(TDatumsSP& tDatums)
    {
        try
        {
            // Run waitAndEmplace + waitAndPop
            if (waitAndEmplace(tDatums))
                return waitAndPop(tDatums);
            return false;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TDatumsSP, typename TWorker>
    TDatumsSP WrapperT<TDatums, TDatumsSP, TWorker>::emplaceAndPop(const cv::Mat& cvMat)
    {
        try
        {
            // Create new datum
            auto datumsPtr = std::make_shared<TDatums>();
            datumsPtr->emplace_back();
            auto& datum = datumsPtr->at(0);
            // Fill datum
            datum.cvInputData = cvMat;
            // Emplace and pop
            emplaceAndPop(datumsPtr);
            // Return result
            return datumsPtr;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return TDatumsSP{};
        }
    }

    extern template class WrapperT<DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE_WRAPPER_WRAPPER_HPP
