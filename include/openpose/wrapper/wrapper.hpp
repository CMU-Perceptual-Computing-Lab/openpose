#ifndef OPENPOSE__WRAPPER__WRAPPER_HPP
#define OPENPOSE__WRAPPER__WRAPPER_HPP

#include "../core/headers.hpp"
#include "../experimental/hands/headers.hpp"
#include "../filestream/headers.hpp"
#include "../pose/headers.hpp"
#include "../producer/headers.hpp"
#include "../thread/headers.hpp"
#include "enumClasses.hpp"

namespace op
{
    struct WrapperPoseStruct
    {
        cv::Size netInputSize;
        cv::Size outputSize;
        ScaleMode scaleMode;
        int gpuNumber;
        int gpuNumberStart;
        int scalesNumber;
        float scaleGap;
        bool renderOutput;
        PoseModel poseModel;
        bool blendOriginalFrame;
        float alphaPose;
        float alphaHeatMap;
        int defaultPartToRender;
        std::string modelFolder;
        std::vector<HeatMapType> heatMapTypes;
        ScaleMode heatMapScaleMode;

        WrapperPoseStruct(const cv::Size& netInputSize = cv::Size{656, 368}, const cv::Size& outputSize = cv::Size{1280, 720},
                          const ScaleMode scaleMode = ScaleMode::InputResolution, const int gpuNumber = 1, const int gpuNumberStart = 0, const int scalesNumber = 1,
                          const float scaleGap = 0.15f, const bool renderOutput = false, const PoseModel poseModel = PoseModel::COCO_18,
                          const bool blendOriginalFrame = true, const float alphaPose = POSE_DEFAULT_ALPHA_POSE, const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEATMAP,
                          const int defaultPartToRender = 0, const std::string& modelFolder = "models/",
                          const std::vector<HeatMapType>& heatMapTypes = {}, const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne);
    };

    namespace experimental
    {
        struct WrapperHandsStruct
        {
            bool extractAndRenderHands;

            WrapperHandsStruct(const bool extractAndRenderHands = false);
        };
    }

    struct WrapperInputStruct
    {
        std::shared_ptr<Producer> producerSharedPtr;
        unsigned long long frameFirst;
        unsigned long long frameLast;
        bool realTimeProcessing;
        bool frameFlip;
        int frameRotate;
        bool framesRepeat;

        WrapperInputStruct(const std::shared_ptr<Producer> producerSharedPtr = nullptr, const unsigned long long frameFirst = 0,
                           const unsigned long long frameLast = -1, const bool realTimeProcessing = false, const bool frameFlip = false,
                           const int frameRotate = 0, const bool framesRepeat = false);
    };

    struct WrapperOutputStruct
    {
        bool displayGui;
        bool guiVerbose;
        bool fullScreen;
        std::string writePose;
        DataFormat dataFormat;
        std::string writePoseJson;
        std::string writeCocoJson;
        std::string writeImages;
        std::string writeImagesFormat;
        std::string writeVideo;
        std::string writeHeatMaps;
        std::string writeHeatMapsFormat;

        WrapperOutputStruct(const bool displayGui = false, const bool guiVerbose = false, const bool fullScreen = false, const std::string& writePose = "",
                            const DataFormat dataFormat = DataFormat::Json, const std::string& writePoseJson = "", const std::string& writeCocoJson = "",
                            const std::string& writeImages = "", const std::string& writeImagesFormat = "", const std::string& writeVideo = "",
                            const std::string& writeHeatMaps = "", const std::string& writeHeatMapsFormat = "");
    };

    // This function can be used in 2 ways:
        // - Synchronous mode: call the full constructor with your desired input and output workers
        // - Asynchronous mode: call the empty constructor Wrapper() + use the emplace and pop functions to push the original frames and retrieve the processed ones
        // - Mix of them:
            // Synchronous input + asynchronous output: call the constructor Wrapper(ThreadMode::Synchronous, workersInput, {}, true)
            // Asynchronous input + synchronous output: call the constructor Wrapper(ThreadMode::Synchronous, nullptr, workersOutput, irrelevantBoolean, true)
    template<typename TDatums, typename TWorker = std::shared_ptr<Worker<std::shared_ptr<TDatums>>>, typename TQueue = Queue<std::shared_ptr<TDatums>>>
    class Wrapper
    {
    public:
        explicit Wrapper(const ThreadMode threadMode = ThreadMode::Synchronous);

        ~Wrapper();

        // Useful for debugging -> all the Workers will run in the same thread (workerOnNewThread will not make any effect)
        void setWrapperMode(const WrapperMode wrapperMode);

        void setWorkerInput(const TWorker& worker, const bool workerOnNewThread = true);

        void setWorkerPostProcessing(const TWorker& worker, const bool workerOnNewThread = true);

        void setWorkerOutput(const TWorker& worker, const bool workerOnNewThread = true);

        // If output is not required, just use this function until the renderOutput argument. Keep the default values for the other parameters in order not to display/save any output.
        void configure(const WrapperPoseStruct& wrapperPoseStruct,
                       // Producer (set producerSharedPtr = nullptr or use the default WrapperInputStruct{} to disable any input)
                       const WrapperInputStruct& wrapperInputStruct = WrapperInputStruct{},
                       // Consumer (keep default values to disable any output)
                       const WrapperOutputStruct& wrapperOutputStruct = WrapperOutputStruct{});

        // Similar to the previos configure, but it includes hand extraction and rendering
        void configure(const WrapperPoseStruct& wrapperPoseStruct = WrapperPoseStruct{},
                       // Hand (use the default WrapperHandsStruct{} to disable any hand detector)
                       const experimental::WrapperHandsStruct& wrapperHandStruct = experimental::WrapperHandsStruct{},
                       // Producer (set producerSharedPtr = nullptr or use the default WrapperInputStruct{} to disable any input)
                       const WrapperInputStruct& wrapperInputStruct = WrapperInputStruct{},
                       // Consumer (keep default values to disable any output)
                       const WrapperOutputStruct& wrapperOutputStruct = WrapperOutputStruct{});

        void exec();

        void start();

        void stop();

        void reset();

        bool isRunning() const;

        // Asynchronous(In) mode
        bool tryEmplace(std::shared_ptr<TDatums>& tDatums);

        // Asynchronous(In) mode
        bool waitAndEmplace(std::shared_ptr<TDatums>& tDatums);

        // Asynchronous(In) mode
        bool tryPush(const std::shared_ptr<TDatums>& tDatums);

        // Asynchronous(In) mode
        bool waitAndPush(const std::shared_ptr<TDatums>& tDatums);

        // Asynchronous(Out) mode
        bool tryPop(std::shared_ptr<TDatums>& tDatums);

        // Asynchronous(Out) mode
        bool waitAndPop(std::shared_ptr<TDatums>& tDatums);

    private:
        const ThreadMode mThreadMode;
        const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
        ThreadManager<std::shared_ptr<TDatums>> mThreadManager;
        int mGpuNumber;
        bool mUserInputWsOnNewThread;
        bool mUserPostProcessingWsOnNewThread;
        bool mUserOutputWsOnNewThread;
        unsigned int mThreadId;
        WrapperMode mWrapperMode;
        // Workers
        std::vector<TWorker> mUserInputWs;
        TWorker wDatumProducer;
        TWorker spWIdGenerator;
        TWorker spWCvMatToOpInput;
        TWorker spWCvMatToOpOutput;
        std::vector<std::vector<TWorker>> spWPoses;
        std::vector<TWorker> mPostProcessingWs;
        std::vector<TWorker> mUserPostProcessingWs;
        std::vector<TWorker> mOutputWs;
        TWorker spWGui;
        std::vector<TWorker> mUserOutputWs;

        void configureThreadManager();

        unsigned int threadIdPP();

        std::vector<TWorker> mergeWorkers(const std::vector<TWorker>& workersA, const std::vector<TWorker>& workersB);

        DELETE_COPY(Wrapper);
    };
}





// Implementation
#include "../experimental/headers.hpp"
#include "../gui/headers.hpp"
#include "../utilities/errorAndLog.hpp"
#include "../utilities/fileSystem.hpp"
namespace op
{
    template<typename TDatums, typename TWorker, typename TQueue>
    Wrapper<TDatums, TWorker, TQueue>::Wrapper(const ThreadMode threadMode) :
        mThreadMode{threadMode},
        spVideoSeek{std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>()},
        mThreadManager{threadMode},
        mWrapperMode{WrapperMode::MultiThread}
    {
        try
        {
            // It cannot be directly included in the constructor, otherwise compiler error for copying std::atomic
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
    void Wrapper<TDatums, TWorker, TQueue>::setWrapperMode(const WrapperMode wrapperMode)
    {
        try
        {
            mWrapperMode = {wrapperMode};
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
    void Wrapper<TDatums, TWorker, TQueue>::setWorkerPostProcessing(const TWorker& worker, const bool workerOnNewThread)
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
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperPoseStruct& wrapperPoseStruct, const WrapperInputStruct& wrapperInputStruct,
                                                      const WrapperOutputStruct& wrapperOutputStruct)
    {
        try
        {
            configure(wrapperPoseStruct, experimental::WrapperHandsStruct{}, wrapperInputStruct, wrapperOutputStruct);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configure(const WrapperPoseStruct& wrapperPoseStruct, const experimental::WrapperHandsStruct& wrapperHandStruct,
                                                      const WrapperInputStruct& wrapperInputStruct, const WrapperOutputStruct& wrapperOutputStruct)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Shortcut
            typedef std::shared_ptr<TDatums> TDatumsPtr;

            // Check no contradictory flags enabled
            if (wrapperPoseStruct.alphaPose < 0. || wrapperPoseStruct.alphaPose > 1. || wrapperPoseStruct.alphaHeatMap < 0. || wrapperPoseStruct.alphaHeatMap > 1.)
                error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
            if (wrapperPoseStruct.scaleGap <= 0.f && wrapperPoseStruct.scalesNumber > 1)
                error("The scale gap must be greater than 0 (it has no effect if the number of scales is 1).", __LINE__, __FUNCTION__, __FILE__);
            if (!wrapperPoseStruct.renderOutput && (!wrapperOutputStruct.writeImages.empty() || !wrapperOutputStruct.writeVideo.empty()))
                error("In order to save the rendered frames (`write_images` or `write_video`), you must set `render_output` to true.", __LINE__, __FUNCTION__, __FILE__);
            if (!wrapperOutputStruct.writeHeatMaps.empty() && wrapperPoseStruct.heatMapTypes.empty())
            {
                const auto message = "In order to save the heatmaps (`write_heatmaps`), you need to pick which heat maps you want to save: `heatmaps_add_X`"
                                     " flags or fill the wrapperPoseStruct.heatMapTypes.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (!wrapperOutputStruct.writeHeatMaps.empty() && wrapperPoseStruct.heatMapScaleMode != ScaleMode::UnsignedChar)
                error("In order to save the heatmaps, you must set wrapperPoseStruct.heatMapScaleMode to ScaleMode::UnsignedChar, i.e. range [0, 255].", __LINE__, __FUNCTION__, __FILE__);
            if (mUserOutputWs.empty() && mThreadMode != ThreadMode::Asynchronous && mThreadMode != ThreadMode::AsynchronousOut)
            {
                const std::string additionalMessage = " You could also set mThreadMode = mThreadMode::Asynchronous(Out) and/or add your own output worker class"
                                                      " before calling this function.";
                const auto savingSomething = (!wrapperOutputStruct.writeImages.empty() || !wrapperOutputStruct.writeVideo.empty() || !wrapperOutputStruct.writePose.empty()
                                              || !wrapperOutputStruct.writePoseJson.empty() || !wrapperOutputStruct.writeCocoJson.empty()
                                              || !wrapperOutputStruct.writeHeatMaps.empty());
                if (!wrapperOutputStruct.displayGui && !savingSomething)
                {
                    const auto message = "No output is selected (`no_display`) and no results are generated (no `write_X` flags enabled). Thus, no output would be "
                                         "generated." + additionalMessage;
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }

                if ((wrapperOutputStruct.displayGui && wrapperOutputStruct.guiVerbose) && !wrapperPoseStruct.renderOutput)
                {
                    const auto message = "No render is enabled (`no_render_output`), so you should also remove the display (set `no_display` or `no_gui_verbose`)."
                                       + additionalMessage;
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                if (wrapperInputStruct.framesRepeat && savingSomething)
                {
                    const auto message = "Frames repetition (`frames_repeat`) is enabled as well as some writing function (`write_X`). This program would"
                                         " never stop recording the same frames over and over. Please, disable repetition or remove writing.";
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                if (wrapperInputStruct.realTimeProcessing && savingSomething)
                {
                    const auto message = "Real time processing is enabled as well as some writing function. Thus, some frames might be skipped. Consider disabling"
                                         " real time processing if you intend to save any results.";
                    log(message, Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                }
            }
            if (!wrapperOutputStruct.writeVideo.empty() && wrapperInputStruct.producerSharedPtr == nullptr)
                error("Writting video is only available if the OpenPose producer is used (i.e. wrapperInputStruct.producerSharedPtr cannot be a nullptr).");

            // Proper format
            const auto writeImagesCleaned = formatAsDirectory(wrapperOutputStruct.writeImages);
            const auto writePoseCleaned = formatAsDirectory(wrapperOutputStruct.writePose);
            const auto writePoseJsonCleaned = formatAsDirectory(wrapperOutputStruct.writePoseJson);
            const auto writeHeatMapsCleaned = formatAsDirectory(wrapperOutputStruct.writeHeatMaps);

            // Common parameters
            auto finalOutputSize = wrapperPoseStruct.outputSize;
            cv::Size producerSize{-1,-1};
            if (wrapperInputStruct.producerSharedPtr != nullptr)
            {
                // 1. Set producer properties
                const auto displayProducerFpsMode = (wrapperInputStruct.realTimeProcessing ? ProducerFpsMode::OriginalFps : ProducerFpsMode::RetrievalFps);
                wrapperInputStruct.producerSharedPtr->setProducerFpsMode(displayProducerFpsMode);
                wrapperInputStruct.producerSharedPtr->set(ProducerProperty::Flip, wrapperInputStruct.frameFlip);
                wrapperInputStruct.producerSharedPtr->set(ProducerProperty::Rotation, wrapperInputStruct.frameRotate);
                wrapperInputStruct.producerSharedPtr->set(ProducerProperty::AutoRepeat, wrapperInputStruct.framesRepeat);
                // 2. Set finalOutputSize
                producerSize = cv::Size{(int)wrapperInputStruct.producerSharedPtr->get(CV_CAP_PROP_FRAME_WIDTH), (int)wrapperInputStruct.producerSharedPtr->get(CV_CAP_PROP_FRAME_HEIGHT)};
                if (wrapperPoseStruct.outputSize.width == -1 || wrapperPoseStruct.outputSize.height == -1)
                {
                    if (producerSize.area() > 0)
                        finalOutputSize = producerSize;
                    else
                        error("Output resolution = input resolution not valid for image reading (size might change between images).", __LINE__, __FUNCTION__, __FILE__);
                }
            }
            else if (finalOutputSize.width == -1 || finalOutputSize.height == -1)
                error("Output resolution cannot be (-1 x -1) unless wrapperInputStruct.producerSharedPtr is also set.", __LINE__, __FUNCTION__, __FILE__);

            // Update global parameter
            mGpuNumber = wrapperPoseStruct.gpuNumber;

            // Producer
            if (wrapperInputStruct.producerSharedPtr != nullptr)
            {
                const auto datumProducer = std::make_shared<DatumProducer<TDatums>>(wrapperInputStruct.producerSharedPtr, wrapperInputStruct.frameFirst, wrapperInputStruct.frameLast, spVideoSeek);
                wDatumProducer = std::make_shared<WDatumProducer<TDatumsPtr, TDatums>>(datumProducer);
            }
            else
                wDatumProducer = nullptr;

            // Pose estimators
            const cv::Size& netOutputSize = wrapperPoseStruct.netInputSize;
            std::vector<std::shared_ptr<PoseExtractor>> poseExtractors;
            for (auto gpuId = 0; gpuId < wrapperPoseStruct.gpuNumber; gpuId++)
                poseExtractors.emplace_back(std::make_shared<PoseExtractorCaffe>(wrapperPoseStruct.netInputSize, netOutputSize, finalOutputSize, wrapperPoseStruct.scalesNumber,
                                                                                 wrapperPoseStruct.scaleGap, wrapperPoseStruct.poseModel, wrapperPoseStruct.modelFolder,
                                                                                 gpuId + wrapperPoseStruct.gpuNumberStart, wrapperPoseStruct.heatMapTypes,
                                                                                 wrapperPoseStruct.heatMapScaleMode));
            // Pose renderers
            std::vector<std::shared_ptr<PoseRenderer>> poseRenderers;
            if (wrapperPoseStruct.renderOutput)
                for (auto gpuId = 0; gpuId < poseExtractors.size(); gpuId++)
                    poseRenderers.emplace_back(std::make_shared<PoseRenderer>(netOutputSize, finalOutputSize, wrapperPoseStruct.poseModel, poseExtractors[gpuId],
                                                                              wrapperPoseStruct.blendOriginalFrame, wrapperPoseStruct.alphaPose,
                                                                              wrapperPoseStruct.alphaHeatMap, wrapperPoseStruct.defaultPartToRender));
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Input cvMat to OpenPose format
            const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(wrapperPoseStruct.netInputSize, wrapperPoseStruct.scalesNumber, wrapperPoseStruct.scaleGap);
            spWCvMatToOpInput = std::make_shared<WCvMatToOpInput<TDatumsPtr>>(cvMatToOpInput);
            const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>(finalOutputSize, wrapperPoseStruct.renderOutput);
            spWCvMatToOpOutput = std::make_shared<WCvMatToOpOutput<TDatumsPtr>>(cvMatToOpOutput);

            // Pose extractor(s)
            spWPoses.clear();
            spWPoses.resize(poseExtractors.size());
            for (auto i = 0; i < spWPoses.size(); i++)
                spWPoses.at(i) = {std::make_shared<WPoseExtractor<TDatumsPtr>>(poseExtractors.at(i))};

            // Hand extractor(s)
            if (wrapperHandStruct.extractAndRenderHands)
            {
                for (auto gpuId = 0; gpuId < spWPoses.size(); gpuId++)
                {
                    const auto handsExtractor = std::make_shared<experimental::HandsExtractor>(wrapperPoseStruct.modelFolder, gpuId + wrapperPoseStruct.gpuNumberStart,
                                                                                               wrapperPoseStruct.poseModel);
                    spWPoses.at(gpuId).emplace_back(std::make_shared<experimental::WHandsExtractor<TDatumsPtr>>(handsExtractor));
                }
            }

            // Pose renderer(s)
            if (!poseRenderers.empty())
                for (auto i = 0; i < spWPoses.size(); i++)
                    spWPoses.at(i).emplace_back(std::make_shared<WPoseRenderer<TDatumsPtr>>(poseRenderers.at(i)));

            // Hands renderer(s)
            if (wrapperHandStruct.extractAndRenderHands)
            {
                for (auto i = 0; i < spWPoses.size(); i++)
                {
                    // Construct hands renderer
                    const auto handsRenderer = std::make_shared<experimental::HandsRenderer>(wrapperPoseStruct.outputSize);
                    // Performance boost -> share spGpuMemoryPtr for all renderers
                    if (!poseRenderers.empty())
                    {
                        const bool isLastRenderer = true;
                        handsRenderer->setGpuMemoryAndSetIfLast(poseRenderers.at(i)->getGpuMemoryAndSetAsFirst(), isLastRenderer);
                    }
                    // Add worker
                    spWPoses.at(i).emplace_back(std::make_shared<experimental::WHandsRenderer<TDatumsPtr>>(handsRenderer));
                }
            }

            // Itermediate workers (e.g. OpenPose format to cv::Mat, json & frames recorder, ...)
            mPostProcessingWs.clear();
            // Frame buffer and ordering
            if (spWPoses.size() > 1)
                mPostProcessingWs.emplace_back(std::make_shared<WQueueOrderer<TDatumsPtr>>());
            // Frames processor (OpenPose format -> cv::Mat format)
            if (wrapperPoseStruct.renderOutput)
            {
                const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>(finalOutputSize);
                mPostProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsPtr>>(opOutputToCvMat));
            }
            // Resize pose to input size if we want to save any results
            if (wrapperPoseStruct.scaleMode != ScaleMode::OutputResolution && (wrapperPoseStruct.scaleMode != ScaleMode::InputResolution || (finalOutputSize != producerSize))
                 && (wrapperPoseStruct.scaleMode != ScaleMode::NetOutputResolution || (finalOutputSize != netOutputSize)))
            {
                auto arrayScaler = std::make_shared<ArrayScaler>(wrapperPoseStruct.scaleMode);
                mPostProcessingWs.emplace_back(std::make_shared<WArrayScaler<TDatumsPtr>>(arrayScaler));
            }

            mOutputWs.clear();
            // Write people pose data on disk (json for OpenCV >= 3, xml, yml...)
            if (!writePoseCleaned.empty())
            {
                const auto poseSaver = std::make_shared<PoseSaver>(writePoseCleaned, wrapperOutputStruct.dataFormat);
                mOutputWs.emplace_back(std::make_shared<WPoseSaver<TDatumsPtr>>(poseSaver));
            }
            // Write people pose data on disk (json format)
            if (!writePoseJsonCleaned.empty())
            {
                const auto poseJsonSaver = std::make_shared<PoseJsonSaver>(writePoseJsonCleaned);
                mOutputWs.emplace_back(std::make_shared<WPoseJsonSaver<TDatumsPtr>>(poseJsonSaver));
            }
            // Write people pose data on disk (COCO validation json format)
            if (!wrapperOutputStruct.writeCocoJson.empty())
            {
                const auto humanFormat = true; // If true, bigger size (and potentially slower to process), but easier for a human to read it
                const auto poseJsonCocoSaver = std::make_shared<PoseJsonCocoSaver>(wrapperOutputStruct.writeCocoJson, humanFormat);
                mOutputWs.emplace_back(std::make_shared<experimental::WPoseJsonCocoSaver<TDatumsPtr>>(poseJsonCocoSaver));
            }
            // Write frames as desired image format on hard disk
            if (!writeImagesCleaned.empty())
            {
                const auto imageSaver = std::make_shared<ImageSaver>(writeImagesCleaned, wrapperOutputStruct.writeImagesFormat);
                mOutputWs.emplace_back(std::make_shared<WImageSaver<TDatumsPtr>>(imageSaver));
            }
            // Write frames as *.avi video on hard disk
            if (!wrapperOutputStruct.writeVideo.empty() && wrapperInputStruct.producerSharedPtr != nullptr)
            {
                const auto originalVideoFps = (wrapperInputStruct.producerSharedPtr->getType() != ProducerType::Webcam && wrapperInputStruct.producerSharedPtr->get(CV_CAP_PROP_FPS) > 0.
                                               ? wrapperInputStruct.producerSharedPtr->get(CV_CAP_PROP_FPS) : 30.);
                const auto videoSaver = std::make_shared<VideoSaver>(wrapperOutputStruct.writeVideo, CV_FOURCC('M','J','P','G'), originalVideoFps, finalOutputSize);
                mOutputWs.emplace_back(std::make_shared<WVideoSaver<TDatumsPtr>>(videoSaver));
            }
            // Write heat maps as desired image format on hard disk
            if (!writeHeatMapsCleaned.empty())
            {
                const auto heatMapSaver = std::make_shared<HeatMapSaver>(writeHeatMapsCleaned, wrapperOutputStruct.writeHeatMapsFormat);
                mOutputWs.emplace_back(std::make_shared<WHeatMapSaver<TDatumsPtr>>(heatMapSaver));
            }
            // Add frame information for GUI
            // If this WGuiInfoAdder instance is placed before the WImageSaver or WVideoSaver, then the resulting recorded frames will look exactly as the final displayed image by the GUI
            if (wrapperOutputStruct.displayGui && wrapperOutputStruct.guiVerbose)
            {
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(finalOutputSize, wrapperPoseStruct.gpuNumber);
                mOutputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsPtr>>(guiInfoAdder));
            }
            // Minimal graphical user interface (GUI)
            spWGui = nullptr;
            if (wrapperOutputStruct.displayGui)
            {
                const auto gui = std::make_shared<Gui>(wrapperOutputStruct.fullScreen, finalOutputSize, mThreadManager.getIsRunningSharedPtr(), spVideoSeek, poseExtractors, poseRenderers);
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
    void Wrapper<TDatums, TWorker, TQueue>::reset()
    {
        try
        {
            mThreadManager.reset();
            // Reset 
            mUserInputWs.clear();
            wDatumProducer = nullptr;
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
                error("Emplace cannot be called if an input worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
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
                error("Emplace cannot be called if an input worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
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
                error("Push cannot be called if an input worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
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
                error("Push cannot be called if an input worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
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
                error("Pop cannot be called if an output worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
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
                error("Pop cannot be called if an output worker was already selected.", __LINE__, __FUNCTION__, __FILE__);
            return mThreadManager.waitAndPop(tDatums);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    void Wrapper<TDatums, TWorker, TQueue>::configureThreadManager()
    {
        try
        {
            // The less number of queues -> the less lag

            // Security checks
            if (spWCvMatToOpInput == nullptr || spWCvMatToOpOutput == nullptr)
                error("Configure the Wrapper class before calling `start()`.", __LINE__, __FUNCTION__, __FILE__);
            if ((wDatumProducer == nullptr) == (mUserInputWs.empty()) && mThreadMode != ThreadMode::Asynchronous && mThreadMode != ThreadMode::AsynchronousIn)
                error("You need to have 1 and only 1 producer selected. You can introduce your own producer by using setWorkerInput() or use the OpenPose default"
                      " producer by configuring it in the configure function) or use the ThreadMode::Asynchronous(In) mode.", __LINE__, __FUNCTION__, __FILE__);
            if (mOutputWs.empty() && mUserOutputWs.empty() && spWGui == nullptr && mThreadMode != ThreadMode::Asynchronous && mThreadMode != ThreadMode::AsynchronousOut)
                error("No output selected.", __LINE__, __FUNCTION__, __FILE__);

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
                mThreadManager.add(mThreadId, mUserInputWs, queueIn++, queueOut++);                             // Thread 0, queues 0 -> 1
                threadIdPP();
                mThreadManager.add(mThreadId, {spWIdGenerator, spWCvMatToOpInput, spWCvMatToOpOutput}, queueIn++, queueOut++);   // Thread 1, queues 1 -> 2
            }
            // If custom user Worker in same thread or producer on same thread
            else
            {
                std::vector<TWorker> workersAux;
                // Custom user Worker
                if (!mUserInputWs.empty())
                    workersAux = mergeWorkers(workersAux, mUserInputWs);
                // OpenPose producer
                else if (wDatumProducer != nullptr)       
                    workersAux = mergeWorkers(workersAux, {wDatumProducer});
                // Otherwise
                else if (mThreadMode != ThreadMode::Asynchronous && mThreadMode != ThreadMode::AsynchronousIn)
                    error("No input selected.", __LINE__, __FUNCTION__, __FILE__);

                workersAux = mergeWorkers(workersAux, {spWIdGenerator, spWCvMatToOpInput, spWCvMatToOpOutput});
                mThreadManager.add(mThreadId, workersAux, queueIn++, queueOut++);                               // Thread 0 or 1, queues 0 -> 1
            }
            threadIdPP();
            // Pose estimation & rendering
            if (!spWPoses.empty())                                                                              // Thread 1 or 2...X, queues 1 -> 2, X = 2 + number GPUs
            {
                if (mWrapperMode == WrapperMode::MultiThread)
                {
                    for (auto& wPose : spWPoses)
                    {
                        mThreadManager.add(mThreadId, wPose, queueIn, queueOut);
                        threadIdPP();
                    }
                }
                else
                    mThreadManager.add(mThreadId, spWPoses.at(0), queueIn, queueOut);
                queueIn++;
                queueOut++;
            }
            // If custom user Worker and uses its own thread
            if (!mUserPostProcessingWs.empty() && mUserPostProcessingWsOnNewThread)
            {
                // Post processing workers
                if (!mPostProcessingWs.empty())
                {
                    mThreadManager.add(mThreadId, mPostProcessingWs, queueIn++, queueOut++);                    // Thread 2 or 3, queues 2 -> 3
                    threadIdPP();
                }
                // User processing workers
                mThreadManager.add(mThreadId, mUserPostProcessingWs, queueIn++, queueOut++);                    // Thread 3 or 4, queues 3 -> 4
                threadIdPP();
                // Output workers
                if (!mOutputWs.empty())
                {
                    mThreadManager.add(mThreadId, mOutputWs, queueIn++, queueOut++);                            // Thread 4 or 5, queues 4 -> 5
                    threadIdPP();
                }
            }
            // If custom user Worker in same thread or producer on same thread
            else
            {
                // Post processing workers + User post processing workers + Output workers
                auto workersAux = mergeWorkers(mPostProcessingWs, mUserPostProcessingWs);
                workersAux = mergeWorkers(workersAux, mOutputWs);
                if (!workersAux.empty())
                {
                    mThreadManager.add(mThreadId, workersAux, queueIn++, queueOut++);                           // Thread 2 or 3, queues 2 -> 3
                    threadIdPP();
                }
            }
            // User output worker
            if (!mUserOutputWs.empty())                                                                         // Thread Y, queues Q -> Q+1
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
                mThreadManager.add(mThreadId, spWGui, queueIn++, queueOut++);                                   // Thread Y+1, queues Q+1 -> Q+2
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
    unsigned int Wrapper<TDatums, TWorker, TQueue>::threadIdPP()
    {
        try
        {
            if (mWrapperMode == WrapperMode::MultiThread)
                mThreadId++;
            return mThreadId;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums, typename TWorker, typename TQueue>
    std::vector<TWorker> Wrapper<TDatums, TWorker, TQueue>::mergeWorkers(const std::vector<TWorker>& workersA, const std::vector<TWorker>& workersB)
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

    extern template class Wrapper<DATUM_BASE_NO_PTR>;
}

#endif // OPENPOSE__WRAPPER__WRAPPER_HPP
