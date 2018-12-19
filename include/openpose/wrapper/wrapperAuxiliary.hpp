#ifndef OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP
#define OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP

#include <openpose/thread/headers.hpp>
#include <openpose/wrapper/enumClasses.hpp>
#include <openpose/wrapper/wrapperStructExtra.hpp>
#include <openpose/wrapper/wrapperStructFace.hpp>
#include <openpose/wrapper/wrapperStructGui.hpp>
#include <openpose/wrapper/wrapperStructHand.hpp>
#include <openpose/wrapper/wrapperStructInput.hpp>
#include <openpose/wrapper/wrapperStructOutput.hpp>
#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    /**
     * It checks that no wrong/contradictory flags are enabled for Wrapper(T)
     * @param wrapperStructPose
     * @param wrapperStructFace
     * @param wrapperStructHand
     * @param wrapperStructExtra
     * @param wrapperStructInput
     * @param wrapperStructOutput
     * @param renderOutput
     * @param userOutputWsEmpty
     * @param producerSharedPtr
     * @param threadManagerMode
     */
    OP_API void wrapperConfigureSanityChecks(
        WrapperStructPose& wrapperStructPose, const WrapperStructFace& wrapperStructFace,
        const WrapperStructHand& wrapperStructHand, const WrapperStructExtra& wrapperStructExtra,
        const WrapperStructInput& wrapperStructInput, const WrapperStructOutput& wrapperStructOutput,
        const WrapperStructGui& wrapperStructGui, const bool renderOutput, const bool userOutputWsEmpty,
        const std::shared_ptr<Producer>& producerSharedPtr, const ThreadManagerMode threadManagerMode);

    /**
     * Thread ID increase (private internal function).
     * If multi-threading mode, it increases the thread ID.
     * If single-threading mode (for debugging), it does not modify it.
     * Note that mThreadId must be re-initialized to 0 before starting a new Wrapper(T) configuration.
     * @param threadId unsigned long long element with the current thread id value. I will be edited to the next
     * `desired thread id number.
     */
    OP_API void threadIdPP(unsigned long long& threadId, const bool multiThreadEnabled);

    /**
     * Set ThreadManager from TWorkers (private internal function).
     * After any configure() has been called, the TWorkers are initialized. This function resets the ThreadManager
     * and adds them.
     * Common code for start() and exec().
     */
    template<typename TDatums,
             typename TDatumsSP = std::shared_ptr<TDatums>,
             typename TWorker = std::shared_ptr<Worker<TDatumsSP>>>
    void configureThreadManager(
        ThreadManager<TDatumsSP>& threadManager, const bool multiThreadEnabled,
        const ThreadManagerMode threadManagerMode, const WrapperStructPose& wrapperStructPose,
        const WrapperStructFace& wrapperStructFace, const WrapperStructHand& wrapperStructHand,
        const WrapperStructExtra& wrapperStructExtra, const WrapperStructInput& wrapperStructInput,
        const WrapperStructOutput& wrapperStructOutput, const WrapperStructGui& wrapperStructGui,
        const std::array<std::vector<TWorker>, int(WorkerType::Size)>& userWs,
        const std::array<bool, int(WorkerType::Size)>& userWsOnNewThread);
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
namespace op
{
    template<typename TDatums, typename TDatumsSP, typename TWorker>
    void configureThreadManager(
        ThreadManager<TDatumsSP>& threadManager, const bool multiThreadEnabledTemp,
        const ThreadManagerMode threadManagerMode, const WrapperStructPose& wrapperStructPoseTemp,
        const WrapperStructFace& wrapperStructFace, const WrapperStructHand& wrapperStructHand,
        const WrapperStructExtra& wrapperStructExtra, const WrapperStructInput& wrapperStructInput,
        const WrapperStructOutput& wrapperStructOutput, const WrapperStructGui& wrapperStructGui,
        const std::array<std::vector<TWorker>, int(WorkerType::Size)>& userWs,
        const std::array<bool, int(WorkerType::Size)>& userWsOnNewThread)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // Create producer
            auto producerSharedPtr = createProducer(
                wrapperStructInput.producerType, wrapperStructInput.producerString,
                wrapperStructInput.cameraResolution, wrapperStructInput.cameraParameterPath,
                wrapperStructInput.undistortImage, wrapperStructInput.numberViews);

            // Editable arguments
            auto wrapperStructPose = wrapperStructPoseTemp;
            auto multiThreadEnabled = multiThreadEnabledTemp;

            // User custom workers
            const auto& userInputWs = userWs[int(WorkerType::Input)];
            const auto& userPostProcessingWs = userWs[int(WorkerType::PostProcessing)];
            const auto& userOutputWs = userWs[int(WorkerType::Output)];
            const auto userInputWsOnNewThread = userWsOnNewThread[int(WorkerType::Input)];
            const auto userPostProcessingWsOnNewThread = userWsOnNewThread[int(WorkerType::PostProcessing)];
            const auto userOutputWsOnNewThread = userWsOnNewThread[int(WorkerType::Output)];

            // Video seek
            const auto spVideoSeek = std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>();
            // It cannot be directly included in the constructor (compiler error for copying std::atomic)
            spVideoSeek->first = false;
            spVideoSeek->second = 0;

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
            const auto userOutputWsEmpty = userOutputWs.empty();
            wrapperConfigureSanityChecks(
                wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructExtra, wrapperStructInput,
                wrapperStructOutput, wrapperStructGui, renderOutput, userOutputWsEmpty, producerSharedPtr,
                threadManagerMode);

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
                multiThreadEnabled = false;
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
                // Sanity check
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
            const auto oPProducer = (producerSharedPtr != nullptr);
            if (oPProducer)
            {
                // 1. Set producer properties
                const auto displayProducerFpsMode = (wrapperStructInput.realTimeProcessing
                                                      ? ProducerFpsMode::OriginalFps : ProducerFpsMode::RetrievalFps);
                producerSharedPtr->setProducerFpsMode(displayProducerFpsMode);
                producerSharedPtr->set(ProducerProperty::Flip, wrapperStructInput.frameFlip);
                producerSharedPtr->set(ProducerProperty::Rotation, wrapperStructInput.frameRotate);
                producerSharedPtr->set(ProducerProperty::AutoRepeat, wrapperStructInput.framesRepeat);
                // 2. Set finalOutputSize
                producerSize = Point<int>{(int)producerSharedPtr->get(CV_CAP_PROP_FRAME_WIDTH),
                                          (int)producerSharedPtr->get(CV_CAP_PROP_FRAME_HEIGHT)};
                // Set finalOutputSize to input size if desired
                if (finalOutputSize.x == -1 || finalOutputSize.y == -1)
                    finalOutputSize = producerSize;
            }

            // Producer
            TWorker datumProducerW;
            if (oPProducer)
            {
                const auto datumProducer = std::make_shared<DatumProducer<TDatums>>(
                    producerSharedPtr, wrapperStructInput.frameFirst, wrapperStructInput.frameStep,
                    wrapperStructInput.frameLast, spVideoSeek
                );
                datumProducerW = std::make_shared<WDatumProducer<TDatumsSP, TDatums>>(datumProducer);
            }
            else
                datumProducerW = nullptr;

            std::vector<std::shared_ptr<PoseExtractorNet>> poseExtractorNets;
            std::vector<std::shared_ptr<FaceExtractorNet>> faceExtractorNets;
            std::vector<std::shared_ptr<HandExtractorNet>> handExtractorNets;
            std::vector<std::shared_ptr<PoseGpuRenderer>> poseGpuRenderers;
            std::shared_ptr<PoseCpuRenderer> poseCpuRenderer;
            // Workers
            TWorker scaleAndSizeExtractorW;
            TWorker cvMatToOpInputW;
            TWorker cvMatToOpOutputW;
            std::vector<std::vector<TWorker>> poseExtractorsWs;
            std::vector<std::vector<TWorker>> poseTriangulationsWs;
            std::vector<std::vector<TWorker>> jointAngleEstimationsWs;
            std::vector<TWorker> postProcessingWs;
            if (numberThreads > 0)
            {
                // Get input scales and sizes
                const auto scaleAndSizeExtractor = std::make_shared<ScaleAndSizeExtractor>(
                    wrapperStructPose.netInputSize, finalOutputSize, wrapperStructPose.scalesNumber,
                    wrapperStructPose.scaleGap
                );
                scaleAndSizeExtractorW = std::make_shared<WScaleAndSizeExtractor<TDatumsSP>>(scaleAndSizeExtractor);

                // Input cvMat to OpenPose input & output format
                const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(wrapperStructPose.poseModel);
                cvMatToOpInputW = std::make_shared<WCvMatToOpInput<TDatumsSP>>(cvMatToOpInput);
                if (renderOutput)
                {
                    const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>();
                    cvMatToOpOutputW = std::make_shared<WCvMatToOpOutput<TDatumsSP>>(cvMatToOpOutput);
                }

                // Pose estimators & renderers
                std::vector<TWorker> cpuRenderers;
                poseExtractorsWs.clear();
                poseExtractorsWs.resize(numberThreads);
                if (wrapperStructPose.enable)
                {
                    // Pose estimators
                    for (auto gpuId = 0; gpuId < numberThreads; gpuId++)
                        poseExtractorNets.emplace_back(std::make_shared<PoseExtractorCaffe>(
                            wrapperStructPose.poseModel, modelFolder, gpuId + gpuNumberStart,
                            wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                            wrapperStructPose.addPartCandidates, wrapperStructPose.maximizePositives,
                            wrapperStructPose.enableGoogleLogging
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
                    poseExtractorsWs.resize(poseExtractorNets.size());
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
                    for (auto i = 0u; i < poseExtractorsWs.size(); i++)
                    {
                        // OpenPose keypoint detector + keepTopNPeople
                        //    + ID extractor (experimental) + tracking (experimental)
                        const auto poseExtractor = std::make_shared<PoseExtractor>(
                            poseExtractorNets.at(i), keepTopNPeople, personIdExtractor, personTrackers,
                            wrapperStructPose.numberPeopleMax, wrapperStructExtra.tracking);
                        poseExtractorsWs.at(i) = {std::make_shared<WPoseExtractor<TDatumsSP>>(poseExtractor)};
                        // // Just OpenPose keypoint detector
                        // poseExtractorsWs.at(i) = {std::make_shared<WPoseExtractorNet<TDatumsSP>>(
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
                    //     for (auto& wPose : poseExtractorsWs)
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
                        for (auto& wPose : poseExtractorsWs)
                            wPose.emplace_back(std::make_shared<WFaceDetector<TDatumsSP>>(faceDetector));
                    }
                    // OpenCV face detector
                    else
                    {
                        log("Body keypoint detection is disabled. Hence, using OpenCV face detector (much less"
                            " accurate but faster).", Priority::High);
                        for (auto& wPose : poseExtractorsWs)
                        {
                            // 1 FaceDetectorOpenCV per thread, OpenCV face detector is not thread-safe
                            const auto faceDetectorOpenCV = std::make_shared<FaceDetectorOpenCV>(modelFolder);
                            wPose.emplace_back(
                                std::make_shared<WFaceDetectorOpenCV<TDatumsSP>>(faceDetectorOpenCV)
                            );
                        }
                    }
                    // Face keypoint extractor
                    for (auto gpu = 0u; gpu < poseExtractorsWs.size(); gpu++)
                    {
                        // Face keypoint extractor
                        const auto netOutputSize = wrapperStructFace.netInputSize;
                        const auto faceExtractorNet = std::make_shared<FaceExtractorCaffe>(
                            wrapperStructFace.netInputSize, netOutputSize, modelFolder,
                            gpu + gpuNumberStart, wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScale,
                            wrapperStructPose.enableGoogleLogging
                        );
                        faceExtractorNets.emplace_back(faceExtractorNet);
                        poseExtractorsWs.at(gpu).emplace_back(
                            std::make_shared<WFaceExtractorNet<TDatumsSP>>(faceExtractorNet));
                    }
                }

                // Hand extractor(s)
                if (wrapperStructHand.enable)
                {
                    const auto handDetector = std::make_shared<HandDetector>(wrapperStructPose.poseModel);
                    for (auto gpu = 0u; gpu < poseExtractorsWs.size(); gpu++)
                    {
                        // Hand detector
                        // If tracking
                        if (wrapperStructHand.tracking)
                            poseExtractorsWs.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorTracking<TDatumsSP>>(handDetector)
                            );
                        // If detection
                        else
                            poseExtractorsWs.at(gpu).emplace_back(
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
                        poseExtractorsWs.at(gpu).emplace_back(
                            std::make_shared<WHandExtractorNet<TDatumsSP>>(handExtractorNet)
                            );
                        // If tracking
                        if (wrapperStructHand.tracking)
                            poseExtractorsWs.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorUpdate<TDatumsSP>>(handDetector)
                            );
                    }
                }

                // Pose renderer(s)
                if (!poseGpuRenderers.empty())
                    for (auto i = 0u; i < poseExtractorsWs.size(); i++)
                        poseExtractorsWs.at(i).emplace_back(std::make_shared<WPoseRenderer<TDatumsSP>>(
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
                        for (auto i = 0u; i < poseExtractorsWs.size(); i++)
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
                            poseExtractorsWs.at(i).emplace_back(
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
                        for (auto i = 0u; i < poseExtractorsWs.size(); i++)
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
                            poseExtractorsWs.at(i).emplace_back(
                                std::make_shared<WHandRenderer<TDatumsSP>>(handRenderer));
                        }
                    }
                    else
                        error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
                }

                // 3-D reconstruction
                poseTriangulationsWs.clear();
                if (wrapperStructExtra.reconstruct3d)
                {
                    // For all (body/face/hands): PoseTriangulations ~30 msec, 8 GPUS ~30 msec for keypoint estimation
                    poseTriangulationsWs.resize(fastMax(1, int(poseExtractorsWs.size() / 4)));
                    for (auto i = 0u ; i < poseTriangulationsWs.size() ; i++)
                    {
                        const auto poseTriangulation = std::make_shared<PoseTriangulation>(
                            wrapperStructExtra.minViews3d);
                        poseTriangulationsWs.at(i) = {std::make_shared<WPoseTriangulation<TDatumsSP>>(
                            poseTriangulation)};
                    }
                }
                // Itermediate workers (e.g., OpenPose format to cv::Mat, json & frames recorder, ...)
                postProcessingWs.clear();
                // // Person ID identification (when no multi-thread and no dependency on tracking)
                // if (wrapperStructExtra.identification)
                // {
                //     const auto personIdExtractor = std::make_shared<PersonIdExtractor>();
                //     postProcessingWs.emplace_back(
                //         std::make_shared<WPersonIdExtractor<TDatumsSP>>(personIdExtractor)
                //     );
                // }
                // Frames processor (OpenPose format -> cv::Mat format)
                if (renderOutput)
                {
                    postProcessingWs = mergeVectors(postProcessingWs, cpuRenderers);
                    const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>();
                    postProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsSP>>(opOutputToCvMat));
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
                    postProcessingWs.emplace_back(std::make_shared<WKeypointScaler<TDatumsSP>>(keypointScaler));
                }
            }

            // IK/Adam
            const auto displayAdam = wrapperStructGui.displayMode == DisplayMode::DisplayAdam
                                     || (wrapperStructGui.displayMode == DisplayMode::DisplayAll
                                         && wrapperStructExtra.ikThreads > 0);
            jointAngleEstimationsWs.clear();
#ifdef USE_3D_ADAM_MODEL
            if (wrapperStructExtra.ikThreads > 0)
            {
                jointAngleEstimationsWs.resize(wrapperStructExtra.ikThreads);
                // Pose extractor(s)
                for (auto i = 0u; i < jointAngleEstimationsWs.size(); i++)
                {
                    const auto jointAngleEstimation = std::make_shared<JointAngleEstimation>(displayAdam);
                    jointAngleEstimationsWs.at(i) = {std::make_shared<WJointAngleEstimation<TDatumsSP>>(
                        jointAngleEstimation)};
                }
            }
#endif

            // Output workers
            std::vector<TWorker> outputWs;
            // Print verbose
            if (wrapperStructOutput.verbose > 0.)
            {
                const auto verbosePrinter = std::make_shared<VerbosePrinter>(
                    wrapperStructOutput.verbose, producerSharedPtr->get(CV_CAP_PROP_FRAME_COUNT));
                outputWs.emplace_back(std::make_shared<WVerbosePrinter<TDatumsSP>>(verbosePrinter));
            }
            // Send information (e.g., to Unity) though UDP client-server communication

#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.udpHost.empty() && !wrapperStructOutput.udpPort.empty())
            {
                const auto udpSender = std::make_shared<UdpSender>(wrapperStructOutput.udpHost,
                                                                   wrapperStructOutput.udpPort);
                outputWs.emplace_back(std::make_shared<WUdpSender<TDatumsSP>>(udpSender));
            }
#endif
            // Write people pose data on disk (json for OpenCV >= 3, xml, yml...)
            if (!writeKeypointCleaned.empty())
            {
                const auto keypointSaver = std::make_shared<KeypointSaver>(writeKeypointCleaned,
                                                                           wrapperStructOutput.writeKeypointFormat);
                outputWs.emplace_back(std::make_shared<WPoseSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructFace.enable)
                    outputWs.emplace_back(std::make_shared<WFaceSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructHand.enable)
                    outputWs.emplace_back(std::make_shared<WHandSaver<TDatumsSP>>(keypointSaver));
            }
            // Write OpenPose output data on disk in json format (body/hand/face keypoints, body part locations if
            // enabled, etc.)
            if (!writeJsonCleaned.empty())
            {
                const auto peopleJsonSaver = std::make_shared<PeopleJsonSaver>(writeJsonCleaned);
                outputWs.emplace_back(std::make_shared<WPeopleJsonSaver<TDatumsSP>>(peopleJsonSaver));
            }
            // Write people pose data on disk (COCO validation json format)
            if (!wrapperStructOutput.writeCocoJson.empty())
            {
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(
                    wrapperStructOutput.writeCocoJson, wrapperStructPose.poseModel, humanFormat,
                    (wrapperStructPose.poseModel != PoseModel::CAR_22
                        && wrapperStructPose.poseModel != PoseModel::CAR_12
                        ? CocoJsonFormat::Body : CocoJsonFormat::Car),
                    wrapperStructOutput.writeCocoJsonVariant);
                outputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsSP>>(cocoJsonSaver));
            }
            // Write people foot pose data on disk (COCO validation json format for foot data)
            if (!wrapperStructOutput.writeCocoFootJson.empty())
            {
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(
                    wrapperStructOutput.writeCocoFootJson, wrapperStructPose.poseModel, humanFormat,
                    CocoJsonFormat::Foot);
                outputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsSP>>(cocoJsonSaver));
            }
            // Write frames as desired image format on hard disk
            if (!writeImagesCleaned.empty())
            {
                const auto imageSaver = std::make_shared<ImageSaver>(writeImagesCleaned,
                                                                     wrapperStructOutput.writeImagesFormat);
                outputWs.emplace_back(std::make_shared<WImageSaver<TDatumsSP>>(imageSaver));
            }
            auto originalVideoFps = 0.;
            if (!wrapperStructOutput.writeVideo.empty() || !wrapperStructOutput.writeVideo3D.empty()
                || !wrapperStructOutput.writeBvh.empty())
            {
                if (wrapperStructOutput.writeVideoFps <= 0
                    && (!oPProducer || producerSharedPtr->get(CV_CAP_PROP_FPS) <= 0))
                    error("The frame rate of the frames producer is unknown. Set `--write_video_fps` to your desired"
                          " FPS if you wanna record video (`--write_video`). E.g., if it is a folder of images, you"
                          " will have to know or guess the frame rate; if it is a webcam, you should use the OpenPose"
                          " displayed FPS as desired value. If you do not care, simply add `--write_video_fps 30`.",
                          __LINE__, __FUNCTION__, __FILE__);
                originalVideoFps = (
                    wrapperStructOutput.writeVideoFps > 0 ?
                    wrapperStructOutput.writeVideoFps : producerSharedPtr->get(CV_CAP_PROP_FPS));
            }
            // Write frames as *.avi video on hard disk
            if (!wrapperStructOutput.writeVideo.empty())
            {
                if (!oPProducer)
                    error("Video file can only be recorded inside `wrapper/wrapper.hpp` if the producer"
                          " is one of the default ones (e.g., video, webcam, ...).",
                          __LINE__, __FUNCTION__, __FILE__);
                const auto videoSaver = std::make_shared<VideoSaver>(
                    wrapperStructOutput.writeVideo, CV_FOURCC('M','J','P','G'), originalVideoFps);
                outputWs.emplace_back(std::make_shared<WVideoSaver<TDatumsSP>>(videoSaver));
            }
            // Write joint angles as *.bvh file on hard disk
#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.writeBvh.empty())
            {
                const auto bvhSaver = std::make_shared<BvhSaver>(
                    wrapperStructOutput.writeBvh, JointAngleEstimation::getTotalModel(), originalVideoFps
                );
                outputWs.emplace_back(std::make_shared<WBvhSaver<TDatumsSP>>(bvhSaver));
            }
#endif
            // Write heat maps as desired image format on hard disk
            if (!writeHeatMapsCleaned.empty())
            {
                const auto heatMapSaver = std::make_shared<HeatMapSaver>(
                    writeHeatMapsCleaned, wrapperStructOutput.writeHeatMapsFormat);
                outputWs.emplace_back(std::make_shared<WHeatMapSaver<TDatumsSP>>(heatMapSaver));
            }
            // Add frame information for GUI
            const bool guiEnabled = (wrapperStructGui.displayMode != DisplayMode::NoDisplay);
            // If this WGuiInfoAdder instance is placed before the WImageSaver or WVideoSaver, then the resulting
            // recorded frames will look exactly as the final displayed image by the GUI
            if (wrapperStructGui.guiVerbose && (guiEnabled || !userOutputWs.empty()
                                                || threadManagerMode == ThreadManagerMode::Asynchronous
                                                || threadManagerMode == ThreadManagerMode::AsynchronousOut))
            {
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(numberThreads, guiEnabled);
                outputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsSP>>(guiInfoAdder));
            }
            // Minimal graphical user interface (GUI)
            TWorker guiW;
            TWorker videoSaver3DW;
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
                const auto numberViews = (intRound(producerSharedPtr->get(ProducerProperty::NumberViews)));
                auto finalOutputSizeGui = finalOutputSize;
                if (numberViews > 1 && finalOutputSizeGui.x > 0)
                    finalOutputSizeGui.x *= numberViews;
                // Adam (+3-D/2-D) display
                if (displayAdam)
                {
#ifdef USE_3D_ADAM_MODEL
                    // Gui
                    const auto gui = std::make_shared<GuiAdam>(
                        finalOutputSizeGui, wrapperStructGui.fullScreen, threadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers,
                        wrapperStructGui.displayMode, JointAngleEstimation::getTotalModel(),
                        wrapperStructOutput.writeVideoAdam
                    );
                    // WGui
                    guiW = {std::make_shared<WGuiAdam<TDatumsSP>>(gui)};
                    // Write 3D frames as *.avi video on hard disk
                    if (!wrapperStructOutput.writeVideo3D.empty())
                        error("3D video can only be recorded if 3D render is enabled.",
                              __LINE__, __FUNCTION__, __FILE__);
#endif
                }
                // 3-D (+2-D) display
                else if (wrapperStructGui.displayMode == DisplayMode::Display3D
                    || wrapperStructGui.displayMode == DisplayMode::DisplayAll)
                {
                    // Gui
                    const auto gui = std::make_shared<Gui3D>(
                        finalOutputSizeGui, wrapperStructGui.fullScreen, threadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers,
                        wrapperStructPose.poseModel, wrapperStructGui.displayMode,
                        !wrapperStructOutput.writeVideo3D.empty()
                    );
                    // WGui
                    guiW = {std::make_shared<WGui3D<TDatumsSP>>(gui)};
                    // Write 3D frames as *.avi video on hard disk
                    if (!wrapperStructOutput.writeVideo3D.empty())
                    {
                        const auto videoSaver = std::make_shared<VideoSaver>(
                            wrapperStructOutput.writeVideo3D, CV_FOURCC('M','J','P','G'), originalVideoFps);
                        videoSaver3DW = std::make_shared<WVideoSaver3D<TDatumsSP>>(videoSaver);
                    }
                }
                // 2-D display
                else if (wrapperStructGui.displayMode == DisplayMode::Display2D)
                {
                    // Gui
                    const auto gui = std::make_shared<Gui>(
                        finalOutputSizeGui, wrapperStructGui.fullScreen, threadManager.getIsRunningSharedPtr(),
                        spVideoSeek, poseExtractorNets, faceExtractorNets, handExtractorNets, renderers
                    );
                    // WGui
                    guiW = {std::make_shared<WGui<TDatumsSP>>(gui)};
                    // Write 3D frames as *.avi video on hard disk
                    if (!wrapperStructOutput.writeVideo3D.empty())
                        error("3D video can only be recorded if 3D render is enabled.",
                              __LINE__, __FUNCTION__, __FILE__);
                }
                else
                    error("Unknown DisplayMode.", __LINE__, __FUNCTION__, __FILE__);
            }
            // Set FpsMax
            TWorker wFpsMax;
            if (wrapperStructPose.fpsMax > 0.)
                wFpsMax = std::make_shared<WFpsMax<TDatumsSP>>(wrapperStructPose.fpsMax);
            // Set wrapper as configured
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);





            // The less number of queues -> the less threads opened, and potentially the less lag

            // Sanity checks
            if ((datumProducerW == nullptr) == (userInputWs.empty())
                && threadManagerMode != ThreadManagerMode::Asynchronous
                && threadManagerMode != ThreadManagerMode::AsynchronousIn)
            {
                const auto message = "You need to have 1 and only 1 producer selected. You can introduce your own"
                                     " producer by using setWorker(WorkerType::Input, ...) or use the OpenPose"
                                     " default producer by configuring it in the configure function) or use the"
                                     " ThreadManagerMode::Asynchronous(In) mode.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (outputWs.empty() && userOutputWs.empty() && guiW == nullptr
                && threadManagerMode != ThreadManagerMode::Asynchronous
                && threadManagerMode != ThreadManagerMode::AsynchronousOut)
            {
                error("No output selected.", __LINE__, __FUNCTION__, __FILE__);
            }

            // Thread Manager
            // Clean previous thread manager (avoid configure to crash the program if used more than once)
            threadManager.reset();
            unsigned long long threadId = 0ull;
            auto queueIn = 0ull;
            auto queueOut = 1ull;
            // After producer
            // ID generator (before any multi-threading or any function that requires the ID)
            const auto wIdGenerator = std::make_shared<WIdGenerator<TDatumsSP>>();
            std::vector<TWorker> workersAux{wIdGenerator};
            // Scale & cv::Mat to OP format
            if (scaleAndSizeExtractorW != nullptr)
                workersAux = mergeVectors(workersAux, {scaleAndSizeExtractorW});
            if (cvMatToOpInputW != nullptr)
                workersAux = mergeVectors(workersAux, {cvMatToOpInputW});
            // cv::Mat to output format
            if (cvMatToOpOutputW != nullptr)
                workersAux = mergeVectors(workersAux, {cvMatToOpOutputW});

            // Producer
            // If custom user Worker and uses its own thread
            if (!userInputWs.empty() && userInputWsOnNewThread)
            {
                // Thread 0, queues 0 -> 1
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, userInputWs, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
            // If custom user Worker in same thread
            else if (!userInputWs.empty())
                workersAux = mergeVectors(userInputWs, workersAux);
            // If OpenPose producer (same thread)
            else if (datumProducerW != nullptr)
                workersAux = mergeVectors({datumProducerW}, workersAux);
            // Otherwise
            else if (threadManagerMode != ThreadManagerMode::Asynchronous
                        && threadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("No input selected.", __LINE__, __FUNCTION__, __FILE__);
            // Thread 0 or 1, queues 0 -> 1
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            threadManager.add(threadId, workersAux, queueIn++, queueOut++);
            // Increase thread
            threadIdPP(threadId, multiThreadEnabled);

            // Pose estimation & rendering
            // Thread 1 or 2...X, queues 1 -> 2, X = 2 + #GPUs
            if (!poseExtractorsWs.empty())
            {
                if (multiThreadEnabled)
                {
                    for (auto& wPose : poseExtractorsWs)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wPose, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames - Required own thread
                    if (poseExtractorsWs.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (poseExtractorsWs.size() > 1)
                        log("Multi-threading disabled, only 1 thread running. All GPUs have been disabled but the"
                            " first one, which is defined by gpuNumberStart (e.g., in the OpenPose demo, it is set"
                            " with the `--num_gpu_start` flag).", Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, poseExtractorsWs.at(0), queueIn++, queueOut++);
                }
            }
            // Assemble all frames from same time instant (3-D module)
            const auto wQueueAssembler = std::make_shared<WQueueAssembler<TDatumsSP, TDatums>>();
            // 3-D reconstruction
            if (!poseTriangulationsWs.empty())
            {
                // Assemble frames
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, wQueueAssembler, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
                // 3-D reconstruction
                if (multiThreadEnabled)
                {
                    for (auto& wPoseTriangulations : poseTriangulationsWs)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wPoseTriangulations, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (poseTriangulationsWs.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (poseTriangulationsWs.size() > 1)
                        log("Multi-threading disabled, only 1 thread running for 3-D triangulation.",
                            Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, poseTriangulationsWs.at(0), queueIn++, queueOut++);
                }
            }
            else
                postProcessingWs = mergeVectors({wQueueAssembler}, postProcessingWs);
            // Adam/IK step
            if (!jointAngleEstimationsWs.empty())
            {
                if (multiThreadEnabled)
                {
                    for (auto& wJointAngleEstimator : jointAngleEstimationsWs)
                    {
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wJointAngleEstimator, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (jointAngleEstimationsWs.size() > 1)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (jointAngleEstimationsWs.size() > 1)
                        log("Multi-threading disabled, only 1 thread running for joint angle estimation.",
                            Priority::High);
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, jointAngleEstimationsWs.at(0), queueIn++, queueOut++);
                }
            }
            // Post processing workers
            if (!postProcessingWs.empty())
            {
                // Combining postProcessingWs and outputWs
                outputWs = mergeVectors(postProcessingWs, outputWs);
                // // If I wanna split them
                // log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // threadManager.add(threadId, postProcessingWs, queueIn++, queueOut++);
                // threadIdPP(threadId, multiThreadEnabled);
            }
            // If custom user Worker and uses its own thread
            if (!userPostProcessingWs.empty())
            {
                // If custom user Worker in its own thread
                if (userPostProcessingWsOnNewThread)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, userPostProcessingWs, queueIn++, queueOut++);
                    threadIdPP(threadId, multiThreadEnabled);
                }
                // If custom user Worker in same thread
                // Merge with outputWs
                else
                    outputWs = mergeVectors(outputWs, userPostProcessingWs);
            }
            // Output workers
            if (!outputWs.empty())
            {
                // Thread 4 or 5, queues 4 -> 5
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, outputWs, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
            // User output worker
            // Thread Y, queues Q -> Q+1
            if (!userOutputWs.empty())
            {
                if (userOutputWsOnNewThread)
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, userOutputWs, queueIn++, queueOut++);
                    threadIdPP(threadId, multiThreadEnabled);
                }
                else
                {
                    log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId-1, userOutputWs, queueIn++, queueOut++);
                }
            }
            // OpenPose GUI
            if (guiW != nullptr)
            {
                // Thread Y+1, queues Q+1 -> Q+2
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, guiW, queueIn++, queueOut++);
                // Saving 3D output
                if (videoSaver3DW != nullptr)
                    threadManager.add(threadId, videoSaver3DW, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Setting maximum speed
            if (wFpsMax != nullptr)
            {
                log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, wFpsMax, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}

#endif // OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP
