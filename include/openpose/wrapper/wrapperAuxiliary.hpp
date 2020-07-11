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
        const WrapperStructGui& wrapperStructGui, const bool renderOutput, const bool userInputAndPreprocessingWsEmpty,
        const bool userOutputWsEmpty, const std::shared_ptr<Producer>& producerSharedPtr,
        const ThreadManagerMode threadManagerMode);

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
    template<typename TDatum,
             typename TDatums = std::vector<std::shared_ptr<TDatum>>,
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

    /**
     * It fills camera parameters and splits the cvMat depending on how many camera parameter matrices are found.
     * For example usage, check `examples/tutorial_api_cpp/11_asynchronous_custom_input_multi_camera.cpp`
     */
    template<typename TDatum,
             typename TDatums = std::vector<std::shared_ptr<TDatum>>,
             typename TDatumsSP = std::shared_ptr<TDatums>>
    void createMultiviewTDatum(
        TDatumsSP& tDatumsSP, unsigned long long& frameCounter,
        const CameraParameterReader& cameraParameterReader, const void* const cvMatPtr);
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
    template<typename TDatum, typename TDatums, typename TDatumsSP, typename TWorker>
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
            opLog("Running configureThreadManager...", Priority::Normal);

            // Create producer
            auto producerSharedPtr = createProducer(
                wrapperStructInput.producerType, wrapperStructInput.producerString.getStdString(),
                wrapperStructInput.cameraResolution, wrapperStructInput.cameraParameterPath.getStdString(),
                wrapperStructInput.undistortImage, wrapperStructInput.numberViews);

            // Editable arguments
            auto wrapperStructPose = wrapperStructPoseTemp;
            auto multiThreadEnabled = multiThreadEnabledTemp;

            // User custom workers
            const auto& userInputWs = userWs[int(WorkerType::Input)];
            const auto& userPreProcessingWs = userWs[int(WorkerType::PreProcessing)];
            const auto& userPostProcessingWs = userWs[int(WorkerType::PostProcessing)];
            const auto& userOutputWs = userWs[int(WorkerType::Output)];
            const auto userInputWsOnNewThread = userWsOnNewThread[int(WorkerType::Input)];
            const auto userPreProcessingWsOnNewThread = userWsOnNewThread[int(WorkerType::PreProcessing)];
            const auto userPostProcessingWsOnNewThread = userWsOnNewThread[int(WorkerType::PostProcessing)];
            const auto userOutputWsOnNewThread = userWsOnNewThread[int(WorkerType::Output)];

            // Video seek
            const auto spVideoSeek = std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>();
            // It cannot be directly included in the constructor (compiler error for copying std::atomic)
            spVideoSeek->first = false;
            spVideoSeek->second = 0;

            // Required parameters
            const auto gpuMode = getGpuMode();
            const auto renderModePose = (
                wrapperStructPose.renderMode != RenderMode::Auto
                    ? wrapperStructPose.renderMode
                    : (gpuMode == GpuMode::Cuda ? RenderMode::Gpu : RenderMode::Cpu));
            const auto renderModeFace = (
                wrapperStructFace.renderMode != RenderMode::Auto
                    ? wrapperStructFace.renderMode
                    : (gpuMode == GpuMode::Cuda ? RenderMode::Gpu : RenderMode::Cpu));
            const auto renderModeHand = (
                wrapperStructHand.renderMode != RenderMode::Auto
                    ? wrapperStructHand.renderMode
                    : (gpuMode == GpuMode::Cuda ? RenderMode::Gpu : RenderMode::Cpu));
            const auto renderOutput = renderModePose != RenderMode::None
                                        || renderModeFace != RenderMode::None
                                        || renderModeHand != RenderMode::None;
            const bool renderOutputGpu = renderModePose == RenderMode::Gpu
                || (wrapperStructFace.enable && renderModeFace == RenderMode::Gpu)
                || (wrapperStructHand.enable && renderModeHand == RenderMode::Gpu);
            const bool renderFace = wrapperStructFace.enable && renderModeFace != RenderMode::None;
            const bool renderHand = wrapperStructHand.enable && renderModeHand != RenderMode::None;
            const bool renderHandGpu = wrapperStructHand.enable && renderModeHand == RenderMode::Gpu;
            opLog("renderModePose = " + std::to_string(int(renderModePose)), Priority::Normal);
            opLog("renderModeFace = " + std::to_string(int(renderModeFace)), Priority::Normal);
            opLog("renderModeHand = " + std::to_string(int(renderModeHand)), Priority::Normal);
            opLog("renderOutput = " + std::to_string(int(renderOutput)), Priority::Normal);
            opLog("renderOutputGpu = " + std::to_string(int(renderOutput)), Priority::Normal);
            opLog("renderFace = " + std::to_string(int(renderFace)), Priority::Normal);
            opLog("renderHand = " + std::to_string(int(renderHand)), Priority::Normal);
            opLog("renderHandGpu = " + std::to_string(int(renderHandGpu)), Priority::Normal);

            // Check no wrong/contradictory flags enabled
            const bool userInputAndPreprocessingWsEmpty = userInputWs.empty() && userPreProcessingWs.empty();
            const bool userOutputWsEmpty = userOutputWs.empty();
            wrapperConfigureSanityChecks(
                wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructExtra, wrapperStructInput,
                wrapperStructOutput, wrapperStructGui, renderOutput, userInputAndPreprocessingWsEmpty,
                userOutputWsEmpty, producerSharedPtr, threadManagerMode);
            opLog("userInputAndPreprocessingWsEmpty = " + std::to_string(int(userInputAndPreprocessingWsEmpty)),
                Priority::Normal);
            opLog("userOutputWsEmpty = " + std::to_string(int(userOutputWsEmpty)), Priority::Normal);

            // Get number threads
            auto numberGpuThreads = wrapperStructPose.gpuNumber;
            auto gpuNumberStart = wrapperStructPose.gpuNumberStart;
            opLog("numberGpuThreads = " + std::to_string(numberGpuThreads), Priority::Normal);
            opLog("gpuNumberStart = " + std::to_string(gpuNumberStart), Priority::Normal);
            // CPU --> 1 thread or no pose extraction
            if (gpuMode == GpuMode::NoGpu)
            {
                numberGpuThreads = (wrapperStructPose.gpuNumber == 0 ? 0 : 1);
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
                if (numberGpuThreads < 0)
                {
                    if (totalGpuNumber <= gpuNumberStart)
                        error("Number of initial GPU (`--number_gpu_start`) must be lower than the total number of"
                              " used GPUs (`--number_gpu`)", __LINE__, __FUNCTION__, __FILE__);
                    numberGpuThreads = totalGpuNumber - gpuNumberStart;
                    // Reset initial GPU to 0 (we want them all)
                    // Logging message
                    opLog("Auto-detecting all available GPUs... Detected " + std::to_string(totalGpuNumber)
                        + " GPU(s), using " + std::to_string(numberGpuThreads) + " of them starting at GPU "
                        + std::to_string(gpuNumberStart) + ".", Priority::High);
                }
                // Sanity check
                if (gpuNumberStart + numberGpuThreads > totalGpuNumber)
                    error("Initial GPU selected (`--number_gpu_start`) + number GPUs to use (`--number_gpu`) must"
                          " be lower or equal than the total number of GPUs in your machine ("
                          + std::to_string(gpuNumberStart) + " + "
                          + std::to_string(numberGpuThreads) + " vs. "
                          + std::to_string(totalGpuNumber) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
            }

            // Proper format
            const auto writeImagesCleaned = formatAsDirectory(wrapperStructOutput.writeImages.getStdString());
            const auto writeKeypointCleaned = formatAsDirectory(wrapperStructOutput.writeKeypoint.getStdString());
            const auto writeJsonCleaned = formatAsDirectory(wrapperStructOutput.writeJson.getStdString());
            const auto writeHeatMapsCleaned = formatAsDirectory(wrapperStructOutput.writeHeatMaps.getStdString());
            const auto modelFolder = formatAsDirectory(wrapperStructPose.modelFolder.getStdString());
            opLog("writeImagesCleaned = " + writeImagesCleaned, Priority::Normal);
            opLog("writeKeypointCleaned = " + writeKeypointCleaned, Priority::Normal);
            opLog("writeJsonCleaned = " + writeJsonCleaned, Priority::Normal);
            opLog("writeHeatMapsCleaned = " + writeHeatMapsCleaned, Priority::Normal);
            opLog("modelFolder = " + modelFolder, Priority::Normal);

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
                producerSize = Point<int>{(int)producerSharedPtr->get(getCvCapPropFrameWidth()),
                                          (int)producerSharedPtr->get(getCvCapPropFrameHeight())};
                // Set finalOutputSize to input size if desired
                if (finalOutputSize.x == -1 || finalOutputSize.y == -1)
                    finalOutputSize = producerSize;
            }
            opLog("finalOutputSize = [" + std::to_string(finalOutputSize.x) + "," + std::to_string(finalOutputSize.y)
                + "]", Priority::Normal);

            // Producer
            TWorker datumProducerW;
            if (oPProducer)
            {
                const auto datumProducer = std::make_shared<DatumProducer<TDatum>>(
                    producerSharedPtr, wrapperStructInput.frameFirst, wrapperStructInput.frameStep,
                    wrapperStructInput.frameLast, spVideoSeek
                );
                datumProducerW = std::make_shared<WDatumProducer<TDatum>>(datumProducer);
            }
            else
                datumProducerW = nullptr;

            std::vector<std::shared_ptr<PoseExtractorNet>> poseExtractorNets;
            std::vector<std::shared_ptr<FaceExtractorNet>> faceExtractorNets;
            std::vector<std::shared_ptr<HandExtractorNet>> handExtractorNets;
            std::vector<std::shared_ptr<PoseGpuRenderer>> poseGpuRenderers;
            // CUDA vs. CPU resize
            std::vector<std::shared_ptr<CvMatToOpOutput>> cvMatToOpOutputs;
            std::vector<std::shared_ptr<OpOutputToCvMat>> opOutputToCvMats;
            std::shared_ptr<PoseCpuRenderer> poseCpuRenderer;
            // Workers
            TWorker scaleAndSizeExtractorW;
            TWorker cvMatToOpInputW;
            TWorker cvMatToOpOutputW;
            bool addCvMatToOpOutput = renderOutput;
            bool addCvMatToOpOutputInCpu = addCvMatToOpOutput;
            std::vector<std::vector<TWorker>> poseExtractorsWs;
            std::vector<std::vector<TWorker>> poseTriangulationsWs;
            std::vector<std::vector<TWorker>> jointAngleEstimationsWs;
            std::vector<TWorker> postProcessingWs;
            if (numberGpuThreads > 0)
            {
                // Get input scales and sizes
                const auto scaleAndSizeExtractor = std::make_shared<ScaleAndSizeExtractor>(
                    wrapperStructPose.netInputSize, finalOutputSize, wrapperStructPose.scalesNumber,
                    wrapperStructPose.scaleGap
                );
                scaleAndSizeExtractorW = std::make_shared<WScaleAndSizeExtractor<TDatumsSP>>(scaleAndSizeExtractor);

                // Input cvMat to OpenPose input & output format
                // Note: resize on GPU reduces accuracy about 0.1%
                bool resizeOnCpu = true;
                // const auto resizeOnCpu = (wrapperStructPose.poseMode != PoseMode::Enabled);
                if (resizeOnCpu)
                {
                    const auto gpuResize = false;
                    const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(
                        wrapperStructPose.poseModel, gpuResize);
                    cvMatToOpInputW = std::make_shared<WCvMatToOpInput<TDatumsSP>>(cvMatToOpInput);
                }
                // Note: We realized that somehow doing it on GPU for any number of GPUs does speedup the whole OP
                resizeOnCpu = false;
                addCvMatToOpOutputInCpu = addCvMatToOpOutput
                    && (resizeOnCpu || !renderOutputGpu || wrapperStructPose.poseMode != PoseMode::Enabled
                        // Resize in GPU causing bug
                        || wrapperStructPose.outputSize.x != -1 || wrapperStructPose.outputSize.y != -1);
                if (addCvMatToOpOutputInCpu)
                {
                    const auto gpuResize = false;
                    const auto cvMatToOpOutput = std::make_shared<CvMatToOpOutput>(gpuResize);
                    cvMatToOpOutputW = std::make_shared<WCvMatToOpOutput<TDatumsSP>>(cvMatToOpOutput);
                }

                // Pose estimators & renderers
                std::vector<TWorker> cpuRenderers;
                poseExtractorsWs.clear();
                poseExtractorsWs.resize(numberGpuThreads);
                if (wrapperStructPose.poseMode != PoseMode::Disabled)
                {
                    // Pose estimators
                    for (auto gpuId = 0; gpuId < numberGpuThreads; gpuId++)
                        poseExtractorNets.emplace_back(std::make_shared<PoseExtractorCaffe>(
                            wrapperStructPose.poseModel, modelFolder, gpuId + gpuNumberStart,
                            wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScaleMode,
                            wrapperStructPose.addPartCandidates, wrapperStructPose.maximizePositives,
                            wrapperStructPose.protoTxtPath.getStdString(),
                            wrapperStructPose.caffeModelPath.getStdString(),
                            wrapperStructPose.upsamplingRatio, wrapperStructPose.poseMode == PoseMode::Enabled,
                            wrapperStructPose.enableGoogleLogging
                        ));

                    // Pose renderers
                    if (renderOutputGpu || renderModePose == RenderMode::Cpu)
                    {
                        // If renderModePose != RenderMode::Gpu but renderOutput, then we create an
                        // alpha = 0 pose renderer in order to keep the removing background option
                        const auto alphaKeypoint = (renderModePose != RenderMode::None
                                                    ? wrapperStructPose.alphaKeypoint : 0.f);
                        const auto alphaHeatMap = (renderModePose != RenderMode::None
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
                        if (renderModePose == RenderMode::Cpu)
                        {
                            poseCpuRenderer = std::make_shared<PoseCpuRenderer>(
                                wrapperStructPose.poseModel, wrapperStructPose.renderThreshold,
                                wrapperStructPose.blendOriginalFrame, alphaKeypoint, alphaHeatMap,
                                wrapperStructPose.defaultPartToRender);
                            cpuRenderers.emplace_back(std::make_shared<WPoseRenderer<TDatumsSP>>(poseCpuRenderer));
                        }
                    }
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

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
                        // If we want the initial image resize on GPU
                        if (cvMatToOpInputW == nullptr)
                        {
                            const auto gpuResize = true;
                            const auto cvMatToOpInput = std::make_shared<CvMatToOpInput>(
                                wrapperStructPose.poseModel, gpuResize);
                            poseExtractorsWs.at(i).emplace_back(
                                std::make_shared<WCvMatToOpInput<TDatumsSP>>(cvMatToOpInput));
                        }
                        // If we want the final image resize on GPU
                        if (addCvMatToOpOutput && cvMatToOpOutputW == nullptr)
                        {
                            const auto gpuResize = true;
                            cvMatToOpOutputs.emplace_back(std::make_shared<CvMatToOpOutput>(gpuResize));
                            poseExtractorsWs.at(i).emplace_back(
                                std::make_shared<WCvMatToOpOutput<TDatumsSP>>(cvMatToOpOutputs.back()));
                        }
                        poseExtractorsWs.at(i).emplace_back(
                            std::make_shared<WPoseExtractor<TDatumsSP>>(poseExtractor));
                        // poseExtractorsWs.at(i) = {std::make_shared<WPoseExtractor<TDatumsSP>>(poseExtractor)};
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
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Pose renderer(s)
                if (!poseGpuRenderers.empty())
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    for (auto i = 0u; i < poseExtractorsWs.size(); i++)
                    {
                        poseExtractorsWs.at(i).emplace_back(std::make_shared<WPoseRenderer<TDatumsSP>>(
                            poseGpuRenderers.at(i)));
                        // Get shared params
                        if (!cvMatToOpOutputs.empty())
                            poseGpuRenderers.at(i)->setSharedParameters(
                                cvMatToOpOutputs.at(i)->getSharedParameters());
                    }
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Face extractor(s)
                if (wrapperStructFace.enable)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Face detector
                    // OpenPose body-based face detector
                    if (wrapperStructFace.detector == Detector::Body)
                    {
                        // Sanity check
                        if (wrapperStructPose.poseMode == PoseMode::Disabled)
                            error("Body keypoint detection is disabled but face Detector is set to Body. Either"
                                  " re-enable OpenPose body or select a different face Detector (`--face_detector`).",
                                  __LINE__, __FUNCTION__, __FILE__);
                        // Constructors
                        const auto faceDetector = std::make_shared<FaceDetector>(wrapperStructPose.poseModel);
                        for (auto& wPose : poseExtractorsWs)
                            wPose.emplace_back(std::make_shared<WFaceDetector<TDatumsSP>>(faceDetector));
                    }
                    // OpenCV face detector
                    else if (wrapperStructFace.detector == Detector::OpenCV)
                    {
                        opLog("Body keypoint detection is disabled. Hence, using OpenCV face detector (much less"
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
                    // If provided by user: We do not need to create a FaceDetector
                    // Unknown face Detector
                    else if (wrapperStructFace.detector != Detector::Provided)
                        error("Unknown face Detector. Select a valid face Detector (`--face_detector`).",
                              __LINE__, __FUNCTION__, __FILE__);
                    // Face keypoint extractor
                    for (auto gpu = 0u; gpu < poseExtractorsWs.size(); gpu++)
                    {
                        // Face keypoint extractor
                        const auto netOutputSize = wrapperStructFace.netInputSize;
                        const auto faceExtractorNet = std::make_shared<FaceExtractorCaffe>(
                            wrapperStructFace.netInputSize, netOutputSize, modelFolder,
                            gpu + gpuNumberStart, wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScaleMode,
                            wrapperStructPose.enableGoogleLogging
                        );
                        faceExtractorNets.emplace_back(faceExtractorNet);
                        poseExtractorsWs.at(gpu).emplace_back(
                            std::make_shared<WFaceExtractorNet<TDatumsSP>>(faceExtractorNet));
                    }
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Hand extractor(s)
                if (wrapperStructHand.enable)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    const auto handDetector = std::make_shared<HandDetector>(wrapperStructPose.poseModel);
                    for (auto gpu = 0u; gpu < poseExtractorsWs.size(); gpu++)
                    {
                        // Sanity check
                        if ((wrapperStructHand.detector == Detector::BodyWithTracking
                             || wrapperStructHand.detector == Detector::Body)
                            && wrapperStructPose.poseMode == PoseMode::Disabled)
                            error("Body keypoint detection is disabled but hand Detector is set to Body. Either"
                                  " re-enable OpenPose body or select a different hand Detector (`--hand_detector`).",
                                  __LINE__, __FUNCTION__, __FILE__);
                        // Hand detector
                        // OpenPose body-based hand detector with tracking
                        if (wrapperStructHand.detector == Detector::BodyWithTracking)
                        {
                            poseExtractorsWs.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorTracking<TDatumsSP>>(handDetector));
                        }
                        // OpenPose body-based hand detector
                        else if (wrapperStructHand.detector == Detector::Body)
                        {
                            poseExtractorsWs.at(gpu).emplace_back(
                                std::make_shared<WHandDetector<TDatumsSP>>(handDetector));
                        }
                        // If provided by user: We do not need to create a FaceDetector
                        // Unknown hand Detector
                        else if (wrapperStructHand.detector != Detector::Provided)
                            error("Unknown hand Detector. Select a valid hand Detector (`--hand_detector`).",
                                  __LINE__, __FUNCTION__, __FILE__);
                        // Hand keypoint extractor
                        const auto netOutputSize = wrapperStructHand.netInputSize;
                        const auto handExtractorNet = std::make_shared<HandExtractorCaffe>(
                            wrapperStructHand.netInputSize, netOutputSize, modelFolder,
                            gpu + gpuNumberStart, wrapperStructHand.scalesNumber, wrapperStructHand.scaleRange,
                            wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScaleMode,
                            wrapperStructPose.enableGoogleLogging
                        );
                        handExtractorNets.emplace_back(handExtractorNet);
                        poseExtractorsWs.at(gpu).emplace_back(
                            std::make_shared<WHandExtractorNet<TDatumsSP>>(handExtractorNet)
                            );
                        // If OpenPose body-based hand detector with tracking
                        if (wrapperStructHand.detector == Detector::BodyWithTracking)
                            poseExtractorsWs.at(gpu).emplace_back(
                                std::make_shared<WHandDetectorUpdate<TDatumsSP>>(handDetector));
                    }
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Face renderer(s)
                if (renderFace)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // CPU rendering
                    if (renderModeFace == RenderMode::Cpu)
                    {
                        // Construct face renderer
                        const auto faceRenderer = std::make_shared<FaceCpuRenderer>(
                            wrapperStructFace.renderThreshold, wrapperStructFace.alphaKeypoint,
                            wrapperStructFace.alphaHeatMap);
                        // Add worker
                        cpuRenderers.emplace_back(std::make_shared<WFaceRenderer<TDatumsSP>>(faceRenderer));
                    }
                    // GPU rendering
                    else if (renderModeFace == RenderMode::Gpu)
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
                                // const bool isLastRenderer = !renderHandGpu;
                                const bool isLastRenderer = !renderHandGpu && !(addCvMatToOpOutput && !addCvMatToOpOutputInCpu);
                                const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(
                                    poseGpuRenderers.at(i));
                                faceRenderer->setSharedParametersAndIfLast(
                                    renderer->getSharedParameters(), isLastRenderer);
                            }
                            // Add worker
                            poseExtractorsWs.at(i).emplace_back(
                                std::make_shared<WFaceRenderer<TDatumsSP>>(faceRenderer));
                        }
                    }
                    else
                        error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Hand renderer(s)
                if (renderHand)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // CPU rendering
                    if (renderModeHand == RenderMode::Cpu)
                    {
                        // Construct hand renderer
                        const auto handRenderer = std::make_shared<HandCpuRenderer>(
                            wrapperStructHand.renderThreshold, wrapperStructHand.alphaKeypoint,
                            wrapperStructHand.alphaHeatMap);
                        // Add worker
                        cpuRenderers.emplace_back(std::make_shared<WHandRenderer<TDatumsSP>>(handRenderer));
                    }
                    // GPU rendering
                    else if (renderModeHand == RenderMode::Gpu)
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
                                // const bool isLastRenderer = true;
                                const bool isLastRenderer = !(addCvMatToOpOutput && !addCvMatToOpOutputInCpu);
                                const auto renderer = std::static_pointer_cast<PoseGpuRenderer>(
                                    poseGpuRenderers.at(i));
                                handRenderer->setSharedParametersAndIfLast(
                                    renderer->getSharedParameters(), isLastRenderer);
                            }
                            // Add worker
                            poseExtractorsWs.at(i).emplace_back(
                                std::make_shared<WHandRenderer<TDatumsSP>>(handRenderer));
                        }
                    }
                    else
                        error("Unknown RenderMode.", __LINE__, __FUNCTION__, __FILE__);
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // Frames processor (OpenPose format -> cv::Mat format)
                if (addCvMatToOpOutput && !addCvMatToOpOutputInCpu)
                {
                    // for (auto& poseExtractorsW : poseExtractorsWs)
                    for (auto i = 0u ; i < poseExtractorsWs.size() ; ++i)
                    {
                        const auto gpuResize = true;
                        opOutputToCvMats.emplace_back(std::make_shared<OpOutputToCvMat>(gpuResize));
                        poseExtractorsWs.at(i).emplace_back(
                            std::make_shared<WOpOutputToCvMat<TDatumsSP>>(opOutputToCvMats.back()));
                        // Assign shared parameters
                        opOutputToCvMats.back()->setSharedParameters(
                            cvMatToOpOutputs.at(i)->getSharedParameters());
                    }
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                // 3-D reconstruction
                poseTriangulationsWs.clear();
                if (wrapperStructExtra.reconstruct3d)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                if (addCvMatToOpOutputInCpu)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    postProcessingWs = mergeVectors(postProcessingWs, cpuRenderers);
                    const auto opOutputToCvMat = std::make_shared<OpOutputToCvMat>();
                    postProcessingWs.emplace_back(std::make_shared<WOpOutputToCvMat<TDatumsSP>>(opOutputToCvMat));
                }
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Re-scale pose if desired
                // If desired scale is not the current input
                if (wrapperStructPose.keypointScaleMode != ScaleMode::InputResolution
                    // and desired scale is not output when size(input) = size(output)
                    && !(wrapperStructPose.keypointScaleMode == ScaleMode::OutputResolution &&
                         (finalOutputSize == producerSize || finalOutputSize.x <= 0 || finalOutputSize.y <= 0))
                    // and desired scale is not net output when size(input) = size(net output)
                    && !(wrapperStructPose.keypointScaleMode == ScaleMode::NetOutputResolution
                         && producerSize == wrapperStructPose.netInputSize))
                {
                    // Then we must rescale the keypoints
                    auto keypointScaler = std::make_shared<KeypointScaler>(wrapperStructPose.keypointScaleMode);
                    postProcessingWs.emplace_back(std::make_shared<WKeypointScaler<TDatumsSP>>(keypointScaler));
                }
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

            // IK/Adam
            const auto displayAdam = wrapperStructGui.displayMode == DisplayMode::DisplayAdam
                                     || (wrapperStructGui.displayMode == DisplayMode::DisplayAll
                                         && wrapperStructExtra.ikThreads > 0);
            jointAngleEstimationsWs.clear();
#ifdef USE_3D_ADAM_MODEL
            if (wrapperStructExtra.ikThreads > 0)
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                jointAngleEstimationsWs.resize(wrapperStructExtra.ikThreads);
                // Pose extractor(s)
                for (auto i = 0u; i < jointAngleEstimationsWs.size(); i++)
                {
                    const auto jointAngleEstimation = std::make_shared<JointAngleEstimation>(displayAdam);
                    jointAngleEstimationsWs.at(i) = {std::make_shared<WJointAngleEstimation<TDatumsSP>>(
                        jointAngleEstimation)};
                }
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
#endif

            // Output workers
            std::vector<TWorker> outputWs;
            // Print verbose
            if (wrapperStructOutput.verbose > 0.)
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto verbosePrinter = std::make_shared<VerbosePrinter>(
                    wrapperStructOutput.verbose, uLongLongRound(producerSharedPtr->get(getCvCapPropFrameCount())));
                outputWs.emplace_back(std::make_shared<WVerbosePrinter<TDatumsSP>>(verbosePrinter));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Send information (e.g., to Unity) though UDP client-server communication

#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.udpHost.empty() && !wrapperStructOutput.udpPort.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto udpSender = std::make_shared<UdpSender>(wrapperStructOutput.udpHost,
                                                                   wrapperStructOutput.udpPort);
                outputWs.emplace_back(std::make_shared<WUdpSender<TDatumsSP>>(udpSender));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
#endif
            // Write people pose data on disk (json for OpenCV >= 3, xml, yml...)
            if (!writeKeypointCleaned.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto keypointSaver = std::make_shared<KeypointSaver>(writeKeypointCleaned,
                                                                           wrapperStructOutput.writeKeypointFormat);
                outputWs.emplace_back(std::make_shared<WPoseSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructFace.enable)
                    outputWs.emplace_back(std::make_shared<WFaceSaver<TDatumsSP>>(keypointSaver));
                if (wrapperStructHand.enable)
                    outputWs.emplace_back(std::make_shared<WHandSaver<TDatumsSP>>(keypointSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Write OpenPose output data on disk in JSON format (body/hand/face keypoints, body part locations if
            // enabled, etc.)
            if (!writeJsonCleaned.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto peopleJsonSaver = std::make_shared<PeopleJsonSaver>(writeJsonCleaned);
                outputWs.emplace_back(std::make_shared<WPeopleJsonSaver<TDatumsSP>>(peopleJsonSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Write people pose/foot/face/hand/etc. data on disk (COCO validation JSON format)
            if (!wrapperStructOutput.writeCocoJson.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // If humanFormat: bigger size (& maybe slower to process), but easier for user to read it
                const auto humanFormat = true;
                const auto cocoJsonSaver = std::make_shared<CocoJsonSaver>(
                    wrapperStructOutput.writeCocoJson.getStdString(), wrapperStructPose.poseModel, humanFormat,
                    wrapperStructOutput.writeCocoJsonVariants,
                    (wrapperStructPose.poseModel != PoseModel::CAR_22
                        && wrapperStructPose.poseModel != PoseModel::CAR_12
                        ? CocoJsonFormat::Body : CocoJsonFormat::Car),
                    wrapperStructOutput.writeCocoJsonVariant);
                outputWs.emplace_back(std::make_shared<WCocoJsonSaver<TDatumsSP>>(cocoJsonSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Write frames as desired image format on hard disk
            if (!writeImagesCleaned.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto imageSaver = std::make_shared<ImageSaver>(
                    writeImagesCleaned, wrapperStructOutput.writeImagesFormat.getStdString());
                outputWs.emplace_back(std::make_shared<WImageSaver<TDatumsSP>>(imageSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            auto originalVideoFps = 0.;
            if (!wrapperStructOutput.writeVideo.empty() || !wrapperStructOutput.writeVideo3D.empty()
                || !wrapperStructOutput.writeBvh.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                if (wrapperStructOutput.writeVideoFps <= 0
                    && (!oPProducer || producerSharedPtr->get(getCvCapPropFrameFps()) <= 0))
                    error("The frame rate of the frames producer is unknown. Set `--write_video_fps` to your desired"
                          " FPS if you wanna record video (`--write_video`). E.g., if it is a folder of images, you"
                          " will have to know or guess the frame rate; if it is a webcam, you should use the OpenPose"
                          " displayed FPS as desired value. If you do not care, simply add `--write_video_fps 30`.",
                          __LINE__, __FUNCTION__, __FILE__);
                originalVideoFps = (
                    wrapperStructOutput.writeVideoFps > 0 ?
                    wrapperStructOutput.writeVideoFps : producerSharedPtr->get(getCvCapPropFrameFps()));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Write frames as *.avi video on hard disk
            if (!wrapperStructOutput.writeVideo.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Sanity checks
                if (!oPProducer)
                    error("Video file can only be recorded inside `wrapper/wrapper.hpp` if the producer"
                          " is one of the default ones (e.g., video, webcam, ...).",
                          __LINE__, __FUNCTION__, __FILE__);
                if (wrapperStructOutput.writeVideoWithAudio && producerSharedPtr->getType() != ProducerType::Video)
                    error("Audio can only be added to the output saved video if the input is also a video (either"
                          " disable `--write_video_with_audio` or use a video as input with `--video`).",
                          __LINE__, __FUNCTION__, __FILE__);
                // Create video saver worker
                const auto videoSaver = std::make_shared<VideoSaver>(
                    wrapperStructOutput.writeVideo.getStdString(), getCvFourcc('M','J','P','G'), originalVideoFps,
                    (wrapperStructOutput.writeVideoWithAudio ? wrapperStructInput.producerString.getStdString() : ""));
                outputWs.emplace_back(std::make_shared<WVideoSaver<TDatumsSP>>(videoSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Write joint angles as *.bvh file on hard disk
#ifdef USE_3D_ADAM_MODEL
            if (!wrapperStructOutput.writeBvh.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto bvhSaver = std::make_shared<BvhSaver>(
                    wrapperStructOutput.writeBvh, JointAngleEstimation::getTotalModel(), originalVideoFps
                );
                outputWs.emplace_back(std::make_shared<WBvhSaver<TDatumsSP>>(bvhSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
#endif
            // Write heat maps as desired image format on hard disk
            if (!writeHeatMapsCleaned.empty())
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto heatMapSaver = std::make_shared<HeatMapSaver>(
                    writeHeatMapsCleaned, wrapperStructOutput.writeHeatMapsFormat.getStdString());
                outputWs.emplace_back(std::make_shared<WHeatMapSaver<TDatumsSP>>(heatMapSaver));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Add frame information for GUI
            const bool guiEnabled = (wrapperStructGui.displayMode != DisplayMode::NoDisplay);
            // If this WGuiInfoAdder instance is placed before the WImageSaver or WVideoSaver, then the resulting
            // recorded frames will look exactly as the final displayed image by the GUI
            if (wrapperStructGui.guiVerbose && (guiEnabled || !userOutputWs.empty()
                                                || threadManagerMode == ThreadManagerMode::Asynchronous
                                                || threadManagerMode == ThreadManagerMode::AsynchronousOut))
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                const auto guiInfoAdder = std::make_shared<GuiInfoAdder>(numberGpuThreads, guiEnabled);
                outputWs.emplace_back(std::make_shared<WGuiInfoAdder<TDatumsSP>>(guiInfoAdder));
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Minimal graphical user interface (GUI)
            TWorker guiW;
            TWorker videoSaver3DW;
            if (guiEnabled)
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // PoseRenderers to Renderers
                std::vector<std::shared_ptr<Renderer>> renderers;
                if (renderModePose == RenderMode::Cpu)
                    renderers.emplace_back(std::static_pointer_cast<Renderer>(poseCpuRenderer));
                else
                    for (const auto& poseGpuRenderer : poseGpuRenderers)
                        renderers.emplace_back(std::static_pointer_cast<Renderer>(poseGpuRenderer));
                // Display
                const auto numberViews = (producerSharedPtr != nullptr
                    ? positiveIntRound(producerSharedPtr->get(ProducerProperty::NumberViews)) : 1);
                auto finalOutputSizeGui = finalOutputSize;
                if (numberViews > 1 && finalOutputSizeGui.x > 0)
                    finalOutputSizeGui.x *= numberViews;
                // Adam (+3-D/2-D) display
                if (displayAdam)
                {
#ifdef USE_3D_ADAM_MODEL
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                            wrapperStructOutput.writeVideo3D.getStdString(), getCvFourcc('M','J','P','G'), originalVideoFps, "");
                        videoSaver3DW = std::make_shared<WVideoSaver3D<TDatumsSP>>(videoSaver);
                    }
                }
                // 2-D display
                else if (wrapperStructGui.displayMode == DisplayMode::Display2D)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Set FpsMax
            TWorker wFpsMax;
            if (wrapperStructPose.fpsMax > 0.)
                wFpsMax = std::make_shared<WFpsMax<TDatumsSP>>(wrapperStructPose.fpsMax);
            // Set wrapper as configured
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);





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
            // If custom user Worker and uses its own thread
            std::vector<TWorker> workersAux;
            if (!userPreProcessingWs.empty())
            {
                // If custom user Worker in its own thread
                if (userPreProcessingWsOnNewThread)
                    opLog("You chose to add your pre-processing function in a new thread. However, OpenPose will"
                        " add it in the same thread than the input frame producer.",
                        Priority::High, __LINE__, __FUNCTION__, __FILE__);
                workersAux = mergeVectors(workersAux, {userPreProcessingWs});
            }
            workersAux = mergeVectors(workersAux, {wIdGenerator});
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
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wPose, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames - Required own thread
                    if (poseExtractorsWs.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (poseExtractorsWs.size() > 1)
                        opLog("Multi-threading disabled, only 1 thread running. All GPUs have been disabled but the"
                            " first one, which is defined by gpuNumberStart (e.g., in the OpenPose demo, it is set"
                            " with the `--num_gpu_start` flag).", Priority::High);
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, poseExtractorsWs.at(0), queueIn++, queueOut++);
                }
            }
            // Assemble all frames from same time instant (3-D module)
            const auto wQueueAssembler = std::make_shared<WQueueAssembler<TDatums>>();
            // 3-D reconstruction
            if (!poseTriangulationsWs.empty())
            {
                // Assemble frames
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, wQueueAssembler, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
                // 3-D reconstruction
                if (multiThreadEnabled)
                {
                    for (auto& wPoseTriangulations : poseTriangulationsWs)
                    {
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wPoseTriangulations, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (poseTriangulationsWs.size() > 1u)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (poseTriangulationsWs.size() > 1)
                        opLog("Multi-threading disabled, only 1 thread running for 3-D triangulation.",
                            Priority::High);
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wJointAngleEstimator, queueIn, queueOut);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                    queueIn++;
                    queueOut++;
                    // Sort frames
                    if (jointAngleEstimationsWs.size() > 1)
                    {
                        const auto wQueueOrderer = std::make_shared<WQueueOrderer<TDatumsSP>>();
                        opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                        threadManager.add(threadId, wQueueOrderer, queueIn++, queueOut++);
                        threadIdPP(threadId, multiThreadEnabled);
                    }
                }
                else
                {
                    if (jointAngleEstimationsWs.size() > 1)
                        opLog("Multi-threading disabled, only 1 thread running for joint angle estimation.",
                            Priority::High);
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, jointAngleEstimationsWs.at(0), queueIn++, queueOut++);
                }
            }
            // Post processing workers
            if (!postProcessingWs.empty())
            {
                // Combining postProcessingWs and outputWs
                outputWs = mergeVectors(postProcessingWs, outputWs);
                // // If I wanna split them
                // opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // threadManager.add(threadId, postProcessingWs, queueIn++, queueOut++);
                // threadIdPP(threadId, multiThreadEnabled);
            }
            // If custom user Worker and uses its own thread
            if (!userPostProcessingWs.empty())
            {
                // If custom user Worker in its own thread
                if (userPostProcessingWsOnNewThread)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, outputWs, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
            // User output worker
            // Thread Y, queues Q -> Q+1
            if (!userOutputWs.empty())
            {
                if (userOutputWsOnNewThread)
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId, userOutputWs, queueIn++, queueOut++);
                    threadIdPP(threadId, multiThreadEnabled);
                }
                else
                {
                    opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    threadManager.add(threadId-1, userOutputWs, queueIn++, queueOut++);
                }
            }
            // OpenPose GUI
            if (guiW != nullptr)
            {
                // Thread Y+1, queues Q+1 -> Q+2
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, guiW, queueIn++, queueOut++);
                // Saving 3D output
                if (videoSaver3DW != nullptr)
                    threadManager.add(threadId, videoSaver3DW, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
            opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Setting maximum speed
            if (wFpsMax != nullptr)
            {
                opLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                threadManager.add(threadId, wFpsMax, queueIn++, queueOut++);
                threadIdPP(threadId, multiThreadEnabled);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatum, typename TDatums, typename TDatumsSP>
    void createMultiviewTDatum(
        TDatumsSP& tDatumsSP, unsigned long long& frameCounter,
        const CameraParameterReader& cameraParameterReader, const void* const cvMatPtr)
    {
        try
        {
            // Sanity check
            if (tDatumsSP == nullptr)
                op::error("tDatumsSP was nullptr, it must be initialized.", __LINE__, __FUNCTION__, __FILE__);
            // Camera parameters
            const std::vector<op::Matrix>& cameraMatrices = cameraParameterReader.getCameraMatrices();
            const std::vector<op::Matrix>& cameraIntrinsics = cameraParameterReader.getCameraIntrinsics();
            const std::vector<op::Matrix>& cameraExtrinsics = cameraParameterReader.getCameraExtrinsics();
            const auto matrixesSize = cameraMatrices.size();
            // More sanity checks
            if (cameraMatrices.size() < 2)
                op::error("There is less than 2 camera parameter matrices.",
                    __LINE__, __FUNCTION__, __FILE__);
            if (cameraMatrices.size() != cameraIntrinsics.size() || cameraMatrices.size() != cameraExtrinsics.size())
                op::error("Camera parameters must have the same size.", __LINE__, __FUNCTION__, __FILE__);
            // Split image to process
            std::vector<op::Matrix> imagesToProcess(matrixesSize);
            op::Matrix::splitCvMatIntoVectorMatrix(imagesToProcess, cvMatPtr);
            // Fill tDatumsSP
            tDatumsSP->resize(cameraMatrices.size());
            for (auto datumIndex = 0 ; datumIndex < matrixesSize ; ++datumIndex)
            {
                auto& datumPtr = tDatumsSP->at(datumIndex);
                datumPtr = std::make_shared<op::Datum>();
                datumPtr->frameNumber = frameCounter;
                datumPtr->cvInputData = imagesToProcess[datumIndex];
                if (matrixesSize > 1)
                {
                    datumPtr->subId = datumIndex;
                    datumPtr->subIdMax = matrixesSize-1;
                    datumPtr->cameraMatrix = cameraMatrices[datumIndex];
                    datumPtr->cameraExtrinsics = cameraExtrinsics[datumIndex];
                    datumPtr->cameraIntrinsics = cameraIntrinsics[datumIndex];
                }
            }
            ++frameCounter;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}

#endif // OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP
