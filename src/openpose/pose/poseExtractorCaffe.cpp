#include <limits> // std::numeric_limits
#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/bodyPartConnectorCaffe.hpp>
#include <openpose/net/maximumCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/netOpenCv.hpp>
#include <openpose/net/nmsCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>

namespace op
{
    const bool TOP_DOWN_REFINEMENT = false; // Note: +5% acc 1 scale, -2% max acc setting

    struct PoseExtractorCaffe::ImplPoseExtractorCaffe
    {
        #ifdef USE_CAFFE
            // Used when increasing spNets
            const PoseModel mPoseModel;
            const int mGpuId;
            const std::string mModelFolder;
            const bool mEnableGoogleLogging;
            // General parameters
            std::vector<std::shared_ptr<Net>> spNets;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
            std::vector<std::vector<int>> mNetInput4DSizes;
            std::vector<double> mScaleInputToNetInputs;
            // Init with thread
            std::vector<boost::shared_ptr<caffe::Blob<float>>> spCaffeNetOutputBlobs;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> spMaximumPeaksBlob;

            ImplPoseExtractorCaffe(const PoseModel poseModel, const int gpuId,
                                   const std::string& modelFolder, const bool enableGoogleLogging) :
                mPoseModel{poseModel},
                mGpuId{gpuId},
                mModelFolder{modelFolder},
                mEnableGoogleLogging{enableGoogleLogging},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()},
                spMaximumCaffe{(TOP_DOWN_REFINEMENT ? std::make_shared<MaximumCaffe<float>>() : nullptr)}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
            std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
        {
            try
            {
                // Prepare spCaffeNetOutputBlobss
                std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
                for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
                    caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
                return caffeNetOutputBlobs;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }

        inline void reshapePoseExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
                                              std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
                                              std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
                                              std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
                                              std::shared_ptr<caffe::Blob<float>>& maximumPeaksBlob,
                                              const float scaleInputToNetInput,
                                              const PoseModel poseModel,
                                              const int gpuID)
        {
            try
            {
                // HeatMaps extractor blob and layer
                // Caffe modifies bottom - Heatmap gets resized
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);
                resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {heatMapsBlob.get()},
                                             getPoseNetDecreaseFactor(poseModel), 1.f/scaleInputToNetInput, true,
                                             gpuID);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(),
                                  getPoseNumberBodyParts(poseModel), gpuID);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()});
                if (TOP_DOWN_REFINEMENT)
                    maximumCaffe->Reshape({heatMapsBlob.get()}, {maximumPeaksBlob.get()});
                // Cuda check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void addCaffeNetOnThread(std::vector<std::shared_ptr<Net>>& net,
                                 std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                 const PoseModel poseModel, const int gpuId,
                                 const std::string& modelFolder, const bool enableGoogleLogging)
        {
            try
            {
                // Add Caffe Net
                net.emplace_back(
                    std::make_shared<NetCaffe>(
                        modelFolder + getPoseProtoTxt(poseModel),
                        modelFolder + getPoseTrainedModel(poseModel),
                        gpuId, enableGoogleLogging));
                // net.emplace_back(
                //     std::make_shared<NetOpenCv>(
                //         modelFolder + getPoseProtoTxt(poseModel),
                //         modelFolder + getPoseTrainedModel(poseModel),
                //         gpuId));
                // UNUSED(enableGoogleLogging);
                // Initializing them on the thread
                net.back()->initializationOnThread();
                caffeNetOutputBlob.emplace_back(((NetCaffe*)net.back().get())->getOutputBlob());
                // caffeNetOutputBlob.emplace_back(((NetOpenCv*)net.back().get())->getOutputBlob());
                // Sanity check
                if (net.size() != caffeNetOutputBlob.size())
                    error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
                // Cuda check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    PoseExtractorCaffe::PoseExtractorCaffe(const PoseModel poseModel, const std::string& modelFolder,
                                           const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale, const bool addPartCandidates,
                                           const bool maximizePositives, const bool enableGoogleLogging) :
        PoseExtractorNet{poseModel, heatMapTypes, heatMapScale, addPartCandidates, maximizePositives}
        #ifdef USE_CAFFE
        , upImpl{new ImplPoseExtractorCaffe{poseModel, gpuId, modelFolder, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
                // Layers parameters
                upImpl->spBodyPartConnectorCaffe->setPoseModel(upImpl->mPoseModel);
                upImpl->spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
            #else
                UNUSED(poseModel);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScale);
                UNUSED(addPartCandidates);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorCaffe::~PoseExtractorCaffe()
    {
    }

    void PoseExtractorCaffe::netInitializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                // Logging
                log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Initialize Caffe net
                addCaffeNetOnThread(upImpl->spNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                    upImpl->mGpuId, upImpl->mModelFolder, upImpl->mEnableGoogleLogging);
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Initialize blobs
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                if (TOP_DOWN_REFINEMENT)
                    upImpl->spMaximumPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Logging
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorCaffe::forwardPass(const std::vector<Array<float>>& inputNetData,
                                         const Point<int>& inputDataSize,
                                         const std::vector<double>& scaleInputToNetInputs)
    {
        try
        {
            #ifdef USE_CAFFE
                // Sanity checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                for (const auto& inputNetDataI : inputNetData)
                    if (inputNetDataI.empty())
                        error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                if (inputNetData.size() != scaleInputToNetInputs.size())
                    error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                          __LINE__, __FUNCTION__, __FILE__);

                // Resize std::vectors if required
                const auto numberScales = inputNetData.size();
                upImpl->mNetInput4DSizes.resize(numberScales);
                while (upImpl->spNets.size() < numberScales)
                    addCaffeNetOnThread(upImpl->spNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                        upImpl->mGpuId, upImpl->mModelFolder, false);

                // Process each image
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    // 1. Caffe deep network
                    // ~80ms
                    upImpl->spNets.at(i)->forwardPass(inputNetData[i]);

                    // Reshape blobs if required
                    // Note: In order to resize to input size to have same results as Matlab, uncomment the commented
                    // lines
                    // Note: For dynamic sizes (e.g., a folder with images of different aspect ratio)
                    const auto changedVectors = !vectorsAreEqual(
                        upImpl->mNetInput4DSizes.at(i), inputNetData[i].getSize());
                    if (changedVectors)
                        // || !vectorsAreEqual(upImpl->mScaleInputToNetInputs, scaleInputToNetInputs))
                    {
                        upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                        // upImpl->mScaleInputToNetInputs = scaleInputToNetInputs;
                        reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                                  upImpl->spBodyPartConnectorCaffe, upImpl->spMaximumCaffe,
                                                  upImpl->spCaffeNetOutputBlobs, upImpl->spHeatMapsBlob,
                                                  upImpl->spPeaksBlob, upImpl->spMaximumPeaksBlob,
                                                  1.f, upImpl->mPoseModel, upImpl->mGpuId);
                                                  // scaleInputToNetInputs[i] vs. 1.f
                    }
                    // Get scale net to output (i.e., image input)
                    if (changedVectors || TOP_DOWN_REFINEMENT)
                        mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
                                                    upImpl->mNetInput4DSizes[0][2]};
                }
                // 2. Resize heat maps + merge different scales
                // ~5ms (GPU) / ~20ms (CPU)
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spCaffeNetOutputBlobs);
                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                upImpl->spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()});
                // Get scale net to output (i.e., image input)
                // Note: In order to resize to input size, (un)comment the following lines
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                         intRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
                // mScaleNetToOutput = 1.f;
                // 3. Get peaks by Non-Maximum Suppression
                // ~2ms (GPU) / ~7ms (CPU)
                const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
                upImpl->spNmsCaffe->setThreshold(nmsThreshold);
                const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
                upImpl->spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                upImpl->spNmsCaffe->Forward({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});
                // 4. Connecting body parts
                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (float)get(PoseProperty::ConnectInterMinAboveThreshold));
                upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
                // Note: BODY_25D will crash (only implemented for CPU version)
                upImpl->spBodyPartConnectorCaffe->Forward(
                    {upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
                // Re-run on each person
                if (TOP_DOWN_REFINEMENT)
                {
                    // Get each person rectangle
                    for (auto person = 0 ; person < mPoseKeypoints.getSize(0) ; person++)
                    {
                        // Get person rectangle resized to input size
                        const auto rectangleF = getKeypointsRectangle(mPoseKeypoints, person, nmsThreshold)
                                              / mScaleNetToOutput;
                        // Make rectangle bigger to make sure the whole body is inside
                        cv::Rect cvRectangle{
                            intRound(rectangleF.x - 0.2*rectangleF.width),
                            intRound(rectangleF.y - 0.2*rectangleF.height),
                            intRound(rectangleF.width*1.4),
                            intRound(rectangleF.height*1.4)
                        };
                        keepRoiInside(cvRectangle, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                        // Input size
                        // // Note: In order to preserve speed but maximize accuracy
                        // // If e.g. rectange = 10x1 and inputSize = 656x368 --> targetSize = 656x368
                        // // Note: If e.g. rectange = 1x10 and inputSize = 656x368 --> targetSize = 368x656
                        // const auto width = ( ? cvRectangle.width : cvRectangle.height);
                        // const auto height = (width == cvRectangle.width ? cvRectangle.height : cvRectangle.width);
                        // const Point<int> inputSize{width, height};
                        // Note: If inputNetData.size = -1x368 --> TargetSize = 368x-1
                        const Point<int> inputSizeInit{cvRectangle.width, cvRectangle.height};
                        // Target size
                        Point<int> targetSize;
                        // Optimal case (using training size)
                        if (inputNetData[0].getSize(2) >= 368 || inputNetData[0].getVolume(2,3) >= 135424) // 368^2
                            targetSize = Point<int>{368, 368};
                        // Low resolution cases: Keep same area than biggest scale
                        else
                        {
                            const auto minSide = fastMin(
                                368, fastMin(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                            const auto maxSide = fastMin(
                                368, fastMax(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                            // Person bounding box is vertical
                            if (cvRectangle.width < cvRectangle.height)
                                targetSize = Point<int>{minSide, maxSide};
                            // Person bounding box is horizontal
                            else
                                targetSize = Point<int>{maxSide, minSide};
                        }
                        // Fill resizedImage
                        /*const*/ auto scaleNetToRoi = resizeGetScaleFactor(inputSizeInit, targetSize);
                        // Update rectangle to avoid black padding and instead take full advantage of the network area
                        const auto padding = Point<int>{
                            intRound((targetSize.x-1) / scaleNetToRoi + 1 - inputSizeInit.x),
                            intRound((targetSize.y-1) / scaleNetToRoi + 1 - inputSizeInit.y)
                        };
                        // Width requires padding
                        if (padding.x > 2 || padding.y > 2) // 2 pixels as threshold
                        {
                            if (padding.x > 2) // 2 pixels as threshold
                            {
                                cvRectangle.x -= padding.x/2;
                                cvRectangle.width += padding.x;
                            }
                            else if (padding.y > 2) // 2 pixels as threshold
                            {
                                cvRectangle.y -= padding.y/2;
                                cvRectangle.height += padding.y;
                            }
                            keepRoiInside(cvRectangle, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                            scaleNetToRoi = resizeGetScaleFactor(
                                Point<int>{cvRectangle.width, cvRectangle.height}, targetSize);
                        }
                        // No if scaleNetToRoi < 1 (image would be shrinked, so we assume best result already obtained)
                        if (scaleNetToRoi > 1)
                        {
                            const auto areaInput = inputNetData[0].getVolume(2,3);
                            const auto areaRoi = targetSize.area();
                            Array<float> inputNetDataRoi{{1, 3, targetSize.y, targetSize.x}};
                            for (auto c = 0u ; c < 3u ; c++)
                            {
                                // Input image
                                const cv::Mat wholeInputCvMat(
                                    inputNetData[0].getSize(2), inputNetData[0].getSize(3), CV_32FC1,
                                    inputNetData[0].getPseudoConstPtr() + c * areaInput);
                                // Input image cropped
                                const cv::Mat inputCvMat(wholeInputCvMat, cvRectangle);
                                // Resize image for inputNetDataRoi
                                cv::Mat resizedImageCvMat(
                                    inputNetDataRoi.getSize(2), inputNetDataRoi.getSize(3), CV_32FC1,
                                    inputNetDataRoi.getPtr() + c * areaRoi);
                                resizeFixedAspectRatio(resizedImageCvMat, inputCvMat, scaleNetToRoi, targetSize);
                            }

                            // Re-Process image
                            // 1. Caffe deep network
                            upImpl->spNets.at(0)->forwardPass(inputNetDataRoi);
                            std::vector<boost::shared_ptr<caffe::Blob<float>>> caffeNetOutputBlob{upImpl->spCaffeNetOutputBlobs[0]};
                            // Reshape blobs
                            if (!vectorsAreEqual(upImpl->mNetInput4DSizes.at(0), inputNetDataRoi.getSize()))
                            {
                                upImpl->mNetInput4DSizes.at(0) = inputNetDataRoi.getSize();
                                reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                                          upImpl->spBodyPartConnectorCaffe, upImpl->spMaximumCaffe,
                                                          // upImpl->spCaffeNetOutputBlobs,
                                                          caffeNetOutputBlob,
                                                          upImpl->spHeatMapsBlob, upImpl->spPeaksBlob,
                                                          upImpl->spMaximumPeaksBlob, 1.f, upImpl->mPoseModel,
                                                          upImpl->mGpuId);
                            }
                            // 2. Resize heat maps + merge different scales
                            // const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spCaffeNetOutputBlobs);
                            const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);
                            // const std::vector<float> floatScaleRatios(
                            //     scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                            const std::vector<float> floatScaleRatios{(float)scaleInputToNetInputs[0]};
                            upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                            upImpl->spResizeAndMergeCaffe->Forward(
                                caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()});
                            // Get scale net to output (i.e., image input)
                            const auto scaleRoiToOutput = float(mScaleNetToOutput / scaleNetToRoi);
                            // 3. Get peaks by Non-Maximum Suppression
                            const auto nmsThresholdRefined = 0.02f;
                            upImpl->spNmsCaffe->setThreshold(nmsThresholdRefined);
                            const auto nmsOffset = float(0.5/double(scaleRoiToOutput));
                            upImpl->spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                            upImpl->spNmsCaffe->Forward({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});
                            // Define poseKeypoints
                            Array<float> poseKeypoints;
                            Array<float> poseScores;
                            // 4. Connecting body parts
                            // Get scale net to output (i.e., image input)
                            upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(scaleRoiToOutput);
                            upImpl->spBodyPartConnectorCaffe->setInterThreshold(0.01f);
                            upImpl->spBodyPartConnectorCaffe->Forward(
                                {upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, poseKeypoints, poseScores);
                            // If detected people in new subnet
                            if (!poseKeypoints.empty())
                            {
                                // // Scale back keypoints
                                const auto xOffset = float(cvRectangle.x*mScaleNetToOutput);
                                const auto yOffset = float(cvRectangle.y*mScaleNetToOutput);
                                scaleKeypoints2d(poseKeypoints, 1.f, 1.f, xOffset, yOffset);
                                // Re-assign person back
                                // // Option a) Just use biggest person (simplest but fails with crowded people)
                                // const auto personRefined = getBiggestPerson(poseKeypoints, nmsThreshold);
                                // Option b) Get minimum keypoint distance
                                // Get min distance
                                int personRefined = -1;
                                float personAverageDistance = std::numeric_limits<float>::max();
                                for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                                {
                                    // Get average distance
                                    const auto currentAverageDistance = getDistanceAverage(
                                        mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                    // Update person
                                    if (personAverageDistance > currentAverageDistance
                                        && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                            >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                    {
                                        personRefined = person2;
                                        personAverageDistance = currentAverageDistance;
                                    }
                                }
                                // Get max ROI
                                int personRefinedRoi = -1;
                                float personRoi = -1.f;
                                for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                                {
                                    // Get ROI
                                    const auto currentRoi = getKeypointsRoi(
                                        mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                    // Update person
                                    if (personRoi < currentRoi
                                        && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                            >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                    {
                                        personRefinedRoi = person2;
                                        personRoi = currentRoi;
                                    }
                                }
                                // If good refined candidate found
                                // I.e., if both max ROI and min dist match on same person id
                                if (personRefined == personRefinedRoi && personRefined > -1)
                                {
                                    // Update only if avg dist is small enough
                                    const auto personRectangle = getKeypointsRectangle(
                                        mPoseKeypoints, person, nmsThreshold);
                                    const auto personRatio = 0.1f * (float)std::sqrt(
                                        personRectangle.x*personRectangle.x + personRectangle.y*personRectangle.y);
                                    // if (mPoseScores[person] < poseScores[personRefined]) // This harms accuracy
                                    if (personAverageDistance < personRatio)
                                    {
                                        const auto personArea = mPoseKeypoints.getVolume(1,2);
                                        const auto personIndex = person * personArea;
                                        const auto personRefinedIndex = personRefined * personArea;
                                        // mPoseKeypoints: Update keypoints
                                        // Option a) Using refined ones
                                        std::copy(
                                            poseKeypoints.getPtr() + personRefinedIndex,
                                            poseKeypoints.getPtr() + personRefinedIndex + personArea,
                                            mPoseKeypoints.getPtr() + personIndex);
                                        mPoseScores[person] = poseScores[personRefined];
                                        // // Option b) Using ones with highest score (-6% acc single scale)
                                        // // Fill gaps
                                        // for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                        // {
                                        //     // For currently empty keypoints
                                        //     const auto partIndex = personIndex+3*part;
                                        //     const auto partRefinedIndex = personRefinedIndex+3*part;
                                        //     const auto scoreDifference = poseKeypoints[partRefinedIndex+2]
                                        //                                - mPoseKeypoints[partIndex+2];
                                        //     if (scoreDifference > 0)
                                        //     {
                                        //         const auto x = poseKeypoints[partRefinedIndex];
                                        //         const auto y = poseKeypoints[partRefinedIndex + 1];
                                        //         mPoseKeypoints[partIndex] = x;
                                        //         mPoseKeypoints[partIndex+1] = y;
                                        //         mPoseKeypoints[partIndex+2] += scoreDifference;
                                        //         mPoseScores[person] += scoreDifference;
                                        //     }
                                        // }

                                        // No acc improvement (-0.05% acc single scale)
                                        // // Finding all missing peaks (CPM-style)
                                        // // Only if no other person in there (otherwise 2% accuracy drop)
                                        // if (getNonZeroKeypoints(mPoseKeypoints, person, nmsThresholdRefined) > 0)
                                        // {
                                        //     // Get whether 0% ROI with other people
                                        //     // Get max ROI
                                        //     bool overlappingPerson = false;
                                        //     for (auto person2 = 0 ; person2 < mPoseKeypoints.getSize(0) ; person2++)
                                        //     {
                                        //         if (person != person2)
                                        //         {
                                        //             // Get ROI
                                        //             const auto currentRoi = getKeypointsRoi(
                                        //                 mPoseKeypoints, person, person2, nmsThreshold);
                                        //             // Update person
                                        //             if (currentRoi > 0.f)
                                        //             {
                                        //                 overlappingPerson = true;
                                        //                 break;
                                        //             }
                                        //         }
                                        //     }
                                        //     if (!overlappingPerson)
                                        //     {
                                        //         // Get keypoint with maximum probability per channel
                                        //         upImpl->spMaximumCaffe->Forward(
                                        //             {upImpl->spHeatMapsBlob.get()}, {upImpl->spMaximumPeaksBlob.get()});
                                        //         // Fill gaps
                                        //         const auto* posePeaksPtr = upImpl->spMaximumPeaksBlob->mutable_cpu_data();
                                        //         for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                        //         {
                                        //             // For currently empty keypoints
                                        //             if (mPoseKeypoints[personIndex+3*part+2] < nmsThresholdRefined)
                                        //             {
                                        //                 const auto xyIndex = 3*part;
                                        //                 const auto x = posePeaksPtr[xyIndex]*scaleRoiToOutput + xOffset;
                                        //                 const auto y = posePeaksPtr[xyIndex + 1]*scaleRoiToOutput + yOffset;
                                        //                 const auto rectangle = getKeypointsRectangle(
                                        //                     mPoseKeypoints, person, nmsThresholdRefined);
                                        //                 if (x >= rectangle.x && x < rectangle.x + rectangle.width
                                        //                     && y >= rectangle.y && y < rectangle.y + rectangle.height)
                                        //                 {
                                        //                     const auto score = posePeaksPtr[xyIndex + 2];
                                        //                     const auto baseIndex = personIndex + 3*part;
                                        //                     mPoseKeypoints[baseIndex] = x;
                                        //                     mPoseKeypoints[baseIndex+1] = y;
                                        //                     mPoseKeypoints[baseIndex+2] = score;
                                        //                     mPoseScores[person] += score;
                                        //                 }
                                        //             }
                                        //         }
                                        //     }
                                        // }
                                    }
                                }
                            }
                        }
                    }
                }

                // 5. CUDA sanity check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(inputNetData);
                UNUSED(inputDataSize);
                UNUSED(scaleInputToNetInputs);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorCaffe::getCandidatesCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spPeaksBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getCandidatesGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spPeaksBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getHeatMapCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getHeatMapGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    std::vector<int> PoseExtractorCaffe::getHeatMapSize() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->shape();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    const float* PoseExtractorCaffe::getPoseGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                checkThread();
                return nullptr;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
