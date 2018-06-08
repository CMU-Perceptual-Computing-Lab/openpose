#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/nmsCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>

namespace op
{
    struct PoseExtractorCaffe::ImplPoseExtractorCaffe
    {
        #ifdef USE_CAFFE
            // Used when increasing spCaffeNets
            const PoseModel mPoseModel;
            const int mGpuId;
            const std::string mModelFolder;
            const bool mEnableGoogleLogging;
            // General parameters
            std::vector<std::shared_ptr<NetCaffe>> spCaffeNets;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            std::vector<std::vector<int>> mNetInput4DSizes;
            std::vector<double> mScaleInputToNetInputs;
            // Init with thread
            std::vector<boost::shared_ptr<caffe::Blob<float>>> spCaffeNetOutputBlobs;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;

            ImplPoseExtractorCaffe(const PoseModel poseModel, const int gpuId,
                                   const std::string& modelFolder, const bool enableGoogleLogging) :
                mPoseModel{poseModel},
                mGpuId{gpuId},
                mModelFolder{modelFolder},
                mEnableGoogleLogging{enableGoogleLogging},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
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
                                              std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
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
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(poseModel),
                                  getPoseNumberBodyParts(poseModel), gpuID);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()});
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

        void addCaffeNetOnThread(std::vector<std::shared_ptr<NetCaffe>>& netCaffe,
                                 std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                 const PoseModel poseModel, const int gpuId,
                                 const std::string& modelFolder, const bool enableGoogleLogging)
        {
            try
            {
                // Add Caffe Net
                netCaffe.emplace_back(
                    std::make_shared<NetCaffe>(modelFolder + getPoseProtoTxt(poseModel),
                                               modelFolder + getPoseTrainedModel(poseModel),
                                               gpuId, enableGoogleLogging)
                );
                // Initializing them on the thread
                netCaffe.back()->initializationOnThread();
                caffeNetOutputBlob.emplace_back(netCaffe.back()->getOutputBlob());
                // Security checks
                if (netCaffe.size() != caffeNetOutputBlob.size())
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
                                           const bool enableGoogleLogging) :
        PoseExtractorNet{poseModel, heatMapTypes, heatMapScale, addPartCandidates}
        #ifdef USE_CAFFE
        , upImpl{new ImplPoseExtractorCaffe{poseModel, gpuId, modelFolder, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
                // Layers parameters
                upImpl->spBodyPartConnectorCaffe->setPoseModel(upImpl->mPoseModel);
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
                addCaffeNetOnThread(upImpl->spCaffeNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                    upImpl->mGpuId, upImpl->mModelFolder, upImpl->mEnableGoogleLogging);
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Initialize blobs
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
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
                // Security checks
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
                while (upImpl->spCaffeNets.size() < numberScales)
                    addCaffeNetOnThread(upImpl->spCaffeNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                        upImpl->mGpuId, upImpl->mModelFolder, false);

                // Process each image
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    // 1. Caffe deep network
                    upImpl->spCaffeNets.at(i)->forwardPass(inputNetData[i]);                                   // ~80ms

                    // Reshape blobs if required
                    // Note: In order to resize to input size to have same results as Matlab, uncomment the commented
                    // lines
                    // Note: For dynamic sizes (e.g. a folder with images of different aspect ratio)
                    if (!vectorsAreEqual(upImpl->mNetInput4DSizes.at(i), inputNetData[i].getSize()))
                        // || !vectorsAreEqual(upImpl->mScaleInputToNetInputs, scaleInputToNetInputs))
                    {
                        upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                        mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
                                                    upImpl->mNetInput4DSizes[0][2]};
                        // upImpl->mScaleInputToNetInputs = scaleInputToNetInputs;
                        reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                                  upImpl->spBodyPartConnectorCaffe, upImpl->spCaffeNetOutputBlobs,
                                                  upImpl->spHeatMapsBlob, upImpl->spPeaksBlob,
                                                  1.f, upImpl->mPoseModel, upImpl->mGpuId);
                                                  // scaleInputToNetInputs[i], upImpl->mPoseModel);
                    }
                }

                // 2. Resize heat maps + merge different scales
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spCaffeNetOutputBlobs);
                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);

                #ifdef USE_CUDA
                    //upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
                    upImpl->spResizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~5ms
                #elif USE_OPENCL
                    //upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
                    upImpl->spResizeAndMergeCaffe->Forward_ocl(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()});
                #else
                    upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
                #endif

                // Get scale net to output (i.e. image input)
                // Note: In order to resize to input size, (un)comment the following lines
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                         intRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
                // mScaleNetToOutput = 1.f;

                // 3. Get peaks by Non-Maximum Suppression
                upImpl->spNmsCaffe->setThreshold((float)get(PoseProperty::NMSThreshold));
                const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
                upImpl->spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                #ifdef USE_CUDA
                    //upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
                    upImpl->spNmsCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});// ~2ms
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #elif USE_OPENCL
                    //upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
                    upImpl->spNmsCaffe->Forward_ocl({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});
                #else
                    upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
                #endif

                // 4. Connecting body parts
                // Get scale net to output (i.e. image input)
                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (float)get(PoseProperty::ConnectInterMinAboveThreshold)
                );
                upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));

                // CUDA version not implemented yet
                // #ifdef USE_CUDA
                //     upImpl->spBodyPartConnectorCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get(),
                //                                                    upImpl->spPeaksBlob.get()},
                //                                                   mPoseKeypoints, mPoseScores);
                // #else
                    upImpl->spBodyPartConnectorCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get(),
                                                                   upImpl->spPeaksBlob.get()},
                                                                  mPoseKeypoints, mPoseScores);
                // #endif
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
