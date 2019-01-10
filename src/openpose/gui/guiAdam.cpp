#ifdef USE_3D_ADAM_MODEL
#ifdef USE_3D_ADAM_MODEL
    #include <adam/KinematicModel.h>
    #include <adam/Renderer.h>
    #include <adam/utils.h>
    #include <adam/VisualizedData.h>
    #define SMPL_VIS_SCALING 100.0f
#endif
#include <openpose/3d/jointAngleEstimation.hpp>
#include <openpose/filestream/videoSaver.hpp>
#include <openpose/gui/guiAdam.hpp>

namespace op
{
    #ifdef USE_3D_ADAM_MODEL
        const int NUMBER_BODY_KEYPOINTS = 20;
        const int NUMBER_HAND_KEYPOINTS = 21;
        const int NUMBER_FACE_KEYPOINTS = 70;
        // targetJoints: Only for Body, LHand, RHand. No Face, no Foot
        const int NUMBER_KEYPOINTS = 3*(NUMBER_BODY_KEYPOINTS + 2*NUMBER_HAND_KEYPOINTS);

        void updateKeypoint(double* targetJoint, const float* const poseKeypoint3D)
        {
            try
            {
                // Keypoint found
                if (poseKeypoint3D[2] > 0.5) // For 3-D keypoint, it's either 0 or 1.
                {
                    targetJoint[0] = poseKeypoint3D[0];
                    targetJoint[1] = poseKeypoint3D[1];
                    targetJoint[2] = poseKeypoint3D[2];
                }
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    struct GuiAdam::ImplGuiAdam
    {
        #ifdef USE_3D_ADAM_MODEL
            // Visualization
            std::unique_ptr<adam::Renderer> spRender;
            std::array<double, NUMBER_KEYPOINTS> mResultBody;
            std::unique_ptr<GLubyte[]> upReadBuffer;

            // Video AVI writing
            const std::string mWriteAdamRenderAsVideo;
            std::shared_ptr<VideoSaver> spVideoSaver;

            // Shared parameters
            const std::shared_ptr<const TotalModel> spTotalModel;

            ImplGuiAdam(const std::shared_ptr<const TotalModel>& totalModel,
                        const std::string& adamRenderedVideoPath = "") :
                mWriteAdamRenderAsVideo{adamRenderedVideoPath},
                spTotalModel{totalModel}
            {
                try
                {
                    // Sanity check
                    if (spTotalModel == nullptr)
                        error("Given totalModel is a nullptr.", __LINE__, __FUNCTION__, __FILE__);
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }
        #endif
    };

    GuiAdam::GuiAdam(const Point<int>& outputSize, const bool fullScreen,
                     const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
                     const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr,
                     const std::vector<std::shared_ptr<PoseExtractorNet>>& poseExtractorNets,
                     const std::vector<std::shared_ptr<FaceExtractorNet>>& faceExtractorNets,
                     const std::vector<std::shared_ptr<HandExtractorNet>>& handExtractorNets,
                     const std::vector<std::shared_ptr<Renderer>>& renderers,
                     const DisplayMode displayMode, const std::shared_ptr<const TotalModel>& totalModel,
                     const std::string& adamRenderedVideoPath) :
        Gui{outputSize, fullScreen, isRunningSharedPtr, videoSeekSharedPtr, poseExtractorNets, faceExtractorNets,
            handExtractorNets, renderers, displayMode},
        spImpl{std::make_shared<ImplGuiAdam>(totalModel, adamRenderedVideoPath)}
    {
        try
        {
            #ifndef USE_3D_ADAM_MODEL
                UNUSED(outputSize);
                UNUSED(fullScreen);
                UNUSED(isRunningSharedPtr);
                UNUSED(videoSeekSharedPtr);
                UNUSED(poseExtractorNets);
                UNUSED(renderers);
                UNUSED(displayMode);
                UNUSED(totalModel);
                // UNUSED(adamRenderedVideoPath);
                error("OpenPose CMake must be compiled with the `USE_3D_ADAM_MODEL` flag in order to use the"
                      " Adam visualization renderer. Alternatively, set 2-D/3-D rendering with `--display 2`"
                      " or `--display 3`.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    GuiAdam::~GuiAdam()
    {
    }

    void GuiAdam::initializationOnThread()
    {
        try
        {
            // Init parent class
            if (mDisplayMode == DisplayMode::DisplayAll || mDisplayMode == DisplayMode::Display2D)
                Gui::initializationOnThread();
            #ifdef USE_3D_ADAM_MODEL
                if (mDisplayMode == DisplayMode::DisplayAll
                    || mDisplayMode == DisplayMode::DisplayAdam)
                {
                    int argc = 0;
                    // char* argv[0];
                    spImpl->spRender.reset(new adam::Renderer{&argc, nullptr});
                    // spImpl->spRender->options.yrot=-45;
                    spImpl->spRender->options.yrot=45;
                    spImpl->spRender->options.xrot=25;
                    spImpl->spRender->options.meshSolid = true;
                    // spImpl->spRender->options.meshSolid = false;
                    spImpl->upReadBuffer.reset(new GLubyte[spImpl->spRender->options.width
                                                           * spImpl->spRender->options.height * 3]);
                }
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void GuiAdam::generateMesh(const Array<float>& poseKeypoints3D, const Array<float>& faceKeypoints3D,
                               const std::array<Array<float>, 2>& handKeypoints3D,
                               const double* const adamPosePtr, const double* const adamTranslationPtr,
                               const double* const vtVecPtr, const int vtVecRows,
                               const double* const j0VecPtr, const int j0VecRows,
                               const double* const adamFaceCoeffsExpPtr)
    {
        try
        {
            // Adam rendering
            #ifdef USE_3D_ADAM_MODEL
                if (mDisplayMode == DisplayMode::DisplayAll
                    || mDisplayMode == DisplayMode::DisplayAdam)
                {
                    const Eigen::Map<const Eigen::Vector3d> adamTranslation(adamTranslationPtr);
                    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> vtVec(
                        vtVecPtr, vtVecRows);
                    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> j0Vec(
                        j0VecPtr, j0VecRows);
                    CMeshModelInstance cMeshModelInstance;
                    VisualizedData visualizedData;
                    // auto& visualizedData = spImpl->mVisualizedData;
                    // GenerateMesh_Fast modifies cMeshModelInstance & spImpl->mResultBody
                    GenerateMesh_Fast(cMeshModelInstance, spImpl->mResultBody.data(), *spImpl->spTotalModel, vtVec,
                                    j0Vec, adamPosePtr, adamFaceCoeffsExpPtr, adamTranslation);
                    CopyMesh(cMeshModelInstance, visualizedData);

                    // Fill data
                    // Body and hands
                    std::array<double, NUMBER_KEYPOINTS> targetJoints;
                    targetJoints.fill(0.);
                    // If keypoints detected
                    if (!poseKeypoints3D.empty())
                    {
                        // Update body
                        for (auto part = 0 ; part < 19; part++)
                            updateKeypoint(&targetJoints[mapOPToAdam(part)*(poseKeypoints3D.getSize(2)-1)],
                                           &poseKeypoints3D[{0, part, 0}]);
                        // Update left/right hand
                        // NUMBER_BODY_KEYPOINTS = #parts_OP + 1 (top_head)
                        const auto bodyOffset = NUMBER_BODY_KEYPOINTS*(poseKeypoints3D.getSize(2)-1);
                        const auto handOffset = handKeypoints3D[0].getSize(1)*(handKeypoints3D[0].getSize(2)-1);
                        for (auto hand = 0u ; hand < handKeypoints3D.size(); hand++)
                            if (!handKeypoints3D.at(hand).empty())
                                for (auto part = 0 ; part < handKeypoints3D[hand].getSize(1); part++)
                                    updateKeypoint(&targetJoints[bodyOffset + hand*handOffset
                                                    + part*(handKeypoints3D[hand].getSize(2)-1)],
                                                   &handKeypoints3D[hand][{0, part, 0}]);

                        // Meters --> cm
                        for (auto i = 0 ; i < NUMBER_KEYPOINTS ; i++)
                            targetJoints[i] *= 1e2;
                    }
                    visualizedData.targetJoint = targetJoints.data();
                    // visualizedData.targetJoint = nullptr;
                    // Update face
                    if (!faceKeypoints3D.empty())
                    {
                        visualizedData.faceKeypoints.resize(faceKeypoints3D.getSize(1), 3);
                        for (auto part = 0 ; part < faceKeypoints3D.getSize(1); part++)
                        {
                            if (faceKeypoints3D[{0, part, faceKeypoints3D.getSize(2)-1}] > 0.5)
                                for (auto xyz = 0 ; xyz < faceKeypoints3D.getSize(2)-1 ; xyz++)
                                    visualizedData.faceKeypoints(part, xyz) = faceKeypoints3D[{0, part, xyz}];
                            else
                            {
                                visualizedData.faceKeypoints(part, 0) = 0;
                                visualizedData.faceKeypoints(part, 1) = 0;
                                visualizedData.faceKeypoints(part, 2) = 0;
                            }
                        }
                        visualizedData.faceKeypoints *= 100;
                    }
                    else
                        visualizedData.faceKeypoints.setZero();
                    visualizedData.resultJoint = spImpl->mResultBody.data();

                    // visualizedData: 2 for full body, 3 for left hand, 4 for right hand, 5 for face
                    visualizedData.vis_type = 2;
                    // If full body --> nRange > whole body object
                    if (visualizedData.vis_type <= 2)
                        spImpl->spRender->options.nRange = 150;
                    // Otherwise (hand/face) --> ~zoom in
                    else
                        spImpl->spRender->options.nRange = 40;
                    visualizedData.read_buffer = spImpl->upReadBuffer.get();

                    // Send display to screen
                    spImpl->spRender->RenderHand(visualizedData);
                    spImpl->spRender->RenderAndRead(); // read the image into read_buffer
                    // spImpl->spRender->Display();

                    // Save/display Adam display in OpenCV window
                    if (!spImpl->mWriteAdamRenderAsVideo.empty())
                    {
                        // img is y-flipped, and in RGB order
                        cv::Mat img(spImpl->spRender->options.height, spImpl->spRender->options.width,
                                    CV_8UC3, spImpl->upReadBuffer.get());
                        cv::flip(img, img, 0);
                        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                        if (spImpl->spVideoSaver == nullptr)
                        {
                            const auto originalVideoFps = 30;
                            spImpl->spVideoSaver = std::make_shared<VideoSaver>(
                                spImpl->mWriteAdamRenderAsVideo, CV_FOURCC('M','J','P','G'),
                                originalVideoFps
                            );
                        }
                        spImpl->spVideoSaver->write(img);
                        // cv::imshow( "Display window", img );
                        // cv::waitKey(16);
                    }
                }
            #else
                UNUSED(poseKeypoints3D);
                UNUSED(faceKeypoints3D);
                UNUSED(handKeypoints3D);
                UNUSED(adamPose);
                UNUSED(adamTranslation);
                UNUSED(vtVec);
                UNUSED(j0Vec);
                UNUSED(adamFaceCoeffsExpPtr);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void GuiAdam::update()
    {
        try
        {   
            // 2-D rendering
            if (mDisplayMode == DisplayMode::DisplayAll || mDisplayMode == DisplayMode::Display2D)
                Gui::update();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
#endif
