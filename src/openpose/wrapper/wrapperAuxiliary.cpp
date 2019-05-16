#include <openpose/gpu/gpu.hpp>
#include <openpose/thread/enumClasses.hpp>
#include <openpose/wrapper/wrapperAuxiliary.hpp>

namespace op
{
    void wrapperConfigureSanityChecks(
        WrapperStructPose& wrapperStructPose, const WrapperStructFace& wrapperStructFace,
        const WrapperStructHand& wrapperStructHand, const WrapperStructExtra& wrapperStructExtra,
        const WrapperStructInput& wrapperStructInput, const WrapperStructOutput& wrapperStructOutput,
        const WrapperStructGui& wrapperStructGui, const bool renderOutput,
        const bool userInputAndPreprocessingWsEmpty, const bool userOutputWsEmpty,
        const std::shared_ptr<Producer>& producerSharedPtr, const ThreadManagerMode threadManagerMode)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

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
                const auto message = "In order to save the rendered frames (`--write_images` or `--write_video`), you"
                                     " cannot disable `--render_pose`.";
                log(message, Priority::High);
            }
            if (!wrapperStructOutput.writeHeatMaps.empty() && wrapperStructPose.heatMapTypes.empty())
            {
                const auto message = "In order to save the heatmaps (`--write_heatmaps`), you need to pick which heat"
                                     " maps you want to save: `--heatmaps_add_X` flags or fill the"
                                     " wrapperStructPose.heatMapTypes.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (!wrapperStructOutput.writeHeatMaps.empty()
                && (wrapperStructPose.heatMapScaleMode != ScaleMode::UnsignedChar &&
                        wrapperStructOutput.writeHeatMapsFormat != "float"))
            {
                const auto message = "In order to save the heatmaps, you must either set"
                                     " wrapperStructPose.heatMapScaleMode to ScaleMode::UnsignedChar (i.e., range"
                                     " [0, 255]) or `--write_heatmaps_format` to `float` to storage floating numbers"
                                     " in binary mode.";
                error(message, __LINE__, __FUNCTION__, __FILE__);
            }
            if (userOutputWsEmpty && threadManagerMode != ThreadManagerMode::Asynchronous
                && threadManagerMode != ThreadManagerMode::AsynchronousOut)
            {
                const std::string additionalMessage{
                    " You could also set mThreadManagerMode = mThreadManagerMode::Asynchronous(Out) and/or add your"
                    " own output worker class before calling this function."
                };
                const auto savingSomething = (
                    !wrapperStructOutput.writeImages.empty() || !wrapperStructOutput.writeVideo.empty()
                        || !wrapperStructOutput.writeKeypoint.empty() || !wrapperStructOutput.writeJson.empty()
                        || !wrapperStructOutput.writeCocoJson.empty() || !wrapperStructOutput.writeHeatMaps.empty()
                );
                const auto savingCvOutput = (
                    !wrapperStructOutput.writeImages.empty() || !wrapperStructOutput.writeVideo.empty()
                );
                const bool guiEnabled = (wrapperStructGui.displayMode != DisplayMode::NoDisplay);
                if (!guiEnabled && !savingCvOutput && renderOutput)
                {
                    const auto message = "GUI is not enabled and you are not saving the output frames. You should then"
                                         " disable rendering for a faster code. I.e., add `--render_pose 0`."
                                         + additionalMessage;
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                if (!guiEnabled && !savingSomething)
                {
                    const auto message = "No output is selected (`--display 0`) and no results are generated (no"
                                         " `--write_X` flags enabled). Thus, no output would be generated."
                                         + additionalMessage;
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                if (wrapperStructInput.framesRepeat && savingSomething)
                {
                    const auto message = "Frames repetition (`--frames_repeat`) is enabled as well as some writing"
                                         " function (`--write_X`). This program would never stop recording the same"
                                         " frames over and over. Please, disable repetition or remove writing.";
                    error(message, __LINE__, __FUNCTION__, __FILE__);
                }
                // Warnings
                if (guiEnabled && wrapperStructGui.guiVerbose && !renderOutput)
                {
                    const auto message = "No render is enabled (e.g., `--render_pose 0`), so you might also want to"
                                         " remove the display (set `--display 0` or `--no_gui_verbose`). If you"
                                         " simply want to use OpenPose to record video/images without keypoints, you"
                                         " only need to set `--num_gpu 0`." + additionalMessage;
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
            if (!wrapperStructOutput.writeVideo.empty() && producerSharedPtr == nullptr)
                error("Writting video is only available if the OpenPose producer is used (i.e."
                      " producerSharedPtr cannot be a nullptr).",
                      __LINE__, __FUNCTION__, __FILE__);
            if (wrapperStructPose.poseMode == PoseMode::Disabled && !wrapperStructFace.enable
                && !wrapperStructHand.enable)
                error("Body, face, and hand keypoint detectors are disabled. You must enable at least one (i.e,"
                      " unselect `--body 0`, select `--face`, or select `--hand`.",
                      __LINE__, __FUNCTION__, __FILE__);
            const auto ownDetectorProvided = (wrapperStructFace.detector == Detector::Provided
                                              || wrapperStructHand.detector == Detector::Provided);
            if (ownDetectorProvided && userInputAndPreprocessingWsEmpty
                && threadManagerMode != ThreadManagerMode::Asynchronous
                && threadManagerMode != ThreadManagerMode::AsynchronousIn)
                error("You have selected to provide your own face and/or hand rectangle detections (`face_detector 2`"
                      " and/or `hand_detector 2`), thus OpenPose will not detect face and/or hand keypoints based on"
                      " the body keypoints. However, you are not providing any information about the location of the"
                      " faces and/or hands. Either provide the location of the face and/or hands (e.g., see the"
                      " `examples/tutorial_api_cpp/` examples, or change the value of `--face_detector` and/or"
                      " `--hand_detector`.", __LINE__, __FUNCTION__, __FILE__);
            // Warning
            if (ownDetectorProvided && wrapperStructPose.poseMode != PoseMode::Disabled)
                log("Warning: Body keypoint estimation is enabled while you have also selected to provide your own"
                    " face and/or hand rectangle detections (`face_detector 2` and/or `hand_detector 2`). Therefore,"
                    " OpenPose will not detect face and/or hand keypoints based on the body keypoints. Are you sure"
                    " you want to keep enabled the body keypoint detector? (disable it with `--body 0`).",
                    Priority::High);
            // If 3-D module, 1 person is the maximum
            if (wrapperStructExtra.reconstruct3d && wrapperStructPose.numberPeopleMax != 1)
            {
                error("Set `--number_people_max 1` when using `--3d`. The 3-D reconstruction demo assumes there is"
                      " at most 1 person on each image.", __LINE__, __FUNCTION__, __FILE__);
            }
            // If CPU mode, #GPU cannot be > 0
            if (getGpuMode() == GpuMode::NoGpu)
                if (wrapperStructPose.gpuNumber > 0)
                    error("GPU number must be negative or 0 if CPU_ONLY is enabled.",
                          __LINE__, __FUNCTION__, __FILE__);
            // If num_gpu 0 --> output_resolution has no effect
            if (wrapperStructPose.gpuNumber == 0 &&
                (wrapperStructPose.outputSize.x > 0 || wrapperStructPose.outputSize.y > 0))
                error("If `--num_gpu 0`, then `--output_resolution` has no effect, so either disable it or use"
                      " `--output_resolution -1x-1`. Current output size: ("
                      + std::to_string(wrapperStructPose.outputSize.x) + "x"
                      + std::to_string(wrapperStructPose.outputSize.y) + ").",
                      __LINE__, __FUNCTION__, __FILE__);
            #ifdef USE_CPU_ONLY
                if (wrapperStructPose.scalesNumber > 1)
                    error("Temporarily, the number of scales (`--scale_number`) cannot be greater than 1 for"
                          " `CPU_ONLY` version.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            // Net input resolution cannot be reshaped for Caffe OpenCL and MKL versions, only for CUDA version
            #if defined USE_MKL || defined USE_OPENCL
                // If image_dir and netInputSize == -1 --> error
                if ((producerSharedPtr == nullptr
                     || producerSharedPtr->getType() == ProducerType::ImageDirectory)
                    // If netInputSize is -1
                    && (wrapperStructPose.netInputSize.x == -1 || wrapperStructPose.netInputSize.y == -1))
                {
                    wrapperStructPose.netInputSize.x = 656;
                    wrapperStructPose.netInputSize.y = 368;
                    log("The default dynamic `--net_resolution` is not supported in MKL (MKL CPU Caffe) and OpenCL"
                        " Caffe versions. Please, use a static `net_resolution` (recommended"
                        " `--net_resolution 656x368`) or use the Caffe CUDA master branch when processing images"
                        " and/or when using your custom image reader. OpenPose has automatically set the resolution"
                        " to 656x368.", Priority::High);
                }
            #endif
            #ifndef USE_CUDA
                log("---------------------------------- WARNING ----------------------------------\n"
                    "We have introduced an additional boost in accuracy of about 0.5% with respect to the official"
                    " OpenPose 1.4.0 (using default settings). Currently, this increase is only applicable to CUDA"
                    " version, but might eventually be ported to CPU and OpenCL versions. Therefore, CPU and OpenCL"
                    " results might vary. Nevertheless, this accuracy boost is almost insignificant so CPU and"
                    " OpenCL versions can be safely used, they will simply provide the exact same accuracy than"
                    " OpenPose 1.4.0."
                    "\n-------------------------------- END WARNING --------------------------------",
                    Priority::High);
            #endif

            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void threadIdPP(unsigned long long& threadId, const bool multiThreadEnabled)
    {
        try
        {
            if (multiThreadEnabled)
                threadId++;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
