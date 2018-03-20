#include <openpose/gpu/gpu.hpp>
#include <openpose/thread/enumClasses.hpp>
#include <openpose/wrapper/wrapperAuxiliary.hpp>

namespace op
{
    void wrapperConfigureSecurityChecks(const WrapperStructPose& wrapperStructPose,
                                        const WrapperStructFace& wrapperStructFace,
                                        const WrapperStructHand& wrapperStructHand,
                                        const WrapperStructInput& wrapperStructInput,
                                        const WrapperStructOutput& wrapperStructOutput,
                                        const bool renderOutput,
                                        const bool userOutputWsEmpty,
                                        const ThreadManagerMode threadManagerMode)
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
                && (wrapperStructPose.heatMapScale != ScaleMode::UnsignedChar &&
                        wrapperStructOutput.writeHeatMapsFormat != "float"))
            {
                const auto message = "In order to save the heatmaps, you must either set"
                                     " wrapperStructPose.heatMapScale to ScaleMode::UnsignedChar (i.e. range [0, 255])"
                                     " or `--write_heatmaps_format` to `float` to storage floating numbers in binary"
                                     " mode.";
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
                const bool guiEnabled = (wrapperStructOutput.displayMode != DisplayMode::NoDisplay);
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
                if (guiEnabled && wrapperStructOutput.guiVerbose && !renderOutput)
                {
                    const auto message = "No render is enabled (e.g. `--render_pose 0`), so you might also want to"
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
            if (!wrapperStructOutput.writeVideo.empty() && wrapperStructInput.producerSharedPtr == nullptr)
                error("Writting video is only available if the OpenPose producer is used (i.e."
                      " wrapperStructInput.producerSharedPtr cannot be a nullptr).",
                      __LINE__, __FUNCTION__, __FILE__);
            if (!wrapperStructPose.enable)
            {
                if (!wrapperStructFace.enable)
                    error("Body keypoint detection must be enabled.", __LINE__, __FUNCTION__, __FILE__);
                if (wrapperStructHand.enable)
                    error("Body keypoint detection must be enabled in order to run hand keypoint detection.",
                          __LINE__, __FUNCTION__, __FILE__);
            }
            // If 3-D module, 1 person is the maximum
            if (wrapperStructPose.reconstruct3d && wrapperStructPose.numberPeopleMax != 1)
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
            if (wrapperStructOutput.writeVideoFps <= 0
                && wrapperStructInput.producerSharedPtr->get(CV_CAP_PROP_FPS) > 0)
                error("Set `--camera_fps` for this producer, as its frame rate is unknown.",
                      __LINE__, __FUNCTION__, __FILE__);
            #ifdef USE_CPU_ONLY
                if (wrapperStructPose.scalesNumber > 1)
                    error("Temporarily, the number of scales (`--scale_number`) cannot be greater than 1 for"
                          " `CPU_ONLY` version.", __LINE__, __FUNCTION__, __FILE__);
            #endif
            // Net input resolution cannot be reshaped for Caffe OpenCL and MKL versions, only for CUDA version
            #if defined USE_MKL || defined USE_CPU_ONLY
                // If image_dir and netInputSize == -1 --> error
                if ((wrapperStructInput.producerSharedPtr == nullptr
                     || wrapperStructInput.producerSharedPtr->getType() == ProducerType::ImageDirectory)
                    // If netInputSize is -1
                    && (wrapperStructPose.netInputSize.x == -1 || wrapperStructPose.netInputSize.y == -1))
                    error("Dynamic `--net_resolution` is not supported in MKL (CPU) and OpenCL Caffe versions. Please"
                          " remove `-1` from `net_resolution` or use the Caffe master branch when processing images"
                          " and when using your custom image reader.",
                          __LINE__, __FUNCTION__, __FILE__);
            #endif

            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
