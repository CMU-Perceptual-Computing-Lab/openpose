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
                if (!wrapperStructOutput.displayGui && !savingSomething)
                {
                    const auto message = "No output is selected (`--no_display`) and no results are generated (no"
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
                if ((wrapperStructOutput.displayGui && wrapperStructOutput.guiVerbose) && !renderOutput)
                {
                    const auto message = "No render is enabled (e.g. `--render_pose 0`), so you might also want to"
                                         " remove the display (set `--no_display` or `--no_gui_verbose`). If you"
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
                      " wrapperStructInput.producerSharedPtr cannot be a nullptr).", __LINE__, __FUNCTION__, __FILE__);
            if (!wrapperStructPose.enable)
            {
                if (!wrapperStructFace.enable)
                    error("Body keypoint detection must be enabled.", __LINE__, __FUNCTION__, __FILE__);
                if (wrapperStructHand.enable)
                    error("Body keypoint detection must be enabled in order to run hand keypoint detection.",
                          __LINE__, __FUNCTION__, __FILE__);
            }
            #ifdef CPU_ONLY
                if (wrapperStructPose.gpuNumber > 0)
                    error("GPU number must be negative or 0 if CPU_ONLY is enabled.", __LINE__, __FUNCTION__, __FILE__);
            #endif

            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
