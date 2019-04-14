// ----------------------- OpenPose Tests - Hand Keypoint Detection from JSON Ground-Truth Data -----------------------
// Example to test hands accuracy given ground-truth bounding boxes.

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include "wrapperHandFromJsonTest.hpp"

// For info about the flags, check `examples/openpose/openpose.bin`.
// Producer
DEFINE_string(image_dir,                "", "");
DEFINE_string(hand_ground_truth,        "", "");
// OpenPose
DEFINE_string(model_folder,             "models/",      "");
DEFINE_int32(num_gpu,                   -1,             "");
DEFINE_int32(num_gpu_start,             0,              "");
// OpenPose Hand
DEFINE_bool(hand,                       true,           "");
DEFINE_string(hand_net_resolution,      "368x368",      "");
DEFINE_int32(hand_scale_number,         1,              "");
DEFINE_double(hand_scale_range,         0.4,            "");
// Display
DEFINE_int32(display,                   -1,             "");
// Result Saving
DEFINE_string(write_json,               "",             "");

int handFromJsonTest()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

        // Applying user defined configuration - GFlags to program variables
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // producerType
        const auto producerSharedPtr = op::createProducer(op::ProducerType::ImageDirectory, FLAGS_image_dir);

        // OpenPose wrapper
        op::log("Configuring OpenPose...", op::Priority::High);
        op::WrapperHandFromJsonTest<op::Datum> opWrapper;
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        op::WrapperStructPose wrapperStructPose{
            op::PoseMode::Disabled, op::flagsToPoint("656x368"), op::flagsToPoint("1280x720"), op::ScaleMode::InputResolution,
            FLAGS_num_gpu, FLAGS_num_gpu_start, 1, 0.15f, op::RenderMode::None, op::PoseModel::BODY_25, true, 0.f, 0.f,
            0, "models/", {}, op::ScaleMode::ZeroToOne, false, 0.05f, -1, false};
        wrapperStructPose.modelFolder = FLAGS_model_folder;
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, op::Detector::Provided, handNetInputSize, FLAGS_hand_scale_number,
            (float)FLAGS_hand_scale_range, op::flagsToRenderMode(1)};
        // Configure wrapper
        opWrapper.configure(wrapperStructPose, wrapperStructHand, producerSharedPtr, FLAGS_hand_ground_truth,
                            FLAGS_write_json, op::flagsToDisplayMode(FLAGS_display, false));

        // Start processing
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

        // Measuring total time
        const auto now = std::chrono::high_resolution_clock::now();
        const auto totalTimeSec = double(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()* 1e-9);
        const auto message = "OpenPose demo successfully finished. Total time: "
                           + std::to_string(totalTimeSec) + " seconds.";
        op::log(message, op::Priority::High);

        return 0;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running handFromJsonTest
    return handFromJsonTest();
}
