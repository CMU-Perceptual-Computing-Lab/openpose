// ------------------------- OpenPose Library Tutorial - Hand Keypoint Detection from JSON Ground-Truth Data -------------------------
// Example to test hands accuracy given ground-truth bounding boxes.

#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
#include <openpose/headers.hpp>
#include "wrapperHandFromJsonTest.hpp"

// For info about the flags, check `examples/openpose/openpose.bin`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "");
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
DEFINE_bool(hand_tracking,              false,          "");
// Display
DEFINE_bool(no_display,                 false,          "");
// Result Saving
DEFINE_string(write_keypoint_json,      "", "");

int handFromJsonTest()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::High);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration - Google flags to program variables
    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
    // producerType
    const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, "", "", 0);
    // Enabling Google Logging
    const bool enableGoogleLogging = true;
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // OpenPose wrapper
    op::log("Configuring OpenPose wrapper.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    op::WrapperHandFromJsonTest<std::vector<op::Datum>> opWrapper;
    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    op::WrapperStructPose wrapperStructPose{false, op::flagsToPoint("656x368"), op::flagsToPoint("1280x720"),
                                            op::ScaleMode::InputResolution, FLAGS_num_gpu, FLAGS_num_gpu_start,
                                            true, enableGoogleLogging};
    wrapperStructPose.modelFolder = FLAGS_model_folder;
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number,
                                                  (float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
                                                  op::flagsToRenderMode(1)};
    // Configure wrapper
    opWrapper.configure(wrapperStructPose, wrapperStructHand, producerSharedPtr, FLAGS_hand_ground_truth,
                        FLAGS_write_keypoint_json, !FLAGS_no_display);

    // Start processing
    op::log("Starting thread(s)", op::Priority::High);
    opWrapper.exec();  // It blocks this thread until all threads have finished

    // Measuring total time
    const auto now = std::chrono::high_resolution_clock::now();
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                            * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: "
                       + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::High);

    return 0;
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running handFromJsonTest
    return handFromJsonTest();
}
