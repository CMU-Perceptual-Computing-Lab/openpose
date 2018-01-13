// ------------------------- OpenPose Library Tutorial - Thread - Example 1 - OpenPose Read And Display -------------------------
// This third example shows the user how to:
    // 1. Read folder of images / video / webcam  (`producer` module)
    // 2. Display the rendered pose (`gui` module)
    // Everything in a multi-thread scenario (`thread` module) 
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module: for the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/producer/headers.hpp>
#include <openpose/thread/headers.hpp>
#include <openpose/utilities/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_int32(camera,                    -1,             "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
                                                        " number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution,        "1280x720",     "Size of the camera frames to ask for.");
DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the"
                                                        " minimum value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
                                                        " example video.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                                                        " images. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_string(ip_camera,                "",             "String with the IP camera URL. It supports protocols like RTSP and HTTP.");
DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
                                                        " too long, it will skip frames. If it is too fast, it will slow it down.");
// OpenPose
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
// Consumer
DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");

int openPoseTutorialThread1()
{
    op::log("OpenPose Library Tutorial - Example 3.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
        // - 0 will output all the logging messages
        // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // producerType
    const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera,
                                                       FLAGS_camera_resolution, FLAGS_camera_fps);
    const auto displayProducerFpsMode = (FLAGS_process_real_time ? op::ProducerFpsMode::OriginalFps : op::ProducerFpsMode::RetrievalFps);
    producerSharedPtr->setProducerFpsMode(displayProducerFpsMode);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Setting producer
    auto videoSeekSharedPtr = std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>();
    videoSeekSharedPtr->first = false;
    videoSeekSharedPtr->second = 0;
    const op::Point<int> producerSize{(int)producerSharedPtr->get(CV_CAP_PROP_FRAME_WIDTH),
                                      (int)producerSharedPtr->get(CV_CAP_PROP_FRAME_HEIGHT)};
    // Step 4 - Setting thread workers && manager
    typedef std::vector<op::Datum> TypedefDatumsNoPtr;
    typedef std::shared_ptr<TypedefDatumsNoPtr> TypedefDatums;
    op::ThreadManager<TypedefDatums> threadManager;
    // Step 5 - Initializing the worker classes
    // Frames producer (e.g. video, webcam, ...)
    auto DatumProducer = std::make_shared<op::DatumProducer<TypedefDatumsNoPtr>>(producerSharedPtr);
    auto wDatumProducer = std::make_shared<op::WDatumProducer<TypedefDatums, TypedefDatumsNoPtr>>(DatumProducer);
    // GUI (Display)
    auto gui = std::make_shared<op::Gui>(outputSize, FLAGS_fullscreen, threadManager.getIsRunningSharedPtr());
    auto wGui = std::make_shared<op::WGui<TypedefDatums>>(gui);

    // ------------------------- CONFIGURING THREADING -------------------------
    // In this simple multi-thread example, we will do the following:
        // 3 (virtual) queues: 0, 1, 2
        // 1 real queue: 1. The first and last queue ids (in this case 0 and 2) are not actual queues, but the
        // beginning and end of the processing sequence
        // 2 threads: 0, 1
        // wDatumProducer will generate frames (there is no real queue 0) and push them on queue 1
        // wGui will pop frames from queue 1 and process them (there is no real queue 2)
    auto threadId = 0ull;
    auto queueIn = 0ull;
    auto queueOut = 1ull;
    threadManager.add(threadId++, wDatumProducer, queueIn++, queueOut++);       // Thread 0, queues 0 -> 1
    threadManager.add(threadId++, wGui, queueIn++, queueOut++);                 // Thread 1, queues 1 -> 2

    // Equivalent single-thread version (option a)
    // const auto threadId = 0ull;
    // auto queueIn = 0ull;
    // auto queueOut = 1ull;
    // threadManager.add(threadId, wDatumProducer, queueIn++, queueOut++);       // Thread 0, queues 0 -> 1
    // threadManager.add(threadId, wGui, queueIn++, queueOut++);                 // Thread 0, queues 1 -> 2

    // Equivalent single-thread version (option b)
    // const auto threadId = 0ull;
    // const auto queueIn = 0ull;
    // const auto queueOut = 1ull;
    // threadManager.add(threadId, {wDatumProducer, wGui}, queueIn, queueOut);     // Thread 0, queues 0 -> 1

    // ------------------------- STARTING AND STOPPING THREADING -------------------------
    op::log("Starting thread(s)", op::Priority::High);
    // Two different ways of running the program on multithread environment
        // Option a) Using the main thread (this thread) for processing (it saves 1 thread, recommended)
    threadManager.exec();  // It blocks this thread until all threads have finished
        // Option b) Giving to the user the control of this thread
    // // VERY IMPORTANT NOTE: if OpenCV is compiled with Qt support, this option will not work. Qt needs the main
    // // thread to plot visual results, so the final GUI (which uses OpenCV) would return an exception similar to:
    // // `QMetaMethod::invoke: Unable to invoke methods with return values in queued connections`
    // // Start threads
    // threadManager.start();
    // // Keep program alive while running threads. Here the user could perform any other desired function
    // while (threadManager.isRunning())
    //     std::this_thread::sleep_for(std::chrono::milliseconds{33});
    // // Stop and join threads
    // op::log("Stopping thread(s)", op::Priority::High);
    // threadManager.stop();

    // ------------------------- CLOSING -------------------------
    // Logging information message
    op::log("Example 1 successfully finished.", op::Priority::High);
    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialThread1
    return openPoseTutorialThread1();
}
