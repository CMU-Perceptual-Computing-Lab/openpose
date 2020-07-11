// ------------------------- OpenPose Library Tutorial - Thread - Example 1 - User Processing Function -------------------------
// This directory only provides examples for the basic OpenPose thread mechanism API, and it is only meant for people
// interested in the multi-thread architecture without been interested in the OpenPose pose estimation algorithm.
// You are most probably looking for the [examples/tutorial_api_cpp/](../tutorial_api_cpp/) or
// [examples/tutorial_api_python/](../tutorial_api_python/), which provide examples of the thread API already applied
// to body pose estimation.

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// This class can be implemented either as a template or as a simple class given
// that the user usually knows which kind of data he will move between the queues,
// in this case we assume a std::shared_ptr of a std::vector of op::Datum
class WUserClass : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
public:
    WUserClass()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        try
        {
            // User's processing here
                // datumPtr->cvInputData: initial cv::Mat obtained from the frames producer (video, webcam, etc.)
                // datumPtr->cvOutputData: final cv::Mat to be displayed
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                for (auto& datumPtr : *datumsPtr)
                {
                    cv::Mat cvOutputData = OP_OP2CVMAT(datumPtr->cvOutputData);
                    cv::bitwise_not(cvOutputData, cvOutputData);
                }
            }
        }
        catch (const std::exception& e)
        {
            op::opLog("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

int openPoseTutorialThread1()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // ------------------------- INITIALIZATION -------------------------
        // Step 1 - Set logging level
            // - 0 will output all the logging messages
            // - 255 will output nothing
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        // Step 2 - Read GFlags (user defined configuration)
        // cameraSize
        const auto cameraSize = op::flagsToPoint(op::String(FLAGS_camera_resolution), "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // producerType
        op::ProducerType producerType;
        op::String producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            op::String(FLAGS_image_dir), op::String(FLAGS_video), op::String(FLAGS_ip_camera), FLAGS_camera,
            FLAGS_flir_camera, FLAGS_flir_camera_index);
        const auto displayProducerFpsMode = (FLAGS_process_real_time
                                          ? op::ProducerFpsMode::OriginalFps : op::ProducerFpsMode::RetrievalFps);
        auto producerSharedPtr = createProducer(
            producerType, producerString.getStdString(), cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort,
            FLAGS_3d_views);
        producerSharedPtr->setProducerFpsMode(displayProducerFpsMode);
        op::opLog("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        // Step 3 - Setting producer
        auto videoSeekSharedPtr = std::make_shared<std::pair<std::atomic<bool>, std::atomic<int>>>();
        videoSeekSharedPtr->first = false;
        videoSeekSharedPtr->second = 0;
        const op::Point<int> producerSize{
            (int)producerSharedPtr->get(op::getCvCapPropFrameWidth()),
            (int)producerSharedPtr->get(op::getCvCapPropFrameHeight())};
        // Step 4 - Setting thread workers && manager
        typedef std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> TypedefDatumsSP;
        op::ThreadManager<TypedefDatumsSP> threadManager;
        // Step 5 - Initializing the worker classes
        // Frames producer (e.g., video, webcam, ...)
        auto DatumProducer = std::make_shared<op::DatumProducer<op::Datum>>(producerSharedPtr);
        auto wDatumProducer = std::make_shared<op::WDatumProducer<op::Datum>>(DatumProducer);
        // Specific WUserClass
        auto wUserClass = std::make_shared<WUserClass>();
        // GUI (Display)
        auto gui = std::make_shared<op::Gui>(outputSize, FLAGS_fullscreen, threadManager.getIsRunningSharedPtr());
        auto wGui = std::make_shared<op::WGui<TypedefDatumsSP>>(gui);

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
        threadManager.add(threadId++, wDatumProducer, queueIn++, queueOut++);   // Thread 0, queues 0 -> 1
        threadManager.add(threadId++, wUserClass, queueIn++, queueOut++);       // Thread 1, queues 1 -> 2
        threadManager.add(threadId++, wGui, queueIn++, queueOut++);             // Thread 2, queues 2 -> 3

        // Equivalent single-thread version
        // const auto threadId = 0ull;
        // auto queueIn = 0ull;
        // auto queueOut = 1ull;
        // threadManager.add(threadId, wDatumProducer, queueIn++, queueOut++);     // Thread 0, queues 0 -> 1
        // threadManager.add(threadId, wUserClass, queueIn++, queueOut++);         // Thread 1, queues 1 -> 2
        // threadManager.add(threadId, wGui, queueIn++, queueOut++);               // Thread 2, queues 2 -> 3

        // Smart multi-thread version
        // Assume wUser is the slowest process, and that wDatumProducer + wGui is faster than wGui itself,
        // then, we can group the last 2 in the same thread and keep wGui in a different thread:
        // const auto threadId = 0ull;
        // auto queueIn = 0ull;
        // auto queueOut = 1ull;
        // threadManager.add(threadId, wDatumProducer, queueIn++, queueOut++);     // Thread 0, queues 0 -> 1
        // threadManager.add(threadId+1, wUserClass, queueIn++, queueOut++);       // Thread 1, queues 1 -> 2
        // threadManager.add(threadId, wGui, queueIn++, queueOut++);               // Thread 0, queues 2 -> 3

        // ------------------------- STARTING AND STOPPING THREADING -------------------------
        op::opLog("Starting thread(s)...", op::Priority::High);
        // Two different ways of running the program on multithread environment
            // Option a) Using the main thread (this thread) for processing (it saves 1 thread, recommended)
        threadManager.exec();
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
        // op::opLog("Stopping thread(s)", op::Priority::High);
        // threadManager.stop();

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
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

    // Running openPoseTutorialThread1
    return openPoseTutorialThread1();
}
