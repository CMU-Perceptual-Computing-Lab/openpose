// -------------------------- OpenPose C++ API Tutorial - Example 2 - Whole body from image --------------------------
// It reads an image, process it, and displays it with the pose, hand, and face keypoints.

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000241.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
{
    // User's displaying/saving/other processing here
        // datum.cvOutputData: rendered frame with pose or heatmaps
        // datum.poseKeypoints: Array<float> with the estimated pose
    if (datumsPtr != nullptr && !datumsPtr->empty())
    {
        // Display image
        cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
        cv::waitKey(0);
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}

void printKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
{
    // Example: How to use the pose keypoints
    if (datumsPtr != nullptr && !datumsPtr->empty())
    {
        op::log("Body keypoints: " + datumsPtr->at(0).poseKeypoints.toString());
        op::log("Face keypoints: " + datumsPtr->at(0).faceKeypoints.toString());
        op::log("Left hand keypoints: " + datumsPtr->at(0).handKeypoints[0].toString());
        op::log("Right hand keypoints: " + datumsPtr->at(0).handKeypoints[1].toString());
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}

int tutorialApiCpp2()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // Add hand and face
        opWrapper.configure(op::WrapperStructFace{true});
        opWrapper.configure(op::WrapperStructHand{true});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
        // Starting OpenPose
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Process and display image
        const auto imageToProcess = cv::imread(FLAGS_image_path);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
        if (datumProcessed != nullptr)
        {
            printKeypoints(datumProcessed);
            display(datumProcessed);
        }
        else
            op::log("Image could not be processed.", op::Priority::High);

        // Return successful message
        op::log("Stopping OpenPose...", op::Priority::High);
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp2
    return tutorialApiCpp2();
}
