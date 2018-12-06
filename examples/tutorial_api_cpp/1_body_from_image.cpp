// ----------------------------- OpenPose C++ API Tutorial - Example 1 - Body from image -----------------------------
// It reads an image, process it, and displays it with the pose keypoints.

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg",
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
        // Alternative 1
        op::log("Body keypoints: " + datumsPtr->at(0).poseKeypoints.toString());

        // // Alternative 2
        // op::log(datumsPtr->at(0).poseKeypoints);

        // // Alternative 3
        // std::cout << datumsPtr->at(0).poseKeypoints << std::endl;

        // // Alternative 4 - Accesing each element of the keypoints
        // op::log("\nKeypoints:");
        // const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
        // op::log("Person pose keypoints:");
        // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
        // {
        //     op::log("Person " + std::to_string(person) + " (x, y, score):");
        //     for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
        //     {
        //         std::string valueToPrint;
        //         for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
        //             valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
        //         op::log(valueToPrint);
        //     }
        // }
        // op::log(" ");
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}

int tutorialApiCpp1()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
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

    // Running tutorialApiCpp1
    return tutorialApiCpp1();
}
