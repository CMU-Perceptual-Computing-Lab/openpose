// ------------------------- OpenPose Calibration Toolbox -------------------------
// Check `doc/calibration_demo.md`.
// Implemented on top of OpenCV.
// It computes and saves the intrinsics parameters of the input images.

// C++ std library dependencies
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread> // std::this_thread
// Other 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Calibration
DEFINE_int32(mode,                      1,              "Select 1 for intrinsic camera parameter calibration, 2 for extrinsic ones.");
DEFINE_string(intrinsics_image_dir,     "images/intrinsics_18079958/", "Directory where the images for intrinsic calibration are placed.");
DEFINE_double(grid_square_size_mm,      40.0,           "Chessboard square length (in millimeters).");
DEFINE_string(grid_number_inner_corners,"9x5",          "Number of inner squares in width and height, i.e., number of total squares in width"
                                                        " and height minus 1.");
DEFINE_string(camera_serial_number,     "18079958",     "Camera serial number.");
// Producer
DEFINE_string(camera_parameter_folder,  "models/cameraParameters/flir/", "String with the folder where the camera parameters are or will be"
                                                        " located.");

int openPoseDemo()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

        // Calibration - Intrinsics
        if (FLAGS_mode == 1)
        {
            op::log("Running calibration (intrinsic parameters)...", op::Priority::High);
            // Obtain & save intrinsics
            const auto gridInnerCorners = op::flagsToPoint(FLAGS_grid_number_inner_corners, "9x5");
            // const auto flags = 0;                                                                                   // 5 parameters
            const auto flags = cv::CALIB_RATIONAL_MODEL;                                                            // 8 parameters
            // const auto flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;                               // 12 parameters
            // const auto flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL;      // 14 parameters
            const auto saveImagesWithCorners = false;
            // Run camera calibration code
            op::estimateAndSaveIntrinsics(gridInnerCorners, FLAGS_grid_square_size_mm, flags,
                                          op::formatAsDirectory(FLAGS_camera_parameter_folder),
                                          op::formatAsDirectory(FLAGS_intrinsics_image_dir),
                                          FLAGS_camera_serial_number, saveImagesWithCorners);
            op::log("Intrinsic calibration completed!", op::Priority::High);
        }

        // Calibration - Extrinsics
        else if (FLAGS_mode == 2)
            op::error("Unimplemented yet.", __LINE__, __FUNCTION__, __FILE__);

        else
            op::error("Unknown `--mode " + std::to_string(FLAGS_mode) + "`.", __LINE__, __FUNCTION__, __FILE__);

        // Measuring total time
        const auto now = std::chrono::high_resolution_clock::now();
        const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                                * 1e-9;
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

    // Running openPoseDemo
    return openPoseDemo();
}
