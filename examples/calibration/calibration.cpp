// ------------------------- OpenPose Calibration Toolbox -------------------------
// Check `doc/advanced/calibration_module.md`.
// Implemented on top of OpenCV.
// It computes and saves the intrinsics parameters of the input images.

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Calibration
DEFINE_int32(mode,                      1,              "Select 1 for intrinsic camera parameter calibration, 2 for extrinsic calibration.");
DEFINE_string(calibration_image_dir,    "images/intrinsics/", "Directory where the images for camera parameter calibration are placed.");
DEFINE_double(grid_square_size_mm,      127.0,          "Chessboard square length (in millimeters).");
DEFINE_string(grid_number_inner_corners,"9x6",          "Number of inner corners in width and height, i.e., number of total squares in width"
                                                        " and height minus 1.");
// Mode 1 - Intrinsics
DEFINE_string(camera_serial_number,     "18079958",     "Camera serial number.");
// Mode 2 - Extrinsics
DEFINE_bool(omit_distortion,            false,          "Set to true if image views are already undistorted (e.g., if recorded from OpenPose"
                                                        " after intrinsic parameter calibration).");
DEFINE_bool(combine_cam0_extrinsics,    false,          "Set to true if cam0 extrinsics are not [R=I, t=0]. I will make no effect if cam0 is"
                                                        " already the origin. See doc/advanced/calibration_module.md for an example.");
DEFINE_int32(cam0,                      1,              "Baseline camera for extrinsic calibration, cam1 will be calibrated assuming cam0 the"
                                                        " world coordinate origin.");
DEFINE_int32(cam1,                      0,              "Target camera to estimate its extrinsic parameters, it will be calibrated assuming cam0"
                                                        " as the world coordinate origin.");
// Modes 3-4
DEFINE_int32(number_cameras,            4,              "Number of cameras (for mode 3-4).");
// Producer
DEFINE_string(camera_parameter_folder,  "models/cameraParameters/flir/", "String with the folder where the camera parameters are or will be"
                                                        " located.");

int openPoseDemo()
{
    try
    {
        op::opLog("Starting OpenPose calibration toolbox...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

        // Common parameters
        const auto gridInnerCorners = op::flagsToPoint(op::String(FLAGS_grid_number_inner_corners), "12x7");
        const auto calibrationImageDir = op::formatAsDirectory(FLAGS_calibration_image_dir);
        const auto gridSqureSizeMm = (float)FLAGS_grid_square_size_mm;

        // Calibration - Intrinsics
        if (FLAGS_mode == 1)
        {
            op::opLog("Running calibration (intrinsic parameters)...", op::Priority::High);
            // Parameters
            // const auto flags = 0;                                                                   // 5 parameters
            const auto flags = cv::CALIB_RATIONAL_MODEL;                                            // 8 parameters
            // const auto flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;               // 12 parameters
            // const auto flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL; // 14
            // const auto saveImagesWithCorners = false;
            const auto saveImagesWithCorners = true;
            // Run calibration
            op::estimateAndSaveIntrinsics(
                gridInnerCorners, gridSqureSizeMm, flags,
                op::formatAsDirectory(FLAGS_camera_parameter_folder), calibrationImageDir, FLAGS_camera_serial_number,
                saveImagesWithCorners);
            op::opLog("Intrinsic calibration completed!", op::Priority::High);
        }

        // Calibration - Extrinsics
        else if (FLAGS_mode == 2)
        {
            op::opLog("Running calibration (extrinsic parameters)...", op::Priority::High);
            // Run calibration
            op::estimateAndSaveExtrinsics(
                FLAGS_camera_parameter_folder, calibrationImageDir, gridInnerCorners, gridSqureSizeMm,
                FLAGS_cam0, FLAGS_cam1, FLAGS_omit_distortion, FLAGS_combine_cam0_extrinsics);
            // Logging
            op::opLog("Extrinsic calibration completed!", op::Priority::High);
        }

        // Calibration - Extrinsics - Bundle Adjustment (BA)
        else if (FLAGS_mode == 3)
        {
            op::opLog("Running calibration (bundle adjustment over extrinsic parameters)...", op::Priority::High);
            // Sanity check
            if (!FLAGS_omit_distortion)
                op::error("This mode assumes that the images are already undistorted (add flag `--omit_distortion`.",
                          __LINE__, __FUNCTION__, __FILE__);
            // Parameters
            const auto saveImagesWithCorners = false;
            // Run calibration
            op::refineAndSaveExtrinsics(
                FLAGS_camera_parameter_folder, calibrationImageDir, gridInnerCorners, gridSqureSizeMm,
                FLAGS_number_cameras, FLAGS_omit_distortion, saveImagesWithCorners);
            // Logging
            op::opLog("Extrinsic calibration (bundle adjustment) completed!", op::Priority::High);
        }

        // // Calibration - Extrinsics Refinement with Visual SFM
        // else if (FLAGS_mode == 4)
        // {
        //     op::opLog("Running calibration (intrinsic parameters)...", op::Priority::High);
        //     // Obtain & save intrinsics
        //     const auto saveImagesWithCorners = false;
        //     // const auto saveImagesWithCorners = true;
        //     // Run camera calibration code
        //     op::estimateAndSaveSiftFile(
        //         gridInnerCorners, calibrationImageDir, FLAGS_number_cameras, saveImagesWithCorners);
        //     op::opLog("Intrinsic calibration completed!", op::Priority::High);
        // }

        else
            op::error("Unknown `--mode " + std::to_string(FLAGS_mode) + "`.", __LINE__, __FUNCTION__, __FILE__);

        // Measuring total time
        op::printTime(
            opTimer, "OpenPose calibration toolbox successfully finished. Total time: ", " seconds.",
            op::Priority::High);

        return 0;
    }
    catch (const std::exception&)
    {
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
