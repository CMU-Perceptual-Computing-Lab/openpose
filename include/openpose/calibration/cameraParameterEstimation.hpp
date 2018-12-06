#ifndef OPENPOSE_CALIBRATION_CAMERA_PARAMETER_ESTIMATION_HPP
#define OPENPOSE_CALIBRATION_CAMERA_PARAMETER_ESTIMATION_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * This function estimate and saves the intrinsic parameters (K and distortion coefficients).
     * @param gridInnerCorners The Point<int> of the board, i.e., the number of squares by width and height
     * @param gridSquareSizeMm Floating number with the size of a square in your defined unit (point, millimeter,etc).
     * @param flags Integer with the OpenCV flags for calibration (e.g., CALIB_RATIONAL_MODEL,
     * CALIB_THIN_PRISM_MODEL, or CALIB_TILTED_MODEL)
     * @param outputFilePath String with the name of the file where to write
     */
    OP_API void estimateAndSaveIntrinsics(
        const Point<int>& gridInnerCorners, const float gridSquareSizeMm, const int flags,
        const std::string& outputParameterFolder, const std::string& imageFolder, const std::string& serialNumber,
        const bool saveImagesWithCorners = false);

    OP_API void estimateAndSaveExtrinsics(
        const std::string& parameterFolder, const std::string& imageFolder, const Point<int>& gridInnerCorners,
        const float gridSquareSizeMm, const int index0, const int index1, const bool imagesAreUndistorted,
        const bool combineCam0Extrinsics);

    OP_API void refineAndSaveExtrinsics(
        const std::string& parameterFolder, const std::string& imageFolder, const Point<int>& gridInnerCorners,
        const float gridSquareSizeMm, const int numberCameras, const bool imagesAreUndistorted,
        const bool saveImagesWithCorners = false);

    OP_API void estimateAndSaveSiftFile(
        const Point<int>& gridInnerCorners, const std::string& imageFolder, const int numberCameras,
        const bool saveImagesWithCorners = false);
}

#endif // OPENPOSE_CALIBRATION_CAMERA_PARAMETER_ESTIMATION_HPP
