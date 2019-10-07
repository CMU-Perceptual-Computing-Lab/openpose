#include <openpose/3d/poseTriangulation.hpp>
#include <numeric> // std::accumulate
#include <openpose/utilities/fastMath.hpp>
#include <openpose_private/3d/poseTriangulationPrivate.hpp>

namespace op
{
    inline bool isValidKeypoint(const float* const keypointPtr, const Point<int>& imageSize)
    {
        try
        {
            const auto threshold = 0.35f;
            return (keypointPtr[2] > threshold
                    // If keypoint in border --> most probably it is actually out of the image,
                    // so removed to reduce that noise
                    && keypointPtr[0] > 8
                    && keypointPtr[0] < imageSize.x - 8
                    && keypointPtr[1] > 8
                    && keypointPtr[1] < imageSize.y - 8);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool reconstructArrayThread(
        Array<float>* keypoints3DPtr, const std::vector<Array<float>>& keypointsVector,
        const std::vector<cv::Mat>& cameraMatrices, const std::vector<Point<int>>& imageSizes, const int minViews3d)
    {
        try
        {
            auto& keypoints3D = *keypoints3DPtr;

            // Sanity check
            if (cameraMatrices.size() < 2)
                error("Only 1 camera detected. The 3-D reconstruction module can only be used with > 1 cameras"
                      " simultaneously. E.g., using FLIR stereo cameras (`--flir_camera`).",
                      __LINE__, __FUNCTION__, __FILE__);
            // Get number body parts and whether at least 2 cameras have keypoints
            auto detectionMissed = 0;
            auto numberBodyParts = 0;
            auto channel0Length = 0;
            for (const auto& keypoints : keypointsVector)
            {
                if (!keypoints.empty())
                {
                    ++detectionMissed;
                    if (detectionMissed > 1)
                    {
                        numberBodyParts = keypoints.getSize(1);
                        channel0Length = keypoints.getSize(2);
                        break;
                    }
                }
            }
            // If at least 2 set of keypoints not empty
            if (numberBodyParts > 0)
            {
                // Create x-y vector from high score results
                std::vector<int> indexesUsed;
                std::vector<std::vector<cv::Point2d>> xyPoints;
                std::vector<std::vector<cv::Mat>> cameraMatricesPerPoint;
                for (auto part = 0; part < numberBodyParts; ++part)
                {
                    // Create vector of points
                    // auto missedPoint = false;
                    std::vector<cv::Point2d> xyPointsElement;
                    std::vector<cv::Mat> cameraMatricesElement;
                    const auto baseIndex = part * channel0Length;
                    // for (auto& keypoints : keypointsVector)
                    for (auto i = 0u ; i < keypointsVector.size() ; i++)
                    {
                        const auto& keypoints = keypointsVector[i];
                        if (!keypoints.empty() && isValidKeypoint(&keypoints[baseIndex], imageSizes[i]))
                        {
                            xyPointsElement.emplace_back(
                                cv::Point2d{keypoints[baseIndex], keypoints[baseIndex+1]});
                            cameraMatricesElement.emplace_back(cameraMatrices[i]);
                        }
                    }
                    const auto minViews3dValue = (minViews3d > 0 ? minViews3d
                        : fastMax(2u, fastMin(4u, (unsigned int)cameraMatrices.size()-1u)));
                    // If visible for at least minViews3dValue views
                    if (minViews3dValue <= xyPointsElement.size())
                    // Old code
                    // // If visible from all views (minViews3d < 0) or if visible for at least minViews3d views
                    // if ((minViews3d < 0 && cameraMatricesElement.size() == cameraMatrices.size())
                    //     || (minViews3d > 1 && minViews3d <= (int)xyPointsElement.size()))
                    {
                        indexesUsed.emplace_back(part);
                        xyPoints.emplace_back(xyPointsElement);
                        cameraMatricesPerPoint.emplace_back(cameraMatricesElement);
                    }
                }
                // 3D reconstruction
                const auto imageRatio = std::sqrt(imageSizes[0].x * imageSizes[0].y / 1310720.);
                const auto reprojectionMaxAcceptable = 25 * imageRatio;
                std::vector<double> reprojectionErrors(xyPoints.size());
                if (!xyPoints.empty())
                {
                    keypoints3D.reset({ 1, numberBodyParts, 4 }, 0.f);
                    // Do 3D reconstruction
                    std::vector<cv::Point3f> xyzPoints(xyPoints.size());
                    for (auto i = 0u; i < xyPoints.size(); i++)
                    {
                        cv::Mat reconstructedPoint;
                        reprojectionErrors[i] = triangulateWithOptimization(
                            reconstructedPoint, cameraMatricesPerPoint[i], xyPoints[i], reprojectionMaxAcceptable);
                        xyzPoints[i] = cv::Point3d{
                            reconstructedPoint.at<double>(0), reconstructedPoint.at<double>(1),
                            reconstructedPoint.at<double>(2)};
                    }
                    const auto reprojectionErrorTotal = std::accumulate(
                        reprojectionErrors.begin(), reprojectionErrors.end(), 0.0) / xyPoints.size();

                    // 3D points to pose
                    // OpenCV alternative:
                    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints
                    // cv::Mat reconstructedPoints{4, firstcv::Points.size(), CV_64F};
                    // cv::triangulatePoints(cv::Mat::eye(3,4, CV_64F), M_3_1, firstcv::Points, secondcv::Points,
                    //                           reconstructedcv::Points);
                    // 20 pixels for 1280x1024 image
                    bool atLeastOnePointProjected = false;
                    const auto lastChannelLength = keypoints3D.getSize(2);
                    for (auto index = 0u; index < indexesUsed.size(); ++index)
                    {
                        if (std::isfinite(xyzPoints[index].x) && std::isfinite(xyzPoints[index].y)
                            && std::isfinite(xyzPoints[index].z)
                            // Remove outliers
                            && (reprojectionErrors[index] < 5 * reprojectionErrorTotal
                                && reprojectionErrors[index] < reprojectionMaxAcceptable))
                        {
                            const auto baseIndex = indexesUsed[index] * lastChannelLength;
                            keypoints3D[baseIndex] = xyzPoints[index].x;
                            keypoints3D[baseIndex + 1] = xyzPoints[index].y;
                            keypoints3D[baseIndex + 2] = xyzPoints[index].z;
                            keypoints3D[baseIndex + 3] = 1.f;
                            atLeastOnePointProjected = true;
                        }
                    }
                    // Warning
                    if (reprojectionErrorTotal > 60)
                        opLog("Unusual high re-projection error (averaged over #keypoints) of value "
                            + std::to_string(reprojectionErrorTotal) + " pixels, while the average for a good OpenPose"
                            " detection from 4 cameras is about 2-3 pixels. It might be simply a wrong OpenPose"
                            " detection. However, if this message appears very frequently, your calibration parameters"
                            " might be wrong. Note: If you have introduced your own camera intrinsics, are they an"
                            " upper triangular matrix (as specified in the OpenPose doc/modules/calibration_module.md"
                            " and 3d_reconstruction_module.md)?", Priority::High);
                    // opLog("Reprojection error: " + std::to_string(reprojectionErrorTotal)); // To debug reprojection error
                    return atLeastOnePointProjected;
                }
                return false;
            }
            // Keypoints in < 2 images
            else
                return true; // True because it is not a 3D problem but rather it is e.g., a scene without people on it
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    PoseTriangulation::PoseTriangulation(const int minViews3d) :
        mMinViews3d{minViews3d}
    {
        try
        {
            // Sanity check
            if (0 <= mMinViews3d && mMinViews3d < 2)
                error("Minimum number of views must be at least 2 (e.g., `--3d_min_views 2`) or negative.",
                      __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseTriangulation::~PoseTriangulation()
    {
    }

    void PoseTriangulation::initializationOnThread()
    {
    }

    Array<float> PoseTriangulation::reconstructArray(
        const std::vector<Array<float>>& keypointsVector, const std::vector<Matrix>& cameraMatrices,
        const std::vector<Point<int>>& imageSizes) const
    {
        try
        {
            return reconstructArray(
                std::vector<std::vector<Array<float>>>{keypointsVector}, cameraMatrices, imageSizes).at(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    const std::string sFlirErrorMessage{
        " If you are simultaneously using FLIR cameras (`--flir_camera`) and the 3-D reconstruction module"
        " (`--3d), you should also enable `--frame_undistort` so their camera parameters are read."};
    std::vector<Array<float>> PoseTriangulation::reconstructArray(
        const std::vector<std::vector<Array<float>>>& keypointsVectors,
        const std::vector<Matrix>& cameraMatrices,
        const std::vector<Point<int>>& imageSizes) const
    {
        try
        {
            OP_OP2CVVECTORMAT(cvCameraMatrices, cameraMatrices);
            // Sanity checks
            if (cvCameraMatrices.size() < 2)
                error("3-D reconstruction (`--3d`) requires at least 2 camera views, only found "
                    + std::to_string(cvCameraMatrices.size()) + "camera parameter matrices." + sFlirErrorMessage,
                    __LINE__, __FUNCTION__, __FILE__);
            for (const auto& cameraMatrix : cvCameraMatrices)
                if (cameraMatrix.empty())
                    error("Camera matrix was found empty during 3-D reconstruction (`--3d`)." + sFlirErrorMessage,
                        __LINE__, __FUNCTION__, __FILE__);
            if (cvCameraMatrices.size() != imageSizes.size())
                error("The camera parameters and number of images must be the same ("
                    + std::to_string(cvCameraMatrices.size()) + " vs. " + std::to_string(imageSizes.size()) + ").",
                    __LINE__, __FUNCTION__, __FILE__);
            // Run 3-D reconstruction
            bool keypointsReconstructed = false;
            std::vector<Array<float>> keypoints3Ds(keypointsVectors.size());
            // std::vector<std::thread> threads(keypointsVectors.size()-1);
            for (auto i = 0u; i < keypointsVectors.size()-1; i++)
            {
                // // Multi-thread option - ~15% slower
                // // Ceres seems to be super slow if run concurrently in different threads
                // threads.at(i) = std::thread{&reconstructArrayThread,
                //                             &keypoints3Ds[i], keypointsVectors[i], cvCameraMatrices,
                //                             imageSizes, mMinViews3d};
                // Single-thread option
                keypointsReconstructed |= reconstructArrayThread(
                    &keypoints3Ds[i], keypointsVectors[i], cvCameraMatrices, imageSizes, mMinViews3d);
            }
            keypointsReconstructed |= reconstructArrayThread(
                &keypoints3Ds.back(), keypointsVectors.back(), cvCameraMatrices, imageSizes, mMinViews3d);
            // // Close threads
            // for (auto& thread : threads)
            //     if (thread.joinable())
            //         thread.join();
            // Warning
            if (!keypointsReconstructed)
                opLog("No keypoints were reconstructed on this frame. It might be simply a challenging frame."
                    " However, if this message appears frequently, OpenPose is facing some unknown issue,"
                    " mabe the calibration parameters are not accurate. Feel free to open a GitHub issue"
                    " (remember to fill all the required information detailed in the GitHub issue template"
                    " when it is created).", Priority::High);
            // Return results
            return keypoints3Ds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
