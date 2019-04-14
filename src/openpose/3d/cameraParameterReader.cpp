#include <openpose/core/macros.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap in OpenCV 4
#endif
#include <opencv2/imgproc/imgproc.hpp> // cv::initUndistortRectifyMap (OpenCV <= 3), cv::undistort
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/3d/cameraParameterReader.hpp>

namespace op
{
    CameraParameterReader::CameraParameterReader() :
        mUndistortImage{false}
    {
    }

    CameraParameterReader::~CameraParameterReader()
    {
    }

    CameraParameterReader::CameraParameterReader(const std::string& serialNumber,
                                                 const cv::Mat& cameraIntrinsics,
                                                 const cv::Mat& cameraDistortion,
                                                 const cv::Mat& cameraExtrinsics,
                                                 const cv::Mat& cameraExtrinsicsInitial) :
        mUndistortImage{false}
    {
        try
        {
            // Sanity checks
            if (serialNumber.empty())
                error("Camera serialNumber cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (cameraIntrinsics.empty())
                error("Camera intrinsics cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            if (cameraDistortion.empty())
                error("Camera distortion cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
            // Add new matrices
            mSerialNumbers.emplace_back(serialNumber);
            mCameraIntrinsics.emplace_back(cameraIntrinsics.clone());
            mCameraDistortions.emplace_back(cameraDistortion.clone());
            // Add extrinsics if not empty
            if (!cameraExtrinsics.empty())
                mCameraExtrinsics.emplace_back(cameraExtrinsics.clone());
            else
                mCameraExtrinsics.emplace_back(cv::Mat::eye(3, 4, cameraIntrinsics.type()));
            // Add extrinsics (initial) if not empty
            if (!cameraExtrinsicsInitial.empty())
                mCameraExtrinsicsInitial.emplace_back(cameraExtrinsicsInitial.clone());
            // Otherwise, add cv::eye
            else
                mCameraExtrinsicsInitial.emplace_back(cv::Mat::eye(3, 4, cameraIntrinsics.type()));
            mCameraMatrices.emplace_back(mCameraIntrinsics.back() * mCameraExtrinsics.back());
            // Undistortion cv::Mats
            mRemoveDistortionMaps1.resize(getNumberCameras());
            mRemoveDistortionMaps2.resize(getNumberCameras());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CameraParameterReader::readParameters(const std::string& cameraParameterPath,
                                               const std::vector<std::string>& serialNumbers)
    {
        try
        {
            // Serial numbers
            if (serialNumbers.empty())
            {
                mSerialNumbers = getFilesOnDirectory(cameraParameterPath, "xml");
                for (auto& serialNumber : mSerialNumbers)
                    serialNumber = getFileNameNoExtension(serialNumber);
            }
            else
                mSerialNumbers = serialNumbers;

            // Commong saving/loading
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion", "CameraMatrixInitial"
            };

            // Load parameters
            mCameraMatrices.clear();
            mCameraDistortions.clear();
            mCameraIntrinsics.clear();
            mCameraExtrinsics.clear();
            mCameraExtrinsicsInitial.clear();
            // log("Camera matrices:");
            for (auto i = 0ull ; i < mSerialNumbers.size() ; i++)
            {
                const auto parameterPath = cameraParameterPath + mSerialNumbers.at(i);
                const auto cameraParameters = loadData(cvMatNames, parameterPath, dataFormat);
                // Error if empty element
                if (cameraParameters.empty() || cameraParameters.at(0).empty()
                    || cameraParameters.at(1).empty() || cameraParameters.at(2).empty()
                    || cameraParameters.at(3).empty())
                {
                    const std::string errorMessage = " of the camera with serial number `" + mSerialNumbers[i]
                                                   + "` (file: " + parameterPath + "." + dataFormatToString(dataFormat)
                                                   + "). Is its format valid? You might want to check the example xml"
                                                   + " file.";
                    if (cameraParameters.empty())
                        error("Error at reading the camera parameters" + errorMessage,
                              __LINE__, __FUNCTION__, __FILE__);
                    if (cameraParameters.at(0).empty())
                        error("Error at reading the camera matrix parameters" + errorMessage,
                              __LINE__, __FUNCTION__, __FILE__);
                    if (cameraParameters.at(1).empty())
                        error("Error at reading the camera intrinsics parameters" + errorMessage,
                              __LINE__, __FUNCTION__, __FILE__);
                    if (cameraParameters.at(2).empty())
                        error("Error at reading the camera distortion parameters" + errorMessage,
                              __LINE__, __FUNCTION__, __FILE__);
                    // Commented for back-compatibility
                    // if (cameraParameters.at(3).empty())
                    //     error("Error at reading the camera distortion parameters" + errorMessage,
                    //           __LINE__, __FUNCTION__, __FILE__);
                }
                mCameraExtrinsics.emplace_back(cameraParameters.at(0));
                mCameraIntrinsics.emplace_back(cameraParameters.at(1));
                mCameraDistortions.emplace_back(cameraParameters.at(2));
                mCameraExtrinsicsInitial.emplace_back(cameraParameters.at(3));
                mCameraMatrices.emplace_back(mCameraIntrinsics.back() * mCameraExtrinsics.back());
                // log(cameraParameters.at(0));
            }
            // Undistortion cv::Mats
            mRemoveDistortionMaps1.resize(getNumberCameras());
            mRemoveDistortionMaps2.resize(getNumberCameras());
            // // mCameraMatrices
            // log("\nFull camera matrices:");
            // for (const auto& cvMat : mCameraMatrices)
            //     log(cvMat);
            // // mCameraIntrinsics
            // log("\nCamera intrinsic parameters:");
            // for (const auto& cvMat : mCameraIntrinsics)
            //     log(cvMat);
            // // mCameraDistortions
            // log("\nCamera distortion parameters:");
            // for (const auto& cvMat : mCameraDistortions)
            //     log(cvMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CameraParameterReader::readParameters(const std::string& cameraParameterPath,
                                               const std::string& serialNumber)
    {
        try
        {
            readParameters(cameraParameterPath, std::vector<std::string>{serialNumber});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CameraParameterReader::writeParameters(const std::string& cameraParameterPath) const
    {
        try
        {
            // Sanity check
            if (mSerialNumbers.size() != mCameraIntrinsics.size() || mSerialNumbers.size() != mCameraDistortions.size()
                || (mSerialNumbers.size() != mCameraIntrinsics.size() && !mCameraExtrinsics.empty()))
                error("Arguments must have same size (mSerialNumbers, mCameraIntrinsics, mCameraDistortions,"
                      " and mCameraExtrinsics).", __LINE__, __FUNCTION__, __FILE__);
            // Commong saving/loading
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion", "CameraMatrixInitial"
            };
            // Saving
            for (auto i = 0ull ; i < mSerialNumbers.size() ; i++)
            {
                std::vector<cv::Mat> cameraParameters;
                cameraParameters.emplace_back(mCameraExtrinsics[i]);
                cameraParameters.emplace_back(mCameraIntrinsics[i]);
                cameraParameters.emplace_back(mCameraDistortions[i]);
                cameraParameters.emplace_back(mCameraExtrinsicsInitial[i]);
                saveData(cameraParameters, cvMatNames, cameraParameterPath + mSerialNumbers[i], dataFormat);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    unsigned long long CameraParameterReader::getNumberCameras() const
    {
        try
        {
            return mSerialNumbers.size();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0ull;
        }
    }

    const std::vector<std::string>& CameraParameterReader::getCameraSerialNumbers() const
    {
        try
        {
            return mSerialNumbers;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mSerialNumbers;
        }
    }

    const std::vector<cv::Mat>& CameraParameterReader::getCameraMatrices() const
    {
        try
        {
            return mCameraMatrices;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCameraMatrices;
        }
    }

    const std::vector<cv::Mat>& CameraParameterReader::getCameraDistortions() const
    {
        try
        {
            return mCameraDistortions;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCameraDistortions;
        }
    }

    const std::vector<cv::Mat>& CameraParameterReader::getCameraIntrinsics() const
    {
        try
        {
            return mCameraIntrinsics;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCameraIntrinsics;
        }
    }

    const std::vector<cv::Mat>& CameraParameterReader::getCameraExtrinsics() const
    {
        try
        {
            return mCameraExtrinsics;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCameraExtrinsics;
        }
    }

    const std::vector<cv::Mat>& CameraParameterReader::getCameraExtrinsicsInitial() const
    {
        try
        {
            return mCameraExtrinsicsInitial;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return mCameraExtrinsicsInitial;
        }
    }

    bool CameraParameterReader::getUndistortImage() const
    {
        try
        {
            return mUndistortImage;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void CameraParameterReader::setUndistortImage(const bool undistortImage)
    {
        try
        {
            mUndistortImage = undistortImage;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CameraParameterReader::undistort(cv::Mat& frame, const unsigned int cameraIndex)
    {
        try
        {
            if (mUndistortImage)
            {
                // Sanity check
                if (mRemoveDistortionMaps1.size() <= cameraIndex || mRemoveDistortionMaps2.size() <= cameraIndex)
                {
                    error("Variable cameraIndex is out of bounds, it should be smaller than mRemoveDistortionMapsX.",
                          __LINE__, __FUNCTION__, __FILE__);
                }
                // Only first time
                if (mRemoveDistortionMaps1[cameraIndex].empty() || mRemoveDistortionMaps2[cameraIndex].empty())
                {
                    const auto cameraIntrinsics = mCameraIntrinsics.at(0);
                    const auto cameraDistorsions = mCameraDistortions.at(0);
                    const auto imageSize = frame.size();
                    // // Option a - 80 ms / 3 images
                    // // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                    // cv::undistort(cvMatDistorted, mCvMats[i], cameraIntrinsics, cameraDistorsions);
                    // // In OpenCV 2.4, cv::undistort is exactly equal than cv::initUndistortRectifyMap
                    // (with CV_16SC2) + cv::remap (with LINEAR). I.e., log(cv::norm(cvMatMethod1-cvMatMethod2)) = 0.
                    // Option b - 15 ms / 3 images (LINEAR) or 25 ms (CUBIC)
                    // Distorsion removal - not required and more expensive (applied to the whole image instead of
                    // only to our interest points)
                    cv::initUndistortRectifyMap(
                        cameraIntrinsics, cameraDistorsions, cv::Mat(),
                        // cameraIntrinsics instead of cv::getOptimalNewCameraMatrix to
                        // avoid black borders
                        cameraIntrinsics,
                        // #include <opencv2/calib3d/calib3d.hpp> for next line
                        // cv::getOptimalNewCameraMatrix(cameraIntrinsics,
                        //                               cameraDistorsions,
                        //                               imageSize, 1,
                        //                               imageSize, 0),
                        imageSize,
                        CV_16SC2, // Faster, less memory
                        // CV_32FC1, // More accurate
                        mRemoveDistortionMaps1[cameraIndex],
                        mRemoveDistortionMaps2[cameraIndex]);
                }
                cv::Mat undistortedCvMat;
                cv::remap(frame, undistortedCvMat,
                          mRemoveDistortionMaps1[cameraIndex], mRemoveDistortionMaps2[cameraIndex],
                          // cv::INTER_NEAREST);
                          cv::INTER_LINEAR);
                          // cv::INTER_CUBIC);
                          // cv::INTER_LANCZOS4); // Smoother, but we do not need this quality & its >>expensive
                std::swap(undistortedCvMat, frame);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
