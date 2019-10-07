#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap in OpenCV 4
#endif
#include <opencv2/imgproc/imgproc.hpp> // cv::initUndistortRectifyMap (OpenCV <= 3), cv::undistort
#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>

namespace op
{
    struct CameraParameterReader::ImplCameraParameterReader
    {
        std::vector<std::string> mSerialNumbers;
        std::vector<Matrix> mCameraMatrices;
        std::vector<Matrix> mCameraDistortions;
        std::vector<Matrix> mCameraIntrinsics;
        std::vector<Matrix> mCameraExtrinsics;
        std::vector<Matrix> mCameraExtrinsicsInitial;

        // Undistortion (optional)
        bool mUndistortImage;
        std::vector<cv::Mat> mRemoveDistortionMaps1;
        std::vector<cv::Mat> mRemoveDistortionMaps2;

        ImplCameraParameterReader(const bool undistortImage) :
            mUndistortImage{undistortImage}
        {
        }
    };

    CameraParameterReader::CameraParameterReader() :
        spImpl{std::make_shared<ImplCameraParameterReader>(false)}
    {
    }

    CameraParameterReader::~CameraParameterReader()
    {
    }

    CameraParameterReader::CameraParameterReader(const std::string& serialNumber,
                                                 const Matrix& cameraIntrinsics,
                                                 const Matrix& cameraDistortion,
                                                 const Matrix& cameraExtrinsics,
                                                 const Matrix& cameraExtrinsicsInitial) :
        spImpl{std::make_shared<ImplCameraParameterReader>(false)}
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
            spImpl->mSerialNumbers.emplace_back(serialNumber);
            spImpl->mCameraIntrinsics.emplace_back(cameraIntrinsics.clone());
            spImpl->mCameraDistortions.emplace_back(cameraDistortion.clone());
            // Add extrinsics if not empty
            if (!cameraExtrinsics.empty())
                spImpl->mCameraExtrinsics.emplace_back(cameraExtrinsics.clone());
            else
                spImpl->mCameraExtrinsics.emplace_back(Matrix::eye(3, 4, cameraIntrinsics.type()));
            // Add extrinsics (initial) if not empty
            if (!cameraExtrinsicsInitial.empty())
                spImpl->mCameraExtrinsicsInitial.emplace_back(cameraExtrinsicsInitial.clone());
            // Otherwise, add cv::eye
            else
                spImpl->mCameraExtrinsicsInitial.emplace_back(Matrix::eye(3, 4, cameraIntrinsics.type()));;
            const cv::Mat cvCameraMatrices = OP_OP2CVCONSTMAT(spImpl->mCameraIntrinsics.back()) * OP_OP2CVCONSTMAT(spImpl->mCameraExtrinsics.back());
            const Matrix opCameraMatrices = OP_CV2OPCONSTMAT(cvCameraMatrices);
            spImpl->mCameraMatrices.emplace_back(opCameraMatrices);
            // Undistortion Mats
            spImpl->mRemoveDistortionMaps1.resize(getNumberCameras());
            spImpl->mRemoveDistortionMaps2.resize(getNumberCameras());
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
                spImpl->mSerialNumbers = getFilesOnDirectory(cameraParameterPath, "xml");
                for (auto& serialNumber : spImpl->mSerialNumbers)
                    serialNumber = getFileNameNoExtension(serialNumber);
            }
            else
                spImpl->mSerialNumbers = serialNumbers;

            // Commong saving/loading
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion", "CameraMatrixInitial"
            };

            // Load parameters
            spImpl->mCameraMatrices.clear();
            spImpl->mCameraDistortions.clear();
            spImpl->mCameraIntrinsics.clear();
            spImpl->mCameraExtrinsics.clear();
            spImpl->mCameraExtrinsicsInitial.clear();
            // opLog("Camera matrices:");
            for (auto i = 0ull ; i < spImpl->mSerialNumbers.size() ; i++)
            {
                const auto parameterPath = cameraParameterPath + spImpl->mSerialNumbers.at(i);
                const auto opCameraParameters = loadData(cvMatNames, parameterPath, dataFormat);
                OP_OP2CVVECTORMAT(cameraParameters, opCameraParameters)
                // Error if empty element
                if (cameraParameters.empty() || cameraParameters.at(0).empty()
                    || cameraParameters.at(1).empty() || cameraParameters.at(2).empty()
                    || cameraParameters.at(3).empty())
                {
                    const std::string errorMessage = " of the camera with serial number `" + spImpl->mSerialNumbers[i]
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
                spImpl->mCameraExtrinsics.emplace_back(opCameraParameters.at(0));
                spImpl->mCameraIntrinsics.emplace_back(opCameraParameters.at(1));
                spImpl->mCameraDistortions.emplace_back(opCameraParameters.at(2));
                spImpl->mCameraExtrinsicsInitial.emplace_back(opCameraParameters.at(3));
                const cv::Mat cvCameraMatrices = OP_OP2CVCONSTMAT(spImpl->mCameraIntrinsics.back()) * OP_OP2CVCONSTMAT(spImpl->mCameraExtrinsics.back());
                const Matrix opCameraMatrices = OP_CV2OPCONSTMAT(cvCameraMatrices);
                spImpl->mCameraMatrices.emplace_back(opCameraMatrices);
                // opLog(cameraParameters.at(0));
            }
            // Undistortion Mats
            spImpl->mRemoveDistortionMaps1.resize(getNumberCameras());
            spImpl->mRemoveDistortionMaps2.resize(getNumberCameras());
            // // spImpl->mCameraMatrices
            // opLog("\nFull camera matrices:");
            // for (const auto& cvMat : spImpl->mCameraMatrices)
            //     opLog(cvMat);
            // // spImpl->mCameraIntrinsics
            // opLog("\nCamera intrinsic parameters:");
            // for (const auto& cvMat : spImpl->mCameraIntrinsics)
            //     opLog(cvMat);
            // // spImpl->mCameraDistortions
            // opLog("\nCamera distortion parameters:");
            // for (const auto& cvMat : spImpl->mCameraDistortions)
            //     opLog(cvMat);
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
            if (spImpl->mSerialNumbers.size() != spImpl->mCameraIntrinsics.size()
                || spImpl->mSerialNumbers.size() != spImpl->mCameraDistortions.size()
                || (spImpl->mSerialNumbers.size() != spImpl->mCameraIntrinsics.size()
                    && !spImpl->mCameraExtrinsics.empty()))
                error("Arguments must have same size (spImpl->mSerialNumbers, spImpl->mCameraIntrinsics, spImpl->mCameraDistortions,"
                      " and spImpl->mCameraExtrinsics).", __LINE__, __FUNCTION__, __FILE__);
            // Commong saving/loading
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion", "CameraMatrixInitial"
            };
            // Saving
            for (auto i = 0ull ; i < spImpl->mSerialNumbers.size() ; i++)
            {
                std::vector<Matrix> cameraParameters;
                cameraParameters.emplace_back(spImpl->mCameraExtrinsics[i]);
                cameraParameters.emplace_back(spImpl->mCameraIntrinsics[i]);
                cameraParameters.emplace_back(spImpl->mCameraDistortions[i]);
                cameraParameters.emplace_back(spImpl->mCameraExtrinsicsInitial[i]);
                saveData(cameraParameters, cvMatNames, cameraParameterPath + spImpl->mSerialNumbers[i], dataFormat);
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
            return spImpl->mSerialNumbers.size();
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
            return spImpl->mSerialNumbers;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mSerialNumbers;
        }
    }

    const std::vector<Matrix>& CameraParameterReader::getCameraMatrices() const
    {
        try
        {
            return spImpl->mCameraMatrices;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mCameraMatrices;
        }
    }

    const std::vector<Matrix>& CameraParameterReader::getCameraDistortions() const
    {
        try
        {
            return spImpl->mCameraDistortions;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mCameraDistortions;
        }
    }

    const std::vector<Matrix>& CameraParameterReader::getCameraIntrinsics() const
    {
        try
        {
            return spImpl->mCameraIntrinsics;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mCameraIntrinsics;
        }
    }

    const std::vector<Matrix>& CameraParameterReader::getCameraExtrinsics() const
    {
        try
        {
            return spImpl->mCameraExtrinsics;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mCameraExtrinsics;
        }
    }

    const std::vector<Matrix>& CameraParameterReader::getCameraExtrinsicsInitial() const
    {
        try
        {
            return spImpl->mCameraExtrinsicsInitial;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return spImpl->mCameraExtrinsicsInitial;
        }
    }

    bool CameraParameterReader::getUndistortImage() const
    {
        try
        {
            return spImpl->mUndistortImage;
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
            spImpl->mUndistortImage = undistortImage;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void CameraParameterReader::undistort(Matrix& frame, const unsigned int cameraIndex)
    {
        try
        {
            if (spImpl->mUndistortImage)
            {
                // Sanity check
                if (spImpl->mRemoveDistortionMaps1.size() <= cameraIndex
                    || spImpl->mRemoveDistortionMaps2.size() <= cameraIndex)
                {
                    error("Variable cameraIndex is out of bounds, it should be smaller than spImpl->mRemoveDistortionMapsX.",
                          __LINE__, __FUNCTION__, __FILE__);
                }
                // Only first time
                if (spImpl->mRemoveDistortionMaps1[cameraIndex].empty()
                    || spImpl->mRemoveDistortionMaps2[cameraIndex].empty())
                {
                    const auto cvCameraIntrinsics = OP_OP2CVCONSTMAT(spImpl->mCameraIntrinsics.at(0));
                    const auto cvCameraDistorsions = OP_OP2CVCONSTMAT(spImpl->mCameraDistortions.at(0));
                    //const auto imageSize = OP_OP2CVMAT(frame).size();
                    cv::Size imageSize;
                    OP_CONST_MAT_RETURN_FUNCTION(imageSize, frame, size()); // = frame.size();
                    // // Option a - 80 ms / 3 images
                    // // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                    // cv::undistort(cvMatDistorted, mCvMats[i], cvCameraIntrinsics, cvCameraDistorsions);
                    // // In OpenCV 2.4, cv::undistort is exactly equal than cv::initUndistortRectifyMap
                    // (with CV_16SC2) + cv::remap (with LINEAR). I.e., opLog(cv::norm(cvMatMethod1-cvMatMethod2)) = 0.
                    // Option b - 15 ms / 3 images (LINEAR) or 25 ms (CUBIC)
                    // Distorsion removal - not required and more expensive (applied to the whole image instead of
                    // only to our interest points)
                    cv::initUndistortRectifyMap(
                        cvCameraIntrinsics, cvCameraDistorsions, cv::Mat(),
                        // cvCameraIntrinsics instead of cv::getOptimalNewCameraMatrix to
                        // avoid black borders
                        cvCameraIntrinsics,
                        // #include <opencv2/calib3d/calib3d.hpp> for next line
                        // cv::getOptimalNewCameraMatrix(cvCameraIntrinsics,
                        //                               cvCameraDistorsions,
                        //                               imageSize, 1,
                        //                               imageSize, 0),
                        imageSize,
                        CV_16SC2, // Faster, less memory
                        // CV_32FC1, // More accurate
                        spImpl->mRemoveDistortionMaps1[cameraIndex],
                        spImpl->mRemoveDistortionMaps2[cameraIndex]);
                }
                cv::Mat undistortedCvMat;
                const cv::Mat cvFrame = OP_OP2CVCONSTMAT(frame);
                cv::remap(cvFrame, undistortedCvMat,
                          spImpl->mRemoveDistortionMaps1[cameraIndex], spImpl->mRemoveDistortionMaps2[cameraIndex],
                          // cv::INTER_NEAREST);
                          cv::INTER_LINEAR);
                          // cv::INTER_CUBIC);
                          // cv::INTER_LANCZOS4); // Smoother, but we do not need this quality & its >>expensive
                Matrix opUndistortedCvMat = OP_CV2OPMAT(undistortedCvMat);
                std::swap(opUndistortedCvMat, frame);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
