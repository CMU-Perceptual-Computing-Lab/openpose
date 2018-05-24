#include <openpose/filestream/fileStream.hpp>
#include <openpose/3d/cameraParameterReader.hpp>

namespace op
{
    // // User configurable parameters
    // // Camera matrices (extrinsic parameters) - rotation and pose orientation between cameras
    // // Camera 1
    // const cv::Mat M_1_1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0);
    // // Not working on Windows
    // // const cv::Mat M_1_1 = cv::Mat::eye(3, 4, CV_64F);
    // // From camera 1 to 2
    // const cv::Mat M_1_2 = (cv::Mat_<double>(3, 4)
    //     << 0.999962504862692, -0.00165862051503619, 0.00849928507093793, -238.301309354482,
    //     0.00176155163779584, 0.999925029704659, -0.0121174215889211, 4.75863886121558,
    //     -0.00847854967298925, 0.0121319391740716, 0.999890459124058, 15.9219925821916);
    // // From camera 1 to 3
    // const cv::Mat M_1_3 = (cv::Mat_<double>(3, 4)
    //     << 0.995809442124071, -0.000473104796892308, 0.0914512501193800, -461.301274485705,
    //     0.00165046455210419, 0.999916727562850, -0.0127989806923977, 6.22648121362088,
    //     -0.0914375794917412, 0.0128962828696210, 0.995727299487585, 63.4911132860733);
    // // From camera 2 to 3
    // const cv::Mat M_2_3 = (cv::Mat_<double>(3, 4)
    //     << 0.999644115423621, -0.00194501088674130, -0.0266056278177532, -235.236375502202,
    //     0.00201646110733780, 0.999994431880356, 0.00265896462686206, 9.52238656728889,
    //     0.0266003079592876, -0.00271166755609303, 0.999642471324391, -4.23534963077479);

    // // Intrinsic and distortion parameters
    // // Camera 1 parameters
    // const cv::Mat INTRINSIC_1 = (cv::Mat_<double>(3, 3) << 817.93481631740565, 0, 600.70689997785121,
    //     0, 816.51774059837908, 517.84529566329593,
    //     0, 0, 1);
    // const cv::Mat DISTORTION_1 = (cv::Mat_<double>(8, 1) <<
    //     -1.8102158829399091, 9.1966147162623262, -0.00044293900343777355, 0.0013638377686816653,
    //     1.3303863414979364, -1.418905163635487, 8.4725535468475819, 4.7911023525901033);
    // // Camera 2 parameters
    // const cv::Mat INTRINSIC_2 = (cv::Mat_<double>(3, 3) << 816.20921132436638, 0, 612.67087968681585,
    //     0, 816.18292222910486, 530.47901782670431,
    //     0, 0, 1);
    // const cv::Mat DISTORTION_2 = (cv::Mat_<double>(8, 1) <<
    //     -5.1088507540294881, 133.63995617304997, -0.0010048069080912836, 0.00018825291386406282,
    //     20.688286893903879, -4.7604289550474768, 132.42412342224557, 70.01195364029752);
    // const cv::Mat INTRINSIC_3 = (cv::Mat_<double>(3, 3) << 798.42980806905666, 0, 646.48130011561727,
    //     0, 798.46535448393979, 523.91590563194586,
    //     0, 0, 1);
    // // Camera 3
    // const cv::Mat DISTORTION_3 = (cv::Mat_<double>(8, 1) <<
    //     -0.57530495294002304, -0.54721992620722582, -0.00037614702677289967, -0.00081995658363481598,
    //     -0.020321660897680775, -0.18040544059116842, -0.87724444571603022, -0.13136636671099691);

    // // Do not modify this code
    // const std::vector<cv::Mat> INTRINSICS{ INTRINSIC_1, INTRINSIC_2, INTRINSIC_3 };
    // const std::vector<cv::Mat> DISTORTIONS{ DISTORTION_1, DISTORTION_2, DISTORTION_3 };
    // const std::vector<cv::Mat> M{ M_1_1, M_1_2, M_1_3 };
    // // Not working on Windows
    // // const std::vector<cv::Mat> M_EACH_CAMERA{
    // //     INTRINSIC_1 * M_1_1,
    // //     INTRINSIC_2 * M_1_2,
    // //     INTRINSIC_3 * M_1_3
    // // };

    CameraParameterReader::CameraParameterReader()
    {
    }

    CameraParameterReader::CameraParameterReader(const std::string& serialNumber,
                                                 const cv::Mat& cameraIntrinsics,
                                                 const cv::Mat& cameraDistortion,
                                                 const cv::Mat& cameraExtrinsics)
    {
        try
        {
            // Security checks
            if (serialNumber.empty() || cameraIntrinsics.empty() || cameraDistortion.empty())
                error("Camera intrinsics, distortion, and/or serialNumber cannot be empty.",
                      __LINE__, __FUNCTION__, __FILE__);
            // Add new matrices
            mSerialNumbers.emplace_back(serialNumber);
            mCameraIntrinsics.emplace_back(cameraIntrinsics.clone());
            mCameraDistortions.emplace_back(cameraDistortion.clone());
            // Add extrinsics if not empty
            if (!cameraExtrinsics.empty())
                mCameraExtrinsics.emplace_back(cameraExtrinsics.clone());
            // Otherwise, add cv::eye
            else
                mCameraExtrinsics.emplace_back(cv::Mat::eye(3, 4, cameraIntrinsics.type()));
            mCameraMatrices.emplace_back(mCameraIntrinsics.back() * mCameraExtrinsics.back());
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
            mSerialNumbers = serialNumbers;

            // Commong saving/loading
            const std::string fileNameNoExtension{cameraParameterPath};
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion"
            };

            // Load parameters
            mCameraMatrices.clear();
            mCameraExtrinsics.clear();
            mCameraIntrinsics.clear();
            mCameraDistortions.clear();
            // log("Camera matrices:");
            for (auto i = 0ull ; i < serialNumbers.size() ; i++)
            {
                const auto parameterPath = fileNameNoExtension + mSerialNumbers.at(i);
                const auto cameraParameters = loadData(cvMatNames, parameterPath, dataFormat);
                // Error if empty element
                if (cameraParameters.empty() || cameraParameters.at(0).empty()
                    || cameraParameters.at(1).empty() || cameraParameters.at(2).empty())
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
                }
                mCameraExtrinsics.emplace_back(cameraParameters.at(0));
                mCameraIntrinsics.emplace_back(cameraParameters.at(1));
                mCameraDistortions.emplace_back(cameraParameters.at(2));
                mCameraMatrices.emplace_back(mCameraIntrinsics.back() * mCameraExtrinsics.back());
                // log(cameraParameters.at(0));
            }
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

    void CameraParameterReader::writeParameters(const std::string& cameraParameterPath) const
    {
        try
        {
            // Security check
            if (mSerialNumbers.size() != mCameraIntrinsics.size() || mSerialNumbers.size() != mCameraDistortions.size()
                || (mSerialNumbers.size() != mCameraIntrinsics.size() && !mCameraExtrinsics.empty()))
                error("Arguments must have same size (mSerialNumbers, mCameraIntrinsics, mCameraDistortions,"
                      " and mCameraExtrinsics).", __LINE__, __FUNCTION__, __FILE__);
            // Commong saving/loading
            const auto dataFormat = DataFormat::Xml;
            const std::vector<std::string> cvMatNames {
                "CameraMatrix", "Intrinsics", "Distortion"
            };
            // Saving
            for (auto i = 0ull ; i < mSerialNumbers.size() ; i++)
            {
                std::vector<cv::Mat> cameraParameters;
                cameraParameters.emplace_back(mCameraExtrinsics[i]);
                cameraParameters.emplace_back(mCameraIntrinsics[i]);
                cameraParameters.emplace_back(mCameraDistortions[i]);
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
}
