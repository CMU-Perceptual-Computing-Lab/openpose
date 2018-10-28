#include <openpose/filestream/fileStream.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/3d/cameraParameterReader.hpp>

namespace op
{
    CameraParameterReader::CameraParameterReader()
    {
    }

    CameraParameterReader::~CameraParameterReader()
    {
    }

    CameraParameterReader::CameraParameterReader(const std::string& serialNumber,
                                                 const cv::Mat& cameraIntrinsics,
                                                 const cv::Mat& cameraDistortion,
                                                 const cv::Mat& cameraExtrinsics)
    {
        try
        {
            // Sanity check
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
                "CameraMatrix", "Intrinsics", "Distortion"
            };

            // Load parameters
            mCameraMatrices.clear();
            mCameraExtrinsics.clear();
            mCameraIntrinsics.clear();
            mCameraDistortions.clear();
            // log("Camera matrices:");
            for (auto i = 0ull ; i < mSerialNumbers.size() ; i++)
            {
                const auto parameterPath = cameraParameterPath + mSerialNumbers.at(i);
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
            // Sanity check
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
