#include <openpose/producer/spinnakerWrapper.hpp>
#include <atomic>
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp> // cv::undistort, cv::initUndistortRectifyMap
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap for OpenCV 4
#endif
#ifdef USE_FLIR_CAMERA
    #include <Spinnaker.h>
#endif
#include <openpose/3d/cameraParameterReader.hpp>

namespace op
{
    #ifdef USE_FLIR_CAMERA
        std::vector<std::string> getSerialNumbers(const Spinnaker::CameraList& cameraList,
                                                  const bool sorted)
        {
            try
            {
                // Get strSerialNumbers
                std::vector<Spinnaker::GenICam::gcstring> strSerialNumbers(cameraList.GetSize());
                for (auto i = 0u; i < strSerialNumbers.size(); i++)
                {
                    // Select camera
                    auto cameraPtr = cameraList.GetByIndex(i);

                    strSerialNumbers[i] = "";

                    Spinnaker::GenApi::CStringPtr ptrStringSerial = cameraPtr->GetTLDeviceNodeMap().GetNode(
                        "DeviceSerialNumber"
                    );

                    if (Spinnaker::GenApi::IsAvailable(ptrStringSerial)
                        && Spinnaker::GenApi::IsReadable(ptrStringSerial))
                    {
                        strSerialNumbers[i] = ptrStringSerial->GetValue();
                    }
                }

                // Spinnaker::GenICam::gcstring to std::string
                std::vector<std::string> serialNumbers(strSerialNumbers.size());
                for (auto i = 0u ; i < serialNumbers.size() ; i++)
                    serialNumbers[i] = strSerialNumbers[i];

                // Sort serial numbers
                if (sorted)
                    std::sort(serialNumbers.begin(), serialNumbers.end());

                // Return result
                return serialNumbers;
            }
            catch (Spinnaker::Exception &e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }

        // This function prints the device information of the camera from the transport
        // layer; please see NodeMapInfo example for more in-depth comments on printing
        // device information from the nodemap.
        int printDeviceInfo(Spinnaker::GenApi::INodeMap &iNodeMap, const unsigned int camNum)
        {
            try
            {
                int result = 0;

                opLog("Printing device information for camera " + std::to_string(camNum) + "...\n", Priority::High);

                Spinnaker::GenApi::FeatureList_t features;
                Spinnaker::GenApi::CCategoryPtr cCategoryPtr = iNodeMap.GetNode("DeviceInformation");
                if (Spinnaker::GenApi::IsAvailable(cCategoryPtr) && Spinnaker::GenApi::IsReadable(cCategoryPtr))
                {
                    cCategoryPtr->GetFeatures(features);

                    Spinnaker::GenApi::FeatureList_t::const_iterator it;
                    for (it = features.begin(); it != features.end(); ++it)
                    {
                        Spinnaker::GenApi::CNodePtr pfeatureNode = *it;
                        const auto cValuePtr = (Spinnaker::GenApi::CValuePtr)pfeatureNode;
                        opLog(pfeatureNode->GetName() + " : " +
                                (IsReadable(cValuePtr) ? cValuePtr->ToString() : "Node not readable"), Priority::High);
                    }
                }
                else
                    opLog("Device control information not available.", Priority::High);
                opLog(" ", Priority::High);

                return result;
            }
            catch (Spinnaker::Exception &e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
        }

        // This function returns the camera to a normal state by turning off trigger
        // mode.
        int resetTrigger(Spinnaker::GenApi::INodeMap &iNodeMap)
        {
            try
            {
                int result = 0;
                //
                // Turn trigger mode back off
                //
                // *** NOTES ***
                // Once all images have been captured, turn trigger mode back off to
                // restore the camera to a clean state.
                //
                Spinnaker::GenApi::CEnumerationPtr ptrTriggerMode = iNodeMap.GetNode("TriggerMode");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerMode)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerMode))
                    error("Unable to disable trigger mode (node retrieval). Non-fatal error...",
                              __LINE__, __FUNCTION__, __FILE__);

                Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOff)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOff))
                    error("Unable to disable trigger mode (enum entry retrieval). Non-fatal error...",
                              __LINE__, __FUNCTION__, __FILE__);

                ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

                // opLog("Trigger mode disabled...", Priority::High);

                return result;
            }
            catch (Spinnaker::Exception &e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
        }

        Spinnaker::ImagePtr spinnakerImagePtrToColor(const Spinnaker::ImagePtr &imagePtr)
        {
            // Original image --> BGR uchar image
            // Print image information
            // Convert image to RGB
            // Interpolation methods
            // http://softwareservices.ptgrey.com/Spinnaker/latest/group___spinnaker_defs.html
            // DEFAULT     Default method.
            // NO_COLOR_PROCESSING     No color processing.
            // NEAREST_NEIGHBOR    Fastest but lowest quality. Equivalent to
            //                     FLYCAPTURE_NEAREST_NEIGHBOR_FAST in FlyCapture.
            // EDGE_SENSING    Weights surrounding pixels based on localized edge orientation.
            // HQ_LINEAR   Well-balanced speed and quality.
            // RIGOROUS    Slowest but produces good results.
            // IPP     Multi-threaded with similar results to edge sensing.
            // DIRECTIONAL_FILTER  Best quality but much faster than rigorous.
            // Colors
            // http://softwareservices.ptgrey.com/Spinnaker/latest/group___camera_defs__h.html#ggabd5af55aaa20bcb0644c46241c2cbad1a33a1c8a1f6dbcb4a4eaaaf6d4d7ff1d1
            // PixelFormat_BGR8

            // // Time tests
            // // DEFAULT
            // const auto reps = 1e2;
            // const auto begin1 = std::chrono::high_resolution_clock::now();
            // for (auto asdf = 0 ; asdf < reps ; asdf++)
            //     const auto imagePtrTemp = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT);
            // const auto durationMs1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //     std::chrono::high_resolution_clock::now()-begin1
            // ).count() * 1e-6;
            // // EDGE_SENSING
            // const auto begin2 = std::chrono::high_resolution_clock::now();
            // for (auto asdf = 0 ; asdf < reps ; asdf++)
            //     const auto imagePtrTemp = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::EDGE_SENSING);
            // const auto durationMs2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //     std::chrono::high_resolution_clock::now()-begin2
            // ).count() * 1e-6;
            // // IPP
            // const auto begin3 = std::chrono::high_resolution_clock::now();
            // for (auto asdf = 0 ; asdf < reps ; asdf++)
            //     const auto imagePtrTemp = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::IPP);
            // const auto durationMs3 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //     std::chrono::high_resolution_clock::now()-begin3
            // ).count() * 1e-6;
            // // Print times
            // opLog("Time (ms) 1: " + std::to_string(durationMs1 / reps), Priority::High);
            // opLog("Time (ms) 2: " + std::to_string(durationMs2 / reps), Priority::High);
            // opLog("Time (ms) 3: " + std::to_string(durationMs3 / reps), Priority::High);

            // Return right one
            // ~ 1.3 ms but pixeled
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT);
            // ~0.5 ms but BW
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::NO_COLOR_PROCESSING);
            // ~6 ms, looks as good as best
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::HQ_LINEAR);
            // ~2.2 ms default << edge << best
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::EDGE_SENSING);
            // ~115, too slow
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::RIGOROUS);
            // ~1.7 ms, slightly worse than HQ_LINEAR
            return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::IPP);
            // ~30 ms, ideally best quality?
            // return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DIRECTIONAL_FILTER);
        }

        /*
         * This function converts between Spinnaker::ImagePtr container to cv::Mat container used in OpenCV.
        */
        cv::Mat spinnakerWrapperToCvMat(const Spinnaker::ImagePtr &imagePtr)
        {
            try
            {
                const auto XPadding = imagePtr->GetXPadding();
                const auto YPadding = imagePtr->GetYPadding();
                const auto rowsize = imagePtr->GetWidth();
                const auto colsize = imagePtr->GetHeight();

                // Image data contains padding. When allocating cv::Mat container size, you need to account for the X,Y
                // image data padding.
                return cv::Mat((int)(colsize + YPadding), (int)(rowsize + XPadding), CV_8UC3, imagePtr->GetData(),
                               imagePtr->GetStride());
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return cv::Mat();
            }
        }

        // This function configures the camera to use a trigger. First, trigger mode is
        // set to off in order to select the trigger source. Once the trigger source
        // has been selected, trigger mode is then enabled, which has the camera
        // capture only a single image upon the execution of the chosen trigger.
        int configureTrigger(Spinnaker::GenApi::INodeMap &iNodeMap)
        {
            try
            {
                int result = 0;
                opLog("*** CONFIGURING TRIGGER ***", Priority::High);
                opLog("Configuring trigger...", Priority::High);
                // opLog("Configuring hardware trigger...", Priority::High);
                // Ensure trigger mode off
                // *** NOTES ***
                // The trigger must be disabled in order to configure whether the source
                // is software or hardware.
                Spinnaker::GenApi::CEnumerationPtr ptrTriggerMode = iNodeMap.GetNode("TriggerMode");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerMode) || !Spinnaker::GenApi::IsReadable(ptrTriggerMode))
                    error("Unable to disable trigger mode (node retrieval). Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);

                Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOff)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOff))
                    error("Unable to disable trigger mode (enum entry retrieval). Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);

                ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

                opLog("Trigger mode disabled...", Priority::High);

                // Select trigger source
                // *** NOTES ***
                // The trigger source must be set to hardware or software while trigger
                // mode is off.
                Spinnaker::GenApi::CEnumerationPtr ptrTriggerSource = iNodeMap.GetNode("TriggerSource");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSource)
                    || !Spinnaker::GenApi::IsWritable(ptrTriggerSource))
                    error("Unable to set trigger mode (node retrieval). Aborting...",
                          __LINE__, __FUNCTION__, __FILE__);

                // // Set trigger mode to hardware ('Line0')
                // Spinnaker::GenApi::CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line0");
                // if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSourceHardware)
                //     || !Spinnaker::GenApi::IsReadable(ptrTriggerSourceHardware))
                //     error("Unable to set trigger mode (enum entry retrieval). Aborting...",
                //               __LINE__, __FUNCTION__, __FILE__);
                // ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());
                // opLog("Trigger source set to hardware...", Priority::High);

                // Set trigger mode to sofware
                Spinnaker::GenApi::CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSourceSoftware)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerSourceSoftware))
                    error("Unable to set trigger mode (enum entry retrieval). Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);
                ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());
                // opLog("Trigger source set to source...", Priority::High);

                // Turn trigger mode on
                // *** LATER ***
                // Once the appropriate trigger source has been set, turn trigger mode
                // on in order to retrieve images using the trigger.
                Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOn)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOn))
                {
                    error("Unable to enable trigger mode (enum entry retrieval). Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);
                    return -1;
                }

                ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

                opLog("Trigger mode turned back on...", Priority::High);

                return result;
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
        }

        int GrabNextImageByTrigger(Spinnaker::GenApi::INodeMap& nodeMap)
        {
            try
            {
                int result = 0;

                // Execute software trigger
                Spinnaker::GenApi::CCommandPtr ptrSoftwareTriggerCommand = nodeMap.GetNode("TriggerSoftware");
                if (!IsAvailable(ptrSoftwareTriggerCommand) || !IsWritable(ptrSoftwareTriggerCommand))
                    error("Unable to enable trigger. Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);

                ptrSoftwareTriggerCommand->Execute();

                return result;
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return -1;
            }
        }
    #else
        const std::string USE_FLIR_CAMERA_ERROR{"OpenPose CMake must be compiled with the `USE_FLIR_CAMERA`"
            " flag in order to use the FLIR camera. Alternatively, disable `--flir_camera`."};
    #endif

    struct SpinnakerWrapper::ImplSpinnakerWrapper
    {
        #ifdef USE_FLIR_CAMERA
            bool mInitialized;
            CameraParameterReader mCameraParameterReader;
            Point<int> mResolution;
            Spinnaker::CameraList mCameraList;
            Spinnaker::SystemPtr mSystemPtr;
            std::vector<cv::Mat> mCvMats;
            std::vector<std::string> mSerialNumbers;
            // Camera index
            const int mCameraIndex;
            // Undistortion
            const bool mUndistortImage;
            std::vector<cv::Mat> mRemoveDistortionMaps1;
            std::vector<cv::Mat> mRemoveDistortionMaps2;
            // Thread
            bool mThreadOpened;
            std::vector<Spinnaker::ImagePtr> mBuffer;
            std::mutex mBufferMutex;
            std::atomic<bool> mCloseThread;
            std::thread mThread;

            ImplSpinnakerWrapper(const bool undistortImage, const int cameraIndex) :
                mInitialized{false},
                mCameraIndex{cameraIndex},
                mUndistortImage{undistortImage}
            {
            }

            void readAndUndistortImage(const int i, const Spinnaker::ImagePtr& imagePtr,
                                       const cv::Mat& cameraIntrinsics = cv::Mat(),
                                       const cv::Mat& cameraDistorsions = cv::Mat())
            {
                try
                {
                    // Original image --> BGR uchar image
                    const auto imagePtrColor = spinnakerImagePtrToColor(imagePtr);
                    // Spinnaker to cv::Mat
                    const auto cvMatDistorted = spinnakerWrapperToCvMat(imagePtrColor);
                    // const auto cvMatDistorted = spinnakerWrapperToCvMat(imagePtr);
                    // Undistort
                    if (mUndistortImage)
                    {
                        // Sanity check
                        if (cameraIntrinsics.empty() || cameraDistorsions.empty())
                            error("Camera intrinsics/distortions were empty.", __LINE__, __FUNCTION__, __FILE__);
                        // // Option a - 80 ms / 3 images
                        // // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                        // cv::undistort(cvMatDistorted, mCvMats[i], cameraIntrinsics, cameraDistorsions);
                        // // In OpenCV 2.4, cv::undistort is exactly equal than cv::initUndistortRectifyMap
                        // (with CV_16SC2) + cv::remap (with LINEAR). I.e., opLog(cv::norm(cvMatMethod1-cvMatMethod2)) = 0.
                        // Option b - 15 ms / 3 images (LINEAR) or 25 ms (CUBIC)
                        // Distorsion removal - not required and more expensive (applied to the whole image instead of
                        // only to our interest points)
                        if (mRemoveDistortionMaps1[i].empty() || mRemoveDistortionMaps2[i].empty())
                        {
                            const auto imageSize = cvMatDistorted.size();
                            cv::initUndistortRectifyMap(cameraIntrinsics,
                                                        cameraDistorsions,
                                                        cv::Mat(),
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
                                                        mRemoveDistortionMaps1[i],
                                                        mRemoveDistortionMaps2[i]);
                        }
                        cv::remap(cvMatDistorted, mCvMats[i],
                                  mRemoveDistortionMaps1[i], mRemoveDistortionMaps2[i],
                                  // cv::INTER_NEAREST);
                                  cv::INTER_LINEAR);
                                  // cv::INTER_CUBIC);
                                  // cv::INTER_LANCZOS4); // Smoother, but we do not need this quality & its >>expensive
                    }
                    // Baseline (do not undistort)
                    else
                        mCvMats[i] = cvMatDistorted.clone();
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void bufferingThread()
            {
                #ifdef USE_FLIR_CAMERA
                    try
                    {
                        mCloseThread = false;
                        // Get cameras - ~0.005 ms (3 cameras)
                        std::vector<Spinnaker::CameraPtr> cameraPtrs(mCameraList.GetSize());
                        for (auto i = 0u; i < cameraPtrs.size(); i++)
                            cameraPtrs.at(i) = mCameraList.GetBySerial(mSerialNumbers.at(i)); // Sorted by Serial Number
                            // cameraPtrs.at(i) = mCameraList.GetByIndex(i); // Sorted by however Spinnaker decided
                        while (!mCloseThread)
                        {
                            // Trigger
                            for (auto i = 0u; i < cameraPtrs.size(); i++)
                            {
                                // Retrieve GenICam nodemap
                                auto& iNodeMap = cameraPtrs[i]->GetNodeMap();
                                Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = iNodeMap.GetNode("AcquisitionMode");
                                const auto result = GrabNextImageByTrigger(iNodeMap);
                                if (result != 0)
                                    error("Error in GrabNextImageByTrigger.", __LINE__, __FUNCTION__, __FILE__);
                            }
                            // Get frame
                            std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                            for (auto i = 0u; i < cameraPtrs.size(); i++)
                                imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                            // Move to buffer
                            bool imagesExtracted = true;
                            for (auto& imagePtr : imagePtrs)
                            {
                                if (imagePtr->IsIncomplete())
                                {
                                    opLog("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus())
                                        + "...", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                                    imagesExtracted = false;
                                    break;
                                }
                            }
                            if (imagesExtracted)
                            {
                                std::unique_lock<std::mutex> lock{mBufferMutex};
                                std::swap(mBuffer, imagePtrs);
                                lock.unlock();
                                std::this_thread::sleep_for(std::chrono::microseconds{1});
                            }
                        }
                    }
                    catch (const std::exception& e)
                    {
                        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    }
                #endif
            }

            // This function acquires and displays images from each device.
            std::vector<Matrix> acquireImages(
                const std::vector<Matrix>& opCameraIntrinsics,
                const std::vector<Matrix>& opCameraDistorsions,
                const int cameraIndex = -1)
            {
                try
                {
                    OP_OP2CVVECTORMAT(cameraIntrinsics, opCameraIntrinsics)
                    OP_OP2CVVECTORMAT(cameraDistorsions, opCameraDistorsions)
                    // std::vector<cv::Mat> cvMats;

                    // Retrieve, convert, and return an image for each camera
                    // In order to work with simultaneous camera streams, nested loops are
                    // needed. It is important that the inner loop be the one iterating
                    // through the cameras; otherwise, all images will be grabbed from a
                    // single camera before grabbing any images from another.

                    // // Get cameras - ~0.005 ms (3 cameras)
                    // std::vector<Spinnaker::CameraPtr> cameraPtrs(cameraList.GetSize());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     cameraPtrs.at(i) = cameraList.GetByIndex(i);

                    // Read raw images - ~0.15 ms (3 cameras)
                    // std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                    std::vector<Spinnaker::ImagePtr> imagePtrs;
                    // Retrieve frame
                    auto cvMatRetrieved = false;
                    while (!cvMatRetrieved)
                    {
                        // Retrieve frame
                        std::unique_lock<std::mutex> lock{mBufferMutex};
                        if (!mBuffer.empty())
                        {
                            std::swap(imagePtrs, mBuffer);
                            cvMatRetrieved = true;
                        }
                        // No frames available -> sleep & wait
                        else
                        {
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::microseconds{5});
                        }
                    }
                    // Getting frames
                    // Retrieve next received image and ensure image completion
                    // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();

                    // All images completed
                    bool imagesExtracted = true;
                    for (auto& imagePtr : imagePtrs)
                    {
                        if (imagePtr->IsIncomplete())
                        {
                            opLog("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus())
                                + "...", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                            imagesExtracted = false;
                            break;
                        }
                    }
                    mCvMats.clear();
                    // Convert to cv::Mat
                    if (imagesExtracted)
                    {
                        // // Original image --> BGR uchar image - ~4 ms (3 cameras)
                        // for (auto& imagePtr : imagePtrs)
                        //     imagePtr = spinnakerImagePtrToColor(imagePtr);

                        // Init anti-distortion matrices first time
                        if (mRemoveDistortionMaps1.empty())
                            mRemoveDistortionMaps1.resize(imagePtrs.size());
                        if (mRemoveDistortionMaps2.empty())
                            mRemoveDistortionMaps2.resize(imagePtrs.size());

                        // Multi-thread undistort (slowest function in the class)
                        //     ~7.7msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + LINEAR)
                        //     ~23.2msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + CUBIC)
                        //     ~35msec (3 cameras + multi-thread + undistort)
                        //     ~59msec (2 cameras + single-thread + undistort)
                        //     ~75msec (3 cameras + single-thread + undistort)
                        mCvMats.resize(imagePtrs.size());
                        // All cameras
                        if (cameraIndex < 0)
                        {
                            // Undistort image
                            if (mUndistortImage)
                            {
                                std::vector<std::thread> threads(imagePtrs.size()-1);
                                for (auto i = 0u; i < threads.size(); i++)
                                {
                                    // Multi-thread option
                                    threads.at(i) = std::thread{&ImplSpinnakerWrapper::readAndUndistortImage, this, i,
                                                                imagePtrs.at(i), cameraIntrinsics.at(i),
                                                                cameraDistorsions.at(i)};
                                    // // Single-thread option
                                    // readAndUndistortImage(i, imagePtrs.at(i), cameraIntrinsics.at(i), cameraDistorsions.at(i));
                                }
                                readAndUndistortImage(imagePtrs.size()-1, imagePtrs.back(), cameraIntrinsics.back(),
                                                      cameraDistorsions.back());
                                // Close threads
                                for (auto& thread : threads)
                                    if (thread.joinable())
                                        thread.join();
                            }
                            // Do not undistort image
                            else
                            {
                                for (auto i = 0u; i < imagePtrs.size(); i++)
                                    readAndUndistortImage(i, imagePtrs.at(i));
                            }
                        }
                        // Only 1 camera
                        else
                        {
                            // Sanity check
                            if ((unsigned int)cameraIndex >= imagePtrs.size())
                                error("There are only " + std::to_string(imagePtrs.size())
                                      + " cameras, but you asked for the "
                                      + std::to_string(cameraIndex+1) +"-th camera (i.e., `--flir_camera_index "
                                      + std::to_string(cameraIndex) +"`), which doesn't exist. Note that the index is"
                                      + " 0-based.", __LINE__, __FUNCTION__, __FILE__);
                            // Undistort image
                            if (mUndistortImage)
                                readAndUndistortImage(cameraIndex, imagePtrs.at(cameraIndex), cameraIntrinsics.at(cameraIndex),
                                                      cameraDistorsions.at(cameraIndex));
                            // Do not undistort image
                            else
                                readAndUndistortImage(cameraIndex, imagePtrs.at(cameraIndex));
                            mCvMats = std::vector<cv::Mat>{mCvMats[cameraIndex]};
                        }
                    }
                    OP_CV2OPVECTORMAT(opMats, mCvMats)
                    return opMats;
                }
                catch (Spinnaker::Exception &e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            }
        #endif
    };

    SpinnakerWrapper::SpinnakerWrapper(const std::string& cameraParameterPath, const Point<int>& resolution,
                                       const bool undistortImage, const int cameraIndex)
        #ifdef USE_FLIR_CAMERA
            : upImpl{new ImplSpinnakerWrapper{undistortImage, cameraIndex}}
        #endif
    {
        #ifdef USE_FLIR_CAMERA
            try
            {
                // Clean previous unclosed builds (e.g., if core dumped in the previous code using the cameras)
                release();

                upImpl->mInitialized = true;

                // Print application build information
                opLog(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, Priority::High);

                // Retrieve singleton reference to upImpl->mSystemPtr object
                upImpl->mSystemPtr = Spinnaker::System::GetInstance();

                // Retrieve list of cameras from the upImpl->mSystemPtr
                upImpl->mCameraList = upImpl->mSystemPtr->GetCameras();

                const unsigned int numCameras = upImpl->mCameraList.GetSize();

                opLog("Number of cameras detected: " + std::to_string(numCameras), Priority::High);

                // Finish if there are no cameras
                if (numCameras == 0)
                {
                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    // opLog("Not enough cameras!\nPress Enter to exit...", Priority::High);
                    // getchar();

                    error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
                }
                opLog("Camera system initialized...", Priority::High);

                //
                // Retrieve transport layer nodemaps and print device information for
                // each camera
                //
                // *** NOTES ***
                // This example retrieves information from the transport layer nodemap
                // twice: once to print device information and once to grab the device
                // serial number. Rather than caching the nodemap, each nodemap is
                // retrieved both times as needed.
                //
                opLog("\n*** DEVICE INFORMATION ***\n", Priority::High);

                for (auto i = 0u; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    // Retrieve TL device nodemap
                    auto& iNodeMapTLDevice = cameraPtr->GetTLDeviceNodeMap();

                    // Print device information
                    auto result = printDeviceInfo(iNodeMapTLDevice, i);
                    if (result < 0)
                        error("Result > 0, error " + std::to_string(result) + " occurred...",
                                  __LINE__, __FUNCTION__, __FILE__);
                }

                for (auto i = 0u; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    // Initialize each camera
                    // You may notice that the steps in this function have more loops with
                    // less steps per loop; this contrasts the acquireImages() function
                    // which has less loops but more steps per loop. This is done for
                    // demonstrative purposes as both work equally well.
                    // Later: Each camera needs to be deinitialized once all images have been
                    // acquired.
                    cameraPtr->Init();

                    // Retrieve GenICam nodemap
                    auto& iNodeMap = cameraPtr->GetNodeMap();

                    // Configure trigger
                    int result = configureTrigger(iNodeMap);
                    if (result < 0)
                        error("Result > 0, error " + std::to_string(result) + " occurred...",
                                  __LINE__, __FUNCTION__, __FILE__);

                    // // Configure chunk data
                    // result = configureChunkData(iNodeMap);
                    // if (result < 0)
                    //     return result;

                    // // Remove buffer --> Always get newest frame
                    // Spinnaker::GenApi::INodeMap& snodeMap = cameraPtr->GetTLStreamNodeMap();
                    // Spinnaker::GenApi::CEnumerationPtr ptrBufferHandlingMode = snodeMap.GetNode(
                    //     "StreamBufferHandlingMode");
                    // if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingMode)
                    //     || !Spinnaker::GenApi::IsWritable(ptrBufferHandlingMode))
                    //     error("Unable to change buffer handling mode", __LINE__, __FUNCTION__, __FILE__);

                    // Spinnaker::GenApi::CEnumEntryPtr ptrBufferHandlingModeNewest = ptrBufferHandlingMode->GetEntryByName(
                    //     "NewestFirstOverwrite");
                    // if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingModeNewest)
                    //     || !IsReadable(ptrBufferHandlingModeNewest))
                    //     error("Unable to set buffer handling mode to newest (entry 'NewestFirstOverwrite' retrieval)."
                    //               " Aborting...", __LINE__, __FUNCTION__, __FILE__);
                    // int64_t bufferHandlingModeNewest = ptrBufferHandlingModeNewest->GetValue();

                    // ptrBufferHandlingMode->SetIntValue(bufferHandlingModeNewest);
                }

                // Prepare each camera to acquire images
                //
                // *** NOTES ***
                // For pseudo-simultaneous streaming, each camera is prepared as if it
                // were just one, but in a loop. Notice that cameras are selected with
                // an index. We demonstrate pseduo-simultaneous streaming because true
                // simultaneous streaming would require multiple process or threads,
                // which is too complex for an example.
                //
                // Serial numbers are the only persistent objects we gather in this
                // example, which is why a std::vector is created.
                for (auto i = 0u; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    // Set acquisition mode to continuous
                    Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = cameraPtr->GetNodeMap().GetNode(
                        "AcquisitionMode");
                    if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionMode)
                        || !Spinnaker::GenApi::IsWritable(ptrAcquisitionMode))
                        error("Unable to set acquisition mode to continuous (node retrieval; camera "
                              + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

                    Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName(
                        "Continuous");
                    if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionModeContinuous)
                        || !Spinnaker::GenApi::IsReadable(ptrAcquisitionModeContinuous))
                        error("Unable to set acquisition mode to continuous (entry 'continuous' retrieval "
                                  + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

                    const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

                    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

                    opLog("Camera " + std::to_string(i) + " acquisition mode set to continuous...", Priority::High);

                    // Set camera resolution
                    // Retrieve GenICam nodemap
                    auto& iNodeMap = cameraPtr->GetNodeMap();
                    // Set offset
                    Spinnaker::GenApi::CIntegerPtr ptrOffsetX = iNodeMap.GetNode("OffsetX");
                    ptrOffsetX->SetValue(0);
                    Spinnaker::GenApi::CIntegerPtr ptrOffsetY = iNodeMap.GetNode("OffsetY");
                    ptrOffsetY->SetValue(0);
                    // Set resolution
                    if (resolution.x > 0 && resolution.y > 0)
                    {
                        // WARNING: setting the obset would require auto-change in the camera matrix parameters
                        // Set offset
                        // const Spinnaker::GenApi::CIntegerPtr ptrWidthMax = iNodeMap.GetNode("WidthMax");
                        // const Spinnaker::GenApi::CIntegerPtr ptrHeightMax = iNodeMap.GetNode("HeightMax");
                        // ptrOffsetX->SetValue((ptrWidthMax->GetValue() - resolution.x) / 2);
                        // ptrOffsetY->SetValue((ptrHeightMax->GetValue() - resolution.y) / 2);
                        // Set Width
                        Spinnaker::GenApi::CIntegerPtr ptrWidth = iNodeMap.GetNode("Width");
                        ptrWidth->SetValue(resolution.x);
                        // Set width
                        Spinnaker::GenApi::CIntegerPtr ptrHeight = iNodeMap.GetNode("Height");
                        ptrHeight->SetValue(resolution.y);
                    }
                    else
                    {
                        const Spinnaker::GenApi::CIntegerPtr ptrWidthMax = iNodeMap.GetNode("WidthMax");
                        const Spinnaker::GenApi::CIntegerPtr ptrHeightMax = iNodeMap.GetNode("HeightMax");
                        // Set Width
                        Spinnaker::GenApi::CIntegerPtr ptrWidth = iNodeMap.GetNode("Width");
                        ptrWidth->SetValue(ptrWidthMax->GetValue());
                        // Set width
                        Spinnaker::GenApi::CIntegerPtr ptrHeight = iNodeMap.GetNode("Height");
                        ptrHeight->SetValue(ptrHeightMax->GetValue());
                        opLog("Choosing maximum resolution for flir camera (" + std::to_string(ptrWidth->GetValue())
                            + " x " + std::to_string(ptrHeight->GetValue()) + ").", Priority::High);
                    }

                    // Begin acquiring images
                    cameraPtr->BeginAcquisition();

                    opLog("Camera " + std::to_string(i) + " started acquiring images...", Priority::High);
                }

                // Retrieve device serial number for filename
                opLog("\nReading (and sorting by) serial numbers...", Priority::High);
                const bool sorted = true;
                upImpl->mSerialNumbers = getSerialNumbers(upImpl->mCameraList, sorted);
                const auto& serialNumbers = upImpl->mSerialNumbers;
                for (auto i = 0u; i < serialNumbers.size(); i++)
                    opLog("Camera " + std::to_string(i) + " serial number set to "
                        + serialNumbers[i] + "...", Priority::High);
                if (upImpl->mCameraIndex >= 0)
                    opLog("Only using camera index " + std::to_string(upImpl->mCameraIndex) + ", i.e., serial number "
                        + serialNumbers[upImpl->mCameraIndex] + "...", Priority::High);

                // Read camera parameters from SN
                if (upImpl->mUndistortImage)
                {
                    // If all images required
                    if (upImpl->mCameraIndex < 0)
                        upImpl->mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);
                    // If only one required
                    else
                    {
                        upImpl->mCameraParameterReader.readParameters(
                            cameraParameterPath,
                            std::vector<std::string>(serialNumbers.size(), serialNumbers.at(upImpl->mCameraIndex)));
                    }
                }

                // Start buffering thread
                upImpl->mThreadOpened = true;
                upImpl->mThread = std::thread{&SpinnakerWrapper::ImplSpinnakerWrapper::bufferingThread, this->upImpl};

                // Get resolution
                const auto cvMats = getRawFrames();
                // Sanity check
                if (cvMats.empty())
                    error("Cameras could not be opened.", __LINE__, __FUNCTION__, __FILE__);
                // Get resolution
                upImpl->mResolution = Point<int>{cvMats[0].cols(), cvMats[0].rows()};

                const std::string numberCameras = std::to_string(upImpl->mCameraIndex < 0 ? serialNumbers.size() : 1);
                opLog("\nRunning for " + numberCameras + " out of " + std::to_string(serialNumbers.size())
                    + " camera(s)...\n\n*** IMAGE ACQUISITION ***\n", Priority::High);
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            UNUSED(cameraParameterPath);
            UNUSED(resolution);
            UNUSED(undistortImage);
            UNUSED(cameraIndex);
            error(USE_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    SpinnakerWrapper::~SpinnakerWrapper()
    {
        try
        {
            release();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getRawFrames()
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                try
                {
                    // Sanity check
                    if (upImpl->mUndistortImage &&
                        (unsigned long long) upImpl->mCameraList.GetSize()
                            != upImpl->mCameraParameterReader.getNumberCameras())
                        error("The number of cameras must be the same as the INTRINSICS vector size.",
                          __LINE__, __FUNCTION__, __FILE__);
                    // Return frames
                    return upImpl->acquireImages(upImpl->mCameraParameterReader.getCameraIntrinsics(),
                                                 upImpl->mCameraParameterReader.getCameraDistortions(),
                                                 upImpl->mCameraIndex);
                }
                catch (const Spinnaker::Exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            #else
                error(USE_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraMatrices() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraMatrices();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraExtrinsics() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraExtrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraIntrinsics() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraIntrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    Point<int> SpinnakerWrapper::getResolution() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mResolution;
            #else
                return Point<int>{};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<int>{};
        }
    }

    bool SpinnakerWrapper::isOpened() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mInitialized;
            #else
                return false;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void SpinnakerWrapper::release()
    {
        #ifdef USE_FLIR_CAMERA
            try
            {
                if (upImpl->mInitialized)
                {
                    // Stop thread
                    // Close and join thread
                    if (upImpl->mThreadOpened)
                    {
                        upImpl->mCloseThread = true;
                        upImpl->mThread.join();
                    }

                    // End acquisition for each camera
                    // Notice that what is usually a one-step process is now two steps
                    // because of the additional step of selecting the camera. It is worth
                    // repeating that camera selection needs to be done once per loop.
                    // It is possible to interact with cameras through the camera list with
                    // GetByIndex(); this is an alternative to retrieving cameras as
                    // Spinnaker::CameraPtr objects that can be quick and easy for small tasks.
                    //
                    for (auto i = 0u; i < upImpl->mCameraList.GetSize(); i++)
                    {
                        // Select camera
                        auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                        cameraPtr->EndAcquisition();

                        // Retrieve GenICam nodemap
                        auto& iNodeMap = cameraPtr->GetNodeMap();

                        // // Disable chunk data
                        // result = disableChunkData(iNodeMap);
                        // // if (result < 0)
                        // //     return result;

                        // Reset trigger
                        auto result = resetTrigger(iNodeMap);
                        if (result < 0)
                            error("Error happened..." + std::to_string(result), __LINE__, __FUNCTION__, __FILE__);

                        // Deinitialize each camera
                        // Each camera must be deinitialized separately by first
                        // selecting the camera and then deinitializing it.
                        cameraPtr->DeInit();
                    }

                    opLog("FLIR (Point-grey) capture completed. Releasing cameras...", Priority::High);

                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    // Setting the class as released
                    upImpl->mInitialized = false;

                    opLog("Cameras released! Exiting program.", Priority::High);
                }
                else
                {
                    // Open general system
                    auto systemPtr = Spinnaker::System::GetInstance();
                    auto cameraList = systemPtr->GetCameras();
                    if (cameraList.GetSize() > 0)
                    {

                        for (auto i = 0u; i < cameraList.GetSize(); i++)
                        {
                            // Select camera
                            auto cameraPtr = cameraList.GetByIndex(i);
                            // Begin
                            cameraPtr->Init();
                            cameraPtr->BeginAcquisition();
                            // End
                            cameraPtr->EndAcquisition();
                            cameraPtr->DeInit();
                        }
                    }
                    cameraList.Clear();
                    systemPtr->ReleaseInstance();
                }
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
    }
}
