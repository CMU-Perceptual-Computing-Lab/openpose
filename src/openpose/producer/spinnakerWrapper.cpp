#ifdef WITH_FLIR_CAMERA
    #include <thread>
#endif
#include <opencv2/imgproc/imgproc.hpp> // cv::undistort
#ifdef WITH_FLIR_CAMERA
    #include <Spinnaker.h>
#endif
#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/producer/spinnakerWrapper.hpp>

namespace op
{
    #ifdef WITH_FLIR_CAMERA
        // This function prints the device information of the camera from the transport
        // layer; please see NodeMapInfo example for more in-depth comments on printing
        // device information from the nodemap.
        int printDeviceInfo(Spinnaker::GenApi::INodeMap &iNodeMap, const unsigned int camNum)
        {
            int result = 0;

            log("Printing device information for camera " + std::to_string(camNum) + "...\n", Priority::High);

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
                    log(pfeatureNode->GetName() + " : " +
                            (IsReadable(cValuePtr) ? cValuePtr->ToString() : "Node not readable"), Priority::High);
                }
            }
            else
                log("Device control information not available.", Priority::High);
            log(" ", Priority::High);

            return result;
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

                // log("Trigger mode disabled...", Priority::High);

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
            // Time tests
            // const auto reps = 1e3;
            // const auto begin = std::chrono::high_resolution_clock::now();
            // for (auto asdf = 0 ; asdf < reps ; asdf++){
            // ~ 1.5 ms but pixeled
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT);
            return imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT);
            // ~0.5 ms but BW
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::NO_COLOR_PROCESSING);
            // ~6 ms, looks as good as best
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::HQ_LINEAR);
            // ~2 ms default << edge << best
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::EDGE_SENSING);
            // ~115, too slow
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::RIGOROUS);
            // ~2 ms, slightly worse than HQ_LINEAR
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::IPP);
            // ~30 ms, ideally best quality?
            // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DIRECTIONAL_FILTER);
            // imagePtr = imagePtr;
            // }
            // durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //     std::chrono::high_resolution_clock::now()-begin
            // ).count() * 1e-6;
            // log("Time conversion (ms): " + std::to_string(durationMs / reps), Priority::High);
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
                log("*** CONFIGURING TRIGGER ***", Priority::High);
                log("Configuring hardware trigger...", Priority::High);
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

                log("Trigger mode disabled...", Priority::High);

                // Select trigger source
                // *** NOTES ***
                // The trigger source must be set to hardware or software while trigger
                // mode is off.
                Spinnaker::GenApi::CEnumerationPtr ptrTriggerSource = iNodeMap.GetNode("TriggerSource");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSource)
                    || !Spinnaker::GenApi::IsWritable(ptrTriggerSource))
                    error("Unable to set trigger mode (node retrieval). Aborting...",
                          __LINE__, __FUNCTION__, __FILE__);

                // Set trigger mode to hardware ('Line0')
                Spinnaker::GenApi::CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line0");
                if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSourceHardware)
                    || !Spinnaker::GenApi::IsReadable(ptrTriggerSourceHardware))
                    error("Unable to set trigger mode (enum entry retrieval). Aborting...",
                              __LINE__, __FUNCTION__, __FILE__);

                ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());

                log("Trigger source set to hardware...", Priority::High);

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

                log("Trigger mode turned back on...", Priority::High);

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
        const std::string WITH_FLIR_CAMERA_ERROR{"OpenPose CMake must be compiled with the `WITH_FLIR_CAMERA`"
            " flag in order to use the FLIR camera. Alternatively, disable `--flir_camera`."};
    #endif

    struct SpinnakerWrapper::ImplSpinnakerWrapper
    {
        #ifdef WITH_FLIR_CAMERA
            bool mInitialized;
            CameraParameterReader mCameraParameterReader;
            Point<int> mResolution;
            Spinnaker::CameraList mCameraList;
            Spinnaker::SystemPtr mSystemPtr;
            std::vector<cv::Mat> mCvMats;

            ImplSpinnakerWrapper() :
                mInitialized{false}
            {
            }

            void undistortImage(const int i, const Spinnaker::ImagePtr& imagePtr,
                                const cv::Mat& cameraIntrinsics, const cv::Mat& cameraDistorsions)
            {
                // Original image --> BGR uchar image
                const auto imagePtrColor = spinnakerImagePtrToColor(imagePtr);
                // Spinnaker to cv::Mat
                const auto cvMatDistorted = spinnakerWrapperToCvMat(imagePtrColor);
                // const auto cvMatDistorted = spinnakerWrapperToCvMat(imagePtr);
                // Baseline
                // mCvMats[i] = cvMatDistorted.clone();
                // Undistort
                // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                cv::undistort(cvMatDistorted, mCvMats[i], cameraIntrinsics, cameraDistorsions);
            }

            // This function acquires and displays images from each device.
            std::vector<cv::Mat> acquireImages(Spinnaker::CameraList &cameraList,
                                               const std::vector<cv::Mat>& cameraIntrinsics,
                                               const std::vector<cv::Mat>& cameraDistorsions)
            {
                try
                {
                    // std::vector<cv::Mat> cvMats;

                    // Retrieve, convert, and return an image for each camera
                    // In order to work with simultaneous camera streams, nested loops are
                    // needed. It is important that the inner loop be the one iterating
                    // through the cameras; otherwise, all images will be grabbed from a
                    // single camera before grabbing any images from another.

                    // Get cameras - ~0.005 ms (3 cameras)
                    std::vector<Spinnaker::CameraPtr> cameraPtrs(cameraList.GetSize());
                    for (auto i = 0u; i < cameraPtrs.size(); i++)
                        cameraPtrs.at(i) = cameraList.GetByIndex(i);

                    // Read raw images - ~0.15 ms (3 cameras)
                    std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                    for (auto i = 0u; i < cameraPtrs.size(); i++)
                        imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                    // Commented code was supposed to clean buffer, but `NewestFirstOverwrite` does that
                    // Getting frames
                    // Retrieve next received image and ensure image completion
                    // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();
                    // Clean buffer + retrieve next received image + ensure image completion
                    // auto durationMs = 0.;
                    // // for (auto counter = 0 ; counter < 10 ; counter++)
                    // while (durationMs < 1.)
                    // {
                    //     const auto begin = std::chrono::high_resolution_clock::now();
                    //     for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //         imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                    //     durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    //         std::chrono::high_resolution_clock::now()-begin
                    //     ).count() * 1e-6;
                    //     // log("Time extraction (ms): " + std::to_string(durationMs), Priority::High);
                    // }

                    // All images completed
                    bool imagesExtracted = true;
                    for (auto& imagePtr : imagePtrs)
                    {
                        if (imagePtr->IsIncomplete())
                        {
                            log("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus())
                                + "...", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                            imagesExtracted = false;
                            break;
                        }
                    }
                    // Convert to cv::Mat
                    if (imagesExtracted)
                    {
                        // // Original image --> BGR uchar image - ~4 ms (3 cameras)
                        // for (auto& imagePtr : imagePtrs)
                        //     imagePtr = spinnakerImagePtrToColor(imagePtr);

                        // Multi-thread undistort (slowest function in the class)
                        //     ~35msec (3 cameras + multi-thread)
                        //     ~59msec (2 cameras + single-thread)
                        //     ~75msec (3 cameras + single-thread)
                        std::vector<std::thread> threads(imagePtrs.size()-1);
                        mCvMats.clear();
                        mCvMats.resize(imagePtrs.size());
                        for (auto i = 0u; i < imagePtrs.size()-1; i++)
                            // Multi-thread option
                            threads.at(i) = std::thread{&ImplSpinnakerWrapper::undistortImage, this, i, imagePtrs.at(i),
                                                        cameraIntrinsics.at(i), cameraDistorsions.at(i)};
                            // // Single-thread option
                            // undistortImage(i, imagePtrs.at(i), cameraIntrinsics.at(i), cameraDistorsions.at(i));
                        undistortImage(imagePtrs.size()-1, imagePtrs.back(), cameraIntrinsics.back(),
                                       cameraDistorsions.back());
                        // Close threads
                        for (auto& thread : threads)
                            if (thread.joinable())
                                thread.join();
                    }
                    return mCvMats;
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

    SpinnakerWrapper::SpinnakerWrapper(const std::string& cameraParameterPath, const Point<int>& resolution)
        #ifdef WITH_FLIR_CAMERA
            : upImpl{new ImplSpinnakerWrapper{}}
        #endif
    {
        #ifdef WITH_FLIR_CAMERA
            try
            {
                // Clean previous unclosed builds (e.g. if core dumped in the previous code using the cameras)
                release();

                upImpl->mInitialized = true;

                // Print application build information
                log(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, Priority::High);

                // Retrieve singleton reference to upImpl->mSystemPtr object
                upImpl->mSystemPtr = Spinnaker::System::GetInstance();

                // Retrieve list of cameras from the upImpl->mSystemPtr
                upImpl->mCameraList = upImpl->mSystemPtr->GetCameras();

                unsigned int numCameras = upImpl->mCameraList.GetSize();

                log("Number of cameras detected: " + std::to_string(numCameras), Priority::High);

                // Finish if there are no cameras
                if (numCameras == 0)
                {
                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    log("Not enough cameras!\nPress Enter to exit...", Priority::High);
                    getchar();

                    error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
                }
                log("Camera system initialized...", Priority::High);

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
                log("\n*** DEVICE INFORMATION ***\n", Priority::High);

                for (int i = 0; i < upImpl->mCameraList.GetSize(); i++)
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

                for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
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

                    // // Retrieve GenICam nodemap
                    // auto& iNodeMap = cameraPtr->GetNodeMap();

                    // // Configure trigger
                    // int result = configureTrigger(iNodeMap);
                    // if (result < 0)
                    //     error("Result > 0, error " + std::to_string(result) + " occurred...",
                    //               __LINE__, __FUNCTION__, __FILE__);

                    // // Configure chunk data
                    // result = configureChunkData(iNodeMap);
                    // if (result < 0)
                    //     return result;

                    // Remove buffer --> Always get newest frame
                    Spinnaker::GenApi::INodeMap& snodeMap = cameraPtr->GetTLStreamNodeMap();
                    Spinnaker::GenApi::CEnumerationPtr ptrBufferHandlingMode = snodeMap.GetNode(
                        "StreamBufferHandlingMode");
                    if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingMode)
                        || !Spinnaker::GenApi::IsWritable(ptrBufferHandlingMode))
                        error("Unable to change buffer handling mode", __LINE__, __FUNCTION__, __FILE__);

                    Spinnaker::GenApi::CEnumEntryPtr ptrBufferHandlingModeNewest = ptrBufferHandlingMode->GetEntryByName(
                        "NewestFirstOverwrite");
                    if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingModeNewest)
                        || !IsReadable(ptrBufferHandlingModeNewest))
                        error("Unable to set buffer handling mode to newest (entry 'NewestFirstOverwrite' retrieval)."
                                  " Aborting...", __LINE__, __FUNCTION__, __FILE__);
                    int64_t bufferHandlingModeNewest = ptrBufferHandlingModeNewest->GetValue();

                    ptrBufferHandlingMode->SetIntValue(bufferHandlingModeNewest);
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
                std::vector<Spinnaker::GenICam::gcstring> strSerialNumbers(upImpl->mCameraList.GetSize());
                for (auto i = 0u; i < strSerialNumbers.size(); i++)
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

                    int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

                    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

                    log("Camera " + std::to_string(i) + " acquisition mode set to continuous...", Priority::High);

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
                        log("Choosing maximum resolution for flir camera (" + std::to_string(ptrWidth->GetValue())
                            + " x " + std::to_string(ptrHeight->GetValue()) + ").", Priority::High);
                    }

                    // Begin acquiring images
                    cameraPtr->BeginAcquisition();

                    log("Camera " + std::to_string(i) + " started acquiring images...", Priority::High);

                    // Retrieve device serial number for filename
                    strSerialNumbers[i] = "";

                    Spinnaker::GenApi::CStringPtr ptrStringSerial = cameraPtr->GetTLDeviceNodeMap().GetNode(
                        "DeviceSerialNumber"
                    );

                    if (Spinnaker::GenApi::IsAvailable(ptrStringSerial)
                        && Spinnaker::GenApi::IsReadable(ptrStringSerial))
                    {
                        strSerialNumbers[i] = ptrStringSerial->GetValue();
                        log("Camera " + std::to_string(i) + " serial number set to "
                                + strSerialNumbers[i].c_str() + "...", Priority::High);
                    }
                    log(" ", Priority::High);
                }

                // Read camera parameters from SN
                std::vector<std::string> serialNumbers(strSerialNumbers.size());
                for (auto i = 0u ; i < serialNumbers.size() ; i++)
                    serialNumbers[i] = strSerialNumbers[i];
                upImpl->mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);

                // Get resolution + security checks
                const auto cvMats = getRawFrames();
                // Security checks
                if (cvMats.empty())
                    error("Cameras could not be opened.", __LINE__, __FUNCTION__, __FILE__);
                // Get resolution
                else
                    upImpl->mResolution = Point<int>{cvMats[0].cols, cvMats[0].rows};

                log("\nRunning for all cameras...\n\n*** IMAGE ACQUISITION ***\n", Priority::High);

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
            error(WITH_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
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
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<cv::Mat> SpinnakerWrapper::getRawFrames()
    {
        try
        {
            #ifdef WITH_FLIR_CAMERA
                try
                {
                    // Security checks
                    if ((unsigned long long) upImpl->mCameraList.GetSize()
                            != upImpl->mCameraParameterReader.getNumberCameras())
                        error("The number of cameras must be the same as the INTRINSICS vector size.",
                          __LINE__, __FUNCTION__, __FILE__);
                    return upImpl->acquireImages(upImpl->mCameraList,
                                                 upImpl->mCameraParameterReader.getCameraIntrinsics(),
                                                 upImpl->mCameraParameterReader.getCameraDistortions());
                }
                catch (const Spinnaker::Exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            #else
                error(WITH_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<cv::Mat> SpinnakerWrapper::getCameraMatrices() const
    {
        try
        {
            #ifdef WITH_FLIR_CAMERA
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

    Point<int> SpinnakerWrapper::getResolution() const
    {
        try
        {
            #ifdef WITH_FLIR_CAMERA
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
            #ifdef WITH_FLIR_CAMERA
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
        #ifdef WITH_FLIR_CAMERA
            try
            {
                if (upImpl->mInitialized)
                {
                    // End acquisition for each camera
                    // Notice that what is usually a one-step process is now two steps
                    // because of the additional step of selecting the camera. It is worth
                    // repeating that camera selection needs to be done once per loop.
                    // It is possible to interact with cameras through the camera list with
                    // GetByIndex(); this is an alternative to retrieving cameras as
                    // Spinnaker::CameraPtr objects that can be quick and easy for small tasks.
                    //
                    for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
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

                    log("FLIR (Point-grey) capture completed. Releasing cameras...", Priority::High);

                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    // Setting the class as released
                    upImpl->mInitialized = false;

                    log("Cameras released! Exiting program.", Priority::High);
                }
                else
                {
                    // Open general system
                    auto systemPtr = Spinnaker::System::GetInstance();
                    auto cameraList = systemPtr->GetCameras();
                    if (cameraList.GetSize() > 0)
                    {

                        for (int i = 0; i < cameraList.GetSize(); i++)
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
