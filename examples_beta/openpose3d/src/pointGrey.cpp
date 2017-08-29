#include <chrono>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <openpose3d/cameraParameters.hpp>
#include <openpose3d/pointGrey.hpp>

/*
 * This function converts between Spinnaker::ImagePtr container to cv::Mat container used in OpenCV.
*/
cv::Mat pointGreyToCvMat(const Spinnaker::ImagePtr &imagePtr)
{
	try
	{
		const auto XPadding = imagePtr->GetXPadding();
		const auto YPadding = imagePtr->GetYPadding();
		const auto rowsize = imagePtr->GetWidth();
		const auto colsize = imagePtr->GetHeight();

		// image data contains padding. When allocating cv::Mat container size, you need to account for the X,Y image data padding.
		return cv::Mat((int)(colsize + YPadding), (int)(rowsize + XPadding), CV_8UC3, imagePtr->GetData(), imagePtr->GetStride());
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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
		op::log("*** CONFIGURING TRIGGER ***", op::Priority::High);
		op::log("Configuring hardware trigger...", op::Priority::High);
        // Ensure trigger mode off
        // *** NOTES ***
        // The trigger must be disabled in order to configure whether the source
        // is software or hardware.
        Spinnaker::GenApi::CEnumerationPtr ptrTriggerMode = iNodeMap.GetNode("TriggerMode");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerMode) || !Spinnaker::GenApi::IsReadable(ptrTriggerMode))
			op::error("Unable to disable trigger mode (node retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);

        Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOff) || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOff))
			op::error("Unable to disable trigger mode (enum entry retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

		op::log("Trigger mode disabled...", op::Priority::High);

        // Select trigger source
        // *** NOTES ***
        // The trigger source must be set to hardware or software while trigger
        // mode is off.
        Spinnaker::GenApi::CEnumerationPtr ptrTriggerSource = iNodeMap.GetNode("TriggerSource");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSource) || !Spinnaker::GenApi::IsWritable(ptrTriggerSource))
			op::error("Unable to set trigger mode (node retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);

        // Set trigger mode to hardware ('Line0')
        Spinnaker::GenApi::CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line0");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerSourceHardware) || !Spinnaker::GenApi::IsReadable(ptrTriggerSourceHardware))
			op::error("Unable to set trigger mode (enum entry retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);

        ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());

		op::log("Trigger source set to hardware...", op::Priority::High);

        // Turn trigger mode on
        // *** LATER ***
        // Once the appropriate trigger source has been set, turn trigger mode
        // on in order to retrieve images using the trigger.
        Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOn) || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOn))
        {
			op::error("Unable to enable trigger mode (enum entry retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

		op::log("Trigger mode turned back on...", op::Priority::High);

		return result;
    }
	catch (const Spinnaker::Exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return -1;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerMode) || !Spinnaker::GenApi::IsReadable(ptrTriggerMode))
			op::error("Unable to disable trigger mode (node retrieval). Non-fatal error...", __LINE__, __FUNCTION__, __FILE__);

        Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOff) || !Spinnaker::GenApi::IsReadable(ptrTriggerModeOff))
			op::error("Unable to disable trigger mode (enum entry retrieval). Non-fatal error...", __LINE__, __FUNCTION__, __FILE__);

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

		// op::log("Trigger mode disabled...", op::Priority::High);

		return result;
    }
    catch (Spinnaker::Exception &e)
    {
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return -1;
	}
}

// This function acquires and displays images from each device.
std::vector<cv::Mat> acquireImages(Spinnaker::CameraList &cameraList)
{
    try
    {
        // Security checks
        if (cameraList.GetSize() != INTRINSICS.size())
            op::checkE(cameraList.GetSize(), INTRINSICS.size(), "The number of cameras must be the same as the INTRINSICS vector size.", __LINE__, __FUNCTION__, __FILE__);

		std::vector<cv::Mat> cvMats;

        // Retrieve, convert, and return an image for each camera
        // In order to work with simultaneous camera streams, nested loops are
        // needed. It is important that the inner loop be the one iterating
        // through the cameras; otherwise, all images will be grabbed from a
        // single camera before grabbing any images from another.

        // Get cameras
        std::vector<Spinnaker::CameraPtr> cameraPtrs(cameraList.GetSize());
        for (auto i = 0u; i < cameraPtrs.size(); i++)
            cameraPtrs.at(i) = cameraList.GetByIndex(i);

        std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());

        // Getting frames
        // Retrieve next received image and ensure image completion
        // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();
        // Clean buffer + retrieve next received image + ensure image completion
        auto durationMs = 0.;
        // for (auto counter = 0 ; counter < 10 ; counter++)
        while (durationMs < 1.)
        {
            const auto begin = std::chrono::high_resolution_clock::now();
            for (auto i = 0u; i < cameraPtrs.size(); i++)
                imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
            durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-begin).count() * 1e-6;
			// op::log("Time extraction (ms): " + std::to_string(durationMs), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }

        // Original format -> RGB8
        bool imagesExtracted = true;
        for (auto& imagePtr : imagePtrs)
        {
            if (imagePtr->IsIncomplete())
            {
                op::log("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus()) + "...",
					    op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
                imagesExtracted = false;
                break;
            }
            else
            {
                // Print image information
                // Convert image to RGB
                // Interpolation methods
                // http://softwareservices.ptgrey.com/Spinnaker/latest/group___spinnaker_defs.html
                // DEFAULT     Default method.
                // NO_COLOR_PROCESSING     No color processing.
                // NEAREST_NEIGHBOR    Fastest but lowest quality. Equivalent to FLYCAPTURE_NEAREST_NEIGHBOR_FAST in FlyCapture.
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
                // // const auto reps = 1e2; // for RIGOROUS & DIRECTIONAL_FILTER
                // const auto begin = std::chrono::high_resolution_clock::now();
                // for (auto asdf = 0 ; asdf < reps ; asdf++){
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT); // ~ 1.5 ms but pixeled
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::NO_COLOR_PROCESSING); // ~0.5 ms but BW
                imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::HQ_LINEAR); // ~6 ms, looks as good as best
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::EDGE_SENSING); // ~2 ms default << edge << best
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::RIGOROUS); // ~115, too slow
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::IPP); // ~2 ms, slightly worse than HQ_LINEAR
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DIRECTIONAL_FILTER); // ~30 ms, ideally best quality?
                // imagePtr = imagePtr;
                // }
                // durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-begin).count() * 1e-6;
				// op::log("Time conversion (ms): " + std::to_string(durationMs / reps), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Convert to cv::Mat
        if (imagesExtracted)
        {
            for (auto i = 0u; i < imagePtrs.size(); i++)
            {
                // Baseline
                // cvMats.emplace_back(pointGreyToCvMat(imagePtrs.at(i)).clone());
                // Undistort
                // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                auto auxCvMat = pointGreyToCvMat(imagePtrs.at(i));
                cvMats.emplace_back();
                cv::undistort(auxCvMat, cvMats[i], INTRINSICS[i], DISTORTIONS[i]);
            }
        }

		return cvMats;
    }
    catch (Spinnaker::Exception &e)
    {
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return {};
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return {};
	}
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int printDeviceInfo(Spinnaker::GenApi::INodeMap &iNodeMap, const unsigned int camNum)
{
    int result = 0;

	op::log("Printing device information for camera " + std::to_string(camNum) + "...\n", op::Priority::High);

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
			op::log(pfeatureNode->GetName() + " : " + (IsReadable(cValuePtr) ? cValuePtr->ToString() : "Node not readable"), op::Priority::High);
        }
    }
    else
		op::log("Device control information not available.", op::Priority::High);
	op::log(" ", op::Priority::High);

    return result;
}

WPointGrey::WPointGrey() :
    initialized{false}
{
    try
    {
        // Security checks
        if (INTRINSICS.size() != DISTORTIONS.size())
            op::checkE(INTRINSICS.size(), DISTORTIONS.size(), "The INTRINSICS and DISTORTIONS vector should have the same size.", __LINE__, __FUNCTION__, __FILE__);
        if (INTRINSICS.size() != M_EACH_CAMERA.size())
            op::checkE(INTRINSICS.size(), M_EACH_CAMERA.size(), "The INTRINSICS and M_EACH_CAMERA vector should have the same size.", __LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

WPointGrey::~WPointGrey()
{
    try
    {
        if (initialized)
        {
            // End acquisition for each camera
            // Notice that what is usually a one-step process is now two steps
            // because of the additional step of selecting the camera. It is worth
            // repeating that camera selection needs to be done once per loop.
            // It is possible to interact with cameras through the camera list with
            // GetByIndex(); this is an alternative to retrieving cameras as
            // Spinnaker::CameraPtr objects that can be quick and easy for small tasks.
            //
            for (auto i = 0; i < mCameraList.GetSize(); i++)
                mCameraList.GetByIndex(i)->EndAcquisition();

            for (auto i = 0; i < mCameraList.GetSize(); i++)
            {
                // Select camera
                auto cameraPtr = mCameraList.GetByIndex(i);

                // Retrieve GenICam nodemap
                auto& iNodeMap = cameraPtr->GetNodeMap();

                // // Disable chunk data
                // result = disableChunkData(iNodeMap);
                // // if (result < 0)
                // //     return result;

                // Reset trigger
                auto result = resetTrigger(iNodeMap);
                if (result < 0)
					op::error("Error happened..." + std::to_string(result), __LINE__, __FUNCTION__, __FILE__);

                // Deinitialize each camera
                // Each camera must be deinitialized separately by first
                // selecting the camera and then deinitializing it.
                cameraPtr->DeInit();
            }

			op::log("Completed. Releasing...", op::Priority::High);

            // Clear camera list before releasing mSystemPtr
            mCameraList.Clear();

            // Release mSystemPtr
            mSystemPtr->ReleaseInstance();
        }

		op::log("Done! Exitting...", op::Priority::High);
    }
    catch (const Spinnaker::Exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void WPointGrey::initializationOnThread()
{
    try
    {
        initialized = true;

        // Print application build information
		op::log(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, op::Priority::High, __LINE__, __FUNCTION__, __FILE__);

        // Retrieve singleton reference to mSystemPtr object
        mSystemPtr = Spinnaker::System::GetInstance();

        // Retrieve list of cameras from the mSystemPtr
        mCameraList = mSystemPtr->GetCameras();

        unsigned int numCameras = mCameraList.GetSize();

		op::log("Number of cameras detected: " + std::to_string(numCameras), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);

        // Finish if there are no cameras
        if (numCameras == 0)
        {
            // Clear camera list before releasing mSystemPtr
            mCameraList.Clear();

            // Release mSystemPtr
            mSystemPtr->ReleaseInstance();

			op::log("Not enough cameras!\nPress Enter to exit...", op::Priority::High);
            getchar();

            op::error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
        }
		op::log("Camera system initialized...", op::Priority::High);

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
		op::log("\n*** DEVICE INFORMATION ***\n", op::Priority::High);

        for (int i = 0; i < mCameraList.GetSize(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Retrieve TL device nodemap
            auto& iNodeMapTLDevice = cameraPtr->GetTLDeviceNodeMap();

            // Print device information
            auto result = printDeviceInfo(iNodeMapTLDevice, i);
            if (result < 0)
                op::error("Result > 0, error " + std::to_string(result) + " occurred...", __LINE__, __FUNCTION__, __FILE__);
        }

        for (auto i = 0; i < mCameraList.GetSize(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Initialize each camera
            // You may notice that the steps in this function have more loops with
            // less steps per loop; this contrasts the acquireImages() function
            // which has less loops but more steps per loop. This is done for
            // demonstrative purposes as both work equally well.
            // Later: Each camera needs to be deinitialized once all images have been
            // acquired.
            cameraPtr->Init();

            // Retrieve GenICam nodemap
            // auto& iNodeMap = cameraPtr->GetNodeMap();

            // // Configure trigger
            // result = configureTrigger(iNodeMap);
            // if (result < 0)
			// op::error("Result > 0, error " + std::to_string(result) + " occurred...", __LINE__, __FUNCTION__, __FILE__);

            // // Configure chunk data
            // result = configureChunkData(iNodeMap);
            // if (result < 0)
            //     return result;

            // Remove buffer --> Always get newest frame
            Spinnaker::GenApi::INodeMap& snodeMap = cameraPtr->GetTLStreamNodeMap();
            Spinnaker::GenApi::CEnumerationPtr ptrBufferHandlingMode = snodeMap.GetNode("StreamBufferHandlingMode");
            if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingMode) || !Spinnaker::GenApi::IsWritable(ptrBufferHandlingMode))
				op::error("Unable to change buffer handling mode", __LINE__, __FUNCTION__, __FILE__);

            Spinnaker::GenApi::CEnumEntryPtr ptrBufferHandlingModeNewest = ptrBufferHandlingMode->GetEntryByName("NewestFirstOverwrite");
            if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingModeNewest) || !IsReadable(ptrBufferHandlingModeNewest))
				op::error("Unable to set buffer handling mode to newest (entry 'NewestFirstOverwrite' retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);
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
        std::vector<Spinnaker::GenICam::gcstring> strSerialNumbers(mCameraList.GetSize());
        for (auto i = 0u; i < strSerialNumbers.size(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Set acquisition mode to continuous
            Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = cameraPtr->GetNodeMap().GetNode("AcquisitionMode");
            if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionMode) || !Spinnaker::GenApi::IsWritable(ptrAcquisitionMode))
				op::error("Unable to set acquisition mode to continuous (node retrieval; camera " + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

            Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
            if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionModeContinuous) || !Spinnaker::GenApi::IsReadable(ptrAcquisitionModeContinuous))
				op::error("Unable to set acquisition mode to continuous (entry 'continuous' retrieval " + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

            int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

            ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

			op::log("Camera " + std::to_string(i) + " acquisition mode set to continuous...", op::Priority::High);

            // Begin acquiring images
            cameraPtr->BeginAcquisition();

			op::log("Camera " + std::to_string(i) + " started acquiring images...", op::Priority::High);

            // Retrieve device serial number for filename
            strSerialNumbers[i] = "";

            Spinnaker::GenApi::CStringPtr ptrStringSerial = cameraPtr->GetTLDeviceNodeMap().GetNode("DeviceSerialNumber");

            if (Spinnaker::GenApi::IsAvailable(ptrStringSerial) && Spinnaker::GenApi::IsReadable(ptrStringSerial))
            {
                strSerialNumbers[i] = ptrStringSerial->GetValue();
                op::log("Camera " + std::to_string(i) + " serial number set to " + strSerialNumbers[i].c_str() + "...", op::Priority::High);
            }
			op::log(" ", op::Priority::High);
        }

		op::log("\nRunning for all cameras...\n\n*** IMAGE ACQUISITION ***\n", op::Priority::High);
    }
    catch (const Spinnaker::Exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

std::shared_ptr<std::vector<Datum3D>> WPointGrey::workProducer()
{
    try
    {
        // Profiling speed
        const auto profilerKey = op::Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
        // Get image from each camera
        const auto cvMats = acquireImages(mCameraList);
        // Images to userDatum
        auto datums3d = std::make_shared<std::vector<Datum3D>>(cvMats.size());
        for (auto i = 0u ; i < cvMats.size() ; i++)
            datums3d->at(i).cvInputData = cvMats.at(i);
        // Profiling speed
        if (!cvMats.empty())
        {
            op::Profiler::timerEnd(profilerKey);
            op::Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 100);
        }
        // Return Datum
        return datums3d;
    }
    catch (const Spinnaker::Exception& e)
    {
        this->stop();
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return nullptr;
    }
    catch (const std::exception& e)
    {
        this->stop();
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return nullptr;
    }
}
