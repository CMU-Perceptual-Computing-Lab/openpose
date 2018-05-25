#include <cstdio> // sscanf
#include <openpose/producer/flirReader.hpp>
#include <openpose/producer/imageDirectoryReader.hpp>
#include <openpose/producer/ipCameraReader.hpp>
#include <openpose/producer/videoReader.hpp>
#include <openpose/producer/webcamReader.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/flagsToOpenPose.hpp>

namespace op
{
    PoseModel flagsToPoseModel(const std::string& poseModeString)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (poseModeString == "COCO")
                return PoseModel::COCO_18;
            else if (poseModeString == "MPI")
                return PoseModel::MPI_15;
            else if (poseModeString == "MPI_4_layers")
                return PoseModel::MPI_15_4;
            else if (poseModeString == "BODY_18")
                return PoseModel::BODY_18;
            else if (poseModeString == "BODY_19")
                return PoseModel::BODY_19;
            else if (poseModeString == "BODY_19b")
                return PoseModel::BODY_19b;
            else if (poseModeString == "BODY_19N")
                return PoseModel::BODY_19N;
            else if (poseModeString == "BODY_19_X2")
                return PoseModel::BODY_19_X2;
            else if (poseModeString == "BODY_23")
                return PoseModel::BODY_23;
            else if (poseModeString == "BODY_59")
                return PoseModel::BODY_59;
            // else
            error("String does not correspond to any model (COCO, MPI, MPI_4_layers)",
                  __LINE__, __FUNCTION__, __FILE__);
            return PoseModel::COCO_18;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return PoseModel::COCO_18;
        }
    }

    ScaleMode flagsToScaleMode(const int keypointScale)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (keypointScale == 0)
                return ScaleMode::InputResolution;
            else if (keypointScale == 1)
                return ScaleMode::NetOutputResolution;
            else if (keypointScale == 2)
                return ScaleMode::OutputResolution;
            else if (keypointScale == 3)
                return ScaleMode::ZeroToOne;
            else if (keypointScale == 4)
                return ScaleMode::PlusMinusOne;
            // else
            const std::string message = "Integer does not correspond to any scale mode: (0, 1, 2, 3, 4) for"
                                        " (InputResolution, NetOutputResolution, OutputResolution, ZeroToOne,"
                                        " PlusMinusOne).";
            error(message, __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
    }

    ScaleMode flagsToHeatMapScaleMode(const int heatMapScale)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (heatMapScale == 0)
                return ScaleMode::PlusMinusOne;
            else if (heatMapScale == 1)
                return ScaleMode::ZeroToOne;
            else if (heatMapScale == 2)
                return ScaleMode::UnsignedChar;
            else if (heatMapScale == 3)
                return ScaleMode::NoScale;
            // else
            const std::string message = "Integer does not correspond to any scale mode: (0, 1, 2, 3) for"
                                        " (PlusMinusOne, ZeroToOne, UnsignedChar, NoScale).";
            error(message, __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::PlusMinusOne;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::PlusMinusOne;
        }
    }

    ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath,
                                     const std::string& ipCameraPath, const int webcamIndex,
                                     const bool flirCamera)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Avoid duplicates (e.g. selecting at the time camera & video)
            if (int(!imageDirectory.empty()) + int(!videoPath.empty()) + int(webcamIndex > 0)
                + int(flirCamera) + int(!ipCameraPath.empty()) > 1)
                error("Selected simultaneously"
                      " image directory (seletected: " + (imageDirectory.empty() ? "no" : imageDirectory) + "),"
                      " video (seletected: " + (videoPath.empty() ? "no" : videoPath) + "),"
                      " camera (selected: " + (webcamIndex > 0 ? std::to_string(webcamIndex) : "no") + "),"
                      " flirCamera (selected: " + (flirCamera ? "yes" : "no") + ","
                      " and/or IP camera (selected: " + (ipCameraPath.empty() ? "no" : ipCameraPath) + ")."
                      " Please, select only one.", __LINE__, __FUNCTION__, __FILE__);

            // Get desired ProducerType
            if (!imageDirectory.empty())
                return ProducerType::ImageDirectory;
            else if (!videoPath.empty())
                return ProducerType::Video;
            else if (!ipCameraPath.empty())
                return ProducerType::IPCamera;
            else if (flirCamera)
                return ProducerType::FlirCamera;
            else
                return ProducerType::Webcam;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ProducerType::Webcam;
        }
    }

    std::shared_ptr<Producer> flagsToProducer(const std::string& imageDirectory, const std::string& videoPath,
                                              const std::string& ipCameraPath, const int webcamIndex,
                                              const bool flirCamera, const std::string& cameraResolution,
                                              const double webcamFps, const std::string& cameraParameterPath,
                                              const bool undistortImage, const unsigned int imageDirectoryStereo,
                                              const int flirCameraIndex)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const auto type = flagsToProducerType(imageDirectory, videoPath, ipCameraPath, webcamIndex, flirCamera);

            if (type == ProducerType::ImageDirectory)
                return std::make_shared<ImageDirectoryReader>(imageDirectory, imageDirectoryStereo,
                                                              cameraParameterPath);
            else if (type == ProducerType::Video)
                return std::make_shared<VideoReader>(videoPath, imageDirectoryStereo, cameraParameterPath);
            else if (type == ProducerType::IPCamera)
                return std::make_shared<IpCameraReader>(ipCameraPath);
            // Flir camera
            if (type == ProducerType::FlirCamera)
            {
                // cameraFrameSize
                const auto cameraFrameSize = flagsToPoint(cameraResolution, "-1x-1");
                return std::make_shared<FlirReader>(cameraParameterPath, cameraFrameSize, undistortImage,
                                                    flirCameraIndex);
            }
            // Webcam
            if (type == ProducerType::Webcam)
            {
                // cameraFrameSize
                auto cameraFrameSize = flagsToPoint(cameraResolution, "1280x720");
                if (cameraFrameSize.x < 0 || cameraFrameSize.y < 0)
                    cameraFrameSize = Point<int>{1280,720};
                if (webcamIndex >= 0)
                {
                    const auto throwExceptionIfNoOpened = true;
                    return std::make_shared<WebcamReader>(webcamIndex, cameraFrameSize, webcamFps,
                                                          throwExceptionIfNoOpened);
                }
                else
                {
                    const auto throwExceptionIfNoOpened = false;
                    std::shared_ptr<WebcamReader> webcamReader;
                    for (auto index = 0 ; index < 10 ; index++)
                    {
                        webcamReader = std::make_shared<WebcamReader>(index, cameraFrameSize, webcamFps,
                                                                      throwExceptionIfNoOpened);
                        if (webcamReader->isOpened())
                        {
                            log("Auto-detecting camera index... Detected and opened camera " + std::to_string(index)
                                + ".", Priority::High);
                            return webcamReader;
                        }
                    }
                    error("No camera found.", __LINE__, __FUNCTION__, __FILE__);
                }
            }
            // else
            error("Undefined Producer selected.", __LINE__, __FUNCTION__, __FILE__);
            return std::shared_ptr<Producer>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::shared_ptr<Producer>{};
        }
    }

    std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg,
                                             const bool heatMapsAddPAFs)
    {
        try
        {
            std::vector<HeatMapType> heatMapTypes;
            if (heatMapsAddParts)
                heatMapTypes.emplace_back(HeatMapType::Parts);
            if (heatMapsAddBkg)
                heatMapTypes.emplace_back(HeatMapType::Background);
            if (heatMapsAddPAFs)
                heatMapTypes.emplace_back(HeatMapType::PAFs);
            return heatMapTypes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    RenderMode flagsToRenderMode(const int renderFlag, const bool gpuBuggy, const int renderPoseFlag)
    {
        try
        {
            // Body: to auto-pick CPU/GPU depending on CPU_ONLY/CUDA
            if (renderFlag == -1 && renderPoseFlag == -2)
            {
                #ifdef USE_CUDA
                    return (gpuBuggy ? RenderMode::Cpu : RenderMode::Gpu);
                #else
                    return RenderMode::Cpu;
                #endif
            }
            // Face and hand: to pick same than body
            else if (renderFlag == -1 && renderPoseFlag != -2)
                return flagsToRenderMode(renderPoseFlag, gpuBuggy, -2);
            // No render
            else if (renderFlag == 0)
                return RenderMode::None;
            // CPU render
            else if (renderFlag == 1)
                return RenderMode::Cpu;
            // GPU render
            else if (renderFlag == 2)
                return RenderMode::Gpu;
            // else
            error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
            return RenderMode::None;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return RenderMode::None;
        }
    }

    DisplayMode flagsToDisplayMode(const int display, const bool enabled3d)
    {
        try
        {
            // Automatic --> All if 3d enabled, 2d otherwise
            if (display == -1)
            {
                if (enabled3d)
                    return DisplayMode::DisplayAll;
                else
                    return DisplayMode::Display2D;
            }
            // No render
            else if (display == 0)
                return DisplayMode::NoDisplay;
            // All (2-D + 3-D)
            else if (display == 1)
                return DisplayMode::DisplayAll;
            // 2-D
            else if (display == 2)
                return DisplayMode::Display2D;
            // 3-D
            else if (display == 3)
                return DisplayMode::Display3D;
            // else
            error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
            return DisplayMode::NoDisplay;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return DisplayMode::NoDisplay;
        }
    }

    Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample)
    {
        try
        {
            Point<int> point;
            const auto nRead = sscanf(pointString.c_str(), "%dx%d", &point.x, &point.y);
            checkE(nRead, 2, "Invalid resolution format: `" +  pointString + "`, it should be e.g. `" + pointExample
                   + "`.", __LINE__, __FUNCTION__, __FILE__);
            return point;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<int>{};
        }
    }
}
