#include <cstdio> // sscanf
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/flagsToOpenPose.hpp>

namespace op
{
    PoseMode flagsToPoseMode(const int poseModeInt)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (poseModeInt >= 0 && poseModeInt < (int)PoseMode::Size)
                return (PoseMode)poseModeInt;
            else
            {
                error("Value (" + std::to_string(poseModeInt) + ") does not correspond with any PoseMode.",
                      __LINE__, __FUNCTION__, __FILE__);
                return PoseMode::Enabled;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return PoseMode::Enabled;
        }
    }

    PoseModel flagsToPoseModel(const std::string& poseModeString)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Body pose
            if (poseModeString == "BODY_25")
                return PoseModel::BODY_25;
            else if (poseModeString == "COCO")
                return PoseModel::COCO_18;
            else if (poseModeString == "MPI")
                return PoseModel::MPI_15;
            else if (poseModeString == "MPI_4_layers")
                return PoseModel::MPI_15_4;
            else if (poseModeString == "BODY_19")
                return PoseModel::BODY_19;
            else if (poseModeString == "BODY_19E")
                return PoseModel::BODY_19E;
            else if (poseModeString == "BODY_19N")
                return PoseModel::BODY_19N;
            else if (poseModeString == "BODY_19_X2")
                return PoseModel::BODY_19_X2;
            else if (poseModeString == "BODY_23")
                return PoseModel::BODY_23;
            else if (poseModeString == "BODY_25B")
                return PoseModel::BODY_25B;
            else if (poseModeString == "BODY_25D")
                return PoseModel::BODY_25D;
            else if (poseModeString == "BODY_25E")
                return PoseModel::BODY_25E;
            else if (poseModeString == "BODY_135")
                return PoseModel::BODY_135;
            // Car pose
            else if (poseModeString == "CAR_12")
                return PoseModel::CAR_12;
            else if (poseModeString == "CAR_22")
                return PoseModel::CAR_22;
            // else
            error("String (`" + poseModeString + "`) does not correspond to any model (BODY_25, COCO, MPI,"
                  " MPI_4_layers).", __LINE__, __FUNCTION__, __FILE__);
            return PoseModel::BODY_25;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return PoseModel::BODY_25;
        }
    }

    ScaleMode flagsToScaleMode(const int keypointScaleMode)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (keypointScaleMode == 0)
                return ScaleMode::InputResolution;
            else if (keypointScaleMode == 1)
                return ScaleMode::NetOutputResolution;
            else if (keypointScaleMode == 2)
                return ScaleMode::OutputResolution;
            else if (keypointScaleMode == 3)
                return ScaleMode::ZeroToOne;
            else if (keypointScaleMode == 4)
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

    ScaleMode flagsToHeatMapScaleMode(const int heatMapScaleMode)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (heatMapScaleMode == 0)
                return ScaleMode::PlusMinusOne;
            else if (heatMapScaleMode == 1)
                return ScaleMode::ZeroToOne;
            else if (heatMapScaleMode == 2)
                return ScaleMode::UnsignedChar;
            else if (heatMapScaleMode == 3)
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

    Detector flagsToDetector(const int detector)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (detector >= 0 && detector < (int)Detector::Size)
                return (Detector)detector;
            else
            {
                error("Value (" + std::to_string(detector) + ") does not correspond with any Detector.",
                      __LINE__, __FUNCTION__, __FILE__);
                return Detector::Body;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Detector::Body;
        }
    }

    ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath,
                                     const std::string& ipCameraPath, const int webcamIndex,
                                     const bool flirCamera)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Avoid duplicates (e.g., selecting at the time camera & video)
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

    std::pair<ProducerType, std::string> flagsToProducer(
        const std::string& imageDirectory, const std::string& videoPath, const std::string& ipCameraPath,
        const int webcamIndex, const bool flirCamera, const int flirCameraIndex)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const auto type = flagsToProducerType(imageDirectory, videoPath, ipCameraPath, webcamIndex, flirCamera);

            if (type == ProducerType::ImageDirectory)
                return std::make_pair(ProducerType::ImageDirectory, imageDirectory);
            else if (type == ProducerType::Video)
                return std::make_pair(ProducerType::Video, videoPath);
            else if (type == ProducerType::IPCamera)
                return std::make_pair(ProducerType::IPCamera, ipCameraPath);
            // Flir camera
            else if (type == ProducerType::FlirCamera)
                return std::make_pair(ProducerType::FlirCamera, std::to_string(flirCameraIndex));
            // Webcam
            else if (type == ProducerType::Webcam)
                return std::make_pair(ProducerType::Webcam, std::to_string(webcamIndex));
            // else
            error("Undefined Producer selected.", __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(ProducerType::None, "");
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(ProducerType::None, "");
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
            // Auto auto-picks CPU/CUDA depending on the compiled version (CPU_ONLY/CUDA)
            if (renderFlag == -1 && renderPoseFlag == -2)
                return (gpuBuggy ? RenderMode::Cpu : RenderMode::Auto);
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
            checkE(nRead, 2, "Invalid resolution format: `" +  pointString + "`, it should be e.g., `" + pointExample
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
