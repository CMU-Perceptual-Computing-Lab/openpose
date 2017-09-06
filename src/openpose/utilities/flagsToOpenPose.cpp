#include <cstdio> // sscanf
#include <openpose/producer/imageDirectoryReader.hpp>
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
            else if (poseModeString == "BODY_22")
                return PoseModel::BODY_22;
            // else
            error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
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
            const std::string message = "String does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution,"
                                        " NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).";
            error(message, __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
    }

    ProducerType flagsToProducerType(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Avoid duplicates (e.g. selecting at the time camera & video)
            if (!imageDirectory.empty() && !videoPath.empty())
                error("Selected simultaneously image directory and video. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);
            else if (!imageDirectory.empty() && webcamIndex > 0)
                error("Selected simultaneously image directory and webcam. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);
            else if (!videoPath.empty() && webcamIndex > 0)
                error("Selected simultaneously video and webcam. Please, select only one.", __LINE__, __FUNCTION__, __FILE__);

            // Get desired ProducerType
            if (!imageDirectory.empty())
                return ProducerType::ImageDirectory;
            else if (!videoPath.empty())
                return ProducerType::Video;
            else
                return ProducerType::Webcam;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ProducerType::Webcam;
        }
    }

    std::shared_ptr<Producer> flagsToProducer(const std::string& imageDirectory, const std::string& videoPath, const int webcamIndex,
                                              const std::string& webcamResolution, const double webcamFps)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            const auto type = flagsToProducerType(imageDirectory, videoPath, webcamIndex);

            if (type == ProducerType::ImageDirectory)
                return std::make_shared<ImageDirectoryReader>(imageDirectory);
            else if (type == ProducerType::Video)
                return std::make_shared<VideoReader>(videoPath);
            else if (type == ProducerType::Webcam)
            {
                // cameraFrameSize
                const auto webcamFrameSize = op::flagsToPoint(webcamResolution, "1280x720");
                if (webcamIndex >= 0)
                {
                    const auto throwExceptionIfNoOpened = true;
                    return std::make_shared<WebcamReader>(webcamIndex, webcamFrameSize, webcamFps, throwExceptionIfNoOpened);
                }
                else
                {
                    const auto throwExceptionIfNoOpened = false;
                    std::shared_ptr<WebcamReader> webcamReader;
                    for (auto index = 0 ; index < 10 ; index++)
                    {
                        webcamReader = std::make_shared<WebcamReader>(index, webcamFrameSize, webcamFps, throwExceptionIfNoOpened);
                        if (webcamReader->isOpened())
                        {
                            log("Auto-detecting camera index... Detected and opened camera " + std::to_string(index) + ".", Priority::High);
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

    std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg, const bool heatMapsAddPAFs)
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

    RenderMode flagsToRenderMode(const int renderFlag, const int renderPoseFlag)
    {
        try
        {
            if (renderFlag == -1 && renderPoseFlag != -2)
                return flagsToRenderMode(renderPoseFlag, -2);
            else if (renderFlag == 0)
                return RenderMode::None;
            else if (renderFlag == 1)
                return RenderMode::Cpu;
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
