#ifndef OPENPOSE_3D_W_POSE_TRIANGULATION_HPP
#define OPENPOSE_3D_W_POSE_TRIANGULATION_HPP

#include <openpose/core/common.hpp>
#include <openpose/3d/poseTriangulation.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WPoseTriangulation : public Worker<TDatums>
    {
    public:
        explicit WPoseTriangulation(const std::shared_ptr<PoseTriangulation>& poseTriangulation);

        virtual ~WPoseTriangulation();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<PoseTriangulation> spPoseTriangulation;

        DELETE_COPY(WPoseTriangulation);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPoseTriangulation<TDatums>::WPoseTriangulation(const std::shared_ptr<PoseTriangulation>& poseTriangulation) :
        spPoseTriangulation{poseTriangulation}
    {
    }

    template<typename TDatums>
    WPoseTriangulation<TDatums>::~WPoseTriangulation()
    {
    }

    template<typename TDatums>
    void WPoseTriangulation<TDatums>::initializationOnThread()
    {
        try
        {
            spPoseTriangulation->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WPoseTriangulation<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // 3-D triangulation and reconstruction
                std::vector<cv::Mat> cameraMatrices;
                std::vector<Array<float>> poseKeypointVector;
                std::vector<Array<float>> faceKeypointVector;
                std::vector<Array<float>> leftHandKeypointVector;
                std::vector<Array<float>> rightHandKeypointVector;
                std::vector<Point<int>> imageSizes;
                for (auto& datumsElement : *tDatums)
                {
                    poseKeypointVector.emplace_back(datumsElement.poseKeypoints);
                    faceKeypointVector.emplace_back(datumsElement.faceKeypoints);
                    leftHandKeypointVector.emplace_back(datumsElement.handKeypoints[0]);
                    rightHandKeypointVector.emplace_back(datumsElement.handKeypoints[1]);
                    cameraMatrices.emplace_back(datumsElement.cameraMatrix);
                    imageSizes.emplace_back(Point<int>{datumsElement.cvInputData.cols,
                                                       datumsElement.cvInputData.rows});
                }
                // Pose 3-D reconstruction
                auto poseKeypoints3Ds = spPoseTriangulation->reconstructArray(
                    {poseKeypointVector, faceKeypointVector, leftHandKeypointVector, rightHandKeypointVector},
                    cameraMatrices, imageSizes);
                // Assign to all tDatums
                for (auto& datumsElement : *tDatums)
                {
                    datumsElement.poseKeypoints3D = poseKeypoints3Ds[0];
                    datumsElement.faceKeypoints3D = poseKeypoints3Ds[1];
                    datumsElement.handKeypoints3D[0] = poseKeypoints3Ds[2];
                    datumsElement.handKeypoints3D[1] = poseKeypoints3Ds[3];
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WPoseTriangulation);
}

#endif // OPENPOSE_3D_W_POSE_TRIANGULATION_HPP
