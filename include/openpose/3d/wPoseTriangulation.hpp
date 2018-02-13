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
        explicit WPoseTriangulation();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        DELETE_COPY(WPoseTriangulation);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPoseTriangulation<TDatums>::WPoseTriangulation()
    {
    }

    template<typename TDatums>
    void WPoseTriangulation<TDatums>::initializationOnThread()
    {
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
                std::vector<cv::Mat> cameraParameterMatrices;
                std::vector<Array<float>> poseKeypointVector;
                std::vector<Array<float>> faceKeypointVector;
                std::vector<Array<float>> leftHandKeypointVector;
                std::vector<Array<float>> rightHandKeypointVector;
                for (auto& datumsElement : *tDatums)
                {
                    poseKeypointVector.emplace_back(datumsElement.poseKeypoints);
                    faceKeypointVector.emplace_back(datumsElement.faceKeypoints);
                    leftHandKeypointVector.emplace_back(datumsElement.handKeypoints[0]);
                    rightHandKeypointVector.emplace_back(datumsElement.handKeypoints[1]);
                    cameraParameterMatrices.emplace_back(datumsElement.cameraParameterMatrix);
                }
                // Pose 3-D reconstruction
                reconstructArray(tDatums->at(0).poseKeypoints3D, poseKeypointVector, cameraParameterMatrices);
                // Face 3-D reconstruction
                reconstructArray(tDatums->at(0).faceKeypoints3D, faceKeypointVector, cameraParameterMatrices);
                // Left hand 3-D reconstruction
                reconstructArray(tDatums->at(0).handKeypoints3D[0], leftHandKeypointVector, cameraParameterMatrices);
                // Right hand 3-D reconstruction
                reconstructArray(tDatums->at(0).handKeypoints3D[1], rightHandKeypointVector, cameraParameterMatrices);
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
