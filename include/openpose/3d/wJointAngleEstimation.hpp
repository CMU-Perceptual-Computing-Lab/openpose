#ifdef USE_3D_ADAM_MODEL
#ifndef OPENPOSE_3D_W_JOINT_ANGLE_ESTIMATION_HPP
#define OPENPOSE_3D_W_JOINT_ANGLE_ESTIMATION_HPP

#include <openpose/core/common.hpp>
#include <openpose/3d/jointAngleEstimation.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WJointAngleEstimation : public Worker<TDatums>
    {
    public:
        explicit WJointAngleEstimation(const std::shared_ptr<JointAngleEstimation>& jointAngleEstimation);

        virtual ~WJointAngleEstimation();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<JointAngleEstimation> spJointAngleEstimation;

        DELETE_COPY(WJointAngleEstimation);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WJointAngleEstimation<TDatums>::WJointAngleEstimation(const std::shared_ptr<JointAngleEstimation>& jointAngleEstimation) :
        spJointAngleEstimation{jointAngleEstimation}
    {
    }

    template<typename TDatums>
    WJointAngleEstimation<TDatums>::~WJointAngleEstimation()
    {
    }

    template<typename TDatums>
    void WJointAngleEstimation<TDatums>::initializationOnThread()
    {
        try
        {
            spJointAngleEstimation->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WJointAngleEstimation<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Input
                auto& datum = tDatums->at(0);
                const auto& poseKeypoints3D = datum.poseKeypoints3D;
                const auto& faceKeypoints3D = datum.faceKeypoints3D;
                const auto& handKeypoints3D = datum.handKeypoints3D;
                // Running Adam model
                spJointAngleEstimation->adamFastFit(
                    datum.adamPose, datum.adamTranslation, datum.vtVec, datum.j0Vec,
                    datum.adamFaceCoeffsExp, poseKeypoints3D, faceKeypoints3D, handKeypoints3D);
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

    COMPILE_TEMPLATE_DATUM(WJointAngleEstimation);
}

#endif // OPENPOSE_3D_W_JOINT_ANGLE_ESTIMATION_HPP
#endif
