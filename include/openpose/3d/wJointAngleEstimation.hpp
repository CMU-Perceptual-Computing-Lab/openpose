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
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Input
                auto& tDatumPtr = tDatums->at(0);
                const auto& poseKeypoints3D = tDatumPtr->poseKeypoints3D;
                const auto& faceKeypoints3D = tDatumPtr->faceKeypoints3D;
                const auto& handKeypoints3D = tDatumPtr->handKeypoints3D;
                // Running Adam model
                spJointAngleEstimation->adamFastFit(
                    tDatumPtr->adamPose, tDatumPtr->adamTranslation, tDatumPtr->vtVec, tDatumPtr->j0Vec,
                    tDatumPtr->adamFaceCoeffsExp, poseKeypoints3D, faceKeypoints3D, handKeypoints3D);
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                opLogIfDebug("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
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
