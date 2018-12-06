#ifndef OPENPOSE_POSE_POSE_RENDERER_HPP
#define OPENPOSE_POSE_POSE_RENDERER_HPP

#include <map>
#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    class OP_API PoseRenderer
    {
    public:
        PoseRenderer(const PoseModel poseModel);

        virtual ~PoseRenderer();

        virtual void initializationOnThread(){};

        virtual std::pair<int, std::string> renderPose(Array<float>& outputData, const Array<float>& poseKeypoints,
                                                       const float scaleInputToOutput,
                                                       const float scaleNetToOutput = -1.f) = 0;

    protected:
        const PoseModel mPoseModel;
        const std::map<unsigned int, std::string> mPartIndexToName;

    private:

        DELETE_COPY(PoseRenderer);
    };
}

#endif // OPENPOSE_POSE_POSE_RENDERER_HPP
