#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_NET_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_NET_HPP

#include <atomic>
#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    class OP_API PoseExtractorNet
    {
    public:
        PoseExtractorNet(const PoseModel poseModel,
                         const std::vector<HeatMapType>& heatMapTypes = {},
                         const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOneFixedAspect,
                         const bool addPartCandidates = false,
                         const bool maximizePositives = false);

        virtual ~PoseExtractorNet();

        void initializationOnThread();

        virtual void forwardPass(
            const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
            const std::vector<double>& scaleRatios = {1.f}, const Array<float>& poseNetOutput = Array<float>{}) = 0;

        virtual const float* getCandidatesCpuConstPtr() const = 0;

        virtual const float* getCandidatesGpuConstPtr() const = 0;

        virtual const float* getHeatMapCpuConstPtr() const = 0;

        virtual const float* getHeatMapGpuConstPtr() const = 0;

        virtual std::vector<int> getHeatMapSize() const = 0;

        Array<float> getHeatMapsCopy() const;

        std::vector<std::vector<std::array<float,3>>> getCandidatesCopy() const;

        virtual const float* getPoseGpuConstPtr() const = 0;

        Array<float> getPoseKeypoints() const;

        Array<float> getPoseScores() const;

        float getScaleNetToOutput() const;

        double get(const PoseProperty property) const;

        void set(const PoseProperty property, const double value);

        void increase(const PoseProperty property, const double value);

        void clear();

    protected:
        const PoseModel mPoseModel;
        Point<int> mNetOutputSize;
        Array<float> mPoseKeypoints;
        Array<float> mPoseScores;
        float mScaleNetToOutput;

        void checkThread() const;

        virtual void netInitializationOnThread() = 0;

    private:
        const std::vector<HeatMapType> mHeatMapTypes;
        const ScaleMode mHeatMapScaleMode;
        const bool mAddPartCandidates;
        std::array<std::atomic<double>, (int)PoseProperty::Size> mProperties;
        std::thread::id mThreadId;

        DELETE_COPY(PoseExtractorNet);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_NET_HPP
