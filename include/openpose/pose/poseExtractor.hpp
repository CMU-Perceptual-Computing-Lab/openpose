#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_HPP

#include <atomic>
#include <thread>
#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/pose/poseParameters.hpp>

namespace op
{
    class OP_API PoseExtractor
    {
    public:
        PoseExtractor(const PoseModel poseModel,
                      const std::vector<HeatMapType>& heatMapTypes = {},
                      const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        virtual ~PoseExtractor();

        void initializationOnThread();

        virtual void forwardPass(const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
                                 const std::vector<double>& scaleRatios = {1.f}) = 0;

        virtual const float* getHeatMapCpuConstPtr() const = 0;

        virtual const float* getHeatMapGpuConstPtr() const = 0;

        virtual std::vector<int> getHeatMapSize() const = 0;

        Array<float> getHeatMaps() const;

        virtual const float* getPoseGpuConstPtr() const = 0;

        Array<float> getPoseKeypoints() const;

        Array<float> getPoseScores() const;

        float getScaleNetToOutput() const;

        double get(const PoseProperty property) const;

        void set(const PoseProperty property, const double value);

        void increase(const PoseProperty property, const double value);

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
        std::array<std::atomic<double>, (int)PoseProperty::Size> mProperties;
        std::thread::id mThreadId;

        DELETE_COPY(PoseExtractor);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_HPP
