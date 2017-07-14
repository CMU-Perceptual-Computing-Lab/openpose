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
        PoseExtractor(const Point<int>& netOutputSize, const Point<int>& outputSize, const PoseModel poseModel, const std::vector<HeatMapType>& heatMapTypes = {},
                      const ScaleMode heatMapScale = ScaleMode::ZeroToOne);

        virtual ~PoseExtractor();

        void initializationOnThread();

        virtual void forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize, const std::vector<float>& scaleRatios = {1.f}) = 0;

        virtual const float* getHeatMapCpuConstPtr() const = 0;

        virtual const float* getHeatMapGpuConstPtr() const = 0;

        Array<float> getHeatMaps() const;

        virtual const float* getPoseGpuConstPtr() const = 0;

        Array<float> getPoseKeypoints() const;

        float getScaleNetToOutput() const;

        double get(const PoseProperty property) const;

        void set(const PoseProperty property, const double value);

        void increase(const PoseProperty property, const double value);

    protected:
        const PoseModel mPoseModel;
        const Point<int> mNetOutputSize;
        const Point<int> mOutputSize;
        Array<float> mPoseKeypoints;
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
