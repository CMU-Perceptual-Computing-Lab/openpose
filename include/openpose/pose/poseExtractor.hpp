#ifndef OPENPOSE__POSE__POSE_EXTRACTOR_HPP
#define OPENPOSE__POSE__POSE_EXTRACTOR_HPP

#include <array>
#include <atomic>
#include <thread>
#include <opencv2/core/core.hpp>
#include "../core/array.hpp"
#include "../utilities/macros.hpp"
#include "poseParameters.hpp"
#include "../core/enumClasses.hpp"

namespace op
{
    class PoseExtractor
    {
    public:
        PoseExtractor(const cv::Size& netOutputSize, const cv::Size& outputSize, const PoseModel poseModel, const std::vector<HeatMapType>& heatMapTypes = {},
                      const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne);

        void initializationOnThread();

        virtual void forwardPass(const Array<float>& inputNetData, const cv::Size& inputDataSize) = 0;

        virtual const float* getHeatMapCpuConstPtr() const = 0;

        virtual const float* getHeatMapGpuConstPtr() const = 0;

        Array<float> getHeatMaps() const;

        virtual const float* getPoseGpuConstPtr() const = 0;

        Array<float> getPoseKeyPoints() const;

        double getScaleNetToOutput() const;

        double get(const PoseProperty property) const;

        void set(const PoseProperty property, const double value);

        void increase(const PoseProperty property, const double value);

    protected:
        const PoseModel mPoseModel;
        const cv::Size mNetOutputSize;
        const cv::Size mOutputSize;
        Array<float> mPoseKeyPoints;
        double mScaleNetToOutput;

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

#endif // OPENPOSE__POSE__POSE_EXTRACTOR_HPP
