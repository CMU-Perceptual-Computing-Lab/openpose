#ifndef OPENPOSE__POSE__POSE_EXTRACTOR_HPP
#define OPENPOSE__POSE__POSE_EXTRACTOR_HPP

#include <array>
#include <atomic>
#include <thread>
#include <opencv2/core/core.hpp>
#include <openpose/core/array.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/utilities/macros.hpp>
#include "poseParameters.hpp"

namespace op
{
    class PoseExtractor
    {
    public:
        PoseExtractor(const cv::Size& netOutputSize, const cv::Size& outputSize, const PoseModel poseModel, const std::vector<HeatMapType>& heatMapTypes = {},
                      const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne);

        virtual ~PoseExtractor();

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

#endif // OPENPOSE__POSE__POSE_EXTRACTOR_HPP
