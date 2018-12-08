#ifndef OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP
#define OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep
    // the compatibility with any generic Caffe version, we keep this 'layer' inside our library rather than in the
    // Caffe code.
    template <typename T>
    class BodyPartConnectorCaffe
    {
    public:
        explicit BodyPartConnectorCaffe();

        virtual ~BodyPartConnectorCaffe();

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom);

        virtual inline const char* type() const { return "BodyPartConnector"; }

        void setPoseModel(const PoseModel poseModel);

        void setMaximizePositives(const bool maximizePositives);

        void setInterMinAboveThreshold(const T interMinAboveThreshold);

        void setInterThreshold(const T interThreshold);

        void setMinSubsetCnt(const int minSubsetCnt);

        void setMinSubsetScore(const T minSubsetScore);

        void setScaleNetToOutput(const T scaleNetToOutput);

        virtual void Forward(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                             Array<T>& poseScores);

        virtual void Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                 Array<T>& poseScores);

        virtual void Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                 Array<T>& poseScores);

        virtual void Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

    private:
        PoseModel mPoseModel;
        bool mMaximizePositives;
        T mInterMinAboveThreshold;
        T mInterThreshold;
        int mMinSubsetCnt;
        T mMinSubsetScore;
        T mScaleNetToOutput;
        std::array<int, 4> mHeatMapsSize;
        std::array<int, 4> mPeaksSize;
        std::array<int, 4> mTopSize;
        // GPU auxiliary
        unsigned int* pBodyPartPairsGpuPtr;
        unsigned int* pMapIdxGpuPtr;
        Array<T> mFinalOutputCpu;
        T* pFinalOutputGpuPtr;

        DELETE_COPY(BodyPartConnectorCaffe);
    };
}

#endif // OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP
