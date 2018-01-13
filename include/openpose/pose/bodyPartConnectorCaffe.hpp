#ifndef OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP
#define OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

// PIMPL does not work here. Alternative:
// stackoverflow.com/questions/13978775/how-to-avoid-include-dependency-to-external-library?answertab=active#tab-top
namespace caffe
{
    template <typename T> class Blob;
}

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep
    // the compatibility with any generic Caffe version, we keep this 'layer' inside our library rather than in the
    // Caffe code.
    template <typename T>
    class OP_API BodyPartConnectorCaffe
    {
    public:
        explicit BodyPartConnectorCaffe();

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom);

        virtual inline const char* type() const { return "BodyPartConnector"; }

        void setPoseModel(const PoseModel poseModel);

        void setInterMinAboveThreshold(const T interMinAboveThreshold);

        void setInterThreshold(const T interThreshold);

        void setMinSubsetCnt(const int minSubsetCnt);

        void setMinSubsetScore(const T minSubsetScore);

        void setScaleNetToOutput(const T scaleNetToOutput);

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
        T mInterMinAboveThreshold;
        T mInterThreshold;
        int mMinSubsetCnt;
        T mMinSubsetScore;
        T mScaleNetToOutput;
        std::array<int, 4> mHeatMapsSize;
        std::array<int, 4> mPeaksSize;
        std::array<int, 4> mTopSize;

        DELETE_COPY(BodyPartConnectorCaffe);
    };
}

#endif // OPENPOSE_POSE_BODY_PART_CONNECTOR_CAFFE_HPP
