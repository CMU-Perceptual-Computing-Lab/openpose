#ifndef OPENPOSE_CORE_ARRAY_CPU_GPU_HPP
#define OPENPOSE_CORE_ARRAY_CPU_GPU_HPP

#include <memory> // std::shared_ptr
#include <vector>
#include <openpose/core/macros.hpp>

namespace op
{
    /**
     * ArrayCpuGpu<T>: Bind of caffe::Blob<T> to avoid Caffe as dependency in the headers. 
     */
    template<typename T>
    class ArrayCpuGpu
    {
    public:
        ArrayCpuGpu();
        explicit ArrayCpuGpu(const void* caffeBlobTPtr);
        explicit ArrayCpuGpu(const int num, const int channels, const int height, const int width);
        // explicit ArrayCpuGpu(const std::vector<int>& shape);

        void Reshape(const int num, const int channels, const int height, const int width);
        void Reshape(const std::vector<int>& shape);
        // // void Reshape(const BlobShape& shape);
        // // void ReshapeLike(const Blob& other);
        // void ReshapeLike(const ArrayCpuGpu& other);
        std::string shape_string() const;
        const std::vector<int>& shape() const;
        int shape(const int index) const;
        int num_axes() const;
        int count() const;
        int count(const int start_axis, const int end_axis) const;
        int count(const int start_axis) const;

        int CanonicalAxisIndex(const int axis_index) const;

        // int num() const;
        // int channels() const;
        // int height() const;
        // int width() const;
        // int LegacyShape(const int index) const;

        // int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const;
        // int offset(const std::vector<int>& indices) const;

        // // void CopyFrom(const Blob<T>& source, bool copy_diff = false, bool reshape = false);
        // void CopyFrom(const ArrayCpuGpu<T>& source, bool copy_diff = false, bool reshape = false);

        T data_at(const int n, const int c, const int h, const int w) const;
        T diff_at(const int n, const int c, const int h, const int w) const;
        // T data_at(const std::vector<int>& index) const; // Caffe warning
        // T diff_at(const std::vector<int>& index) const;

        // // const boost::shared_ptr<SyncedMemory>& data() const;
        // // const boost::shared_ptr<SyncedMemory>& diff() const;

        const T* cpu_data() const;
        void set_cpu_data(T* data);
        const int* gpu_shape() const;
        const T* gpu_data() const;
        void set_gpu_data(T* data);
        const T* cpu_diff() const;
        const T* gpu_diff() const;
        T* mutable_cpu_data();
        T* mutable_gpu_data();
        T* mutable_cpu_diff();
        T* mutable_gpu_diff();
        void Update();
        // void FromProto(const BlobProto& proto, bool reshape = true);
        // void ToProto(BlobProto* proto, bool write_diff = false) const;

        // T asum_data() const;
        // T asum_diff() const;
        // T sumsq_data() const;
        // T sumsq_diff() const;

        // void scale_data(T scale_factor);
        // void scale_diff(T scale_factor);

        // // void ShareData(const Blob& other);
        // // void ShareDiff(const Blob& other);

        // // bool ShapeEquals(const BlobProto& other);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplArrayCpuGpu;
        std::shared_ptr<ImplArrayCpuGpu> spImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(ArrayCpuGpu);
    };

    // // Static methods
    // OVERLOAD_C_OUT(ArrayCpuGpu)
}

#endif // OPENPOSE_CORE_ARRAY_CPU_GPU_HPP
