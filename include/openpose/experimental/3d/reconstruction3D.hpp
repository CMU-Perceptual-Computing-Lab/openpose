#ifndef OPENPOSE_EXPERIMENTAL_3D_RECONSTRUCTION_3D_HPP
#define OPENPOSE_EXPERIMENTAL_3D_RECONSTRUCTION_3D_HPP

#include <openpose/core/common.hpp>
#include <openpose/experimental/3d/datum3D.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    // Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from Worker.
    // This worker will do 3-D reconstruction
    // We apply the simple Direct linear transformation (DLT) algorithm, asumming each keypoint (e.g. right hip) is seen
    // by all the cameras.
    // No non-linear minimization used, and if some camera misses the point, it is not reconstructed.
    // See `examples/tutorial_wrapper/` for more details about inhering the Worker class and using it for post-processing
    // purposes.
    class OP_API WReconstruction3D : public Worker<std::shared_ptr<std::vector<Datum3D>>>
    {
    public:
        void initializationOnThread() {}

        void work(std::shared_ptr<std::vector<Datum3D>>& datumsPtr);
    };
}

#endif // OPENPOSE_EXPERIMENTAL_3D_RECONSTRUCTION_3D_HPP
