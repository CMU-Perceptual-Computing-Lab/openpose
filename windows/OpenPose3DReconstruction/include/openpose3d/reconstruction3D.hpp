#ifndef RECONSTRUCTION_3D_HPP
#define RECONSTRUCTION_3D_HPP

#include <openpose3d/datum3D.hpp>

// Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from Worker.
// This worker will do 3-D reconstruction
// We apply the simple Direct linear transformation (DLT) algorithm, asumming each keypoint (e.g. right hip) is seen by all the cameras.
// No non-linear minimization used, and if some camera misses the point, it is not reconstructed.
// See `examples/tutorial_wrapper/` for more details about inhering the Worker class and using it for post-processing purposes.
class WReconstruction3D : public op::Worker<std::shared_ptr<std::vector<Datum3D>>>
{
public:
    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<Datum3D>>& datumsPtr);
};

#endif // RECONSTRUCTION_3D_HPP
