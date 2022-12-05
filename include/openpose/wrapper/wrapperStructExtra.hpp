#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_EXTRA_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_EXTRA_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * WrapperStructExtra: Pose estimation and rendering configuration struct.
     * WrapperStructExtra allows the user to set up the pose estimation and rendering parameters that will be used for
     * the OpenPose WrapperT template and Wrapper class.
     */
    struct OP_API WrapperStructExtra
    {
        /**
         * Whether to run the 3-D reconstruction demo, i.e.,
         * 1) Reading from a stereo camera system.
         * 2) Performing 3-D reconstruction from the multiple views.
         * 3) Displaying 3-D reconstruction results.
         */
        bool reconstruct3d;

        /**
         * Minimum number of views required to reconstruct each keypoint.
         * By default (-1), it will require max(2, min(4, #cameras-1)) cameras to see the keypoint in order to
         * reconstruct it.
         */
        int minViews3d;

        /**
         * Whether to return a person ID for each body skeleton, providing temporal consistency.
         */
        bool identification;

        /**
         * Whether to enable people tracking across frames. The value indicates the number of frames where tracking
         * is run between each OpenPose keypoint detection. Select -1 (default) to disable it or 0 to run
         * simultaneously OpenPose keypoint detector and tracking for potentially higher accuracy than only OpenPose.
         */
        int tracking;

        /**
         * Whether to enable inverse kinematics (IK) from 3-D keypoints to obtain 3-D joint angles. By default
         * (0 threads), it is disabled. Increasing the number of threads will increase the speed but also the
         * global system latency.
         */
        int ikThreads;

        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructExtra(
            const bool reconstruct3d = false, const int minViews3d = -1, const bool identification = false,
            const int tracking = -1, const int ikThreads = 0);
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_EXTRA_HPP
