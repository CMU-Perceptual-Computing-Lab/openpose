#ifndef OPENPOSE_POSE_ENUM_CLASSES_HPP
#define OPENPOSE_POSE_ENUM_CLASSES_HPP

namespace op
{
    /**
     * An enum class in which all the possible type of pose estimation models are included.
     */
    enum class PoseModel : unsigned char
    {
        COCO_18 = 0,    /**< COCO model, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15 = 1,     /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4 = 2,   /**< Variation of the MPI model, reduced number of CNN stages to 4: faster but less accurate.*/
        BODY_18 = 3,    /**< Experimental. Do not use. */
        BODY_22 = 4,    /**< Experimental. Do not use. */
        Size,
    };

    enum class PoseProperty : unsigned char
    {
        NMSThreshold = 0,
        ConnectInterMinAboveThreshold,
        ConnectInterThreshold,
        ConnectMinSubsetCnt,
        ConnectMinSubsetScore,
        Size,
    };
}

#endif // OPENPOSE_POSE_ENUM_CLASSES_HPP
