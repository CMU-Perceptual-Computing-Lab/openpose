#ifndef OPENPOSE_POSE_ENUM_CLASSES_HPP
#define OPENPOSE_POSE_ENUM_CLASSES_HPP

namespace op
{
    /**
     * An enum class in which all the possible type of pose estimation models are included.
     */
    enum class PoseModel : unsigned char
    {
        /**
         * COCO + 6 foot keypoints + neck + lower abs model, with 25+1 components (see poseParameters.hpp for details).
         */
        BODY_25 = 0,
        COCO_18,        /**< COCO model + neck, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15,         /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4,       /**< Variation of the MPI model, reduced number of CNN stages to 4: faster but less accurate.*/
        BODY_19,        /**< Experimental. Do not use. */
        BODY_19_X2,     /**< Experimental. Do not use. */
        BODY_59,        /**< Experimental. Do not use. */
        BODY_19N,       /**< Experimental. Do not use. */
        BODY_25E,       /**< Experimental. Do not use. */
        BODY_25_19,     /**< Experimental. Do not use. */
        BODY_65,        /**< Experimental. Do not use. */
        CAR_12,         /**< Experimental. Do not use. */
        BODY_25D,       /**< Experimental. Do not use. */
        BODY_23,        /**< Experimental. Do not use. */
        CAR_22,         /**< Experimental. Do not use. */
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
