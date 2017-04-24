#ifndef OPENPOSE__POSE__ENUM_CLASSES_HPP
#define OPENPOSE__POSE__ENUM_CLASSES_HPP

namespace op
{
    /**
     * An enum class in which all the possible type of pose estimation models are included.
     */
    enum class PoseModel : unsigned char
    {
        COCO_18 = 0,    /**< COCO model, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15,         /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4,       /**< Same MPI model, but reducing the number of CNN stages to 4 (see poseModel.cpp for details). It should increase speed and reduce accuracy.*/
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

#endif // OPENPOSE__POSE__ENUM_CLASSES_HPP
