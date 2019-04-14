#ifndef OPENPOSE_CORE_DATUM_HPP
#define OPENPOSE_CORE_DATUM_HPP

#ifdef USE_EIGEN
    #include <Eigen/Core>
#endif
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>

namespace op
{
    /**
     * Datum: The OpenPose Basic Piece of Information Between Threads
     * Datum is one the main OpenPose classes/structs. The workers and threads share by default a
     * std::shared_ptr<std::vector<Datum>>. It contains all the parameters that the different workers and threads need
     * to exchange.
     */
    struct OP_API Datum
    {
        // ---------------------------------------- ID parameters ---------------------------------------- //
        unsigned long long id; /**< Datum ID. Internally used to sort the Datums if multi-threading is used. */

        unsigned long long subId; /**< Datum sub-ID. Internally used to sort the Datums if multi-threading is used. */

        unsigned long long subIdMax; /**< Datum maximum sub-ID. Used to sort the Datums if multi-threading is used. */

        /**
         * Name used when saving the data to disk (e.g., `write_images` or `write_keypoint` flags in the demo).
         */
        std::string name;

        /**
         * Corresponding frame number.
         * If the producer (e.g., video) starts from frame 0 and does not repeat any frame, then frameNumber should
         * match the field id.
         */
        unsigned long long frameNumber;

        // ------------------------------ Input image and rendered version parameters ------------------------------ //
        /**
         * Original image to be processed in cv::Mat uchar format.
         * Size: (input_width x input_height) x 3 channels
         */
        cv::Mat cvInputData;

        /**
         * Original image to be processed in Array<float> format.
         * It has been resized to the net input resolution, as well as reformatted Array<float> format to be compatible
         * with the net.
         * If >1 scales, each scale is right- and bottom-padded to fill the greatest resolution. The
         * scales are sorted from bigger to smaller.
         * Vector size: #scales
         * Each array size: 3 x input_net_height x input_net_width
         */
        std::vector<Array<float>> inputNetData;

        /**
         * Rendered image in Array<float> format.
         * It consists of a blending of the cvInputData and the pose/body part(s) heatmap/PAF(s).
         * If rendering is disabled (e.g., `no_render_pose` flag in the demo), outputData will be empty.
         * Size: 3 x output_net_height x output_net_width
         */
        Array<float> outputData;

        /**
         * Rendered image in cv::Mat uchar format.
         * It has been resized to the desired output resolution (e.g., `resolution` flag in the demo).
         * If outputData is empty, cvOutputData will also be empty.
         * Size: (output_height x output_width) x 3 channels
         */
        cv::Mat cvOutputData;

        /**
         * Rendered 3D image in cv::Mat uchar format.
         */
        cv::Mat cvOutputData3D;

        // ------------------------------ Resulting Array<float> data parameters ------------------------------ //
        /**
         * Body pose (x,y,score) locations for each person in the image.
         * It has been resized to the desired output resolution (e.g., `resolution` flag in the demo).
         * Size: #people x #body parts (e.g., 18 for COCO or 15 for MPI) x 3 ((x,y) coordinates + score)
         */
        Array<float> poseKeypoints;

        /**
         * People ID
         * It returns a person ID for each body pose, providing temporal consistency. The ID will be the same one
         * for a person across frames. I.e. this ID allows to keep track of the same person in time.
         * If either person identification is disabled or poseKeypoints is empty, poseIds will also be empty.
         * Size: #people
         */
        Array<long long> poseIds;

        /**
         * Body pose global confidence/score for each person in the image.
         * It does not only consider the score of each body keypoint, but also the score of each PAF association.
         * Optimized for COCO evaluation metric.
         * It will highly penalyze people with missing body parts (e.g., cropped people on the borders of the image).
         * If poseKeypoints is empty, poseScores will also be empty.
         * Size: #people
         */
        Array<float> poseScores;

        /**
         * Body pose heatmaps (body parts, background and/or PAFs) for the whole image.
         * This parameter is by default empty and disabled for performance. Each group (body parts, background and
         * PAFs) can be individually enabled.
         * #heatmaps = #body parts (if enabled) + 1 (if background enabled) + 2 x #PAFs (if enabled). Each PAF has 2
         * consecutive channels, one for x- and one for y-coordinates.
         * Order heatmaps: body parts + background (as appears in POSE_BODY_PART_MAPPING) + (x,y) channel of each PAF
         * (sorted as appears in POSE_BODY_PART_PAIRS). See `pose/poseParameters.hpp`.
         * The user can choose the heatmaps normalization: ranges [0, 1], [-1, 1] or [0, 255]. Check the
         * `heatmaps_scale` flag in {OpenPose_path}doc/demo_overview.md for more details.
         * Size: #heatmaps x output_net_height x output_net_width
         */
        Array<float> poseHeatMaps;

        /**
         * Body pose candidates for the whole image.
         * This parameter is by default empty and disabled for performance. It can be enabled with `candidates_body`.
         * Candidates refer to all the detected body parts, before being assembled into people. Note that the number
         * of candidates is equal or higher than the number of body parts after being assembled into people.
         * Size: #body parts x min(part candidates, POSE_MAX_PEOPLE) x 3 (x,y,score).
         * Rather than vector, it should ideally be:
         * std::array<std::vector<std::array<float,3>>, #BP> poseCandidates;
         */
        std::vector<std::vector<std::array<float,3>>> poseCandidates;

        /**
         * Face detection locations (x,y,width,height) for each person in the image.
         * It is resized to cvInputData.size().
         * Size: #people
         */
        std::vector<Rectangle<float>> faceRectangles;

        /**
         * Face keypoints (x,y,score) locations for each person in the image.
         * It has been resized to the same resolution as `poseKeypoints`.
         * Size: #people x #face parts (70) x 3 ((x,y) coordinates + score)
         */
        Array<float> faceKeypoints;

        /**
         * Face pose heatmaps (face parts and/or background) for the whole image.
         * Analogous of bodyHeatMaps applied to face. However, there is no PAFs and the size is different.
         * Size: #people x #face parts (70) x output_net_height x output_net_width
         */
        Array<float> faceHeatMaps;

        /**
         * Hand detection locations (x,y,width,height) for each person in the image.
         * It is resized to cvInputData.size().
         * Size: #people
         */
        std::vector<std::array<Rectangle<float>, 2>> handRectangles;

        /**
         * Hand keypoints (x,y,score) locations for each person in the image.
         * It has been resized to the same resolution as `poseKeypoints`.
         * handKeypoints[0] corresponds to left hands, and handKeypoints[1] to right ones.
         * Size each Array: #people x #hand parts (21) x 3 ((x,y) coordinates + score)
         */
        std::array<Array<float>, 2> handKeypoints;

        /**
         * Hand pose heatmaps (hand parts and/or background) for the whole image.
         * Analogous of faceHeatMaps applied to face.
         * Size each Array: #people x #hand parts (21) x output_net_height x output_net_width
         */
        std::array<Array<float>, 2> handHeatMaps;

        // ---------------------------------------- 3-D Reconstruction parameters ---------------------------------------- //
        /**
         * Body pose (x,y,z,score) locations for each person in the image.
         * Size: #people x #body parts (e.g., 18 for COCO or 15 for MPI) x 4 ((x,y,z) coordinates + score)
         */
        Array<float> poseKeypoints3D;

        /**
         * Face keypoints (x,y,z,score) locations for each person in the image.
         * It has been resized to the same resolution as `poseKeypoints3D`.
         * Size: #people x #face parts (70) x 4 ((x,y,z) coordinates + score)
         */
        Array<float> faceKeypoints3D;

        /**
         * Hand keypoints (x,y,z,score) locations for each person in the image.
         * It has been resized to the same resolution as `poseKeypoints3D`.
         * handKeypoints[0] corresponds to left hands, and handKeypoints[1] to right ones.
         * Size each Array: #people x #hand parts (21) x 4 ((x,y,z) coordinates + score)
         */
        std::array<Array<float>, 2> handKeypoints3D;

        /**
         * 3x4 camera matrix of the camera (equivalent to cameraIntrinsics * cameraExtrinsics).
         */
        cv::Mat cameraMatrix;

        /**
         * 3x4 extrinsic parameters of the camera.
         */
        cv::Mat cameraExtrinsics;

        /**
         * 3x3 intrinsic parameters of the camera.
         */
        cv::Mat cameraIntrinsics;

        /**
         * If it is not empty, OpenPose will not run its internal body pose estimation network and will instead use
         * this data as the substitute of its network. The size of this element must match the size of the output of
         * its internal network, or it will lead to core dumped (segmentation) errors. You can modify the pose
         * estimation flags to match the dimension of both elements (e.g., `--net_resolution`, `--scale_number`, etc.).
         */
        Array<float> poseNetOutput;

        // ---------------------------------------- Other (internal) parameters ---------------------------------------- //
        /**
         * Scale ratio between the input Datum::cvInputData and the net input size.
         */
        std::vector<double> scaleInputToNetInputs;

        /**
         * Size(s) (width x height) of the image(s) fed to the pose deep net.
         * The size of the std::vector corresponds to the number of scales. 
         */
        std::vector<Point<int>> netInputSizes;

        /**
         * Scale ratio between the input Datum::cvInputData and the output Datum::cvOutputData.
         */
        double scaleInputToOutput;

        /**
         * Size (width x height) of the image returned by the deep net.
         */
        Point<int> netOutputSize;

        /**
         * Scale ratio between the net output and the final output Datum::cvOutputData.
         */
        double scaleNetToOutput;

        /**
         * Pair with the element key id POSE_BODY_PART_MAPPING on `pose/poseParameters.hpp` and its mapped value (e.g.
         * 1 and "Neck").
         */
        std::pair<int, std::string> elementRendered;

        // 3D/Adam parameters (experimental code not meant to be publicly used)
        #ifdef USE_3D_ADAM_MODEL
            // Adam/Unity params
            std::vector<double> adamPosePtr;
            int adamPoseRows;
            std::vector<double> adamTranslationPtr;
            std::vector<double> vtVecPtr;
            int vtVecRows;
            std::vector<double> j0VecPtr;
            int j0VecRows;
            std::vector<double> adamFaceCoeffsExpPtr;
            int adamFaceCoeffsExpRows;
            #ifdef USE_EIGEN
                // Adam/Unity params
                Eigen::Matrix<double, 62, 3, Eigen::RowMajor> adamPose;
                Eigen::Vector3d adamTranslation;
                // Adam params (Jacobians)
                Eigen::Matrix<double, Eigen::Dynamic, 1> vtVec;
                Eigen::Matrix<double, Eigen::Dynamic, 1> j0Vec;
                Eigen::VectorXd adamFaceCoeffsExp;
            #endif
        #endif





        // ---------------------------------------- Functions ---------------------------------------- //
        /**
         * Default constructor struct.
         * It simply initializes the struct, id is temporary set to 0 and each other variable is assigned to its
         * default value.
         */
        explicit Datum();

        /**
         * Copy constructor.
         * It performs `fast copy`: For performance purpose, copying a Datum or Array<T> or cv::Mat just copies the
         * reference, it still shares the same internal data.
         * Modifying the copied element will modify the original one.
         * Use clone() for a slower but real copy, similarly to cv::Mat and Array<T>.
         * @param datum Datum to be copied.
         */
        Datum(const Datum& datum);

        /**
         * Copy assignment.
         * Similar to Datum::Datum(const Datum& datum).
         * @param datum Datum to be copied.
         * @return The resulting Datum.
         */
        Datum& operator=(const Datum& datum);

        /**
         * Move constructor.
         * It destroys the original Datum to be moved.
         * @param datum Datum to be moved.
         */
        Datum(Datum&& datum);

        /**
         * Move assignment.
         * Similar to Datum::Datum(Datum&& datum).
         * @param datum Datum to be moved.
         * @return The resulting Datum.
         */
        Datum& operator=(Datum&& datum);

        /**
         * Destructor class.
         * Declared virtual so that Datum can be inherited.
         */
        virtual ~Datum();

        /**
         * Clone function.
         * Similar to cv::Mat::clone and Array<T>::clone.
         * It performs a real but slow copy of the data, i.e., even if the copied element is modified, the original
         * one is not.
         * @return The resulting Datum.
         */
        Datum clone() const;





        // ---------------------------------------- Comparison operators ---------------------------------------- //
        /**
         * Less comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator<(const Datum& datum) const
        {
            // return id < datum.id;
            return id < datum.id || (id == datum.id && subId < datum.subId);
        }
        /**
         * Greater comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator>(const Datum& datum) const
        {
            // return id > datum.id;
            return id > datum.id || (id == datum.id && subId > datum.subId);
        }
        /**
         * Less or equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator<=(const Datum& datum) const
        {
            // return id <= datum.id;
            return id < datum.id || (id == datum.id && subId <= datum.subId);
        }
        /**
         * Greater or equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator>=(const Datum& datum) const
        {
            // return id >= datum.id;
            return id > datum.id || (id == datum.id && subId >= datum.subId);
        }
        /**
         * Equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator==(const Datum& datum) const
        {
            // return id == datum.id;
            return id == datum.id && subId == datum.subId;
        }
        /**
         * Not equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator!=(const Datum& datum) const
        {
            // return id != datum.id;
            return id != datum.id || subId != datum.subId;
        }
    };

    // Defines for Datum. Added here rather than in `macros.hpp` to avoid circular dependencies
    #define BASE_DATUM Datum
    #define BASE_DATUMS std::vector<std::shared_ptr<BASE_DATUM>>
    #define BASE_DATUMS_SH std::shared_ptr<BASE_DATUMS>
    #define DEFINE_TEMPLATE_DATUM(templateName) template class OP_API templateName<BASE_DATUMS_SH>
    #define COMPILE_TEMPLATE_DATUM(templateName) extern template class templateName<BASE_DATUMS_SH>
}

#endif // OPENPOSE_CORE_DATUM_HPP
