#ifndef OPENPOSE_CORE_DATUM_HPP
#define OPENPOSE_CORE_DATUM_HPP

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

        /**
         * Name used when saving the data to disk (e.g. `write_images` or `write_keypoint` flags in the demo).
         */
        std::string name;

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
         * If rendering is disabled (e.g. `no_render_pose` flag in the demo), outputData will be empty.
         * Size: 3 x output_net_height x output_net_width
         */
        Array<float> outputData;

        /**
         * Rendered image in cv::Mat uchar format.
         * It has been resized to the desired output resolution (e.g. `resolution` flag in the demo).
         * If outputData is empty, cvOutputData will also be empty.
         * Size: (output_height x output_width) x 3 channels
         */
        cv::Mat cvOutputData;

        // ------------------------------ Resulting Array<float> data parameters ------------------------------ //
        /**
         * Body pose (x,y,score) locations for each person in the image.
         * It has been resized to the desired output resolution (e.g. `resolution` flag in the demo).
         * Size: #people x #body parts (e.g. 18 for COCO or 15 for MPI) x 3 ((x,y) coordinates + score)
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
         * It will highly penalyze people with missing body parts (e.g. cropped people on the borders of the image).
         * If poseKeypoints is empty, poseScores will also be empty.
         * Size: #people
         */
        Array<float> poseScores;

        /**
         * Body pose heatmaps (body parts, background and/or PAFs) for the whole image.
         * This parameters is by default empty and disabled for performance. Each group (body parts, background and
         * PAFs) can be individually enabled.
         * #heatmaps = #body parts (if enabled) + 1 (if background enabled) + 2 x #PAFs (if enabled). Each PAF has 2
         * consecutive channels, one for x- and one for y-coordinates.
         * Order heatmaps: body parts + background (as appears in POSE_BODY_PART_MAPPING) + (x,y) channel of each PAF
         * (sorted as appears in POSE_BODY_PART_PAIRS). See `pose/poseParameters.hpp`.
         * The user can choose the heatmaps normalization: ranges [0, 1], [-1, 1] or [0, 255]. Check the
         * `heatmaps_scale` flag in the examples/tutorial_wrapper/ for more details.
         * Size: #heatmaps x output_net_height x output_net_width
         */
        Array<float> poseHeatMaps;

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
         * Face pose heatmaps (hand parts and/or background) for the whole image.
         * Analogous of faceHeatMaps applied to face.
         * Size each Array: #people x #hand parts (21) x output_net_height x output_net_width
         */
        std::array<Array<float>, 2> handHeatMaps;

        // ---------------------------------------- Other parameters ---------------------------------------- //
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
            return id < datum.id;
        }
        /**
         * Greater comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator>(const Datum& datum) const
        {
            return id > datum.id;
        }
        /**
         * Less or equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator<=(const Datum& datum) const
        {
            return id <= datum.id;
        }
        /**
         * Greater or equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator>=(const Datum& datum) const
        {
            return id >= datum.id;
        }
        /**
         * Equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator==(const Datum& datum) const
        {
            return id == datum.id;
        }
        /**
         * Not equal comparison operator.
         * @param datum Datum to be compared.
         * @result Whether the instance satisfies the condition with respect to datum.
         */
        inline bool operator!=(const Datum& datum) const
        {
            return id != datum.id;
        }
    };
}

#endif // OPENPOSE_CORE_DATUM_HPP
