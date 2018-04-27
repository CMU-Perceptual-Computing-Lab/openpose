OpenPose Demo - Output
====================================



## Contents
1. [Output Format](#output-format)
    1. [Keypoint Ordering](#keypoint-ordering)
    2. [Heatmap Ordering](#heatmap-ordering)
    3. [Face and Hands](#face-and-hands)
    4. [Pose Output Format](#pose-output-format)
    5. [Face Output Format](#face-output-format)
    6. [Hand Output Format](#hand-output-format)
3. [Reading Saved Results](#reading-saved-results)
4. [Keypoint Format in the C++ API](#keypoint-format-in-the-c-api)



## Output Format
There are 2 alternatives to save the OpenPose output.

1. The `write_json` flag saves the people pose data using a custom JSON writer. Each JSON file has a `people` array of objects, where each object has:
    1. An array `pose_keypoints_2d` containing the body part locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`. The coordinates `x` and `y` can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag `keypoint_scale`, while `c` is the confidence score in the range [0,1].
    2. The arrays `face_keypoints_2d`, `hand_left_keypoints_2d`, and `hand_right_keypoints_2d`, analogous to `pose_keypoints_2d`.
    3. The analogous 3-D arrays `body_keypoints_3d`, `face_keypoints_3d`, `hand_left_keypoints_2d`, and `hand_right_keypoints_2d` (if `--3d` is enabled, otherwise they will be empty). Instead of `x1,y1,c1,x2,y2,c2,...`, their format is `x1,y1,z1,c1,x2,y2,z2,c2,...`, where `c` is simply 1 or 0 depending on whether the 3-D reconstruction was successful or not.
    4. The body part candidates before being assembled into people (if `--part_candidates` is enabled).
```
{
    "version":1.1,
    "people":[
        {
            "pose_keypoints_2d":[582.349,507.866,0.845918,746.975,631.307,0.587007,...],
            "face_keypoints_2d":[468.725,715.636,0.189116,554.963,652.863,0.665039,...],
            "hand_left_keypoints_2d":[746.975,631.307,0.587007,615.659,617.567,0.377899,...],
            "hand_right_keypoints_2d":[617.581,472.65,0.797508,0,0,0,723.431,462.783,0.88765,...]
            "pose_keypoints_3d":[582.349,507.866,507.866,0.845918,507.866,746.975,631.307,0.587007,...],
            "face_keypoints_3d":[468.725,715.636,715.636,0.189116,715.636,554.963,652.863,0.665039,...],
            "hand_left_keypoints_3d":[746.975,631.307,631.307,0.587007,631.307,615.659,617.567,0.377899,...],
            "hand_right_keypoints_3d":[617.581,472.65,472.65,0.797508,472.65,0,0,0,723.431,462.783,0.88765,...]
        }
    ],
    // If `--part_candidates` enabled
    "part_candidates":[
        {
            "0":[296.994,258.976,0.845918,238.996,365.027,0.189116],
            "1":[381.024,321.984,0.587007],
            "2":[313.996,314.97,0.377899],
            "3":[238.996,365.027,0.189116],
            "4":[283.015,332.986,0.665039],
            "5":[457.987,324.003,0.430488,283.015,332.986,0.665039],
            "6":[],
            "7":[],
            "8":[],
            "9":[],
            "10":[],
            "11":[],
            "12":[],
            "13":[],
            "14":[293.001,242.991,0.674305],
            "15":[314.978,241,0.797508],
            "16":[],
            "17":[369.007,235.964,0.88765]
        }
    ]
}
```

2. (Deprecated) The `write_keypoint` flag uses the OpenCV cv::FileStorage default formats, i.e. JSON (available after OpenCV 3.0), XML, and YML. Note that it does not include any other information othern than keypoints.

Both of them follow the keypoint ordering described in the [Keypoint Ordering](#keypoint-ordering) section.



### Keypoint Ordering
The body part mapping order of any body model (e.g. COCO, MPI) can be extracted from the C++ API by using the `getPoseBodyPartMapping(const PoseModel poseModel)` function available in [poseParameters.hpp](../include/openpose/pose/poseParameters.hpp):
```
// C++ API call
#include <openpose/pose/poseParameters.hpp>
const auto& poseBodyPartMappingCoco = getPoseBodyPartMapping(PoseModel::COCO_18);
const auto& poseBodyPartMappingMpi = getPoseBodyPartMapping(PoseModel::MPI_15);

// Result for COCO (18 body parts)
// POSE_COCO_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "RHip"},
//     {9,  "RKnee"},
//     {10, "RAnkle"},
//     {11, "LHip"},
//     {12, "LKnee"},
//     {13, "LAnkle"},
//     {14, "REye"},
//     {15, "LEye"},
//     {16, "REar"},
//     {17, "LEar"},
//     {18, "Background"},
// }
```



### Heatmap Ordering
For the **heat maps storing format**, instead of saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenates them into a huge (width x #heat maps) x (height) matrix (i.e., concatenated by columns). E.g., columns [0, individual heat map width] contains the first heat map, columns [individual heat map width + 1, 2 * individual heat map width] contains the second heat map, etc. Note that some image viewers are not able to display the resulting images due to the size. However, Chrome and Firefox are able to properly open them.

The saving order is body parts + background + PAFs. Any of them can be disabled with program flags. If background is disabled, then the final image will be body parts + PAFs. The body parts and background follow the order of `getPoseBodyPartMapping(const PoseModel poseModel)`.

The PAFs follow the order specified on `getPosePartPairs(const PoseModel poseModel)` together with `getPoseMapIndex(const PoseModel poseModel)`. E.g., assuming COCO (see example code below), the PAF channels in COCO start in 19 (smallest number in `getPoseMapIndex`, equal to #body parts + 1), and end up in 56 (highest one). Then, we can match its value from `getPosePartPairs`. For instance, 19 (x-channel) and 20 (y-channel) in `getPoseMapIndex` correspond to PAF from body part 1 to 8; 21 and 22 correspond to x,y channels in the joint from body part 8 to 9, etc. Note that if the smallest channel is odd (19), then all the x-channels are odd, and all the y-channels even. If the smallest channel is even, then the opposite will happen.
```
// C++ API call
#include <openpose/pose/poseParameters.hpp>
const auto& posePartPairsCoco = getPosePartPairs(PoseModel::COCO_18);
const auto& posePartPairsMpi = getPosePartPairs(PoseModel::MPI_15);

// getPosePartPairs(PoseModel::COCO_18) result
// Each index is the key value corresponding to each body part in `getPoseBodyPartMapping`. E.g., 1 for "Neck", 2 for "RShoulder", etc.
// 1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17

// getPoseMapIndex(PoseModel::COCO_18) result
// 31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46
```



### Face and Hands
The output format is analogous for hand (`hand_left_keypoints`, `hand_right_keypoints`) and face (`face_keypoints`) JSON files.



### Pose Output Format
<p align="center">
    <img src="media/keypoints_pose.png", width="480">
</p>



### Face Output Format
<p align="center">
    <img src="media/keypoints_face.png", width="480">
</p>



### Hand Output Format
<p align="center">
    <img src="media/keypoints_hand.png", width="480">
</p>



## Reading Saved Results
We use standard formats (JSON, XML, PNG, JPG, ...) to save our results, so there are many open-source libraries to read them in most programming languages. From C++, but you might the functions in [include/openpose/filestream/fileStream.hpp](../include/openpose/filestream/fileStream.hpp). In particular, `loadData` (for JSON, XML and YML files) and `loadImage` (for image formats such as PNG or JPG) to load the data into cv::Mat format.



## Keypoint Format in the C++ API
There are 3 different keypoint `Array<float>` elements in the `Datum` class:

1. Array<float> **poseKeypoints**: In order to access person `person` and body part `part` (where the index matches `POSE_COCO_BODY_PARTS` or `POSE_MPI_BODY_PARTS`), you can simply output:
```
    // Common parameters needed
    const auto numberPeopleDetected = poseKeypoints.getSize(0);
    const auto numberBodyParts = poseKeypoints.getSize(1);
    // Easy version
    const auto x = poseKeypoints[{person, part, 0}];
    const auto y = poseKeypoints[{person, part, 1}];
    const auto score = poseKeypoints[{person, part, 2}];
    // Slightly more efficient version
    // If you want to access these elements on a huge loop, you can get the index
    // by your own, but it is usually not faster enough to be worthy
    const auto baseIndex = poseKeypoints.getSize(2)*(person*numberBodyParts + part);
    const auto x = poseKeypoints[baseIndex];
    const auto y = poseKeypoints[baseIndex + 1];
    const auto score = poseKeypoints[baseIndex + 2];
```
2. Array<float> **faceKeypoints**: It is completely analogous to poseKeypoints.
```
    // Common parameters needed
    const auto numberPeopleDetected = faceKeypoints.getSize(0);
    const auto numberFaceParts = faceKeypoints.getSize(1);
    // Easy version
    const auto x = faceKeypoints[{person, part, 0}];
    const auto y = faceKeypoints[{person, part, 1}];
    const auto score = faceKeypoints[{person, part, 2}];
    // Slightly more efficient version
    const auto baseIndex = faceKeypoints.getSize(2)*(person*numberFaceParts + part);
    const auto x = faceKeypoints[baseIndex];
    const auto y = faceKeypoints[baseIndex + 1];
    const auto score = faceKeypoints[baseIndex + 2];
```
3. std::array<Array<float>, 2> **handKeypoints**, where handKeypoints[0] corresponds to the left hand and handKeypoints[1] to the right one. Each handKeypoints[i] is analogous to poseKeypoints and faceKeypoints:
```
    // Common parameters needed
    const auto numberPeopleDetected = handKeypoints[0].getSize(0); // = handKeypoints[1].getSize(0)
    const auto numberHandParts = handKeypoints[0].getSize(1); // = handKeypoints[1].getSize(1)

    // Easy version
    // Left Hand
    const auto xL = handKeypoints[0][{person, part, 0}];
    const auto yL = handKeypoints[0][{person, part, 1}];
    const auto scoreL = handKeypoints[0][{person, part, 2}];
    // Right Hand
    const auto xR = handKeypoints[1][{person, part, 0}];
    const auto yR = handKeypoints[1][{person, part, 1}];
    const auto scoreR = handKeypoints[1][{person, part, 2}];

    // Slightly more efficient version
    const auto baseIndex = handKeypoints[0].getSize(2)*(person*numberHandParts + part);
    // Left Hand
    const auto xL = handKeypoints[0][baseIndex];
    const auto yL = handKeypoints[0][baseIndex + 1];
    const auto scoreL = handKeypoints[0][baseIndex + 2];
    // Right Hand
    const auto xR = handKeypoints[1][baseIndex];
    const auto yR = handKeypoints[1][baseIndex + 1];
    const auto scoreR = handKeypoints[1][baseIndex + 2];
```
