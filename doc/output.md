OpenPose Demo - Output
====================================



## Output Format
There are 2 alternatives to save the **(x,y,score) body part locations**. The `write_keypoint` flag uses the OpenCV cv::FileStorage default formats (JSON, XML and YML). However, the JSON format is only available after OpenCV 3.0. Hence, `write_keypoint_json` saves the people pose data using a custom JSON writer. For the latter, each JSON file has a `people` array of objects, where each object has an array `pose_keypoints` containing the body part locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`. The coordinates `x` and `y` can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag `keypoint_scale`. In addition, `c` is the confidence in the range [0,1].

```
{
    "version":0.1,
    "people":[
        {"pose_keypoints":[1114.15,160.396,0.846207,...]},
        {"pose_keypoints":[...]},
    ]
}
```

The body part order of the COCO (18 body parts) and MPI (15 body parts) keypoints is described in `POSE_BODY_PART_MAPPING` in [include/openpose/pose/poseParameters.hpp](../include/openpose/pose/poseParameters.hpp). E.g., for COCO:
```
    POSE_COCO_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Bkg"},
    }
```

For the **heat maps storing format**, instead of individually saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenates them into a huge (width x #heat maps) x (height) matrix, i.e. it concats the heat maps by columns. E.g., columns [0, individual heat map width] contains the first heat map, columns [individual heat map width + 1, 2 * individual heat map width] contains the second heat map, etc. Note that some image viewers are not able to display the resulting images due to the size. However, Chrome and Firefox are able to properly open them.

The saving order is body parts + background + PAFs. Any of them can be disabled with program flags. If background is disabled, then the final image will be body parts + PAFs. The body parts and background follow the order of `POSE_COCO_BODY_PARTS` or `POSE_MPI_BODY_PARTS`, while the PAFs follow the order specified on POSE_BODY_PART_PAIRS in `poseParameters.hpp`. E.g., for COCO:
```
    POSE_COCO_PAIRS    {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,   2,16,  5,17};
```

Where each index is the key value corresponding to each body part in `POSE_COCO_BODY_PARTS`, e.g., 0 for "Neck", 1 for "RShoulder", etc.

### Face and Hands
The output format is analogous for hand (`hand_left_keypoints` and `hand_right_keypoints`) and face (`face_keypoints`) JSON files.



## Keypoint Format on the op::Datum Class
There are 3 different keypoint Array<float> elements on this class:

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



## Reading Saved Results
We use standard formats (JSON, XML, PNG, JPG, ...) to save our results, so there will be lots of frameworks to read them later, but you might also directly use our functions in [include/openpose/filestream.hpp](../include/openpose/filestream.hpp). In particular, `loadData` (for JSON, XML and YML files) and `loadImage` (for image formats such as PNG or JPG) to load the data into cv::Mat format.



## Pose Output Format
<p align="center">
    <img src="media/keypoints_pose.png", width="480">
</p>



## Face Output Format
<p align="center">
    <img src="media/keypoints_face.png", width="480">
</p>



## Hand Output Format
<p align="center">
    <img src="media/keypoints_hand.png", width="480">
</p>
