OpenPose Advanced Doc - Heatmap Output
====================================



## Contents
1. [Keypoints](#keypoints)
2. [UI and Visual Heatmap Output](#ui-and-visual-heatmap-output)
3. [Heatmap Ordering](#heatmap-ordering)
4. [Heatmap Saving in Float Format](#heatmap-saving-in-float-format)
5. [Heatmap Scaling](#heatmap-scaling)





## Keypoints
Check [doc/output_keypoints.md](../02_output.md) for the basic output information. This document is for users that want to use the heatmaps.





## UI and Visual Heatmap Output
If you choose to visualize a body part or a PAF (Part Affinity Field) heat map with the command option `--part_to_show`, the visual GUI should show something similar to one of the following images:
<p align="center">
    <img src="../.github/media/body_heat_maps.png" width="720">
</p>

<p align="center">
    <img src="../.github/media/paf_heat_maps.png" width="720">
</p>





## Heatmap Ordering
For the **heat maps storing format**, instead of saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenates them into a huge (width x #heat maps) x (height) matrix (i.e., concatenated by columns). E.g., columns [0, individual heat map width] contain the first heat map, columns [individual heat map width + 1, 2 * individual heat map width] contain the second heat map, etc. Note that some image viewers are not able to display the resulting images due to the size. However, Chrome and Firefox are able to properly open them.

The saving order is body parts + background + PAFs. Any of them can be disabled with program flags. If background is disabled, then the final image will be body parts + PAFs. The body parts and background follow the order of `getPoseBodyPartMapping(const PoseModel poseModel)`.

The PAFs follow the order specified on `getPosePartPairs(const PoseModel poseModel)` together with `getPoseMapIndex(const PoseModel poseModel)`. E.g., assuming COCO (see example code below), the PAF channels in COCO start in 19 (smallest number in `getPoseMapIndex`, equal to #body parts + 1), and end up in 56 (highest one). Then, we can match its value from `getPosePartPairs`. For instance, 19 (x-channel) and 20 (y-channel) in `getPoseMapIndex` correspond to PAF from body part 1 to 8; 21 and 22 correspond to x,y channels in the joint from body part 8 to 9, etc. Note that if the smallest channel is odd (19), then all the x-channels are odd, and all the y-channels even. If the smallest channel is even, then the opposite will happen.
```
// C++ API call
#include <openpose/pose/poseParameters.hpp>
const auto& posePartPairsBody25 = getPosePartPairs(PoseModel::BODY_25);
const auto& posePartPairsCoco = getPosePartPairs(PoseModel::COCO_18);
const auto& posePartPairsMpi = getPosePartPairs(PoseModel::MPI_15);

// getPosePartPairs(PoseModel::BODY_25) result
// Each index is the key value corresponding to each body part in `getPoseBodyPartMapping`. E.g., 1 for "Neck", 2 for "RShoulder", etc.
// 1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24

// getPoseMapIndex(PoseModel::BODY_25) result
// 0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
```



## Heatmap Saving in Float Format
If you save the heatmaps in floating format by using the flag `--write_heatmaps_format float`, you can later read them in Python with:
```
# Load custom float format - Example in Python, assuming a (18 x 300 x 500) size Array
x = np.fromfile(heatMapFullPath, dtype=np.float32)
assert x[0] == 3 # First parameter saves the number of dimensions (18x300x500 = 3 dimensions)
shape_x = x[1:1+int(x[0])]
assert len(shape_x[0]) == 3 # Number of dimensions
assert shape_x[0] == 18 # Size of the first dimension
assert shape_x[1] == 300 # Size of the second dimension
assert shape_x[2] == 500 # Size of the third dimension
arrayData = x[1+int(round(x[0])):]
```



## Heatmap Scaling
Note that `--net_resolution` sets the size of the network, thus also the size of the output heatmaps. This heatmaps are resized while keeping the aspect ratio. When aspect ratio of the input and network are not the same, padding is added at the bottom and/or right part of the output heatmaps.
