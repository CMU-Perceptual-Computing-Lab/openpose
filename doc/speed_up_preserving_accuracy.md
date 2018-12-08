OpenPose - Maximizing the OpenPose Speed
========================================================================================

## Contents
1. [OpenPose Benchmark](#openpose-benchmark)
2. [Profiling Speed](#profiling-speed)
3. [Speed Up Preserving Accuracy](#speed-up-preserving-accuracy)
4. [Speed Up and Memory Reduction](#speed-up-and-memory-reduction)
5. [CPU Version](#cpu-version)





## OpenPose Benchmark
Check the [OpenPose Benchmark](https://docs.google.com/spreadsheets/d/1-DynFGvoScvfWDA1P4jDInCkbD4lg0IKOYbXgEq0sK0/edit#gid=0) to discover the approximate expected speed of your graphics card.



### Profiling Speed
Check the [doc/installation.md#profiling-speed](./installation.md#profiling-speed) section to measure the bottlenecks in your OpenPose distribution and make sure everything is working as expected.



## Speed Up Preserving Accuracy
Some speed tips to maximize the OpenPose runtime speed while preserving the accuracy (do not expect miracles, but it might help a bit boosting the framerate):

    1. Enable the `WITH_OPENCV_WITH_OPENGL` flag in CMake to have a much faster GUI display (but you must also compile OpenCV with OpenGL support). Note: Default OpenCV in Ubuntu (from apt-get install) does have OpenGL support included. Nevertheless, default Windows portable binaries do not.
    2. Change GPU rendering by CPU rendering to get approximately +0.5 FPS (`--render_pose 1`).
    3. Use cuDNN 5.1 (cuDNN 6 is ~10% slower).
    4. Use the `BODY_25` model for simultaneously maximum speed and accuracy (both COCO and MPII models are slower and less accurate).



## Speed Up and Memory Reduction
Some speed tips to highly maximize the OpenPose speed, but keep in mind the accuracy trade-off:

    1. Reduce the `--net_resolution` (e.g., to 320x176) (lower accuracy). Note: For maximum accuracy, follow [doc/quick_start.md#maximum-accuracy-configuration](./quick_start.md#maximum-accuracy-configuration).
    2. For face, reduce the `--face_net_resolution`. The resolution 320x320 usually works pretty decently.
    3. Points 1-2 will also reduce the GPU memory usage (or RAM memory for CPU version).
    4. Use the `BODY_25` model for maximum speed. Use `MPI_4_layers` model for minimum GPU memory usage (but lower accuracy, speed, and number of parts).



### CPU Version
The CPU version runs at about 0.3 FPS on the COCO model, and at about 0.1 FPS (i.e., about 15 sec / frame) on the default BODY_25 model. Switch to COCO model and/or reduce the `net_resolution` as indicated above. Contradictory fact: BODY_25 model is about 5x slower than COCO on CPU-only version, but it is about 40% faster on GPU version.
